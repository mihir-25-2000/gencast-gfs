# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# Licensed under the Apache Licence Version 2.0.

import datetime
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

CF_NAME_SFC = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2t": "2m_temperature",
    "sst": "sea_surface_temperature",
    "lsm": "land_sea_mask",
    "msl": "mean_sea_level_pressure",
    "tp": "total_precipitation_12hr",
    "z": "geopotential_at_surface",
}

CF_NAME_PL = {
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "z": "geopotential",
}


def create_training_xarray(
    *,
    fields_sfc: Any,
    fields_pl: Any,
    lagged,
    start_date,
    hour_steps,
    lead_time,
    forcing_variables,
    constants,
    timer,
):
    """Convert GRIB input fields into xarray Dataset expected by GenCast."""
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]

    all_datetimes = [start_date + time_delta for time_delta in time_deltas]

    with timer("Converting GRIB to xarray"):
        lat = fields_sfc[0].metadata("distinctLatitudes")
        lon = fields_sfc[0].metadata("distinctLongitudes")

        fields_sfc = fields_sfc.order_by("param", "valid_datetime")
        sfc = defaultdict(list)
        given_datetimes = set()
        for field in fields_sfc:
            given_datetimes.add(field.metadata("valid_datetime"))
            sfc[field.metadata("param")].append(field)

        fields_pl = fields_pl.order_by("param", "valid_datetime", "level")
        pl = defaultdict(list)
        levels = set()
        given_datetimes = set()
        for field in fields_pl:
            given_datetimes.add(field.metadata("valid_datetime"))
            pl[field.metadata("param")].append(field)
            levels.add(field.metadata("level"))

        data_vars = {}

        for param, fields in sfc.items():
            if param not in CF_NAME_SFC:
                LOG.warning("Skipping unknown surface param '%s'", param)
                continue

            if param in ("z", "lsm"):
                data_vars[CF_NAME_SFC[param]] = (["lat", "lon"], fields[0].to_numpy())
                continue

            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(lat),
                len(lon),
            )

            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[CF_NAME_SFC[param]] = (["batch", "time", "lat", "lon"], data)

        # Ensure SST exists even if the input source does not provide it
        if CF_NAME_SFC["sst"] not in data_vars:
            t2m_key = CF_NAME_SFC["2t"]
            if t2m_key not in data_vars:
                LOG.warning("Sea surface temperature missing and 2m_temperature unavailable; filling with zeros")
                data_vars[CF_NAME_SFC["sst"]] = (
                    ["batch", "time", "lat", "lon"],
                    np.zeros((1, len(all_datetimes), len(lat), len(lon)), dtype=np.float32),
                )
            else:
                LOG.info("Filling sea surface temperature from 2m_temperature and zeroing over land")
                sst_data = np.array(data_vars[t2m_key][1], copy=True)

                lsm_key = CF_NAME_SFC["lsm"]
                lsm = data_vars.get(lsm_key)
                if lsm is not None:
                    land_mask = np.array(lsm[1], copy=False) >= 0.5
                    sst_data = np.where(land_mask[None, None, :, :], 0.0, sst_data)

                data_vars[CF_NAME_SFC["sst"]] = (
                    ["batch", "time", "lat", "lon"],
                    sst_data,
                )

        for param, fields in pl.items():
            if param not in CF_NAME_PL:
                LOG.warning("Skipping unknown pressure-level param '%s'", param)
                continue

            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(levels),
                len(lat),
                len(lon),
            )
            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[CF_NAME_PL[param]] = (
                ["batch", "time", "level", "lat", "lon"],
                data,
            )

        training_xarray = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=lon,
                lat=lat,
                time=time_deltas,
                datetime=(
                    ("batch", "time"),
                    [all_datetimes],
                ),
                level=sorted(levels),
            ),
        )

    with timer("Reindexing"):
        training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values), copy=False)

    if constants:
        x = xr.load_dataset(constants)
        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas

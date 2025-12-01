# Copyright 2024 ECMWF. Licensed under Apache 2.0.
"""Post-processing helpers for GenCast outputs in ai-models-gfs."""

import logging
from collections import defaultdict
from typing import Mapping

import numpy as np

from .convert import GRIB_TO_CF, GRIB_TO_XARRAY_PL, GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)

ACCUMULATION_VALUES = defaultdict(lambda: defaultdict(lambda: 0))


def accumulate(values, param: str, ensemble_number: int):
    """Accumulate values for a given parameter and ensemble member."""
    ACCUMULATION_VALUES[param][ensemble_number] += values
    return ACCUMULATION_VALUES[param][ensemble_number]


def save_output_xarray(
    *,
    output,
    target_variables,
    write,
    all_fields,
    ordering,
    time,
    hour_steps,
    num_ensemble_members,
    lagged,
    oper_fcst,
    member_numbers,
):
    """Convert GenCast xarray outputs back into GRIB via ai-models-gfs write()."""
    # Earthkit's order_by raises if a param/level is not present in the ordering
    # list. GFS inputs can contain extra fields (e.g., 100u/100v) that GenCast
    # does not use, so we extend the ordering to cover them to keep sorting happy.
    ordering_extended = list(ordering)
    for fs in all_fields:
        param = fs.metadata("shortName")
        level = fs.metadata("levelist", default=None)
        key = f"{param}{level}" if level is not None else param
        if key not in ordering_extended:
            ordering_extended.append(key)

    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering_extended,
        remapping={"param_level": "{param}{levelist}"},
    )

    for fs in all_fields[: len(all_fields) // len(lagged)]:
        param, level = fs.metadata("shortName"), fs.metadata("levelist", default=None)
        for i in range(num_ensemble_members):
            ensemble_member = member_numbers[i]
            time_idx = 0

            if level is not None:
                param = GRIB_TO_XARRAY_PL.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time_idx).sel(level=level).isel(sample=i).data_vars[param].values
            else:
                param = GRIB_TO_CF.get(param, param)
                param = GRIB_TO_XARRAY_SFC.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time_idx).isel(sample=i).data_vars[param].values

            values = np.flipud(values.reshape(fs.shape))

            extra_write_kwargs = {} if oper_fcst else dict(number=ensemble_member)

            if param == "total_precipitation_12hr":
                values = accumulate(values, param, ensemble_member)
                write(
                    values,
                    template=fs,
                    stepType="accum",
                    startStep=0,
                    endStep=time * hour_steps,
                    **extra_write_kwargs,
                )
            else:
                write(
                    values,
                    template=fs,
                    step=time * hour_steps,
                    **extra_write_kwargs,
                )

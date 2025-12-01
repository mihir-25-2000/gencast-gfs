# Mapping helpers for aligning GRIB/CF names with GenCast expectations.

GRIB_TO_XARRAY_SFC = {
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    # GenCast target is 12h accumulation; map GRIB tp accordingly so it is written.
    "tp": "total_precipitation_12hr",
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
    "latitude": "lat",
    "longitude": "lon",
    "valid_time": "datetime",
}

GRIB_TO_XARRAY_PL = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
    "isobaricInhPa": "level",
    "latitude": "lat",
    "longitude": "lon",
    "valid_time": "datetime",
}

GRIB_TO_CF = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
}

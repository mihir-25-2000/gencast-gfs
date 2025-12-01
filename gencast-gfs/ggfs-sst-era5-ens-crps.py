#!/usr/bin/env python
import os
# Avoid loading ~/.local packages with incompatible numpy
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("XR_DISABLE_CHUNK_MANAGER_REFACTOR", "1")

import glob
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
import matplotlib.pyplot as plt
from xskillscore import crps_ensemble

# --------------------
# CONFIG
# --------------------
gencast_dir = "/Datastorage/mihir.more/gencast-gfs-run-20251110"
gencast_sst_dir = "/Datastorage/mihir.more/gencast-gfs-run-20251110-sst"
gencast_era5_dir = "/home/mihir.more/myenv/gencast-era5-20251110"
ens_path = "/Datastorage/mihir.more/ens-data-20251110/ens_tp_20251110_00_pf.grib2"
imerg_dir = "/Datastorage/mihir.more/IMERGHHL-20251110to20251123"

OUT_FIG = "/home/mihir.more/myenv/ggfs-sst-gera5_vs_ens_vs_imerg_crps.png"



# Use mm (True) or meters (False) for CRPS units
CRPS_IN_MM = True


def load_gencast_tp_inc(run_dir: str, pattern: str = "gc_gfs_*.grib2"):
    paths = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No GenCast files found in {run_dir} with pattern {pattern}")

    ds_list = []
    for p in paths:
        ds = xr.open_dataset(
            p,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
        )
        if "number" in ds.data_vars:
            ds = ds.drop_vars("number")
        if "number" in ds.coords and "number" not in ds.dims:
            ds = ds.reset_coords("number", drop=True)
        if "number" in ds.indexes:
            ds = ds.reset_index("number", drop=True)
        if "valid_time" in ds:
            ds = ds.drop_vars("valid_time")
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim="step").sortby("step")
    if "number" not in ds.dims:
        ds = ds.expand_dims("number")
    if "valid_time" not in ds.coords and "valid_time" not in ds.data_vars:
        if "time" in ds.coords:
            ds = ds.assign_coords(valid_time=ds["time"] + ds["step"])
        else:
            raise ValueError("GenCast dataset has no 'valid_time' and no 'time' to build it from.")

    tp = ds["tp"]
    if CRPS_IN_MM:
        tp = tp * 1000.0  # m -> mm

    tp_inc = tp.diff("step", label="upper")
    new_step = ds["step"].isel(step=slice(1, None))
    new_vtime = ds["valid_time"].isel(step=slice(1, None))
    lead_hours = (new_step.values / np.timedelta64(1, "h")).astype("float32")

    tp_inc = tp_inc.assign_coords(
        step=new_step.values,
        valid_time=("step", new_vtime.values),
        lead_time=("step", lead_hours),
    )
    tp_inc = tp_inc.transpose("number", "step", "latitude", "longitude", ...)
    tp_inc = tp_inc.swap_dims({"step": "valid_time"})
    return tp_inc


# --------------------
# 1) Load GenCast forecasts (baseline, SST-injected, ERA5-initialized)
# --------------------
gc_tp_inc = load_gencast_tp_inc(gencast_dir, pattern="gc_gfs_*.grib2")
gc_sst_tp_inc = load_gencast_tp_inc(gencast_sst_dir)
gc_era5_tp_inc = load_gencast_tp_inc(gencast_era5_dir, pattern="gencast-025-oper-*.grib")

gc_era5_tp_inc = load_gencast_tp_inc(
    "/home/mihir.more/myenv/gencast-era5-20251110",
    pattern="gencast-025-oper-*.grib",
)

# shift 6h so valid_time lines up with 12Z,24Z,... like GFS/ENS/IMERG
gc_era5_tp_inc = gc_era5_tp_inc.assign_coords(
    valid_time=gc_era5_tp_inc.valid_time + np.timedelta64(6, "h")
)


# --------------------
# 2) Load ENS forecast
# --------------------
ens = xr.open_dataset(
    ens_path,
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
)

if "number" in ens.data_vars:
    ens = ens.drop_vars("number")
if "number" in ens.coords and "number" not in ens.dims:
    ens = ens.reset_coords("number", drop=True)
if "number" in ens.indexes:
    ens = ens.reset_index("number", drop=True)
if "number" not in ens.dims:
    ens = ens.expand_dims("number")

if "time" in ens.dims:
    if ens.sizes["time"] != 1:
        raise ValueError(f"ENS dataset has time dimension size {ens.sizes['time']}, expected 1 init time.")
    init_time_ens = ens["time"].isel(time=0)
    ens = ens.squeeze("time", drop=True)
else:
    if "time" in ens.coords:
        init_time_ens = ens["time"]
    else:
        raise ValueError("No 'time' dimension or coord in ENS dataset.")

ens_tp = ens["tp"]
if CRPS_IN_MM:
    ens_tp = ens_tp * 1000.0  # m -> mm

if "valid_time" not in ens.coords and "valid_time" not in ens.data_vars:
    ens = ens.assign_coords(valid_time=init_time_ens + ens["step"])

ens_tp_inc = ens_tp.diff("step", label="upper")
ens_new_step = ens["step"].isel(step=slice(1, None))
ens_new_vtime = ens["valid_time"].isel(step=slice(1, None))
ens_lead_hours = (ens_new_step.values / np.timedelta64(1, "h")).astype("float32")

ens_tp_inc = ens_tp_inc.assign_coords(
    step=ens_new_step.values,
    valid_time=("step", ens_new_vtime.values),
    lead_time=("step", ens_lead_hours),
)
ens_tp_inc = ens_tp_inc.transpose("number", "step", "latitude", "longitude", ...)
ens_tp_inc = ens_tp_inc.swap_dims({"step": "valid_time"})

# --------------------
# 3) Determine common time window
# --------------------
vt_min = max(
    gc_tp_inc["valid_time"].min().item(),
    gc_sst_tp_inc["valid_time"].min().item(),
    gc_era5_tp_inc["valid_time"].min().item(),
    ens_tp_inc["valid_time"].min().item(),
)
vt_max = min(
    gc_tp_inc["valid_time"].max().item(),
    gc_sst_tp_inc["valid_time"].max().item(),
    gc_era5_tp_inc["valid_time"].max().item(),
    ens_tp_inc["valid_time"].max().item(),
)

# --------------------
# 4) Load IMERG and make 12h totals
# --------------------
imerg_files = sorted(glob.glob(os.path.join(imerg_dir, "*.HDF5")))
if not imerg_files:
    raise FileNotFoundError(f"No IMERG files found in {imerg_dir}")

imerg_dsets = [
    xr.open_dataset(f, engine="h5netcdf", group="Grid", chunks={}) for f in imerg_files
]
imerg = xr.concat(imerg_dsets, dim="time")

precip_var_name = None
for cand in ("precipitation", "precipitationCal"):
    if cand in imerg.data_vars:
        precip_var_name = cand
        break
if precip_var_name is None:
    raise KeyError("No precipitation variable found; tried 'precipitation' and 'precipitationCal'")

imerg_30min = imerg[precip_var_name] * 0.5  # mm/hr -> mm per 30 min

try:
    t_idx = imerg_30min.indexes["time"].to_datetimeindex()
except Exception:
    t_idx = pd.to_datetime(imerg_30min.time.values)
imerg_30min = imerg_30min.assign_coords(time=t_idx)

t_start = pd.Timestamp(vt_min) - pd.Timedelta("12h")
t_stop = pd.Timestamp(vt_max)
imerg_30min = imerg_30min.sel(time=slice(t_start, t_stop))

imerg_12h = imerg_30min.resample(time="12h", label="right", closed="right").sum()

# --------------------
# 5) Regrid IMERG → forecast grid (0.25°)
# --------------------
target_lat = gc_tp_inc["latitude"]
target_lon = gc_tp_inc["longitude"]

regridder = xe.Regridder(
    imerg_12h,
    xr.Dataset({"lat": target_lat, "lon": target_lon}),
    method="bilinear",
    periodic=True,
    reuse_weights=False,
)
imerg_rg = regridder(imerg_12h)

name_map = {}
for src_name, dst_name in (("lat", "latitude"), ("lon", "longitude")):
    if src_name in imerg_rg.dims or src_name in imerg_rg.coords:
        name_map[src_name] = dst_name
if name_map:
    imerg_rg = imerg_rg.rename(name_map)

if not CRPS_IN_MM:
    gc_tp_inc = gc_tp_inc / 1000.0
    gc_sst_tp_inc = gc_sst_tp_inc / 1000.0
    gc_era5_tp_inc = gc_era5_tp_inc / 1000.0
    ens_tp_inc = ens_tp_inc / 1000.0
    imerg_rg = imerg_rg / 1000.0

imerg_rg = imerg_rg.assign_coords(valid_time=("time", imerg_rg["time"].values))
imerg_rg = imerg_rg.swap_dims({"time": "valid_time"})
imerg_rg = imerg_rg.rename("tp")

# --------------------
# 6) Align each forecast with IMERG on valid_time
# --------------------
forecasts = {
    "GenCast-GFS": gc_tp_inc,
    "GenCast-GFS-SST": gc_sst_tp_inc,
    "GenCast-ERA5": gc_era5_tp_inc,
    "ENS": ens_tp_inc,
}

aligned = {}
imerg_aligned = {}
for k, fc in forecasts.items():
    aligned[k], imerg_aligned[k] = xr.align(
        fc,
        imerg_rg,
        join="inner",
        exclude=["number"],
    )

# sanity-check overlap and force dense chunks to avoid Dask auto-chunk issues
for k in list(aligned):
    vt_len = aligned[k].sizes.get("valid_time", 0)
    if vt_len == 0:
        raise RuntimeError(f"No overlapping valid_time for {k}; check vt_min/vt_max and filenames.")
    aligned[k] = aligned[k].transpose("number", "valid_time", "latitude", "longitude").chunk(
        {"number": -1, "valid_time": -1, "latitude": -1, "longitude": -1}
    ).load()
    imerg_aligned[k] = imerg_aligned[k].transpose("valid_time", "latitude", "longitude").chunk(
        {"valid_time": -1, "latitude": -1, "longitude": -1}
    ).load()

# --------------------
# 7) Compute CRPS for each forecast vs IMERG
# --------------------
crps_map = {
    k: crps_ensemble(imerg_aligned[k], aligned[k], member_dim="number", dim=[])
    for k in forecasts
}
crps_mean = {k: v.mean(dim=["latitude", "longitude"]) for k, v in crps_map.items()}
lead = {
    k: aligned[k]["lead_time"].sel(valid_time=crps_mean[k]["valid_time"]).values
    for k in forecasts
}

# --------------------
# 8) Plot comparison
# --------------------
plt.figure(figsize=(8, 4))
plt.plot(lead["GenCast-GFS"], crps_mean["GenCast-GFS"], marker="o", label="GenCast-GFS vs IMERG")
plt.plot(
    lead["GenCast-GFS-SST"],
    crps_mean["GenCast-GFS-SST"],
    marker="^",
    label="GenCast-GFS-SST vs IMERG",
)
plt.plot(
    lead["GenCast-ERA5"],
    crps_mean["GenCast-ERA5"],
    marker="D",
    label="GenCast-ERA5 vs IMERG",
)
plt.plot(lead["ENS"], crps_mean["ENS"], marker="s", label="ENS vs IMERG")
plt.xlabel("Lead time (h)")
plt.ylabel(f"CRPS ({'mm' if CRPS_IN_MM else 'm'})")
plt.title("CRPS: GenCast-GFS (+SST), GenCast-ERA5, ENS vs IMERG (12h TP)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
plt.show()

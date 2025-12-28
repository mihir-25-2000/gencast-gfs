#!/usr/bin/env python
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("XR_DISABLE_CHUNK_MANAGER_REFACTOR", "1")

import glob
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
from xskillscore import crps_ensemble
import geopandas as gpd
import regionmask

# Paths
GENCAST_ROOT = "/Datastorage/mihir.more/gencast-ggfs-masked-run-june-2025"
ENS_ROOT = "/Datastorage/mihir.more/ens_data_june2025"
IMERG_DIR = "/Datastorage/mihir.more/IMERG-JUNE-2025"
SHAPEFILE_PATH = "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/Central_Northeast.shp"

OUT_FIG = "/home/mihir.more/myenv/crps_gencast_vs_ens_june2025_Central_Northeast.png"
OUT_CSV = "/home/mihir.more/myenv/crps_gencast_vs_ens_june2025_Central_Northeast.csv"

# Settings
CRPS_IN_MM = True      # convert m -> mm for CRPS units
MAX_INITS = 15         # use first 15 initializations
INIT_HOUR = "00"       # expected init hour for ENS filenames
REGION_INDEX = 0       # pick polygon index from shapefile


def discover_inits():
    """Find matching GenCast + ENS init dates; keep first MAX_INITS."""
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(GENCAST_ROOT, "gencast-gfs-run-*-masked-sst-5dayslead"))
        if os.path.isdir(d)
    )
    pairs = []
    for rd in run_dirs:
        parts = os.path.basename(rd).split("-")
        if len(parts) < 5:
            continue
        date_str = parts[3]  # e.g., 20250601
        ens_file = os.path.join(ENS_ROOT, date_str, f"ens_tp_{date_str}_{INIT_HOUR}_pf.grib2")
        if os.path.exists(ens_file):
            pairs.append((date_str, rd, ens_file))
        if len(pairs) >= MAX_INITS:
            break
    if not pairs:
        raise FileNotFoundError("No matching GenCast/ENS initializations found.")
    return pairs


def load_gencast_tp_inc(run_dir: str):
    """Load GenCast TP, skip corrupted GRIBs, trim ensemble members to a common size, and make 12h increments."""
    paths = sorted(glob.glob(os.path.join(run_dir, "gc_gfs_*.grib2")))
    if not paths:
        raise FileNotFoundError(f"No GenCast files in {run_dir}")

    dsets = []
    num_sizes = []
    for p in paths:
        try:
            ds = xr.open_dataset(
                p,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
            )
        except Exception as exc:
            print(f"Skipping {p}: {exc}")
            continue

        if "number" in ds.data_vars:
            ds = ds.drop_vars("number")
        if "number" in ds.coords and "number" not in ds.dims:
            ds = ds.reset_coords("number", drop=True)
        if "number" in ds.indexes:
            ds = ds.reset_index("number", drop=True)
        if "valid_time" in ds:
            ds = ds.drop_vars("valid_time")

        num_sizes.append(ds.sizes.get("number", 1))
        dsets.append(ds)

    if not dsets:
        raise RuntimeError(f"All GenCast files in {run_dir} were unreadable.")

    min_num = max(1, min(num_sizes))
    trimmed = []
    for ds in dsets:
        if "number" not in ds.dims:
            ds = ds.expand_dims("number")
        elif ds.sizes["number"] > min_num:
            ds = ds.isel(number=slice(0, min_num))
        trimmed.append(ds)

    ds = xr.concat(trimmed, dim="step").sortby("step")
    if "valid_time" not in ds.coords and "valid_time" not in ds.data_vars:
        if "time" in ds.coords:
            ds = ds.assign_coords(valid_time=ds["time"] + ds["step"])
        else:
            raise ValueError("GenCast dataset missing 'time' and 'valid_time'.")

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


def load_ens_tp_inc(path: str):
    ens = xr.open_dataset(
        path,
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
            raise ValueError(f"ENS has {ens.sizes['time']} init times, expected 1.")
        init_time = ens["time"].isel(time=0)
        ens = ens.squeeze("time", drop=True)
    else:
        init_time = ens["time"] if "time" in ens.coords else None
        if init_time is None:
            raise ValueError("ENS dataset missing init 'time'.")

    tp = ens["tp"]
    if CRPS_IN_MM:
        tp = tp * 1000.0

    if "valid_time" not in ens.coords and "valid_time" not in ens.data_vars:
        ens = ens.assign_coords(valid_time=init_time + ens["step"])

    tp_inc = tp.diff("step", label="upper")
    new_step = ens["step"].isel(step=slice(1, None))
    new_vtime = ens["valid_time"].isel(step=slice(1, None))
    lead_hours = (new_step.values / np.timedelta64(1, "h")).astype("float32")

    tp_inc = tp_inc.assign_coords(
        step=new_step.values,
        valid_time=("step", new_vtime.values),
        lead_time=("step", lead_hours),
    )
    tp_inc = tp_inc.transpose("number", "step", "latitude", "longitude", ...)
    tp_inc = tp_inc.swap_dims({"step": "valid_time"})
    return tp_inc


def load_imerg_regridded(imerg_dir: str, target_lat, target_lon):
    imerg_files = sorted(glob.glob(os.path.join(imerg_dir, "*.HDF5")))
    if not imerg_files:
        raise FileNotFoundError(f"No IMERG files in {imerg_dir}")
    imerg_dsets = [xr.open_dataset(f, engine="h5netcdf", group="Grid", chunks={}) for f in imerg_files]
    imerg = xr.concat(imerg_dsets, dim="time")

    precip_var = None
    for cand in ("precipitation", "precipitationCal"):
        if cand in imerg.data_vars:
            precip_var = cand
            break
    if precip_var is None:
        raise KeyError("Could not find 'precipitation' or 'precipitationCal' in IMERG.")

    # mm/hr -> mm per 30 min
    imerg_30min = imerg[precip_var] * 0.5
    try:
        t_idx = imerg_30min.indexes["time"].to_datetimeindex()
    except Exception:
        t_idx = pd.to_datetime(imerg_30min.time.values)
    imerg_30min = imerg_30min.assign_coords(time=t_idx)

    imerg_12h = imerg_30min.resample(time="12h", label="right", closed="right").sum()

    regridder = xe.Regridder(
        imerg_12h,
        xr.Dataset({"lat": target_lat, "lon": target_lon}),
        method="bilinear",
        periodic=True,
        reuse_weights=False,
    )
    imerg_rg = regridder(imerg_12h)

    rename_map = {}
    for src, dst in (("lat", "latitude"), ("lon", "longitude")):
        if src in imerg_rg.dims or src in imerg_rg.coords:
            rename_map[src] = dst
    if rename_map:
        imerg_rg = imerg_rg.rename(rename_map)

    if not CRPS_IN_MM:
        imerg_rg = imerg_rg / 1000.0

    imerg_rg = imerg_rg.assign_coords(valid_time=("time", imerg_rg["time"].values))
    imerg_rg = imerg_rg.swap_dims({"time": "valid_time"})
    imerg_rg = imerg_rg.rename("tp")
    return imerg_rg


def build_region_mask(lat, lon, shp_path=SHAPEFILE_PATH, region_index=REGION_INDEX):
    """Create region mask on target grid and return mask + label."""
    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    gdf_region = gdf.iloc[[region_index]]
    region_label = gdf_region.iloc[0].get("NAME", "Region")
    grid = xr.Dataset({"lat": lat, "lon": lon})
    wrap = 360 if float(grid["lon"].max()) > 180 else None
    mask = regionmask.mask_geopandas(gdf_region, grid, wrap_lon=wrap)
    if "lat" in mask.dims:
        mask = mask.rename({"lat": "latitude", "lon": "longitude"})
    return mask, region_label


def main():
    init_info = discover_inits()

    # Load first run to establish target grid for IMERG regridding and mask
    sample_gc = load_gencast_tp_inc(init_info[0][1])
    imerg_rg = load_imerg_regridded(IMERG_DIR, sample_gc["latitude"], sample_gc["longitude"])
    region_mask, region_label = build_region_mask(sample_gc["latitude"], sample_gc["longitude"])

    # Apply region mask to IMERG
    imerg_rg = imerg_rg.where(region_mask == 0)

    records = []
    for date_str, gc_dir, ens_file in init_info:
        try:
            gc = load_gencast_tp_inc(gc_dir).where(region_mask == 0)
            ens = load_ens_tp_inc(ens_file).where(region_mask == 0)
        except Exception as exc:
            print(f"Skipping init {date_str}: {exc}")
            continue

        gc_aligned, ens_aligned, imerg_aligned = xr.align(
            gc, ens, imerg_rg, join="inner", exclude=["number"]
        )
        if gc_aligned.sizes.get("valid_time", 0) == 0:
            print(f"Skipping init {date_str}: no overlapping valid_time")
            continue

        crps_gc = crps_ensemble(imerg_aligned, gc_aligned, member_dim="number", dim=[])
        crps_ens = crps_ensemble(imerg_aligned, ens_aligned, member_dim="number", dim=[])

        crps_gc_mean = crps_gc.mean(dim=["latitude", "longitude"])
        crps_ens_mean = crps_ens.mean(dim=["latitude", "longitude"])

        lead = gc_aligned["lead_time"].sel(valid_time=crps_gc_mean["valid_time"])
        for lt, gc_val, ens_val in zip(
            lead.values,
            crps_gc_mean.values,
            crps_ens_mean.sel(valid_time=crps_gc_mean.valid_time).values,
        ):
            records.append(
                {
                    "init": pd.Timestamp(date_str + "T00:00:00"),
                    "lead_time_h": float(lt),
                    "crps_gencast": float(gc_val),
                    "crps_ens": float(ens_val),
                    "region": region_label,
                }
            )

    if not records:
        raise RuntimeError("No CRPS records computed; check inputs/logs.")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    agg = df.groupby("lead_time_h")[["crps_gencast", "crps_ens"]].mean().reset_index().sort_values("lead_time_h")

    plt.figure(figsize=(8, 4))
    plt.plot(agg["lead_time_h"], agg["crps_gencast"], marker="^", label="GenCast-GFS masked vs IMERG")
    plt.plot(agg["lead_time_h"], agg["crps_ens"], marker="s", label="ENS vs IMERG")
    plt.xlabel("Lead time (h)")
    plt.ylabel(f"CRPS ({'mm' if CRPS_IN_MM else 'm'})")
    plt.title(f"CRPS comparison (June 2025, 12h TP, 0–120h lead) — Central_Northeast")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.show()

    print(f"Saved per-init CRPS to {OUT_CSV}")
    print(f"Saved plot to {OUT_FIG}")


if __name__ == "__main__":
    main()

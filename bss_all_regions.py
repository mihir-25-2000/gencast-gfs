#!/usr/bin/env python
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("XR_DISABLE_CHUNK_MANAGER_REFACTOR", "1")

import glob
import re
from collections import Counter
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import geopandas as gpd
import regionmask
import dask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
GENCAST_ROOT = "/Datastorage/mihir.more/tp-gencast-ggfs-masked-run-june-2025/"
ENS_ROOT     = "/Datastorage/mihir.more/ens_data_june2025"
IMERG_DIR    = "/Datastorage/mihir.more/IMERG-JUNE-2025"
CLIMATOLOGY_PATH = os.environ.get(
    "WB_CLIM_PATH",
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr",
)

SHAPEFILE_PATHS = [
    "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/Northeast.shp",
    "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/South_Peninsular.shp",
    "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/Central_Northeast.shp",
    "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/West_Central.shp",
    "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/Northwest.shp",
    "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/Hilly_Regions.shp",
]

OUT_DIR = "/home/mihir.more/myenv"


# ----------------------------
# Settings
# ----------------------------
PRECIP_IN_MM = True
MAX_INITS = 15
INIT_HOUR = "00"

# Fix A bbox
BBOX_LAT_MIN, BBOX_LAT_MAX = 5.0, 38.0
BBOX_LON_MIN, BBOX_LON_MAX = 65.0, 100.0

REGION_INDEX = 0  # set None to pick largest polygon

EVENT_THRESHOLD_MM = 10.0
SHIFT_IMERG_TIME_BY_30MIN = False
SCAN_THRESHOLDS = (0.5, 1, 2, 5, 10, 20, 30, 50)


# ----------------------------
# Helpers
# ----------------------------
def _to_mm(da: xr.DataArray) -> xr.DataArray:
    return da * 1000.0 if PRECIP_IN_MM else da

def _print_stats(name: str, da: xr.DataArray):
    mn = float(da.min(skipna=True).compute() if hasattr(da.data, "compute") else da.min(skipna=True))
    me = float(da.mean(skipna=True).compute() if hasattr(da.data, "compute") else da.mean(skipna=True))
    mx = float(da.max(skipna=True).compute() if hasattr(da.data, "compute") else da.max(skipna=True))
    print(f"{name}: min={mn:.4g}, mean={me:.4g}, max={mx:.4g}")

def _normalize_bbox_lon_for_grid(grid_lon: xr.DataArray, lon_min: float, lon_max: float):
    gmax = float(grid_lon.max().compute() if hasattr(grid_lon.data, "compute") else grid_lon.max())
    if gmax > 180 and lon_min < 0:
        return lon_min % 360, lon_max % 360
    if gmax <= 180 and lon_min > 180:
        def to_pm180(x): return ((x + 180) % 360) - 180
        return to_pm180(lon_min), to_pm180(lon_max)
    return lon_min, lon_max

def subset_bbox(obj, lat_name="latitude", lon_name="longitude"):
    if lat_name not in obj.coords or lon_name not in obj.coords:
        return obj
    lat = obj[lat_name]
    lon = obj[lon_name]
    lon_min2, lon_max2 = _normalize_bbox_lon_for_grid(lon, BBOX_LON_MIN, BBOX_LON_MAX)

    lat_asc = bool((lat.values[-1] - lat.values[0]) > 0)
    lon_asc = bool((lon.values[-1] - lon.values[0]) > 0)

    lat_slice = slice(BBOX_LAT_MIN, BBOX_LAT_MAX) if lat_asc else slice(BBOX_LAT_MAX, BBOX_LAT_MIN)
    lon_slice = slice(lon_min2, lon_max2) if lon_asc else slice(lon_max2, lon_min2)
    return obj.sel({lat_name: lat_slice, lon_name: lon_slice})

def scan_event_rates(da_tp_12h_mm, thresholds=SCAN_THRESHOLDS):
    print("Event-rate scan (fraction of grid×times exceeding threshold):")
    for th in thresholds:
        rate = (da_tp_12h_mm > th).mean(skipna=True)
        rate = float(rate.compute()) if hasattr(rate.data, "compute") else float(rate)
        print(f"  > {th:>5} mm : {rate:.6f}")

def discover_inits():
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(GENCAST_ROOT, "gencast-gfs-run-*-masked-sst-5dayslead"))
        if os.path.isdir(d)
    )
    pairs = []
    for rd in run_dirs:
        parts = os.path.basename(rd).split("-")
        if len(parts) < 5:
            continue
        date_str = parts[3]
        ens_file = os.path.join(ENS_ROOT, date_str, f"ens_tp_{date_str}_{INIT_HOUR}_pf.grib2")
        if os.path.exists(ens_file):
            pairs.append((date_str, rd, ens_file))
        if len(pairs) >= MAX_INITS:
            break
    if not pairs:
        raise FileNotFoundError("No matching GenCast/ENS initializations found.")
    return pairs

def _is_cumulative(tp: xr.DataArray, step_dim: str = "step", eps: float = 1e-6) -> bool:
    if step_dim not in tp.dims or tp.sizes.get(step_dim, 0) < 2:
        return False
    d = tp.diff(step_dim)
    frac_nonneg = float((d >= -eps).mean(skipna=True))
    return frac_nonneg >= 0.95

def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text or "region"

def _unique_paths(paths):
    seen = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique

def _output_paths(shp_path: str):
    base = os.path.splitext(os.path.basename(shp_path))[0]
    tag = _slugify(base)
    fig_bs = os.path.join(OUT_DIR, f"brier_score_{tag}.png")
    fig_bss = os.path.join(OUT_DIR, f"brier_skill_{tag}.png")
    csv = os.path.join(OUT_DIR, f"bss_{tag}.csv")
    return fig_bs, fig_bss, csv


# ----------------------------
# Region mask
# ----------------------------
def build_region_mask(lat, lon, shp_path, region_index=REGION_INDEX):
    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)

    if region_index is None:
        gdf_area = gdf.to_crs(epsg=6933)
        idx = gdf_area.geometry.area.idxmax()
        gdf_region = gdf.loc[[idx]]
    else:
        gdf_region = gdf.iloc[[region_index]]

    region_label = None
    for cand in ["NAME", "Name", "region", "Region", "STATE", "DISTRICT"]:
        if cand in gdf_region.columns:
            region_label = str(gdf_region.iloc[0][cand])
            break
    if region_label is None:
        region_label = os.path.splitext(os.path.basename(shp_path))[0]

    grid = xr.Dataset({"lat": lat, "lon": lon})
    wrap = 360 if float(grid["lon"].max()) > 180 else None
    mask = regionmask.mask_geopandas(gdf_region, grid, wrap_lon=wrap)
    if "lat" in mask.dims:
        mask = mask.rename({"lat": "latitude", "lon": "longitude"})
    return mask, region_label


# ----------------------------
# Forecast loaders
# ----------------------------
def load_gencast_tp_inc(run_dir: str) -> xr.DataArray:
    paths = sorted(glob.glob(os.path.join(run_dir, "gc_gfs_*.grib2")))
    if not paths:
        raise FileNotFoundError(f"No GenCast files in {run_dir}")

    dsets, sizes = [], []
    for p in paths:
        try:
            ds = xr.open_dataset(
                p, engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
            )
        except Exception as exc:
            print(f"Skipping {p}: {exc}")
            continue

        if "number" in ds.data_vars:
            ds = ds.drop_vars("number")
        if "valid_time" in ds:
            ds = ds.drop_vars("valid_time")

        sizes.append(ds.sizes.get("number", None))
        dsets.append(ds)

    if not dsets:
        raise RuntimeError(f"All GenCast files in {run_dir} were unreadable.")

    sizes_nn = [int(s) for s in sizes if s is not None]
    target_num = Counter(sizes_nn).most_common(1)[0][0] if sizes_nn else 1
    target_num = max(1, int(target_num))

    trimmed = []
    for ds in dsets:
        if "number" not in ds.dims:
            if target_num > 1:
                continue
            ds = ds.expand_dims("number")
        else:
            if ds.sizes["number"] < target_num:
                continue
            if ds.sizes["number"] > target_num:
                ds = ds.isel(number=slice(0, target_num))
        trimmed.append(ds)

    if not trimmed:
        raise RuntimeError(f"No GenCast files left after ensemble-size filtering in {run_dir}.")

    ds = xr.concat(trimmed, dim="step").sortby("step")

    if "valid_time" not in ds.coords and "valid_time" not in ds.data_vars:
        if "time" in ds.coords:
            ds = ds.assign_coords(valid_time=ds["time"] + ds["step"])
        else:
            raise ValueError("GenCast dataset missing 'time' and 'valid_time'.")

    tp = _to_mm(ds["tp"])

    if _is_cumulative(tp, "step"):
        tp_inc = tp.diff("step", label="upper")
        new_step = ds["step"].isel(step=slice(1, None))
        new_vtime = ds["valid_time"].isel(step=slice(1, None))
    else:
        tp_inc = tp
        new_step = ds["step"]
        new_vtime = ds["valid_time"]

    tp_inc = tp_inc.clip(min=0)
    lead_hours = (new_step.values / np.timedelta64(1, "h")).astype("float32")

    tp_inc = tp_inc.assign_coords(
        step=new_step.values,
        valid_time=("step", new_vtime.values),
        lead_time=("step", lead_hours),
    ).transpose("number", "step", "latitude", "longitude", ...)

    return tp_inc.swap_dims({"step": "valid_time"})

def load_ens_tp_inc(path: str) -> xr.DataArray:
    ens = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
    )

    if "number" in ens.data_vars:
        ens = ens.drop_vars("number")
    if "number" not in ens.dims:
        ens = ens.expand_dims("number")

    if "time" in ens.dims:
        if ens.sizes["time"] != 1:
            raise ValueError(f"ENS has {ens.sizes['time']} init times, expected 1.")
        init_time = ens["time"].isel(time=0)
        ens = ens.squeeze("time", drop=True)
    else:
        if "time" in ens.coords:
            init_time = ens["time"]
        else:
            raise ValueError("ENS dataset missing init 'time'.")

    tp = _to_mm(ens["tp"])

    if "valid_time" not in ens.coords and "valid_time" not in ens.data_vars:
        ens = ens.assign_coords(valid_time=init_time + ens["step"])

    if _is_cumulative(tp, "step"):
        tp_inc = tp.diff("step", label="upper")
        new_step = ens["step"].isel(step=slice(1, None))
        new_vtime = ens["valid_time"].isel(step=slice(1, None))
    else:
        tp_inc = tp
        new_step = ens["step"]
        new_vtime = ens["valid_time"]

    tp_inc = tp_inc.clip(min=0)
    lead_hours = (new_step.values / np.timedelta64(1, "h")).astype("float32")

    tp_inc = tp_inc.assign_coords(
        step=new_step.values,
        valid_time=("step", new_vtime.values),
        lead_time=("step", lead_hours),
    ).transpose("number", "step", "latitude", "longitude", ...)

    return tp_inc.swap_dims({"step": "valid_time"})


# ----------------------------
# IMERG loader
# ----------------------------
def load_imerg_regridded(imerg_dir: str, target_lat, target_lon) -> xr.DataArray:
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

    pr = imerg[precip_var]
    units = str(pr.attrs.get("units", "")).lower()
    if ("mm" in units) and (("hr" in units) or ("/h" in units)):
        imerg_30min = pr * 0.5
    else:
        imerg_30min = pr

    try:
        t_idx = imerg_30min.indexes["time"].to_datetimeindex()
    except Exception:
        t_idx = pd.to_datetime(imerg_30min.time.values)
    imerg_30min = imerg_30min.assign_coords(time=t_idx)

    if SHIFT_IMERG_TIME_BY_30MIN:
        imerg_30min = imerg_30min.assign_coords(time=imerg_30min["time"] + np.timedelta64(30, "m"))

    imerg_12h = imerg_30min.resample(time="12h", label="right", closed="right").sum()

    target = xr.Dataset({"lat": ("lat", target_lat.values), "lon": ("lon", target_lon.values)})
    regridder = xe.Regridder(imerg_12h, target, method="bilinear", periodic=False, reuse_weights=False)
    imerg_rg = regridder(imerg_12h).rename({"lat": "latitude", "lon": "longitude"})

    if not PRECIP_IN_MM:
        imerg_rg = imerg_rg / 1000.0

    imerg_rg = imerg_rg.assign_coords(valid_time=("time", imerg_rg["time"].values))
    return imerg_rg.swap_dims({"time": "valid_time"}).rename("tp")


# ----------------------------
# WB2 climatology: handle (hour, dayofyear, lat, lon) with no time
# ----------------------------
def load_climatology_prob(clim_path: str, target_lat, target_lon, threshold_mm: float) -> xr.DataArray:
    storage_opts = {"token": "anon"} if clim_path.startswith("gs://") else None
    clim = xr.open_zarr(clim_path, consolidated=True, storage_options=storage_opts)

    precip_var = None
    for cand in ("tp", "total_precipitation", "total_precipitation_6hr", "precipitation"):
        if cand in clim.data_vars:
            precip_var = cand
            break
    if precip_var is None:
        raise KeyError(f"No precip variable found in climatology: {list(clim.data_vars)}")

    precip = _to_mm(clim[precip_var])

    if "longitude" in precip.coords and "lon" not in precip.coords:
        precip = precip.rename({"latitude": "lat", "longitude": "lon"})
    precip = precip.sortby("lat").sortby("lon")

    print("Climatology precip dims:", precip.dims)

    if "time" not in precip.dims:
        if "dayofyear" not in precip.dims:
            raise KeyError(f"Climatology has no 'time' and no 'dayofyear'. dims={precip.dims}")

        dims_to_stack = ["dayofyear"]
        if "hour" in precip.dims:
            dims_to_stack.append("hour")

        stacked = precip.stack(time=dims_to_stack)

        # DROP the multiindex level coords BEFORE assigning a new time coordinate (avoid future error)
        drop_list = []
        for v in ["time", "dayofyear", "hour"]:
            if v in stacked.coords:
                drop_list.append(v)
        if drop_list:
            stacked = stacked.drop_vars(drop_list)

        base = np.datetime64("2000-01-01")  # leap year base supports day 366
        times = []
        for idx in stacked["time"].values:
            if isinstance(idx, tuple):
                day = int(idx[0])
                hour = int(idx[1]) if len(idx) > 1 else 0
            else:
                day = int(idx)
                hour = 0
            times.append(base + np.timedelta64(day - 1, "D") + np.timedelta64(hour, "h"))

        precip = stacked.assign_coords(time=("time", np.array(times, dtype="datetime64[ns]")))

    precip_12h = precip.resample(time="12h", label="right", closed="right").sum()
    event = precip_12h > threshold_mm
    prob_doy = event.groupby("time.dayofyear").mean("time")

    target = xr.Dataset({"lat": ("lat", target_lat.values), "lon": ("lon", target_lon.values)})
    regridder = xe.Regridder(prob_doy, target, method="bilinear", periodic=False, reuse_weights=False)
    prob_rg = regridder(prob_doy).rename({"lat": "latitude", "lon": "longitude"})
    return prob_rg

def climatology_for_valid_times(prob_doy: xr.DataArray, valid_time: xr.DataArray) -> xr.DataArray:
    doy = xr.DataArray(
        valid_time.dt.dayofyear.values,
        coords={"valid_time": valid_time.values},
        dims="valid_time",
    )
    prob = prob_doy.sel(dayofyear=doy)
    return prob.assign_coords(valid_time=valid_time.values)


# ----------------------------
# Per-region run
# ----------------------------
def run_region(shp_path: str, init_info, sample_gc, imerg_rg, clim_prob_doy, threshold_mm: float):
    out_fig_bs, out_fig_bss, out_csv = _output_paths(shp_path)
    os.makedirs(os.path.dirname(out_fig_bs), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig_bss), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    region_mask, region_label = build_region_mask(sample_gc["latitude"], sample_gc["longitude"], shp_path=shp_path)
    inside_frac = float((region_mask == 0).mean(skipna=True).compute() if hasattr(region_mask.data, "compute") else (region_mask == 0).mean(skipna=True))
    print(f"\nRegion: {region_label} | inside_frac={inside_frac:.4f} | file={shp_path}")

    imerg_masked = imerg_rg.where(region_mask == 0)
    _print_stats("IMERG regridded 12h (mm) on bbox", imerg_masked)

    print("\nEvent-rate scan on IMERG (masked region, India bbox):")
    scan_event_rates(imerg_masked)

    records = []
    for date_str, gc_dir, ens_file in init_info:
        print("\n==============================")
        print("INIT:", date_str)

        try:
            gc  = subset_bbox(load_gencast_tp_inc(gc_dir)).where(region_mask == 0)
            ens = subset_bbox(load_ens_tp_inc(ens_file)).where(region_mask == 0)
        except Exception as exc:
            print(f"Skipping init {date_str}: {exc}")
            continue

        gc_aligned, ens_aligned, imerg_aligned = xr.align(gc, ens, imerg_masked, join="inner", exclude=["number"])
        nvt = int(gc_aligned.sizes.get("valid_time", 0))
        print("Overlap valid_time count:", nvt)
        print("Ensemble members | GenCast:", gc_aligned.sizes.get("number"), "ENS:", ens_aligned.sizes.get("number"))
        if nvt == 0:
            print("No overlap; skipping.")
            continue

        clim_prob = climatology_for_valid_times(clim_prob_doy, imerg_aligned["valid_time"]).where(region_mask == 0)

        obs_event = (imerg_aligned > threshold_mm).astype("float32")

        # ---- DASK-SAFE scalar extraction ----
        event_rate = obs_event.mean(skipna=True)
        event_rate = float(event_rate.compute()) if hasattr(event_rate.data, "compute") else float(event_rate)

        any_events_da = (obs_event.sum(skipna=True) > 0)
        any_events = bool(any_events_da.compute().item()) if hasattr(any_events_da.data, "compute") else bool(any_events_da.item())
        print(f"Chosen threshold: {threshold_mm:.3f} mm | event_rate={event_rate:.6f} | any_events={any_events}")

        gc_prob  = (gc_aligned  > threshold_mm).mean(dim="number")
        ens_prob = (ens_aligned > threshold_mm).mean(dim="number")

        bs_gc   = ((gc_prob   - obs_event) ** 2).mean(dim=["latitude", "longitude"])
        bs_ens  = ((ens_prob  - obs_event) ** 2).mean(dim=["latitude", "longitude"])
        bs_clim = ((clim_prob - obs_event) ** 2).mean(dim=["latitude", "longitude"])

        valid_bs = bs_clim > 1e-8
        bss_gc  = xr.where(valid_bs, 1.0 - (bs_gc  / bs_clim), np.nan)
        bss_ens = xr.where(valid_bs, 1.0 - (bs_ens / bs_clim), np.nan)

        lead = gc_aligned["lead_time"].sel(valid_time=bss_gc["valid_time"])

        # ---- COMPUTE small per-lead arrays before converting to Python floats ----
        bs_gc, bs_ens, bss_gc, bss_ens, lead = dask.compute(bs_gc, bs_ens, bss_gc, bss_ens, lead)

        for lt, bs_gc_val, bs_ens_val, bss_gc_val, bss_ens_val in zip(
            lead.values,
            bs_gc.values,
            bs_ens.sel(valid_time=bs_gc.valid_time).values,
            bss_gc.values,
            bss_ens.sel(valid_time=bss_gc.valid_time).values,
        ):
            records.append(
                dict(
                    init=pd.Timestamp(date_str + "T00:00:00"),
                    lead_time_h=float(lt),
                    bs_gencast=float(bs_gc_val),
                    bs_ens=float(bs_ens_val),
                    bss_gencast=float(bss_gc_val),
                    bss_ens=float(bss_ens_val),
                    region=region_label,
                    threshold_mm=float(threshold_mm),
                    event_rate=float(event_rate),
                )
            )

    if not records:
        print("No records computed for this region; skipping outputs.")
        return

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print("\nSaved CSV:", out_csv)

    agg_bs = (
        df.groupby("lead_time_h")[["bs_gencast", "bs_ens"]]
        .mean().reset_index().sort_values("lead_time_h")
    )
    agg_bss = (
        df.groupby("lead_time_h")[["bss_gencast", "bss_ens"]]
        .mean().reset_index().sort_values("lead_time_h")
    )

    bs_vals = agg_bs[["bs_gencast", "bs_ens"]].to_numpy()
    bs_min = np.nanmin(bs_vals)
    bs_max = np.nanmax(bs_vals)

    bss_vals = agg_bss[["bss_gencast", "bss_ens"]].to_numpy()
    bss_min = np.nanmin(bss_vals)
    bss_max = np.nanmax(bss_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(agg_bs["lead_time_h"], agg_bs["bs_gencast"], marker="^", label="GenCast vs IMERG")
    plt.plot(agg_bs["lead_time_h"], agg_bs["bs_ens"], marker="s", label="ENS vs IMERG")
    plt.xlabel("Lead time (h)")
    plt.ylabel("Brier Score")
    plt.title(f"BS (12h TP > {threshold_mm:.2f} mm, June 2025) — {region_label} ")
    plt.grid(True)
    plt.legend()
    if np.isfinite(bs_min) and np.isfinite(bs_max):
        pad = 0.05 * (bs_max - bs_min) if bs_max > bs_min else 0.02
        y0 = max(0.0, bs_min - pad)
        y1 = min(1.0, bs_max + pad)
        if (y1 - y0) < 1e-6:
            y1 = min(1.0, y0 + 0.05)
        plt.ylim(y0, y1)
    plt.tight_layout()
    plt.savefig(out_fig_bs, dpi=200)
    plt.close()
    print("Saved BS plot:", out_fig_bs, "| exists?", os.path.exists(out_fig_bs))

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, linewidth=1, linestyle="--", label="No skill")
    plt.plot(agg_bss["lead_time_h"], agg_bss["bss_gencast"], marker="^", label="GenCast vs IMERG")
    plt.plot(agg_bss["lead_time_h"], agg_bss["bss_ens"], marker="s", label="ENS vs IMERG")
    plt.xlabel("Lead time (h)")
    plt.ylabel("Brier Skill Score")
    plt.title(f"BSS (12h TP > {threshold_mm:.2f} mm, June 2025) — {region_label} (India bbox)")
    plt.grid(True)
    plt.legend()
    if np.isfinite(bss_min) and np.isfinite(bss_max):
        pad = 0.05 * (bss_max - bss_min) if bss_max > bss_min else 0.05
        y0 = bss_min - pad
        y1 = bss_max + pad
        if y0 > 0:
            y0 = -0.05
        if y1 < 0:
            y1 = 0.05
        if (y1 - y0) < 1e-6:
            y1 = y0 + 0.1
        plt.ylim(y0, y1)
    plt.tight_layout()
    plt.savefig(out_fig_bss, dpi=200)
    plt.close()
    print("Saved BSS plot:", out_fig_bss, "| exists?", os.path.exists(out_fig_bss))


# ----------------------------
# Main
# ----------------------------
def main():
    init_info = discover_inits()
    sample_gc = subset_bbox(load_gencast_tp_inc(init_info[0][1]))

    imerg_rg = load_imerg_regridded(IMERG_DIR, sample_gc["latitude"], sample_gc["longitude"])
    _print_stats("IMERG regridded 12h (mm) on bbox", imerg_rg)

    threshold_mm = float(EVENT_THRESHOLD_MM)
    print(f"\nThreshold mode=fixed | threshold={threshold_mm:.3f} mm (12h)")

    clim_prob_doy = load_climatology_prob(CLIMATOLOGY_PATH, sample_gc["latitude"], sample_gc["longitude"], threshold_mm)

    for shp_path in _unique_paths(SHAPEFILE_PATHS):
        run_region(shp_path, init_info, sample_gc, imerg_rg, clim_prob_doy, threshold_mm)


if __name__ == "__main__":
    main()

import re
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# User inputs
# ------------------------------------------------------------
shp_path = "/Datastorage/saptarishi.dhanuka_asp25/shapefile_data/india_homog/Northeast.shp"
ens_path = "/Datastorage/mihir.more/ens_20250819_3dayslead/ens_tp_20250819_00_pf.grib2"
gencast_dir = Path("/Datastorage/mihir.more/gencast-ggfs-run-20250819-0000-masked-sst-3dayslead")
imerg_dir = Path("/Datastorage/mihir.more/GPM_3IMERGHHL_07-20251101_143604")
var_name = "tp"
target_start = np.datetime64("2025-08-19T00:00:00")
target_end   = np.datetime64("2025-08-22T00:00:00")
imerg_pattern = "3B-HHR-L.MS.MRG.3IMERG.*.HDF5"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def lead_from_name(path: Path):
    m = re.search(r"f(\d{3})", path.name) or re.search(r"(\d{2,3})h", path.name)
    return int(m.group(1)) if m else None

def normalize_units(tp_raw):
    if getattr(tp_raw, "units", "").lower() in ["m", "meter", "meters", "metre", "metres"]:
        out = tp_raw * 1000.0
        out.attrs["units"] = "mm"
        return out
    return tp_raw

def region_quantiles(ds, tp_da, gdf_region):
    wrap = 360 if float(ds["longitude"].max()) > 180 else None
    mask = regionmask.mask_geopandas(gdf_region, ds, wrap_lon=wrap)
    lat_w = np.cos(np.deg2rad(ds["latitude"]))
    lat_w = lat_w / lat_w.mean()
    w2d = xr.broadcast(lat_w, ds["longitude"])[0]
    w_reg = w2d.where(mask == 0)
    tp_reg = tp_da.where(mask == 0)
    reg_mean = (tp_reg * w_reg).sum(("latitude", "longitude")) / w_reg.sum(("latitude", "longitude"))
    ens_dim = "number" if "number" in reg_mean.dims else ("member" if "member" in reg_mean.dims else None)
    if ens_dim is None:
        ens_dim = "number"
        reg_mean = reg_mean.expand_dims({ens_dim: [0]})
    q = reg_mean.quantile([0.05, 0.5, 0.95], dim=ens_dim)
    q05, q50, q95 = (q.sel(quantile=x) for x in [0.05, 0.5, 0.95])
    return reg_mean, (q05, q50, q95)

def imerg_file_start(path: Path):
    name = path.name
    after_tag = name.split("3IMERG.")[1]
    ymd = after_tag.split("-S")[0]
    hms = after_tag.split("-S")[1][:6]
    return np.datetime64(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}T{hms[:2]}:{hms[2:4]}:{hms[4:]}")

# ------------------------------------------------------------
# Shapefile / region label
# ------------------------------------------------------------
gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
gdf_region = gdf.iloc[[0]]
region_label = gdf_region.iloc[0].get("NAME", "Region")

# ------------------------------------------------------------
# ENS
# ------------------------------------------------------------
ens = xr.open_dataset(ens_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
tp_ens = normalize_units(ens[var_name])
_, (e05, e50, e95) = region_quantiles(ens, tp_ens, gdf_region)
lead_hours_ens = (ens["step"] / np.timedelta64(1, "h")).astype(float)
valid_times = ens["valid_time"] if "valid_time" in ens else ens["time"].isel(step=0) + ens["step"]
if "number" in valid_times.dims:
    valid_times = valid_times.isel(number=0)
valid_times_np = np.asarray(valid_times.values).astype("datetime64[ns]")

# ------------------------------------------------------------
# GenCast-GFS (multiple GRIBs)
# ------------------------------------------------------------
datasets = []
for f in sorted(gencast_dir.glob("*.grib*")):
    dsf = None
    for fbk in [
        {"filter_by_keys": {"typeOfLevel": "surface", "shortName": "tp"}},
        {"filter_by_keys": {"typeOfLevel": "surface", "paramId": 228228}},
        {"filter_by_keys": {"typeOfLevel": "surface"}},
    ]:
        try:
            tmp = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": "", **fbk})
        except Exception:
            continue
        if var_name not in tmp:
            continue
        dsf = tmp[[var_name]]
        break
    if dsf is None:
        print(f"Skipping {f.name}: no {var_name}")
        continue
    lead_h = float(dsf["step"].values / np.timedelta64(1, "h")) if "step" in dsf else lead_from_name(f)
    if lead_h is None:
        print(f"Skipping {f.name}: no lead")
        continue
    if "number" in dsf.dims:
        dsf = dsf.reset_index("number", drop=True)
        dsf = dsf.assign_coords(number=("number", np.asarray(dsf["number"].values)))
    else:
        dsf = dsf.expand_dims({"number": [0]})
    dsf = dsf.assign_coords(lead_time=lead_h)
    datasets.append(dsf)

if not datasets:
    raise ValueError("No GenCast datasets loaded; check filters/variable name.")

gencast = xr.concat(datasets, dim="lead_time").sortby("lead_time")
tp_gc = normalize_units(gencast[var_name])
_, (g05, g50, g95) = region_quantiles(gencast, tp_gc, gdf_region)
lead_hours_gc = gencast["lead_time"]

# ------------------------------------------------------------
# IMERG cumulative (ground truth)
# ------------------------------------------------------------
file_list = []
for p in imerg_dir.glob(imerg_pattern):
    ts = imerg_file_start(p)
    if (ts >= target_start) and (ts < target_end):
        file_list.append((ts, p))
imerg_files = [p for ts, p in sorted(file_list)]
if not imerg_files:
    raise FileNotFoundError("No IMERG files found in the requested time window.")

expected_time = np.arange(target_start, target_end, np.timedelta64(30, "m"))

imerg = xr.open_mfdataset(
    imerg_files,
    engine="h5netcdf",
    group="Grid",
    concat_dim="time",
    combine="nested",
    decode_times=True,
    mask_and_scale=True,
    data_vars="minimal",
    coords="minimal",
    compat="override",
)
imerg = imerg.rename({"lat": "latitude", "lon": "longitude"})
imerg = imerg.convert_calendar("proleptic_gregorian", use_cftime=False).sortby("time")

var_imerg = "precipitationCal" if "precipitationCal" in imerg.data_vars else "precipitation"
imerg_rate = imerg[var_imerg]  # mm/hr
imerg_rate = imerg_rate.reindex(time=expected_time, method="nearest", tolerance=np.timedelta64(15, "m"))

dt_hours = float(np.median(np.diff(expected_time)) / np.timedelta64(1, "h"))
imerg_amount = imerg_rate * dt_hours  # mm per 30‑min step

mask_imerg = regionmask.mask_geopandas(gdf_region, imerg_rate.to_dataset(name="tmp"), wrap_lon=360)
w_im_lat = np.cos(np.deg2rad(imerg["latitude"]))
w_im_lat = w_im_lat / w_im_lat.mean()
w_im = xr.broadcast(w_im_lat, imerg["longitude"])[0]
w_im_reg = w_im.where(mask_imerg == 0)

imerg_reg_step = (imerg_amount.where(mask_imerg == 0) * w_im_reg).sum(
    ("latitude", "longitude")
) / w_im_reg.sum(("latitude", "longitude"))
imerg_reg_step = imerg_reg_step.clip(min=0)

imerg_reg_cum = imerg_reg_step.cumsum("time")

tol = np.timedelta64(90, "m")
imerg_on_leads = imerg_reg_cum.reindex(time=valid_times_np, method="nearest", tolerance=tol)

# ------------------------------------------------------------
# Plot: q05–q95 + median for ENS/GenCast, IMERG cumulative line
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.fill_between(lead_hours_ens, e05, e95, color="tab:blue", alpha=0.2, label="ENS q05–q95")
plt.plot(lead_hours_ens, e50, color="tab:blue", marker="o", label="ENS median")
plt.fill_between(lead_hours_gc, g05, g95, color="tab:orange", alpha=0.2, label="GenCast-GFS q05–q95")
plt.plot(lead_hours_gc, g50, color="tab:orange", marker="s", label="GenCast-GFS median")

finite_mask = np.isfinite(imerg_on_leads.values)
plt.plot(
    lead_hours_ens[finite_mask],
    imerg_on_leads.values[finite_mask],
    color="k",
    lw=2,
    label="IMERG 12 hr accum.",
)

plt.xlabel("Lead time (hours)")
plt.ylabel("Region-mean precip (mm)")
plt.title(f"Region-mean precip (q05–q95, median) — Northeast")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

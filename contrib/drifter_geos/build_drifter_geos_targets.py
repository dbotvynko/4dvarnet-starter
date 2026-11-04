"""Daily geostrophic-velocity targets from drogued GDP drifters, for fine-tuning the SLA UNet with a drifter-velocity loss.

Drifters measure the total current at their drogue depth (15 m). Their geostrophic part is estimated as in the CNES-CLS MDT
processing:
  1. GDP 6-hourly positions/velocities (NOAA S3 gdp6h_ragged_may25.zarr), drogued only (per-observation drogue_status == 1);
  2. daily means per drifter (the 4 positions of a UTC day required): removes most inertial / tidal motion;
  3. minus the Ekman current at 15 m of GlobCurrent (CMEMS MULTIOBS_GLO_PHY_MYNRT_015_003 daily, ue/ve at 15 m: MY product,
     NRT after it ends), sampled at the nearest 0.25 deg cell.
GlobCurrent ugos/vgos are stored at the same points for comparison. Near the equator geostrophy fails: filter |lat| >= 5 when
using the targets.

Output: <out>/gdp_drogued_daily_geos_<year>.nc, one row per drifter-day:
  id, time (day), lat, lon, u_drifter, v_drifter, u_ekman15, v_ekman15, u_geos = u_drifter - u_ekman15, v_geos,
  u_gc_geos, v_gc_geos (GlobCurrent ugos/vgos).
Usage: python build_drifter_geos_targets.py YEAR [--out DIR]   (Copernicus credentials from the environment)
"""
import argparse
import os

import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr

URL = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/experimental/gdp6h_ragged_may25.zarr"
GC = {"my": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m", "nrt": "cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m"}


def drifter_days(year):
    z = xr.open_zarr(URL, decode_times=False)
    t = z.time.values
    t0, t1 = pd.Timestamp(f"{year}-01-01").timestamp(), pd.Timestamp(f"{year + 1}-01-01").timestamp()
    idx = np.where((t >= t0) & (t < t1))[0]
    traj = np.repeat(np.arange(z.sizes["traj"]), z.rowsize.values)[idx]
    sub = z[["lat", "lon", "ve", "vn", "drogue_status"]].isel(obs=idx).load()
    df = pd.DataFrame({"id": z.id.values[traj], "time": pd.to_datetime(t[idx], unit="s"), "lat": sub.lat.values,
                       "lon": (sub.lon.values + 180) % 360 - 180, "ve": sub.ve.values, "vn": sub.vn.values,
                       "drogued": sub.drogue_status.values})
    df = df[(df.drogued == 1) & np.isfinite(df.ve) & np.isfinite(df.vn) & np.isfinite(df.lat)]
    df["day"] = df.time.dt.floor("D")
    # circular mean of the longitude (drifters crossing the dateline)
    df["cx"], df["cy"] = np.cos(np.radians(df.lon)), np.sin(np.radians(df.lon))
    g = df.groupby(["id", "day"]).agg(n=("ve", "size"), lat=("lat", "mean"), cx=("cx", "mean"), cy=("cy", "mean"),
                                       u=("ve", "mean"), v=("vn", "mean")).reset_index()
    g = g[g.n == 4]
    g["lon"] = np.degrees(np.arctan2(g.cy, g.cx))
    return g[["id", "day", "lat", "lon", "u", "v"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--out", default="/Odyssey/private/d21botvy/DATA/drifters_geos")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = drifter_days(a.year)
    print(f"{a.year}: {len(d)} drogued drifter-days, {d.id.nunique()} drifters", flush=True)
    gcs = {k: cm.open_dataset(dataset_id=v, variables=["ue", "ve", "ugos", "vgos"], minimum_depth=15, maximum_depth=15)
           for k, v in GC.items()}
    my_end = pd.Timestamp(gcs["my"].time.values[-1])
    cols = {k: np.full(len(d), np.nan, np.float32) for k in ("ue", "ve", "ugos", "vgos")}
    for k, (day, rows) in enumerate(d.groupby("day").groups.items()):
        src = gcs["my"] if day <= my_end else gcs["nrt"]
        f = src.sel(time=day).isel(depth=0).load()
        pts = dict(latitude=xr.DataArray(d.lat.values[rows]), longitude=xr.DataArray(d.lon.values[rows]))
        for v in cols:
            cols[v][rows] = f[v].sel(**pts, method="nearest").values
        if k % 30 == 0:
            print(f"  {day.date()}", flush=True)
    ds = xr.Dataset({"id": ("obs", d.id.values), "time": ("obs", d.day.values), "lat": ("obs", d.lat.values),
                     "lon": ("obs", d.lon.values), "u_drifter": ("obs", d.u.values.astype(np.float32)),
                     "v_drifter": ("obs", d.v.values.astype(np.float32)), "u_ekman15": ("obs", cols["ue"]),
                     "v_ekman15": ("obs", cols["ve"]), "u_gc_geos": ("obs", cols["ugos"]), "v_gc_geos": ("obs", cols["vgos"])})
    ds["u_geos"] = ds.u_drifter - ds.u_ekman15
    ds["v_geos"] = ds.v_drifter - ds.v_ekman15
    ds.attrs = {"source": "GDP 6h (NOAA S3 gdp6h_ragged_may25), drogued, daily means of 4 fixes; Ekman 15 m and ugos from "
                          "GlobCurrent (MY up to its end, NRT after)", "script": __file__}
    out = f"{a.out}/gdp_drogued_daily_geos_{a.year}.nc"
    ds.to_netcdf(out)
    ok = np.isfinite(ds.u_geos.values) & (np.abs(ds.lat.values) >= 5)
    r = np.corrcoef(ds.u_geos.values[ok], ds.u_gc_geos.values[ok])[0, 1]
    print(f"written {out}; |lat|>=5: {ok.sum()} rows, corr(u_geos drifter, GlobCurrent ugos) = {r:.2f}, "
          f"rms u_geos {100 * np.sqrt(np.mean(ds.u_geos.values[ok] ** 2)):.1f} cm/s, "
          f"rms diff vs ugos {100 * np.sqrt(np.mean((ds.u_geos.values - ds.u_gc_geos.values)[ok] ** 2)):.1f} cm/s")


if __name__ == "__main__":
    main()

"""2023 gridded NRT L3 input with ALL satellites: the 6 input satellites of the pipeline (al, c2n, h2b, j3n, s3a, s3b,
concat/concatenated_input.nc) + Sentinel-6A (s6a-hr, concat/concatenated_ref.nc, the independent reference of the
original runs), on the grid of /Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc (1/4 deg, lat -90..89.75,
lon -180..179.75, daily 2023).

Gridding = the pipeline's coord_based_to_grid convention, re-implemented (ocn_tools is not installed any more): points with
time in [day - 12 h, day + 12 h), nearest grid cell, mean of the points. Rebuilding the 6-satellite grid this way matches
the original file exactly in 99.9% of the cells (checked on 2023-03-01; printed per month below as a control).

Usage: python build_gridded_allsats_2023.py [output]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CONCAT = "/Odyssey/public/altimetry_traces/nrt_sla/concat/concatenated_{}.nc"
ORIG = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/Odyssey/public/altimetry_traces/nrt_sla_allsats/2023/gridded_input.nc"
VARS = ["ssh", "sla_filtered", "sla_unfiltered", "lwe", "mdt"]
DAYS = pd.date_range("2023-01-01", "2023-12-31", freq="D")
NLAT, NLON, RES = 720, 1440, 0.25


def grid_day(ds, t, day):
    i0, i1 = np.searchsorted(t, [np.datetime64(day - pd.Timedelta(hours=12)), np.datetime64(day + pd.Timedelta(hours=12))])
    s = ds.isel(time=slice(i0, i1))[VARS + ["lat", "lon"]].load()
    ilat = np.round((s.lat.values + 90) / RES).astype(int)
    ilon = np.round((s.lon.values + 180) / RES).astype(int) % NLON
    ok = (ilat >= 0) & (ilat < NLAT)
    return {v: (ilat[ok] * NLON + ilon[ok], s[v].values[ok]) for v in VARS}


def main():
    srcs = {k: xr.open_dataset(CONCAT.format(k)) for k in ("input", "ref")}
    times = {k: d.time.values for k, d in srcs.items()}
    orig = xr.open_dataset(ORIG)
    out = {v: np.full((len(DAYS), NLAT, NLON), np.nan, np.float32) for v in VARS}
    nobs = np.zeros((len(DAYS), 2), np.int64)
    for j, day in enumerate(DAYS):
        parts = {k: grid_day(srcs[k], times[k], day) for k in srcs}
        for v in VARS:
            k = np.concatenate([parts["input"][v][0], parts["ref"][v][0]])
            x = np.concatenate([parts["input"][v][1], parts["ref"][v][1]])
            f = np.isfinite(x)
            sm = np.bincount(k[f], x[f], NLAT * NLON)
            n = np.bincount(k[f], minlength=NLAT * NLON)
            out[v][j] = np.where(n > 0, sm / np.maximum(n, 1), np.nan).reshape(NLAT, NLON)
        nobs[j] = [len(parts["input"]["sla_filtered"][0]), len(parts["ref"]["sla_filtered"][0])]
        if day.day == 1:   # control: the 6 input satellites alone vs the original gridded file
            k, x = parts["input"]["sla_filtered"]
            f = np.isfinite(x)
            sm, n = np.bincount(k[f], x[f], NLAT * NLON), np.bincount(k[f], minlength=NLAT * NLON)
            m6 = np.where(n > 0, sm / np.maximum(n, 1), np.nan).reshape(NLAT, NLON)
            ref = orig.sla_filtered.sel(time=day).values
            both = np.isfinite(m6) & np.isfinite(ref)
            print(f"{day.date()}: points 6 sats {nobs[j, 0]}, s6a {nobs[j, 1]} | 6-sat rebuild vs original: "
                  f"{100 * np.mean(np.abs(m6 - ref)[both] < 1e-5):.2f}% cells identical, cells {np.isfinite(m6).sum()} vs {np.isfinite(ref).sum()} | "
                  f"all-sat cells {np.isfinite(out['sla_filtered'][j]).sum()}", flush=True)
    ds = xr.Dataset({v: (("time", "lat", "lon"), out[v]) for v in VARS},
                    coords={"time": DAYS, "lat": np.arange(-90, 90, RES), "lon": np.arange(-180, 180, RES)})
    ds.attrs = {"satellites": "al c2n h2b j3n s3a s3b + s6a-hr", "source": "concat/concatenated_input.nc + concat/concatenated_ref.nc",
                "gridding": "points in [day-12h, day+12h), nearest 1/4 deg cell, mean", "script": __file__}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"written {OUT}; mean points/day: 6 sats {nobs[:, 0].mean():.0f}, s6a {nobs[:, 1].mean():.0f}")


if __name__ == "__main__":
    main()

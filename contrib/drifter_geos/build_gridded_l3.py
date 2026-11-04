"""Rebuild the gridded L3 input of the OSE UNet training (the original 2010_2023/gridded/gridded_input.nc no longer exists and
its compressed copy has NaN longitudes) from the along-track concat file, with the pipeline's gridding convention
(ocn_tools coord_based_to_grid = pyinterp Binning2D): points with time in (day - 12 h, day + 12 h], nearest 1/4 deg cell,
mean. The same re-implementation reproduced the 2023 pipeline grid in 99.93% of the cells.
Variable 'sla' (= along-track sla_filtered, what open_glorys12_data_sla_OSE reads) on lat -90..89.75, lon -180..179.75.
Usage: python build_gridded_l3.py [--concat ...] [--start 2017-01-01] [--end 2022-12-31] [--out ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

NLAT, NLON, RES = 720, 1440, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concat", default="/Odyssey/public/altimetry_traces/2010_2023/concat/concatenated_input.nc")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--var", default="sla_filtered")
    ap.add_argument("--out", default="/Odyssey/public/altimetry_traces/2010_2023/gridded/gridded_input_2017_2022_rebuilt.nc")
    a = ap.parse_args()
    days = pd.date_range(a.start, a.end, freq="D")
    c = xr.open_dataset(a.concat)
    t = c.time.values
    out = np.full((len(days), NLAT, NLON), np.nan, np.float32)
    for j, day in enumerate(days):
        i0, i1 = np.searchsorted(t, [np.datetime64(day - pd.Timedelta(hours=12)), np.datetime64(day + pd.Timedelta(hours=12))],
                                 side="right")
        s = c.isel(time=slice(i0, i1))[[a.var, "lat", "lon"]].load()
        v = s[a.var].values
        ilat = np.round((s.lat.values + 90) / RES).astype(int)
        ilon = np.round((((s.lon.values + 180) % 360) - 180 + 180) / RES).astype(int) % NLON
        ok = np.isfinite(v) & (ilat >= 0) & (ilat < NLAT)
        k = ilat[ok] * NLON + ilon[ok]
        sm = np.bincount(k, v[ok], NLAT * NLON)
        n = np.bincount(k, minlength=NLAT * NLON)
        out[j] = np.where(n > 0, sm / np.maximum(n, 1), np.nan).reshape(NLAT, NLON)
        if day.day == 1 and day.month in (1, 7):
            print(f"{day.date()}: {ok.sum()} points, {np.isfinite(out[j]).sum()} cells", flush=True)
    ds = xr.Dataset({"sla": (("time", "lat", "lon"), out)},
                    coords={"time": days, "lat": np.arange(-90, 90, RES), "lon": np.arange(-180, 180, RES)},
                    attrs={"source": a.concat, "variable": a.var,
                           "gridding": "points in (day-12h, day+12h], nearest 1/4 deg cell, mean (pipeline convention)",
                           "script": __file__})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()

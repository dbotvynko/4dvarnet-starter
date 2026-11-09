"""MUR25 L4 SST (PO.DAAC MUR25-JPL-L4-GLOB-v04.2, daily, 0.25 deg, foundation SST) for one year, cut to the NE Pacific
training domain (10-55N, 170-115W) and bilinearly interpolated on the 0.25 deg grid of the DUACS training file, as an
extra input channel of the regional UNet (contrib/sst_regional). MUR is available about 1 day after its date, so a
forecast issued at t0 can use SST up to t0-1, like the altimetry.
Output: /Odyssey/private/d21botvy/DATA/mur25_nepac/mur25_sst_0.25deg_nepac_<year>.nc (var sst in deg C, NaN on land/ice
and on missing days).
Usage: python download_mur25_nepac.py YEAR [--nthreads 8]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

CMR = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MUR25-JPL-L4-GLOB-v04.2"
       "&temporal={}T00:00:00Z,{}T23:59:59Z&page_size=2000&page_num={}")
DUACS = "/Odyssey/public/duacs/2023/duacs_2017_2022_0.25deg.nc"
NETRC = "/Odyssey/private/d21botvy/.netrc_earthdata"
OUTDIR = "/Odyssey/private/d21botvy/DATA/mur25_nepac"
LAT, LON = slice(10, 54.9), slice(-170, -115.1)        # 180 x 220 points of the DUACS grid


def granule_urls(days):
    urls, page = {}, 1
    start, end = (days[0] - pd.Timedelta(days=1)).date(), (days[-1] + pd.Timedelta(days=1)).date()
    while True:
        e = json.load(urllib.request.urlopen(CMR.format(start, end, page)))["feed"]["entry"]
        if not e:
            break
        for x in e:
            for link in x["links"]:
                m = re.search(r"/(\d{8})090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB.*\.nc$", link["href"])
                if m and link["href"].startswith("https://archive"):
                    d = pd.Timestamp(m.group(1))
                    if d in days:
                        urls[d] = link["href"]
        page += 1
    return urls


def fetch(args):
    day, url, tmpdir = args
    f = os.path.join(tmpdir, os.path.basename(url))
    cookie = os.path.join(tmpdir, f"ck_{day:%Y%m%d}.txt")
    for _ in range(4):
        if subprocess.run(["curl", "-s", "-f", "--netrc-file", NETRC, "-L", "-c", cookie, "-b", cookie, "-o", f, url]).returncode == 0:
            return day, f
    return day, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--nthreads", type=int, default=8)
    a = ap.parse_args()
    days = pd.date_range(f"{a.year}-01-01", f"{a.year}-12-31", freq="D")
    g = xr.open_dataset(DUACS)
    lat, lon = g.lat.sel(lat=LAT).values, g.lon.sel(lon=LON).values
    out = f"{OUTDIR}/mur25_sst_0.25deg_nepac_{a.year}.nc"
    urls = granule_urls(days)
    print(f"{a.year}: {len(urls)}/{len(days)} days available; grid {len(lat)} x {len(lon)}", flush=True)
    data = np.full((len(days), len(lat), len(lon)), np.nan, np.float32)
    idx = {d: i for i, d in enumerate(days)}
    tmpdir = tempfile.mkdtemp(prefix=f"tmp_{a.year}_", dir=OUTDIR)
    failed = [d for d in days if d not in urls]
    items = sorted(urls.items())
    chunks = [items[i:i + 4 * a.nthreads] for i in range(0, len(items), 4 * a.nthreads)]
    with ThreadPoolExecutor(a.nthreads) as ex:
        for c in chunks:
            for day, f in ex.map(fetch, [(d, u, tmpdir) for d, u in c]):
                if f is None:
                    failed.append(day)
                    continue
                with xr.open_dataset(f) as d:      # main thread only (HDF5 reads are not thread-safe)
                    sst = d["analysed_sst"].isel(time=0)
                    sst = sst.sel(lat=slice(lat.min() - 0.5, lat.max() + 0.5), lon=slice(lon.min() - 0.5, lon.max() + 0.5)).load()
                    data[idx[day]] = (sst.interp(lat=lat, lon=lon).values - 273.15).astype(np.float32)
                os.remove(f)
            print(f"  {c[-1][0].date()} done", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    xr.Dataset({"sst": (("time", "lat", "lon"), data)}, coords={"time": days, "lat": lat, "lon": lon},
               attrs={"source": "PO.DAAC MUR25-JPL-L4-GLOB-v04.2 analysed_sst (foundation SST), bilinear to the DUACS 0.25 deg training grid",
                      "units": "degC"}).to_netcdf(out)
    print("written", out, "missing days:", sorted(str(d.date()) for d in failed), flush=True)


if __name__ == "__main__":
    main()

"""NeurOST SSH-SST L4 v2024.0 (PO.DAAC NEUROST_SSH-SST_L4_V2024.0, daily, 0.1 deg) SLA for one year, cut to the NE Pacific
training domain (10-55N, 170-115W) and bilinearly interpolated on the 0.25 deg grid of the DUACS training file
(duacs_2017_2022_0.25deg.nc), so that it can replace the DUACS target of the UNet fine-tune one-to-one.
Global files are downloaded (Earthdata login), cut, and deleted. Latest processing version of each day.
Output: /Odyssey/private/d21botvy/DATA/neurost_nepac/neurost_sla_0.25deg_nepac_<year>.nc (var sla, NaN on missing days).
Usage: python download_neurost_nepac.py YEAR [--nthreads 8] [--fill]   (--fill: only re-fetch the all-NaN days of an existing file)
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

CMR = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=NEUROST_SSH-SST_L4_V2024.0"
       "&temporal={}T00:00:00Z,{}T23:59:59Z&page_size=2000&page_num={}")
DUACS = "/Odyssey/public/duacs/2023/duacs_2017_2022_0.25deg.nc"
NETRC = "/Odyssey/private/d21botvy/.netrc_earthdata"
OUTDIR = "/Odyssey/private/d21botvy/DATA/neurost_nepac"
LAT, LON = slice(10, 54.9), slice(-170, -115.1)        # 180 x 220 points of the DUACS grid


def granule_urls(days):
    urls = {}
    start, end = (days[0] - pd.Timedelta(days=1)).date(), (days[-1] + pd.Timedelta(days=1)).date()
    page = 1
    while True:
        e = json.load(urllib.request.urlopen(CMR.format(start, end, page)))["feed"]["entry"]
        if not e:
            break
        for x in e:
            for link in x["links"]:
                m = re.search(r"NeurOST_SSH-SST_(\d{8})_(\d{8})\.nc$", link["href"])
                if m and link["href"].startswith("https://archive"):
                    d, v = pd.Timestamp(m.group(1)), m.group(2)
                    if d in days and (d not in urls or v > urls[d][0]):
                        urls[d] = (v, link["href"])
        page += 1
    return {d: u for d, (v, u) in urls.items()}


def fetch(args):
    day, url, tmpdir = args
    f = os.path.join(tmpdir, os.path.basename(url))
    cookie = os.path.join(tmpdir, f"ck_{day:%Y%m%d}.txt")
    for _ in range(3):
        if subprocess.run(["curl", "-s", "-f", "--netrc-file", NETRC, "-L", "-c", cookie, "-b", cookie, "-o", f, url]).returncode == 0:
            return day, f
    return day, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args()
    days = pd.date_range(f"{a.year}-01-01", f"{a.year}-12-31", freq="D")
    g = xr.open_dataset(DUACS)
    lat, lon = g.lat.sel(lat=LAT).values, g.lon.sel(lon=LON).values
    out = f"{OUTDIR}/neurost_sla_0.25deg_nepac_{a.year}.nc"
    urls = granule_urls(days)
    old = None
    if a.fill:
        with xr.open_dataset(out) as f_old:
            old = f_old["sla"].load()
        missing = set(pd.DatetimeIndex(old.time.values[~np.isfinite(old.values).any(axis=(1, 2))]))
        urls = {d: u for d, u in urls.items() if d in missing}
        print(f"{a.year}: refilling {len(urls)} of {len(missing)} missing days", flush=True)
    print(f"{a.year}: {len(urls)}/{len(days)} days available; grid {len(lat)} x {len(lon)}", flush=True)
    data = old.values.copy() if old is not None else np.full((len(days), len(lat), len(lon)), np.nan, np.float32)
    idx = {d: i for i, d in enumerate(days)}
    tmpdir = tempfile.mkdtemp(prefix=f"tmp_{a.year}_", dir=OUTDIR)
    failed = []
    items = sorted(urls.items())
    chunks = [items[i:i + 2 * a.nthreads] for i in range(0, len(items), 2 * a.nthreads)]    # bounded disk use
    with ThreadPoolExecutor(a.nthreads) as ex:
        for c in chunks:
            for k, (day, f) in enumerate(ex.map(fetch, [(d, u, tmpdir) for d, u in c])):
                if f is None:
                    failed.append(day)
                    continue
                with xr.open_dataset(f) as d:      # main thread only (HDF5 reads are not thread-safe)
                    sla = d["sla"].isel(time=0).assign_coords(longitude=((d.longitude + 180) % 360) - 180).sortby("longitude")
                    box = sla.sel(latitude=slice(lat.min() - 0.5, lat.max() + 0.5), longitude=slice(lon.min() - 0.5, lon.max() + 0.5)).load()
                    data[idx[day]] = box.interp(latitude=lat, longitude=lon).values.astype(np.float32)
                os.remove(f)
            print(f"  {c[-1][0].date()} done", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    xr.Dataset({"sla": (("time", "lat", "lon"), data)}, coords={"time": days, "lat": lat, "lon": lon},
               attrs={"source": "PO.DAAC NEUROST_SSH-SST_L4_V2024.0, latest version per day, bilinear 0.1 -> 0.25 deg (DUACS training grid)"}
               ).to_netcdf(out)
    print("written", out, "failed days:", [str(d.date()) for d in failed], flush=True)


if __name__ == "__main__":
    main()

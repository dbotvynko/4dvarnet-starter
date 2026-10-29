"""Crossover cells of the SWOT 1-day repeat orbit (2023 fast-sampling phase), on the 1/4-degree grid of the gridded L3 calval file.

A cell is a crossover when it is covered by an ascending pass (odd pass number) and by a descending pass (even pass number)
of the same orbit, i.e. it is seen twice a day. Coverage = valid ssha_filtered pixels of the L3 v3.0 Basic passes, binned to
the nearest grid cell, union over a few cycles (the ground track repeats every day, so a few cycles are enough; several
guard against data gaps). Also writes the great-circle distance (km) from each cell to the nearest crossover cell.

Usage: python contrib/swot_osse/crossover_mask_calval_2023.py OUT.npz [--cycles 480,490,500,510]
Output: npz with lat, lon, asc (bool), desc (bool), crossover (bool), dist_km (float32)
"""
import argparse
import os
from glob import glob

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

PASSES = "/Odyssey/public/swot_traces/l3_calval_2023/v3_0_basic/cycle_{}/SWOT_L3_LR_SSH_Basic_*_v3.0.nc"
GRID = "/Odyssey/public/swot_traces/l3_calval_2023/gridded_l3_calval_2023_0.25deg_sla_filtered.nc"
EARTH_KM = 6371.0


def unit_vectors(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--cycles", default="480,490,500,510")
    args = p.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")
    g = xr.open_dataset(GRID)
    lat, lon = g.lat.values, g.lon.values
    cover = {"asc": np.zeros((lat.size, lon.size), bool), "desc": np.zeros((lat.size, lon.size), bool)}
    for cycle in args.cycles.split(","):
        files = sorted(glob(PASSES.format(cycle)))
        for f in files:
            number = int(os.path.basename(f).split("_")[6])
            d = xr.open_dataset(f)
            ok = np.isfinite(d.ssha_filtered.values)
            la, lo = d.latitude.values[ok], ((d.longitude.values[ok] + 180) % 360) - 180
            i = np.clip(np.round((la - lat[0]) / 0.25).astype(int), 0, lat.size - 1)
            j = np.round((lo - lon[0]) / 0.25).astype(int) % lon.size
            cover["asc" if number % 2 else "desc"][i, j] = True
        print(f"cycle {cycle}: {len(files)} passes", flush=True)
    crossover = cover["asc"] & cover["desc"]
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    tree = cKDTree(unit_vectors(glat[crossover], glon[crossover]))
    chord, _ = tree.query(unit_vectors(glat.ravel(), glon.ravel()))
    dist = (2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))).reshape(lat.size, lon.size).astype(np.float32)
    np.savez_compressed(args.out, lat=lat, lon=lon, asc=cover["asc"], desc=cover["desc"], crossover=crossover, dist_km=dist)
    print("cells: asc", cover["asc"].sum(), "desc", cover["desc"].sum(), "crossover", crossover.sum(), "->", args.out)


if __name__ == "__main__":
    main()

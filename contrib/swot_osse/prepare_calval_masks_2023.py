"""Daily nadir and SWOT-swath observation masks of the SWOT 1-day (fast-sampling) orbit, 30 March - 8 July 2023, for the OSSE
training on the 1-day orbit (same format as prepare_cls_masks_2024.py, so contrib.swot_osse.data_epoch can use either).
  nadir: finite pixels of the 2023 DT nadir input (processed_2023_global_4/gridded/gridded_input.nc)
  swot:  finite pixels of the gridded SWOT L3 v3.0 Basic swaths (|SLA| <= 1 m; grid_l3_calval_2023.py) that are not nadir pixels
Both grids are already the 680 x 1440 training grid (rows -80 .. 89.75 deg, columns -180 .. 179.75 deg). Days are kept in
chronological order (data_epoch sequence="orbit" walks through them day by day).
Usage: python -m contrib.swot_osse.prepare_calval_masks_2023 [output.npz]   (refuses to overwrite)
"""
import os
import sys

import numpy as np
import xarray as xr

NADIR = "/Odyssey/public/altimetry_traces/processed_2023_global_4/gridded/gridded_input.nc"
SWOT = "/Odyssey/public/swot_traces/l3_calval_2023/gridded_l3_calval_2023_0.25deg_sla_filtered.nc"
OUTPUT = "/Odyssey/public/swot_traces/mask/calval_2023_1day_orbit_nadir_swot_masks.npz"
DAYS = slice("2023-03-30", "2023-07-08")


def main(output: str = OUTPUT) -> None:
    if os.path.exists(output):
        raise FileExistsError(output)
    sw = xr.open_dataset(SWOT).sla_filtered.sel(time=DAYS)
    na = xr.open_dataset(NADIR).sla_filtered.sel(time=sw.time)
    assert sw.shape[1:] == (680, 1440) and na.shape[1:] == (680, 1440), (sw.shape, na.shape)
    assert np.allclose(sw.lat.values, na.lat.values) and np.allclose(sw.lon.values, na.lon.values)
    n = np.isfinite(na.values)
    s = sw.values
    s = np.isfinite(s) & (np.abs(np.nan_to_num(s)) <= 1) & ~n
    np.savez(output, dates=sw.time.values, shape=np.array(n.shape), nadir=np.packbits(n, axis=-1), swot=np.packbits(s, axis=-1))
    cov = s.mean(axis=(1, 2))
    print(output, n.shape, "nadir %.3f, swot %.3f of the grid per day (swot min %.3f, max %.3f)" % (n.mean(), s.mean(), cov.min(), cov.max()))


if __name__ == "__main__":
    main(*sys.argv[1:])

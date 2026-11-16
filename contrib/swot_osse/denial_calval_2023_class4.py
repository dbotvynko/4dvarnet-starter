"""2023 SWOT daily-repeat test scored the OceanBench way (Class-4 against DT along-track altimetry), daily starts.

Same forecasts as denial_calval_2023.py (same UNets, inputs, normalisation, starts 2023-04-12 .. 2023-07-03, lead day 1 = t0),
but the truth follows the OceanBench 2024-v2 observation policy instead of AltiKa only:
  - DUACS L3 DT (MY) 1 Hz `sla_filtered` of the OceanBench missions available in the window: alg, c2n, h2b, s3a, s3b from the
    public folder altimetry_traces/processed/2023_2025 (version 202411), s6a_lr downloaded by
    oceanbench_eval/download_obs2023_calval.sh (swon, SWOT's nadir, has no DT data before late July 2023);
  - |sla| <= 2 m, observation on the valid day (UTC), forecast interpolated bilinearly to the observation position. OceanBench
    compares forecast SSH - MDT with the observed SLA, so the CNES-CLS22 MDT added to the challenger cancels: forecast SLA
    vs observed SLA is the same comparison;
  - all points (not only the SWOT strips); strip / off-strip split kept as extra point sets.
AltiKa is withheld from the input (as in denial_calval_2023.py); the other missions are inputs on past days only, never on the
valid day (as in the 2024 OceanBench evaluation).
Experiments: Nadir only (last_k 14), SWOT on all 14 days (last_k 0), SWOT on one day t0-j only (keep_j, j = 1, 2, 3, 7).
Areas: global (all latitudes), the latitude bands and boxes of the 2024 paper tables.
Output: OUT/scores.csv with per (start, experiment, lead, truth set, point set, area) n, sum_err, sum_sq_err (refuses an
existing folder). Truth sets: all6 (the six missions), alg (AltiKa only).

Usage (GPU): python contrib/swot_osse/denial_calval_2023_class4.py OUT_DIR [--ckpt CKPT] [--max-starts N]
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from contrib.swot_osse.explain_tropics import Batch, load_model  # noqa: E402
from contrib.swot_osse.denial_calval_2023 import DT, MEAN, N_LEAD, N_OBS, NADIR, STARTS, STD, SWOT  # noqa: E402

PUBLIC = "/Odyssey/public/altimetry_traces/processed/2023_2025/SEALEVEL_GLO_PHY_L3_MY_008_062"  # alg, c2n, h2b, s3a, s3b (202411)
OBS = "/Odyssey/private/d21botvy/oceanbench_eval/data_obs2023_calval"  # s6a_lr only (not in the public folder)
MISSIONS = ["alg", "c2n", "h2b", "s3a", "s3b", "s6a_lr"]
SLA_ABS_MAX = 2.0
BANDS = {"n40_n82": (40, 82), "n20_n40": (20, 40), "s20_n20": (-20, 20), "s40_s20": (-40, -20), "s80_s40": (-80, -40)}
BOXES = {"gulf_stream": (30, 45, -80, -45), "kuroshio": (25, 45, 125, 170), "agulhas": (-45, -25, 10, 45)}
EXPERIMENTS = [("last_k", 14), ("last_k", 0), ("keep_j", 1), ("keep_j", 2), ("keep_j", 3), ("keep_j", 7)]


def areas(lat, lon):
    out = {"global": np.ones(lat.shape, bool)}
    for name, (a, b) in BANDS.items():
        out[name] = (lat >= a) & (lat < b) if b < 82 else (lat >= a) & (lat <= b)
    for name, (a0, a1, b0, b1) in BOXES.items():
        out[name] = (lat >= a0) & (lat <= a1) & (lon >= b0) & (lon <= b1)
    return out


def mission_of(path):
    name = os.path.basename(path)  # dt_global_<mission>_phy_l3_1hz_YYYYMMDD_*.nc
    return name.split("_phy_l3")[0].replace("dt_global_", "")


def load_tracks(days):
    files = sorted(glob(f"{PUBLIC}/*/2023/0[3-7]/dt_global_*_phy_l3_1hz_2023*.nc"))
    files = [f for f in files if mission_of(f) in MISSIONS and mission_of(f) != "s6a_lr"]
    files += sorted(glob(f"{OBS}/**/dt_global_s6a_lr_phy_l3_1hz_2023*.nc", recursive=True))
    parts = []
    for f in files:
        m = mission_of(f)
        if m not in MISSIONS:
            continue
        ds = xr.open_dataset(f)
        t = pd.DatetimeIndex(ds.time.values)
        v = ds.sla_filtered.values.astype(np.float64)
        parts.append(pd.DataFrame({"day": t.normalize(), "lat": ds.latitude.values, "lon": ((ds.longitude.values + 180) % 360) - 180,
                                   "val": v, "alg": m == "alg"}))
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isfinite(d.val) & (np.abs(d.val) <= SLA_ABS_MAX) & d.day.isin(days)].reset_index(drop=True)
    found = sorted({mission_of(f) for f in files} & set(MISSIONS))
    print("missions found:", found, "points:", len(d), flush=True)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max-starts", type=int, default=None)
    args = p.parse_args()
    device = "cuda"
    if args.ckpt:
        import contrib.swot_osse.explain_tropics as et
        et.CKPT = args.ckpt
    model = load_model(device)
    print("checkpoint:", args.ckpt or "default (Nadir-trained UNet)", flush=True)
    os.makedirs(args.out, exist_ok=False)

    days = pd.date_range(STARTS[0] - pd.Timedelta(days=N_OBS), STARTS[-1] + pd.Timedelta(days=N_LEAD - 1))
    nad = xr.open_dataset(NADIR).sla_filtered.sel(time=days).values.astype(np.float32)
    ds_sw = xr.open_dataset(SWOT)
    lat, lon = ds_sw.lat.values, ds_sw.lon.values
    sw = ds_sw.sla_filtered.reindex(time=days).values.astype(np.float32)
    sw[np.abs(sw) > 1] = np.nan
    sw_only = np.isfinite(sw) & ~np.isfinite(nad)
    combined = np.where(np.isfinite(nad), nad, sw)
    phase = (days >= "2023-03-30") & (days <= "2023-07-08")
    strip = np.isfinite(sw[phase]).mean(0) >= 0.8

    obs = load_tracks(days)
    y = (obs.lat.values - lat[0]) / 0.25
    x = (obs.lon.values - lon[0]) / 0.25
    gi, gj = np.floor(y).astype(int), np.floor(x).astype(int)
    wy, wx = y - gi, x - gj
    inside = (gi >= 0) & (gi < lat.size - 1)
    on_strip = strip[np.clip(np.round(y).astype(int), 0, lat.size - 1), np.round(x).astype(int) % lon.size]
    area = areas(obs.lat.values, obs.lon.values)
    per_day = {}
    for k, day in enumerate(days):
        idx = np.where((obs.day.values == np.datetime64(day)) & inside)[0]
        if idx.size:
            per_day[k] = (gi[idx], gj[idx] % lon.size, (gj[idx] + 1) % lon.size, wy[idx], wx[idx], obs.val.values[idx],
                          obs.alg.values[idx], on_strip[idx], {a: m[idx] for a, m in area.items()})

    rows = []
    starts = STARTS if args.max_starts is None else STARTS[:args.max_starts]
    for t0 in starts:
        i0 = days.get_loc(t0)
        base = combined[i0 - N_OBS:i0].copy()
        so = sw_only[i0 - N_OBS:i0]
        for kind, level in EXPERIMENTS:
            removed = np.arange(N_OBS - level, N_OBS) if kind == "last_k" else np.array([d for d in range(N_OBS) if d != N_OBS - level])
            xin = np.full((DT,) + base.shape[1:], np.nan, np.float32)
            xin[:N_OBS] = base
            for d in removed:
                xin[d][so[d]] = np.nan
            xn = np.nan_to_num((xin - MEAN) / STD)
            with torch.no_grad():
                f = model(Batch(torch.from_numpy(xn)[None].to(device)))[0, N_OBS:N_OBS + N_LEAD].cpu().numpy() * STD + MEAN
            for L in range(1, N_LEAD + 1):
                iv = i0 + L - 1
                if iv not in per_day:
                    continue
                fc = f[L - 1]
                g0, j0, j1, a, b, val, is_alg, st, ar = per_day[iv]
                fi = (1 - a) * ((1 - b) * fc[g0, j0] + b * fc[g0, j1]) + a * ((1 - b) * fc[g0 + 1, j0] + b * fc[g0 + 1, j1])
                e = fi - val
                good = np.isfinite(e)
                for tname, tm in (("all6", good), ("alg", good & is_alg)):
                    for pname, pm in (("all", tm), ("strip", tm & st), ("off_strip", tm & ~st)):
                        for aname, am in ar.items():
                            v = e[pm & am]
                            rows.append((t0, kind, level, L, tname, pname, aname, v.size, v.sum(), (v ** 2).sum()))
        print(f"{pd.Timestamp.now():%H:%M} {t0.date()} done", flush=True)
    pd.DataFrame(rows, columns=["t0", "experiment", "level", "lead_day", "truth", "points", "area", "n", "sum_err", "sum_sq_err"]
                 ).to_csv(f"{args.out}/scores.csv", index=False)
    print("written", args.out)


if __name__ == "__main__":
    main()

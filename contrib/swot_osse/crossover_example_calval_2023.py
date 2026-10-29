"""One crossover of the SWOT 1-day orbit, shown in full: where it is, which cells are crossover / nearby swath / off swath, the
input of day t0-1, the truths of the valid day, and the day-1 forecasts with Nadir only, SWOT on t0-1 only and SWOT on all
14 input days (same forward pass as denial_calval_2023_crossover.py).

Region: the crossover diamond off Cape Hatteras (35.5 N, 74 W), where the Gulf Stream leaves the coast. Start date: the start of
the 83 (12 Apr - 3 Jul 2023) whose valid day t0 has the most AltiKa points within 150 km of the diamond, so that the
independent truth crosses it (chosen on the truth sampling only, not on the forecast scores).

Usage: python contrib/swot_osse/crossover_example_calval_2023.py OUT_DIR --model nadir|swotscratch|1day_orbit [--device cpu]
Writes OUT_DIR/crossover_example_<model>.png and crossover_example_<model>.csv (RMSE per class in the box).
"""
import argparse
import os
import sys
from glob import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import contrib.swot_osse.explain_tropics as et  # noqa: E402
from contrib.swot_osse.denial_calval_2023_crossover import (ALG, CLASSES, DT, MEAN, N_OBS, NADIR, STARTS, STD, SWOT,  # noqa: E402
                                                            point_classes)

CKPTS = {"nadir": None,
         "swotscratch": "/Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_errors_scratch/*/checkpoints/val_mse=*.ckpt",
         "1day_orbit": "/Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_1day_orbit/*/checkpoints/val_mse=*.ckpt"}
LABELS = {"nadir": "Nadir-trained UNet", "swotscratch": "UNet trained with SWOT errors (2024 orbit)", "1day_orbit": "UNet trained on the 1-day orbit"}
CENTER = (35.5, -74.0)
BOX = (31.0, 40.0, -78.0, -69.5)  # lat_min, lat_max, lon_min, lon_max
MASK = "/Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_crossover/crossover_mask.npz"
SHOWN = [("strip_xover", "crossover", "#d62728"), ("strip_0_150", "swath < 150 km", "#2a78d6"), ("strip_150_300", "swath 150–300 km", "#8fb8ea"),
         ("off_0_150", "off swath < 150 km", "#b9b8b3")]
CASES = [("nadir", "Nadir only"), ("t0m1", "Nadir + SWOT on t0−1 only"), ("all", "Nadir + SWOT on all 14 days")]
CASE_COLORS = {"nadir": "#52514e", "t0m1": "#2a78d6", "all": "#eb6834"}
SLA_KW = dict(cmap="RdBu_r", vmin=-0.4, vmax=0.4)
ERR_KW = dict(cmap="PuOr_r", vmin=-0.25, vmax=0.25)


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", required=True, choices=list(CKPTS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--replace", action="store_true", help="redraw an existing figure of this script")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    png = f"{args.out}/crossover_example_{args.model}.png"
    if args.replace and os.path.exists(png):
        os.remove(png)  # --replace: this script's own figure, redrawn
    if os.path.exists(png):
        raise SystemExit(f"{png} exists")
    if CKPTS[args.model]:
        et.CKPT = sorted(glob(CKPTS[args.model]), key=lambda f: float(f.split("val_mse=")[1].split("-")[0]))[0]
    model = et.load_model(args.device)
    print("checkpoint:", et.CKPT, flush=True)

    # truth sampling: AltiKa points near the diamond, per day
    files = sorted(f for f in glob(ALG) if any(f"/2023/{m:02d}/" in f for m in (4, 5, 6, 7)))
    alg = xr.open_mfdataset(files, combine="nested", concat_dim="time")
    a_lat, a_lon = alg.latitude.values, ((alg.longitude.values + 180) % 360) - 180
    a_val = (alg.sla_unfiltered - alg.lwe).values
    a_time = pd.DatetimeIndex(alg.time.values)
    near = np.isfinite(a_val) & (haversine_km(a_lat, a_lon, *CENTER) <= 150)
    counts = pd.Series(1, index=a_time[near].normalize()).groupby(level=0).sum().reindex(STARTS, fill_value=0)
    t0 = counts.idxmax()
    print("start", t0.date(), "AltiKa points within 150 km on t0:", counts.max(), flush=True)

    days = pd.date_range(t0 - pd.Timedelta(days=N_OBS), t0)
    nad = xr.open_dataset(NADIR).sla_filtered.sel(time=days).values.astype(np.float32)
    ds_sw = xr.open_dataset(SWOT)
    lat, lon = ds_sw.lat.values, ds_sw.lon.values
    sw = ds_sw.sla_filtered.reindex(time=days).values.astype(np.float32)
    sw[np.abs(sw) > 1] = np.nan
    sw_only = np.isfinite(sw) & ~np.isfinite(nad)
    combined = np.where(np.isfinite(nad), nad, sw)
    phase = ds_sw.sla_filtered.sel(time=slice("2023-03-30", "2023-07-08")).values
    phase[np.abs(phase) > 1] = np.nan
    strip = np.isfinite(phase).mean(0) >= 0.8
    xm = np.load(MASK)
    cell_class = point_classes(strip, xm["crossover"], xm["dist_km"])

    forecasts = {}
    for case, _ in CASES:
        x = np.full((DT,) + combined.shape[1:], np.nan, np.float32)
        x[:N_OBS] = combined[:N_OBS]
        removed = {"nadir": range(N_OBS), "t0m1": range(N_OBS - 1), "all": []}[case]
        for d in removed:
            x[d][sw_only[d]] = np.nan
        with torch.no_grad():
            f = model(et.Batch(torch.from_numpy(np.nan_to_num((x - MEAN) / STD))[None].to(args.device)))[0, N_OBS].cpu().numpy()
        forecasts[case] = f * STD + MEAN  # lead day 1 = day t0
        print(case, "done", flush=True)

    # crop to the box
    la0, la1, lo0, lo1 = BOX
    iy = (lat >= la0) & (lat <= la1)
    ix = (lon >= lo0) & (lon <= lo1)
    blat, blon = lat[iy], lon[ix]
    crop = lambda a: a[np.ix_(iy, ix)]  # noqa: E731
    cls = crop(cell_class)
    truth = crop(sw[N_OBS])  # SWOT on t0, never an input of this forecast
    in_nadir = crop(nad[N_OBS - 1])
    in_swot = np.where(crop(sw_only[N_OBS - 1]), crop(sw[N_OBS - 1]), np.nan)
    fc = {k: crop(v) for k, v in forecasts.items()}

    # AltiKa on t0 in the box, forecasts interpolated (nearest cell is enough at this scale)
    on_t0 = (a_time.normalize() == t0) & np.isfinite(a_val) & (a_lat >= la0) & (a_lat <= la1) & (a_lon >= lo0) & (a_lon <= lo1)
    t_lat, t_lon, t_val = a_lat[on_t0], a_lon[on_t0], a_val[on_t0]
    gi = np.clip(np.round((t_lat - blat[0]) / 0.25).astype(int), 0, blat.size - 1)
    gj = np.clip(np.round((t_lon - blon[0]) / 0.25).astype(int), 0, blon.size - 1)
    t_cls = cls[gi, gj]
    order = np.argsort(t_lat)

    rows = []
    for case, label in CASES:
        for name, _, _ in SHOWN:
            k = CLASSES.index(name)
            m = (cls == k) & np.isfinite(truth)
            e = (fc[case] - truth)[m]
            a = fc[case][gi, gj][t_cls == k] - t_val[t_cls == k]
            rows.append(dict(model=args.model, t0=t0.date(), case=case, points=name, rmse_swot_cm=100 * np.sqrt(np.mean(e ** 2)) if e.size else np.nan,
                             n_swot=e.size, rmse_altika_cm=100 * np.sqrt(np.mean(a ** 2)) if a.size else np.nan, n_altika=a.size))
    table = pd.DataFrame(rows)
    table.to_csv(f"{args.out}/crossover_example_{args.model}.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False))

    # figure
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 19), layout="constrained")
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.8])
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="#d9d8d4", edgecolor="#52514e", linewidth=0.4)

    def zoom(r, c, title):
        ax = fig.add_subplot(gs[r, c], projection=proj)
        ax.set_extent([lo0, lo1, la0, la1], proj)
        ax.add_feature(land, zorder=3)
        ax.contour(blon, blat, (cls == CLASSES.index("strip_xover")).astype(float), [0.5], colors="k", linewidths=1.2, transform=proj, zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#9a9994")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        return ax

    # row 1: where, which cells, input, truth
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="#d9d8d4"), zorder=2)
    gy, gx = np.where(strip[::2, ::2])
    ax.scatter(lon[::2][gx], lat[::2][gy], s=0.05, c="#8fb8ea", transform=proj, rasterized=True)
    cy, cx = np.where(xm["crossover"])
    ax.scatter(lon[cx], lat[cy], s=0.4, c="#d62728", transform=proj, rasterized=True)
    ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], "k", lw=1.5, transform=proj, zorder=5)
    ax.set_title("(a) Swath (blue), crossovers (red), region", loc="left", fontsize=9.5, fontweight="bold")

    ax = zoom(0, 1, "(b) Point classes in the scores")
    cmap = ListedColormap(["#ffffff"] + [c for _, _, c in SHOWN])
    shown = np.zeros(cls.shape)
    for i, (name, _, _) in enumerate(SHOWN, 1):
        shown[cls == CLASSES.index(name)] = i
    ax.pcolormesh(blon, blat, shown, cmap=cmap, vmin=-0.5, vmax=len(SHOWN) + 0.5, transform=proj)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SHOWN], labels=[lab for _, lab, _ in SHOWN], loc="lower right", fontsize=7)

    ax = zoom(0, 2, f"(c) Input on t0−1 ({(t0 - pd.Timedelta(days=1)):%d %b}):\nnadir tracks + SWOT")
    ax.pcolormesh(blon, blat, in_swot, transform=proj, **SLA_KW)
    ax.pcolormesh(blon, blat, in_nadir, transform=proj, **SLA_KW)

    ax = zoom(0, 3, f"(d) Truth on {t0:%d %b}:\nSWOT + AltiKa track (dots)")
    im = ax.pcolormesh(blon, blat, truth, transform=proj, **SLA_KW)
    ax.scatter(t_lon, t_lat, c=t_val, s=6, edgecolors="k", linewidths=0.2, transform=proj, zorder=5, **SLA_KW)
    fig.colorbar(im, ax=ax, shrink=0.75, label="SLA (m)")

    # row 2: day-1 forecasts and the SWOT imprint
    dy, dx = np.where(np.isfinite(truth))
    keep = (dy % 2 == 0) & (dx % 2 == 0)
    dot_lat, dot_lon, dot_val = blat[dy[keep]], blon[dx[keep]], truth[dy[keep], dx[keep]]
    for c, (case, label) in enumerate(CASES):
        ax = zoom(1, c, f"({'efg'[c]}) Forecast, {label.replace('Nadir + ', '')}\ndots: SWOT observed {t0:%d %b}")
        im = ax.pcolormesh(blon, blat, fc[case], transform=proj, **SLA_KW)
        # the SWOT truth of the valid day as dots on the same colour scale (every second cell): a dot that blends in = forecast right
        ax.scatter(dot_lon, dot_lat, c=dot_val, s=22, edgecolors="k", linewidths=0.4, transform=proj, zorder=5, **SLA_KW)
    fig.colorbar(im, ax=ax, shrink=0.8, label="SLA (m)", extend="both")
    ax = zoom(1, 3, "(h) SWOT imprint:\n(f) − (e)")
    im = ax.pcolormesh(blon, blat, fc["t0m1"] - fc["nadir"], transform=proj, **ERR_KW)
    fig.colorbar(im, ax=ax, shrink=0.8, label="m")

    # row 3: errors against the SWOT truth, and the AltiKa profile
    sel = table.set_index(["case", "points"])
    for c, (case, label) in enumerate(CASES):
        r = sel.loc[(case, "strip_xover")], sel.loc[(case, "strip_0_150")]
        ax = zoom(2, c, f"({'ijk'[c]}) ({'efg'[c]}) − SWOT; RMSE (cm):\ncrossover {r[0].rmse_swot_cm:.1f}, swath <150 km {r[1].rmse_swot_cm:.1f}")
        im = ax.pcolormesh(blon, blat, fc[case] - truth, transform=proj, **ERR_KW)
    fig.colorbar(im, ax=ax, shrink=0.8, label="forecast − SWOT (m)")
    ax = fig.add_subplot(gs[3, :2])
    ax.plot(t_lat[order], t_val[order], "k-", lw=1.8, label="AltiKa (truth)")
    for case, label in CASES:
        v = fc[case][gi, gj]
        rm = 100 * np.sqrt(np.nanmean((v - t_val) ** 2))
        ax.plot(t_lat[order], v[order], "-", color=CASE_COLORS[case], lw=1.4, label=f"{label} · RMSE {rm:.1f} cm")
    xo = t_cls == CLASSES.index("strip_xover")
    if xo.any():
        ax.axvspan(t_lat[xo].min(), t_lat[xo].max(), color="#d62728", alpha=0.12, lw=0, label="crossover cells")
    ax.set_xlabel("latitude along the AltiKa track (°N)")
    ax.set_ylabel("SLA (m)")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"(l) Along the AltiKa track, {t0:%d %b} (never an input)", loc="left", fontsize=9.5, fontweight="bold")

    # SWOT truth of the valid day against the forecasts along the swath: mean across the swath per latitude, on the SWOT cells
    ax = fig.add_subplot(gs[3, 2:])
    ok = np.isfinite(truth)
    rows_ok = ok.sum(1) >= 3
    mean_row = lambda a: np.where(rows_ok, np.nansum(np.where(ok, a, 0), 1) / np.maximum(ok.sum(1), 1), np.nan)  # noqa: E731
    ax.plot(blat, mean_row(truth), "k-", lw=1.8, label="SWOT (truth)")
    for case, label in CASES:
        rm = 100 * np.sqrt(np.mean((fc[case] - truth)[ok] ** 2))
        ax.plot(blat, mean_row(fc[case]), "-", color=CASE_COLORS[case], lw=1.4, label=f"{label} · RMSE {rm:.1f} cm (all SWOT cells)")
    xo_rows = blat[(cls == CLASSES.index("strip_xover")).any(1)]
    if xo_rows.size:
        ax.axvspan(xo_rows.min(), xo_rows.max(), color="#d62728", alpha=0.12, lw=0, label="crossover latitudes")
    ax.set_xlabel("latitude (°N)")
    ax.set_ylabel("SLA (m), mean across the swath")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"(m) Along the SWOT swath, {t0:%d %b}: SWOT vs day-1 forecasts", loc="left", fontsize=9.5, fontweight="bold")

    fig.suptitle(f"One crossover of the SWOT 1-day orbit, off Cape Hatteras · {LABELS[args.model]}\nStart t0 = {t0:%d %b %Y}, lead day 1. "
                 "Black contour: crossover cells; black line: AltiKa track", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(png, dpi=120)
    print(png)


if __name__ == "__main__":
    main()

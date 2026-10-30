"""One tropical swath of the SWOT 1-day orbit: the forecast of the first day (t0) with Nadir only, then with SWOT on one input
day only (t0-1, t0-2, ..., t0-10), and with SWOT on all 14 days. The 1-day orbit repeats every day, so each case puts the same
swath into the input, only older; the maps show how its imprint on the forecast fades with its age.

Region: the swath through the eastern equatorial Pacific near 106 W (tropical instability waves), 8 S - 8 N. Start date: the
start of the 83 (12 Apr - 3 Jul 2023) whose valid day t0 has the most AltiKa points on the swath in the box (chosen on the truth
sampling only, not on the forecast scores); with --pick signal, the start whose SWOT observation of t0 on the swath in the box
has the largest SLA variance (strongest signal; output names end in _signal). Same forward pass as crossover_example_calval_2023.py.

Usage: python contrib/swot_osse/tropics_example_calval_2023.py OUT_DIR --model nadir|swotscratch|1day_orbit [--pick altika|signal] [--device cpu]
Writes OUT_DIR/tropics_example_<model>[_signal].png and .csv (RMSE per case against SWOT and AltiKa in the box).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import contrib.swot_osse.explain_tropics as et  # noqa: E402
from contrib.swot_osse.crossover_example_calval_2023 import CKPTS, LABELS  # noqa: E402
from contrib.swot_osse.denial_calval_2023_crossover import ALG, DT, MEAN, N_OBS, NADIR, STARTS, STD, SWOT  # noqa: E402

BOX = (-8.0, 8.0, -112.0, -100.0)  # lat_min, lat_max, lon_min, lon_max
AGES = [1, 2, 3, 4, 5, 7, 10]
CASES = [("nadir", "Nadir only")] + [(f"t0m{j}", f"SWOT on t0−{j} only") for j in AGES] + [("all", "SWOT on all 14 days")]
AGE_COLORS = dict(zip(AGES, plt.cm.Blues(np.linspace(0.95, 0.35, len(AGES)))))
COLORS = {"nadir": "#52514e", "all": "#eb6834"} | {f"t0m{j}": AGE_COLORS[j] for j in AGES}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", required=True, choices=list(CKPTS))
    p.add_argument("--pick", default="altika", choices=["altika", "signal"], help="start date: most AltiKa points on the swath, or strongest SWOT signal")
    p.add_argument("--device", default="cpu")
    p.add_argument("--replace", action="store_true", help="redraw an existing figure of this script")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.out}/tropics_example_{args.model}" + ("_signal" if args.pick == "signal" else "")
    png = f"{stem}.png"
    if args.replace and os.path.exists(png):
        os.remove(png)  # --replace: this script's own figure, redrawn
    if os.path.exists(png):
        raise SystemExit(f"{png} exists")
    if CKPTS[args.model]:
        et.CKPT = sorted(glob(CKPTS[args.model]), key=lambda f: float(f.split("val_mse=")[1].split("-")[0]))[0]
    model = et.load_model(args.device)
    print("checkpoint:", et.CKPT, flush=True)

    ds_sw = xr.open_dataset(SWOT)
    lat, lon = ds_sw.lat.values, ds_sw.lon.values
    phase = ds_sw.sla_filtered.sel(time=slice("2023-03-30", "2023-07-08")).values
    phase[np.abs(phase) > 1] = np.nan
    strip = np.isfinite(phase).mean(0) >= 0.8
    la0, la1, lo0, lo1 = BOX
    iy = (lat >= la0) & (lat <= la1)
    ix = (lon >= lo0) & (lon <= lo1)
    blat, blon = lat[iy], lon[ix]
    crop = lambda a: a[..., iy, :][..., ix]  # noqa: E731
    bstrip = crop(strip)

    # truth sampling: AltiKa points on the swath in the box, per day
    files = sorted(f for f in glob(ALG) if any(f"/2023/{m:02d}/" in f for m in (4, 5, 6, 7)))
    alg = xr.open_mfdataset(files, combine="nested", concat_dim="time")
    a_lat, a_lon = alg.latitude.values, ((alg.longitude.values + 180) % 360) - 180
    a_val = (alg.sla_unfiltered - alg.lwe).values
    a_time = pd.DatetimeIndex(alg.time.values).normalize()
    inbox = np.isfinite(a_val) & (a_lat >= la0) & (a_lat <= la1) & (a_lon >= lo0) & (a_lon <= lo1)
    gi = np.clip(np.round((a_lat - blat[0]) / 0.25).astype(int), 0, blat.size - 1)
    gj = np.clip(np.round((a_lon - blon[0]) / 0.25).astype(int), 0, blon.size - 1)
    on_strip = inbox & bstrip[gi, gj]
    counts = pd.Series(1, index=a_time[on_strip]).groupby(level=0).sum().reindex(STARTS, fill_value=0)
    t0 = counts.idxmax()
    if args.pick == "signal":
        # SLA variance of the SWOT observation of t0 on the swath in the box, starts with at least half the swath observed
        obs = crop(ds_sw.sla_filtered.reindex(time=STARTS).values)
        obs[(np.abs(obs) > 1) | ~bstrip] = np.nan
        seen = np.isfinite(obs).sum((1, 2)) >= 0.5 * bstrip.sum()
        var = pd.Series(np.where(seen, np.nanvar(obs, (1, 2)), np.nan), index=STARTS)
        t0 = var.idxmax()
        print("start", t0.date(), f"SWOT SLA std on the swath on t0: {100 * np.sqrt(var.max()):.1f} cm (median over starts "
              f"{100 * np.sqrt(var.median()):.1f} cm)", flush=True)
    print("start", t0.date(), "AltiKa points on the swath in the box on t0:", counts[t0], flush=True)

    days = pd.date_range(t0 - pd.Timedelta(days=N_OBS), t0)
    nad = xr.open_dataset(NADIR).sla_filtered.sel(time=days).values.astype(np.float32)
    sw = ds_sw.sla_filtered.reindex(time=days).values.astype(np.float32)
    sw[np.abs(sw) > 1] = np.nan
    sw_only = np.isfinite(sw) & ~np.isfinite(nad)
    combined = np.where(np.isfinite(nad), nad, sw)

    forecasts, n_input = {}, {}
    for case, _ in CASES:
        x = np.full((DT,) + combined.shape[1:], np.nan, np.float32)
        x[:N_OBS] = combined[:N_OBS]
        kept = {"nadir": [], "all": list(range(N_OBS))}[case] if case in ("nadir", "all") else [N_OBS - int(case[3:])]
        n_input[case] = int(sum(crop(sw_only[d]).sum() for d in kept))
        for d in range(N_OBS):
            if d not in kept:
                x[d][sw_only[d]] = np.nan
        with torch.no_grad():
            f = model(et.Batch(torch.from_numpy(np.nan_to_num((x - MEAN) / STD))[None].to(args.device)))[0, N_OBS].cpu().numpy()
        forecasts[case] = crop(f * STD + MEAN)  # first forecast day = day t0 ("lead day 1" in the scores)
        print(case, "done", flush=True)

    truth = crop(sw[N_OBS])  # SWOT on t0, never an input
    in_nadir = crop(nad[N_OBS - 1])
    in_swot = np.where(crop(sw_only[N_OBS - 1]), crop(sw[N_OBS - 1]), np.nan)
    on_t0 = inbox & (a_time == t0)
    t_lat, t_lon, t_val, t_strip = a_lat[on_t0], a_lon[on_t0], a_val[on_t0], bstrip[gi[on_t0], gj[on_t0]]
    ti, tj = gi[on_t0], gj[on_t0]

    rows = []
    ok = np.isfinite(truth)
    for case, label in CASES:
        e = (forecasts[case] - truth)[ok]
        a = forecasts[case][ti, tj] - t_val
        rows.append(dict(model=args.model, t0=t0.date(), case=case, n_swot_input=n_input[case], rmse_swot_cm=100 * np.sqrt(np.mean(e ** 2)), n_swot=e.size,
                         rmse_altika_swath_cm=100 * np.sqrt(np.mean(a[t_strip] ** 2)), n_altika_swath=int(t_strip.sum()),
                         rmse_altika_box_cm=100 * np.sqrt(np.mean(a ** 2)), n_altika_box=a.size))
    table = pd.DataFrame(rows)
    table.to_csv(f"{stem}.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False))

    # colour scales from the truth and the t0-1 imprint; SLA is centred on the box median (in 2023 the eastern equatorial Pacific
    # is high everywhere, El Nino onset, and a scale centred on zero saturates)
    mid = np.round(np.nanmedian(forecasts["nadir"]) / 0.01) * 0.01
    half = max(0.04, np.ceil(np.nanpercentile(np.abs(truth - mid), 98) / 0.01) * 0.01)
    sla_kw = dict(cmap="RdBu_r", vmin=mid - half, vmax=mid + half)
    imax = max(0.02, np.ceil(np.nanpercentile(np.abs(forecasts["t0m1"] - forecasts["nadir"]), 99) / 0.01) * 0.01)
    imp_kw = dict(cmap="PuOr_r", vmin=-imax, vmax=imax)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 27), layout="constrained")
    gs = fig.add_gridspec(6, 4, height_ratios=[1, 1, 1, 1, 1, 0.75])
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="#d9d8d4", edgecolor="#52514e", linewidth=0.4)
    dy, dx = np.where(ok)
    keep = (dy % 2 == 0) & (dx % 2 == 0)
    dot_lat, dot_lon, dot_val = blat[dy[keep]], blon[dx[keep]], truth[dy[keep], dx[keep]]

    def zoom(r, c, title):
        ax = fig.add_subplot(gs[r, c], projection=proj)
        ax.set_extent([lo0, lo1, la0, la1], proj)
        ax.add_feature(land, zorder=3)
        ax.contour(blon, blat, bstrip.astype(float), [0.5], colors="#52514e", linewidths=0.6, transform=proj, zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#9a9994")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        return ax

    def forecast_panel(r, c, case, label, letter):
        none = case.startswith("t0m") and n_input[case] == 0
        ax = zoom(r, c, f"({letter}) Forecast, {label}" + (" — no SWOT\nin the box that day (= Nadir only)" if none else
                                                          f"\nRMSE vs SWOT {table.set_index('case').rmse_swot_cm[case]:.1f} cm"))
        im = ax.pcolormesh(blon, blat, forecasts[case], transform=proj, **sla_kw)
        # the SWOT truth of t0 as dots on the same colour scale (every second cell): a dot that blends in = forecast right
        ax.scatter(dot_lon, dot_lat, c=dot_val, s=14, edgecolors="k", linewidths=0.3, transform=proj, zorder=5, **sla_kw)
        return im

    # row 1: where, input, truth, reference forecast
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="#d9d8d4"), zorder=2)
    gy, gx = np.where(strip[::2, ::2])
    ax.scatter(lon[::2][gx], lat[::2][gy], s=0.05, c="#8fb8ea", transform=proj, rasterized=True)
    ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], "k", lw=1.5, transform=proj, zorder=5)
    ax.set_title("(a) 1-day-orbit swath (blue) and region", loc="left", fontsize=9.5, fontweight="bold")

    ax = zoom(0, 1, f"(b) Input on t0−1 ({(t0 - pd.Timedelta(days=1)):%d %b}):\nnadir tracks + SWOT")
    ax.pcolormesh(blon, blat, in_swot, transform=proj, **sla_kw)
    ax.pcolormesh(blon, blat, in_nadir, transform=proj, **sla_kw)

    ax = zoom(0, 2, f"(c) Truth on t0 = {t0:%d %b}:\nSWOT + AltiKa track (dots)")
    ax.pcolormesh(blon, blat, truth, transform=proj, **sla_kw)
    ax.scatter(t_lon, t_lat, c=t_val, s=6, edgecolors="k", linewidths=0.2, transform=proj, zorder=5, **sla_kw)

    im = forecast_panel(0, 3, "nadir", "Nadir only (reference)", "d")
    fig.colorbar(im, ax=ax, shrink=0.75, label="SLA (m)", extend="both")

    # rows 2-3: forecasts with SWOT on one day only, then on all days; rows 4-5: their difference from Nadir only
    others, imprint_axes = CASES[1:], []
    for i, (case, label) in enumerate(others):
        r, c = 1 + i // 4, i % 4
        im = forecast_panel(r, c, case, label, "efghijkl"[i])
        ax = zoom(r + 2, c, f"({'mnopqrst'[i]}) SWOT imprint: ({'efghijkl'[i]}) − (d)")
        im2 = ax.pcolormesh(blon, blat, forecasts[case] - forecasts["nadir"], transform=proj, **imp_kw)
        imprint_axes.append(ax)
    fig.colorbar(im2, ax=imprint_axes, shrink=0.5, label="forecast − Nadir-only forecast (m)", extend="both")

    # row 6: along the swath, and RMSE against the age of the SWOT day
    ax = fig.add_subplot(gs[5, :2])
    rows_ok = ok.sum(1) >= 3
    mean_row = lambda a: np.where(rows_ok, np.nansum(np.where(ok, a, 0), 1) / np.maximum(ok.sum(1), 1), np.nan)  # noqa: E731
    ax.plot(blat, mean_row(truth), "k-", lw=2, label="SWOT on t0 (truth)")
    for case, label in CASES:
        if case in ("nadir", "all") or case in ("t0m1", "t0m2", "t0m3", "t0m5", "t0m10"):
            ax.plot(blat, mean_row(forecasts[case]), "-", color=COLORS[case], lw=1.4, label=label)
    ax.set_xlabel("latitude (°N)")
    ax.set_ylabel("SLA (m), mean across the swath")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_title(f"(u) Along the SWOT swath, {t0:%d %b}: SWOT vs forecasts", loc="left", fontsize=9.5, fontweight="bold")

    ax = fig.add_subplot(gs[5, 2:])
    t = table.set_index("case")
    curves = [("rmse_swot_cm", "vs SWOT (all swath cells)", "#2a78d6")]
    if t.n_altika_swath.iloc[0] > 0:
        curves.append(("rmse_altika_swath_cm", "vs AltiKa on the swath", "#1f9e74"))
    if t.n_altika_box.iloc[0] > 1.1 * t.n_altika_swath.iloc[0]:  # AltiKa can run along the swath; then the box curve is the same
        curves.append(("rmse_altika_box_cm", "vs AltiKa, whole box", "#9a9994"))
    for col, lab, color in curves:
        ax.plot(AGES, [t.loc[f"t0m{j}", col] for j in AGES], "o-", color=color, lw=1.6, label=lab)
        ax.axhline(t.loc["nadir", col], color=color, ls="--", lw=1)
        ax.axhline(t.loc["all", col], color=color, ls=":", lw=1.2)
    ax.plot([], [], "k--", lw=1, label="Nadir only")
    ax.plot([], [], "k:", lw=1.2, label="SWOT on all 14 days")
    ax.set_xticks(AGES)
    ax.set_xlabel("age of the single SWOT day in the input: SWOT on t0−j only")
    ax.set_ylabel("RMSE of the t0 forecast (cm)")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("(v) Error against the age of the SWOT day", loc="left", fontsize=9.5, fontweight="bold")

    fig.suptitle(f"A tropical swath of the SWOT 1-day orbit, eastern equatorial Pacific · {LABELS[args.model]}\nStart t0 = {t0:%d %b %Y} "
                 f"({'strongest SWOT signal on the swath' if args.pick == 'signal' else 'most AltiKa points on the swath'}), "
                 "first forecast day (t0). Grey contour: swath; dots on the forecasts: SWOT observed on t0", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(png, dpi=100)
    print(png)


if __name__ == "__main__":
    main()

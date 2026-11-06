"""Regional NE Pacific inference of the SLA UNet (same windows as the global OSE pipeline, one regional patch):
for every forecast issued on t0 the window covers t0-14..t0+14, observations of days t0-14..t0-1 only (model.mask_batch),
and output day 14+k is the forecast of lead k, valid on t0+k. Writes test_data_<13..20>.nc (var out, time = valid date,
regional lat/lon) like the pipeline, so parcels_3day_sla_test.py can read it.
Inputs: 2023 = NRT gridded L3 (altimetry_traces/nrt_sla/gridded/gridded_input.nc, sla_filtered, same as the global 2023
runs); 2024 = CLS nadir gridded L3 without SWOT (swot_traces/cls/gridded_obs_sla_0.25_nadirs_2024_without_swon.nc).
Usage: python contrib/neurost_regional/infer_regional.py CKPT OUT_XP_PREFIX [--years 2023 2024]
  -> /Odyssey/public/glorys/rec/<OUT_XP_PREFIX>_<year>/nrt_sla/test_data_<k>.nc
Observation-perturbed ensemble: --members N [--drop-frac 0.2 --drop-km 200] writes <OUT_XP_PREFIX>_m<i>_<year> for i < N:
member i removes, independently for every day, the observations where a random Gaussian field smoothed at ~drop-km
(seed i) is below its drop-frac quantile over the observed points, i.e. ~drop-frac of the data in pieces of track of a
few hundred km (the gridded input has no satellite identity, so whole passes cannot be dropped). The same perturbed input
serves every window of the member, so a dropped piece is missing for all the forecasts that see that day.
"""
import argparse
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from hydra import compose, initialize_config_dir

from contrib.drifter_geos.finetune import TrainingItemOSE

INPUTS = {2023: ("/Odyssey/public/altimetry_traces/nrt_sla/gridded/gridded_input.nc", "sla_filtered"),
          2024: ("/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_nadirs_2024_without_swon.nc", "sla_filtered")}
REC = "/Odyssey/public/glorys/rec"
LEADS = range(-1, 7)       # test_data_13 (analysis of t0-1) .. test_data_20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_prefix")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--members", type=int, default=0)
    ap.add_argument("--drop-frac", type=float, default=0.2)
    ap.add_argument("--drop-km", type=float, default=200.0)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        cfg = compose(config_name="main", overrides=["xp=finetune_SLA_UNet_OSE_2017_2022_nepac_regional", "model.init_ckpt=null"])
    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(a.ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt}: missing {missing}, unexpected {unexpected}", flush=True)
    model = model.cuda().eval()
    m, s = cfg.datamodule.norm_stats
    lat_sl = slice(*cfg.domain.train.lat._args_)
    lon_sl = slice(*cfg.domain.train.lon._args_)
    for year in a.years:
        path, var = INPUTS[year]
        da = xr.open_dataset(path)[var]
        if "latitude" in da.dims:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
        t0s = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        days = pd.date_range(t0s[0] - pd.Timedelta(days=14), t0s[-1] + pd.Timedelta(days=14), freq="D")
        inp = da.sel(lat=lat_sl, lon=lon_sl).reindex(time=days).load()
        lat, lon = inp.lat.values, inp.lon.values
        assert inp.shape[1:] == (180, 220), inp.shape
        x_full = (inp.values - m) / s
        for member in (range(a.members) if a.members else [None]):
            x = torch.tensor(x_full if member is None else perturb(x_full, member, a.drop_frac, a.drop_km, lat), dtype=torch.float32)
            run_year(model, x, t0s, lat, lon, m, s, a, year, a.out_prefix if member is None else f"{a.out_prefix}_m{member}")


def perturb(x, seed, frac, km, lat):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    out = x.copy()
    sy = km / 4 / (111.0 * 0.25)                                     # sigma ~ km/4 -> structures of ~km
    sx = sy / np.cos(np.radians(lat.mean()))
    for t in range(x.shape[0]):
        obs = np.isfinite(x[t])
        if obs.sum() < 10:
            continue
        f = gaussian_filter(rng.standard_normal(x.shape[1:]), (sy, sx), mode="wrap")
        out[t][obs & (f < np.quantile(f[obs], frac))] = np.nan
    return out


def run_year(model, x, t0s, lat, lon, m, s, a, year, prefix):
    if True:
        out = {k: np.full((len(t0s), len(lat), len(lon)), np.nan, np.float32) for k in LEADS}      # indexed by valid date
        vidx = {d: i for i, d in enumerate(t0s)}
        with torch.no_grad():
            for b0 in range(0, len(t0s), a.batch):
                idx = list(range(b0, min(b0 + a.batch, len(t0s))))       # window of t0s[i] = days[i : i + 29]
                xb = torch.stack([x[i:i + 29] for i in idx]).cuda()
                item = model.mask_batch(TrainingItemOSE(xb, xb, xb))
                with torch.set_grad_enabled(False):
                    y = model(batch=item).float().cpu().numpy() * s + m
                for j, i in enumerate(idx):
                    for k in LEADS:
                        v = t0s[i] + pd.Timedelta(days=k)
                        if v in vidx:
                            out[k][vidx[v]] = y[j, 14 + k]
        odir = f"{REC}/{prefix}_{year}/nrt_sla"
        os.makedirs(odir, exist_ok=True)
        for k in LEADS:
            xr.Dataset({"out": (("time", "lat", "lon"), out[k])}, coords={"time": t0s, "lat": lat, "lon": lon}).to_netcdf(f"{odir}/test_data_{14 + k}.nc")
        print(f"{year}: written {odir}/test_data_13..20.nc", flush=True)


if __name__ == "__main__":
    main()

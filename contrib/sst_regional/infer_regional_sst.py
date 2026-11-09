"""Regional NE Pacific inference of the SLA+SST UNet (contrib/sst_regional/sst_finetune.py), same windows and outputs as
contrib/neurost_regional/infer_regional.py: forecast issued on t0 sees SLA and SST of t0-14..t0-1 only (model.mask_batch),
output day 14+k = lead k, valid on t0+k -> /Odyssey/public/glorys/rec/<OUT_PREFIX>_<year>/nrt_sla/test_data_<13..20>.nc
Usage: python contrib/sst_regional/infer_regional_sst.py CKPT OUT_PREFIX [--years 2023 2024]
"""
import argparse
import glob
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from hydra import compose, initialize_config_dir

from contrib.neurost_regional.infer_regional import INPUTS, LEADS, REC
from contrib.sst_regional.sst_finetune import TrainingItemSST, load_sst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_prefix")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--xp", default="finetune_SLA_UNet_OSE_2017_2022_nepac_regional_sst")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        cfg = compose(config_name="main", overrides=[f"xp={a.xp}", "model.init_ckpt=null"])
    model = hydra.utils.instantiate(cfg.model)
    sd = torch.load(a.ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt}: missing {missing}, unexpected {unexpected}", flush=True)
    model = model.cuda().eval()
    m, s = cfg.datamodule.norm_stats
    ms, ss = cfg.datamodule.sst_stats
    lat_sl, lon_sl = slice(*cfg.domain.train.lat._args_), slice(*cfg.domain.train.lon._args_)
    sst_all = load_sst(cfg.paths.sst, cfg.datamodule.input_da.get("sst_highpass_deg"))
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
        sst = sst_all.sel(lat=lat, lon=lon, method="nearest").reindex(time=days)
        assert np.abs(sst.lat.values - lat).max() < 1e-3 and np.abs(sst.lon.values - lon).max() < 1e-3
        x = torch.tensor((inp.values - m) / s, dtype=torch.float32)
        z = torch.tensor((sst.values - ms) / ss, dtype=torch.float32)
        out = {k: np.full((len(t0s), len(lat), len(lon)), np.nan, np.float32) for k in LEADS}
        vidx = {d: i for i, d in enumerate(t0s)}
        with torch.no_grad():
            for b0 in range(0, len(t0s), a.batch):
                idx = list(range(b0, min(b0 + a.batch, len(t0s))))
                xb = torch.stack([x[i:i + 29] for i in idx]).cuda()
                zb = torch.stack([z[i:i + 29] for i in idx]).cuda()
                item = model.mask_batch(TrainingItemSST(xb, xb, xb, zb))
                y = model(batch=item).float().cpu().numpy() * s + m
                for j, i in enumerate(idx):
                    for k in LEADS:
                        v = t0s[i] + pd.Timedelta(days=k)
                        if v in vidx:
                            out[k][vidx[v]] = y[j, 14 + k]
        odir = f"{REC}/{a.out_prefix}_{year}/nrt_sla"
        os.makedirs(odir, exist_ok=True)
        for k in LEADS:
            xr.Dataset({"out": (("time", "lat", "lon"), out[k])}, coords={"time": t0s, "lat": lat, "lon": lon}).to_netcdf(f"{odir}/test_data_{14 + k}.nc")
        print(f"{year}: written {odir}/test_data_13..20.nc", flush=True)


if __name__ == "__main__":
    main()

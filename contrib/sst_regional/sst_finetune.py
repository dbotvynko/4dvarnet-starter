"""Regional NE Pacific UNet forecast with an SST input channel (step 1 of a NeurOST-style forecast).

Same set-up as the regional velocity fine-tune (contrib/drifter_geos/finetune.py, config
finetune_SLA_UNet_OSE_2017_2022_nepac_regional.yaml) plus MUR25 SST (contrib/sst_regional/download_mur25_nepac.py) as 29
extra input channels: the solver sees [SLA(29 days), SST(29 days)]. SST is masked exactly like the altimetry (window days
>= 14, i.e. t0 onwards, are NaN -> 0), so a forecast issued at t0 uses SST up to t0-1 (MUR is available ~1 day after its
date). The first convolution of the pre-trained 29-channel UNet is widened with zero weights for the SST channels, so the
fine-tune starts from exactly the SLA-only model and the gain can be attributed to SST.
"""
import functools as ft
import glob
from collections import namedtuple

import numpy as np
import torch
import xarray as xr

from contrib.drifter_geos.finetune import GeosDataModule, GeosDrifterFineTune, XrDatasetWithTime, open_duacs_tgt_l3_input
from src.models_UNet import UNet

TrainingItemSST = namedtuple("TrainingItemSST", ["input", "input_complete", "tgt", "sst"])


def highpass_sst(sst, sigma_deg):
    """SST minus its Gaussian smoothing (sigma in degrees of latitude, land excluded by normalised convolution), day by day:
    keeps fronts and eddies, removes the large-scale meridional gradient and the seasonal cycle. Land stays NaN."""
    from scipy.ndimage import gaussian_filter
    dlat = float(abs(sst.lat.values[1] - sst.lat.values[0]))
    sig = (0, sigma_deg / dlat, sigma_deg / dlat / np.cos(np.radians(float(sst.lat.mean()))))
    v = sst.values.astype(np.float64)
    ok = np.isfinite(v)
    num = gaussian_filter(np.where(ok, v, 0.0), sig, mode="nearest")
    den = gaussian_filter(ok.astype(np.float64), sig, mode="nearest")
    out = np.where(ok, v - num / np.maximum(den, 1e-6), np.nan)
    return sst.copy(data=out.astype(np.float32))


def load_sst(sst_files, sst_highpass_deg=None):
    sst = xr.concat([xr.open_dataset(f)["sst"].load() for f in sorted(glob.glob(sst_files))], "time")
    if sst_highpass_deg:
        sst = highpass_sst(sst, sst_highpass_deg)
        print(f"SST high-passed (sigma {sst_highpass_deg} deg)", flush=True)
    return sst


def open_tgt_l3_sst_input(path, real_traces, domain, sst_files, tgt_files=None, var_tgt="sla", var_input="sla", sst_highpass_deg=None):
    """(variable = input, input_complete, tgt, sst; time; lat; lon). SST in deg C on the same grid (nearest = same points).
    sst_highpass_deg: if set, SST minus its Gaussian smoothing at that scale (highpass_sst)."""
    da = open_duacs_tgt_l3_input(path, real_traces, domain, var_tgt=var_tgt, var_input=var_input, tgt_files=tgt_files)
    sst = load_sst(sst_files, sst_highpass_deg)
    sst = sst.sel(lat=da.lat, lon=da.lon, method="nearest")
    assert np.abs(sst.lat.values - da.lat.values).max() < 1e-3 and np.abs(sst.lon.values - da.lon.values).max() < 1e-3
    sst = sst.assign_coords(lat=da.lat.values, lon=da.lon.values).reindex(time=da.time)
    print(f"SST: {int(np.isfinite(sst.isel(lat=90, lon=110)).sum())}/{sst.sizes['time']} days with data at the domain centre", flush=True)
    sst = sst.transpose("time", "lat", "lon").expand_dims(variable=["sst"])
    return xr.concat([da, sst.astype(da.dtype)], "variable")


class GeosDataModuleSST(GeosDataModule):
    def __init__(self, *args, sst_stats=(15.0, 7.0), **kwargs):
        self.sst_stats = tuple(sst_stats)
        super().__init__(*args, **kwargs)

    def post_fn(self):
        m, s = self.norm_stats()
        ms, ss = self.sst_stats

        def normalize(x):
            return (x - m) / s
        return ft.partial(ft.reduce, lambda i, f: f(i), [
            TrainingItemSST._make,
            lambda it: it._replace(tgt=normalize(it.tgt), input=normalize(it.input), input_complete=normalize(it.input_complete),
                                   sst=(it.sst - ms) / ss),
        ])


class UNetSST(UNet):
    """UNet whose first block takes 2 * n_channels inputs: [SLA window, SST window]."""

    def __init__(self, n_channels=29, n_classes=29, **kwargs):
        super().__init__(n_channels=2 * n_channels, n_classes=n_classes, **kwargs)
        self.n_sla = n_channels

    def forward(self, x):
        z = x._replace(input=torch.cat([torch.nan_to_num(x.input), torch.nan_to_num(x.sst)], 1))
        return super().forward(z)


class GeosDrifterFineTuneSST(GeosDrifterFineTune):
    def __init__(self, *args, init_ckpt=None, **kwargs):
        super().__init__(*args, init_ckpt=None, **kwargs)
        if init_ckpt:
            sd = torch.load(init_ckpt, map_location="cpu")["state_dict"]
            k = "solver.inc.double_conv.0.weight"
            w_old, w_new = sd[k], self.state_dict()[k]
            if w_old.shape != w_new.shape:            # SLA-only checkpoint: zero weights for the SST channels
                w = torch.zeros_like(w_new)
                w[:, :w_old.shape[1]] = w_old
                sd = dict(sd, **{k: w})
                print(f"widened {k}: {tuple(w_old.shape)} -> {tuple(w.shape)} (SST weights = 0)", flush=True)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"init from {init_ckpt}: missing {missing}, unexpected {unexpected}", flush=True)

    @staticmethod
    def mask_batch(batch):
        """observations (SLA and SST) of window days >= T // 2 (t0 onwards) removed"""
        inp, sst = batch.input.clone(), batch.sst.clone()
        T = inp.shape[1]
        inp[:, T // 2:] = np.nan
        sst[:, T // 2:] = np.nan
        return batch._replace(input=inp, sst=sst)

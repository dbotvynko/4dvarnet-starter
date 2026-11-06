"""Fine-tuning of the OSE SLA UNet (2017-2022, DUACS losses) with two velocity terms:

  1. geostrophic-current loss: geostrophic currents of the predicted SLA vs those of the DUACS target maps (same finite
     differences on both, so the MDT cancels), on the forecast days of the window;
  2. drifter loss: absolute geostrophic currents of the prediction (SLA + CNES-CLS22 MDT) bilinearly sampled at the daily
     positions of drogued GDP drifters, vs their geostrophic velocity (daily mean minus GlobCurrent Ekman 15 m,
     contrib/drifter_geos/build_drifter_geos_targets.py), Huber loss (ageostrophic noise ~10 cm/s rms).
Both only where 5 <= |lat| <= 75 (geostrophy) and on the forecast days of the window (window days 14..20 = leads 0..6).

The SLA terms are those of the August 2025 training (commit 5b5c485): 50 * (MSE + 1 deg coarsened MSE + Sobel gradient + DoG)
vs the DUACS targets. Training windows cover the full grid (patch 680 x 1440 = domain), so a window is located by its start
date only: the dataset returns (item, start time index).
"""
import functools as ft
from collections import namedtuple

import kornia.filters as kfilts
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from contrib.forecast_plus.models import Plus4dVarNetForecastPatchGPU_UNet
from contrib.moving_patches.movpatch_data_tests import MovingPatchDataModuleFastRecGPU, XrDatasetMovingPatchFastRecGPU
from src.DoG import dog_kornia
from src.utils import get_forecast_wei_adaptable_per_resolution

TrainingItemOSE = namedtuple("TrainingItemOSE", ["input", "input_complete", "tgt"])
G, OMEGA, M_PER_DEG = 9.81, 7.2921e-5, 111.0e3


def open_duacs_tgt_l3_input(path, real_traces, domain, var_tgt="sla", var_input="sla", tgt_files=None):
    """Like contrib.data_loading.data.open_glorys12_data_sla_OSE but opens only the SLA of the (55 GB, 6-variable) DUACS
    file: input = input_complete = gridded L3, tgt = DUACS SLA. Returns a DataArray (variable, time, lat, lon).
    tgt_files: optional glob of files on the same grid (e.g. NeurOST, contrib/neurost_regional/download_neurost_nepac.py)
    that replace the DUACS target; the DUACS file then only gives the grid and the time axis."""
    tgt = xr.open_dataset(path)[var_tgt]
    if tgt_files:
        import glob
        alt = xr.concat([xr.open_dataset(f)[var_tgt].load() for f in sorted(glob.glob(tgt_files))], "time")
        if "latitude" in tgt.dims:
            tgt = tgt.rename({"latitude": "lat", "longitude": "lon"})
        ref = tgt.sel(domain)
        tgt = alt.sel(lat=ref.lat, lon=ref.lon, method="nearest").assign_coords(lat=ref.lat.values, lon=ref.lon.values)
        tgt = tgt.reindex(time=ref.time)
        print(f"target replaced by {tgt_files}: {int(np.isfinite(tgt.isel(lat=len(ref.lat) // 2, lon=len(ref.lon) // 2)).sum())} valid days")
    inp = xr.open_dataset(real_traces)[var_input]
    if "latitude" in tgt.dims:
        tgt = tgt.rename({"latitude": "lat", "longitude": "lon"})
    if "latitude" in inp.dims:
        inp = inp.rename({"latitude": "lat", "longitude": "lon"})
    tgt = tgt.sel(domain).load()
    inp = inp.sel(domain).reindex(time=tgt.time).load()
    inp = inp.assign_coords(lat=tgt.lat.values, lon=tgt.lon.values)
    ds = xr.Dataset({"input": inp, "input_complete": inp, "tgt": tgt})
    return ds[list(TrainingItemOSE._fields)].transpose("time", "lat", "lon").to_array()


class XrDatasetWithTime(XrDatasetMovingPatchFastRecGPU):
    """Same patches, plus the window's first day (days since 1970-01-01)."""

    def __getitem__(self, item):
        t_idx = int(np.unravel_index(item, tuple(self.ds_size.values()))[list(self.ds_size).index("time")]) * self.strides.get("time", 1)
        day = int((pd.Timestamp(self.da.time.values[t_idx]) - pd.Timestamp("1970-01-01")).days)
        return super().__getitem__(item), day


class GeosDataModule(MovingPatchDataModuleFastRecGPU):
    def post_fn(self):
        m, s = self.norm_stats()

        def normalize(x):
            return (x - m) / s
        return ft.partial(ft.reduce, lambda i, f: f(i), [
            TrainingItemOSE._make,
            lambda it: it._replace(tgt=normalize(it.tgt), input=normalize(it.input), input_complete=normalize(it.input_complete)),
        ])

    def setup(self, stage="test"):
        post_fn = self.post_fn()
        self.train_ds = XrDatasetWithTime(self.input_da.sel(self.domains["train"]), **self.xrds_kw, postpro_fn=post_fn, rand=True)
        self.val_ds = XrDatasetWithTime(self.input_da.sel(self.domains["val"]), **self.xrds_kw, postpro_fn=post_fn, rand=False)
        self.test_ds = XrDatasetWithTime(self.input_da.sel(self.domains["test"]), **self.xrds_kw, postpro_fn=post_fn, rand=False)


class GeosDrifterFineTune(Plus4dVarNetForecastPatchGPU_UNet):
    def __init__(self, *args, init_ckpt=None, mdt_path=None, drifter_dir=None, drifter_years=(2017, 2018, 2019, 2020, 2021, 2022),
                 w_sla=1.0, w_geos=1.0, w_drift=1.0, vel_scale=0.1, huber_delta=2.0, lead_days=(14, 15, 16, 17, 18, 19, 20),
                 lat_band=(5.0, 75.0), periodic_lon=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_sla, self.w_geos, self.w_drift = w_sla, w_geos, w_drift
        self.vel_scale, self.huber_delta = vel_scale, huber_delta      # velocities divided by vel_scale (m/s) in the losses
        self.lead_days, self.lat_band = list(lead_days), lat_band
        self.mdt_path, self.drifter_dir, self.drifter_years = mdt_path, drifter_dir, list(drifter_years)
        self._grid_ready = False
        self.periodic_lon = periodic_lon        # False for a regional domain: no wrap-around in longitude
        if init_ckpt:
            sd = torch.load(init_ckpt, map_location="cpu")["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"init from {init_ckpt}: missing {missing}, unexpected {unexpected}")

    # ---------------------------------------------------------------- grids, MDT, drifters (once, on the right device)
    def _prepare(self, device):
        if self._grid_ready:
            return
        da = self.trainer.datamodule.input_da
        lat, lon = da.lat.values.astype(np.float64), da.lon.values.astype(np.float64)
        self._lat0, self._lon0 = lat[0], lon[0]
        self._dlat, self._dlon = float(lat[1] - lat[0]), float(lon[1] - lon[0])
        latt = torch.tensor(lat, dtype=torch.float32, device=device)
        f = 2 * OMEGA * torch.sin(torch.deg2rad(latt))
        band = (latt.abs() >= self.lat_band[0]) & (latt.abs() <= self.lat_band[1])
        self._fy = torch.where(band, -G / f / (self._dlat * M_PER_DEG), torch.zeros_like(f))[:, None]            # u = fy dh/di
        self._fx = torch.where(band, G / f / (self._dlon * M_PER_DEG * torch.cos(torch.deg2rad(latt))), torch.zeros_like(f))[:, None]
        self._band = band[:, None].expand(len(lat), len(lon))
        mdt = xr.open_dataset(self.mdt_path)["mdt"].isel(time=0).rename({"latitude": "lat", "longitude": "lon"})
        mdt = mdt.interp(lat=lat, lon=lon).values
        self._mdt = torch.tensor(np.nan_to_num(mdt), dtype=torch.float32, device=device)
        # drifters: per day (days since 1970) -> tensors (row, col fractional indices, u, v)
        self._drift = {}
        for y in self.drifter_years:
            d = xr.open_dataset(f"{self.drifter_dir}/gdp_drogued_daily_geos_{y}.nc")
            ok = (np.isfinite(d.u_geos.values) & np.isfinite(d.v_geos.values) & (np.abs(d.lat.values) >= self.lat_band[0])
                  & (np.abs(d.lat.values) <= self.lat_band[1]))
            day = (pd.DatetimeIndex(d.time.values[ok]).normalize() - pd.Timestamp("1970-01-01")).days.values
            r = (d.lat.values[ok] - self._lat0) / self._dlat
            c = ((d.lon.values[ok] - self._lon0) % 360.0) / self._dlon
            u, v = d.u_geos.values[ok], d.v_geos.values[ok]
            order = np.argsort(day)
            day, r, c, u, v = day[order], r[order], c[order], u[order], v[order]
            bounds = np.flatnonzero(np.diff(day)) + 1
            for chunk in np.split(np.arange(len(day)), bounds):
                if len(chunk):
                    self._drift[int(day[chunk[0]])] = torch.tensor(np.stack([r[chunk], c[chunk], u[chunk], v[chunk]]), dtype=torch.float32)
        print(f"velocity losses ready: {len(self._drift)} drifter days", flush=True)
        self._grid_ready = True

    def _geos(self, h):
        """h (..., lat, lon) in m -> u, v (m/s), centred differences, longitude periodic; 0 outside the latitude band."""
        dhdi = torch.zeros_like(h)
        dhdi[..., 1:-1, :] = (h[..., 2:, :] - h[..., :-2, :]) / 2
        if self.periodic_lon:
            dhdj = (torch.roll(h, -1, -1) - torch.roll(h, 1, -1)) / 2
        else:
            dhdj = torch.zeros_like(h)
            dhdj[..., 1:-1] = (h[..., 2:] - h[..., :-2]) / 2
        return self._fy * dhdi, self._fx * dhdj

    def _valid_stencil(self, finite):
        """point and its 4 neighbours finite (centred differences meaningful)"""
        v = finite.clone()
        v[..., 1:-1, :] &= finite[..., 2:, :] & finite[..., :-2, :]
        v[..., 0, :] = False
        v[..., -1, :] = False
        v &= torch.roll(finite, 1, -1) & torch.roll(finite, -1, -1)
        if not self.periodic_lon:
            v[..., 0] = False
            v[..., -1] = False
        return v

    # ---------------------------------------------------------------- losses
    def sla_losses(self, batch, phase):
        """The August 2025 training loss (commit 5b5c485)."""
        out = self(batch=batch)
        loss = self.weighted_mse(out - torch.nan_to_num(batch.tgt), self.rec_weight)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        pd_ = {"time": out.shape[1], "lat": out.shape[2] // 4, "lon": out.shape[3] // 4}
        w = torch.tensor(get_forecast_wei_adaptable_per_resolution(patch_dims=pd_, coarsen_factor=(4, 4), base_crop={"lat": 4, "lon": 4}),
                         dtype=out.dtype, device=out.device)
        coarsen_loss = self.weighted_mse(torch.nn.AvgPool2d(4)(out) - torch.nn.AvgPool2d(4)(torch.nan_to_num(batch.tgt)), w)
        dog_loss = self.weighted_mse(dog_kornia(out, 1, 2), dog_kornia(torch.nan_to_num(batch.tgt), 1, 2))
        total = 50 * loss + 50 * coarsen_loss + 50 * grad_loss + 50 * dog_loss
        with torch.no_grad():
            self.log(f"{phase}_mse", 50 * loss * self.norm_stats[1] ** 2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{phase}_sla_total", total, on_step=False, on_epoch=True, sync_dist=True)
        return total, out

    def velocity_losses(self, out, batch, t0_day, phase):
        m, s = self.norm_stats
        L = self.lead_days
        h_out = out[:, L] * s + m                                     # (B, nL, lat, lon) SLA in m
        h_tgt = batch.tgt[:, L] * s + m
        fin = torch.isfinite(h_tgt)
        # 1. vs DUACS geostrophic currents (MDT cancels in the difference)
        uo, vo = self._geos(h_out)
        ut, vt = self._geos(torch.nan_to_num(h_tgt))
        ok = self._valid_stencil(fin) & self._band
        err2 = ((uo - ut) ** 2 + (vo - vt) ** 2)[ok] / self.vel_scale ** 2
        geos = err2.mean() if err2.numel() else out.sum() * 0
        # 2. vs drifters: absolute currents of the prediction at the drifter positions
        ua, va = self._geos(h_out + self._mdt)
        land = ~self._valid_stencil(fin)
        ua, va = ua.masked_fill(land, float("nan")), va.masked_fill(land, float("nan"))
        preds, obs = [], []
        for b in range(out.shape[0]):
            for k, lead in enumerate(L):
                key = int(t0_day[b]) + lead
                if key not in self._drift:
                    continue
                r, c, u, v = self._drift[key].to(out.device)
                r0, c0 = torch.floor(r).long(), torch.floor(c).long()
                keep = (r0 >= 0) & (r0 < ua.shape[-2] - 1)
                if not self.periodic_lon:
                    keep &= (c0 >= 0) & (c0 < ua.shape[-1] - 1)
                r, c, u, v, r0, c0 = r[keep], c[keep], u[keep], v[keep], r0[keep], c0[keep]
                wr, wc = r - r0, c - c0
                c1 = (c0 + 1) % ua.shape[-1]
                c0 = c0 % ua.shape[-1]
                for comp, field in ((0, ua), (1, va)):
                    fld = field[b, k]
                    val = (fld[r0, c0] * (1 - wr) * (1 - wc) + fld[r0 + 1, c0] * wr * (1 - wc)
                           + fld[r0, c1] * (1 - wr) * wc + fld[r0 + 1, c1] * wr * wc)
                    if comp == 0:
                        pu = val
                    else:
                        pv = val
                good = torch.isfinite(pu) & torch.isfinite(pv)
                preds.append(torch.stack([pu[good], pv[good]]))
                obs.append(torch.stack([u[good], v[good]]))
        if preds:
            p, o = torch.cat(preds, 1), torch.cat(obs, 1)
            drift = F.huber_loss(p / self.vel_scale, o / self.vel_scale, delta=self.huber_delta)
            with torch.no_grad():
                self.log(f"{phase}_drift_rmse_cms", 100 * torch.sqrt(((p - o) ** 2).sum(0).mean()), on_step=False, on_epoch=True, sync_dist=True)
                self.log(f"{phase}_drift_n", float(p.shape[1]), on_step=False, on_epoch=True, sync_dist=True)
        else:
            drift = out.sum() * 0
        with torch.no_grad():
            self.log(f"{phase}_geos_rmse_cms", 100 * self.vel_scale * torch.sqrt(geos.detach()), on_step=False, on_epoch=True, sync_dist=True)
        return geos, drift

    def step(self, batch, phase=""):
        batch, t_idx = batch
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None
        self._prepare(batch.tgt.device)
        sla_total, out = self.sla_losses(batch, phase)
        geos, drift = self.velocity_losses(out, batch, t_idx, phase)
        total = self.w_sla * sla_total + self.w_geos * geos + self.w_drift * drift
        self.log(f"{phase}_geos_loss", geos, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{phase}_drift_loss", drift, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{phase}_total", total, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return total, out

    def training_step(self, batch, batch_idx):
        item, t_idx = batch
        return self.step((self.mask_batch(item), t_idx), "train")[0]

    def validation_step(self, batch, batch_idx):
        item, t_idx = batch
        return self.step((self.mask_batch(item), t_idx), "val")[0]


def finetune_training(trainer, dm, lit_mod):
    """fit only (the template's base_training also runs a test on a dummy 1-day test domain)"""
    trainer.fit(lit_mod, datamodule=dm)

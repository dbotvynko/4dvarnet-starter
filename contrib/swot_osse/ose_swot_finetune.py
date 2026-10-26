"""Fine-tuning of the OSE SLA UNet trained on 2017-2022 (DUACS losses, val_mse=0.01999-epoch=151) with synthetic SWOT swaths.

Data (OseSwotErrors): the setting of the OSE training, real nadir along-track input and DUACS target, plus SWOT:
  - input = real gridded nadir L3 (MY, `nadir_path`, variable `nadir_variable`), as in the OSE training;
  - on the swath pixels of a random 2024 day (CLS masks, prepare_cls_masks_2024.py) that have no nadir observation,
    input = DUACS + SWOT error, with the error model calibrated on the real 2024 swaths (swot_errors_2024_calibrated.yaml,
    same profiles and Gaussian-filtered pools as data_epoch.SwotErrorsOSSE);
  - SWOT removed on a share `swot_dropout` of the days (0.5: half the inputs are nadir only, as the OSE training);
  - errors, sampling days and SWOT-free days re-drawn every `resample_every` epochs (seed = epoch), validation drawn once.
The swaths are DUACS + error, so they carry no scale finer than DUACS: the network learns to use SWOT where it adds
DUACS-resolved information (positions and amplitudes of mesoscale structures between nadir tracks).

Model (Plus4dVarNetForecastPatchGPU_UNet_DuacsLosses): Plus4dVarNetForecastPatchGPU_UNet with the training loss of the
OSE UNet as it was trained in August 2025 (src/models.py at commit 5b5c485, Lit4dVarNet_UNet.step / base_step):
50 * (MSE + 1-deg coarse MSE + Sobel-gradient MSE + DoG term). The present Lit4dVarNet_UNet trains on input_complete
(L3 fine-tuning), which this datamodule does not provide.
"""
import kornia.filters as kfilts
import numpy as np
import torch
import xarray as xr

from contrib.forecast_plus.models import Plus4dVarNetForecastPatchGPU_UNet
from src.data import TrainingItem
from src.DoG import dog_kornia
from src.utils import get_forecast_wei_adaptable_per_resolution

from .data_epoch import SwotErrorsMovingPatchDataModule, _pool, calibrated_profiles
from .prepare_cls_masks_2024 import load_masks


class OseSwotErrors:
    """DUACS truth and real nadir input (loaded once); synthetic SWOT swaths re-drawn in place by draw(seed, time_index).

    self.da is a DataArray (variable: input, tgt; time; lat; lon) on top of one numpy buffer; the real nadir input is
    kept apart (self.nadir_input) and copied back into `input` at every draw.
    """

    def __init__(self, path, nadir_path, masks_path, domain, calibration, variables="sla", nadir_variable="sla_filtered",
                 swot_dropout=0.5, scale_buckets_deg=(1.0, 1.5, 2.0), n_pool=48, time_range=("2017-01-01", "2022-12-31")):
        print("LOADING DUACS SLA truth and real nadir input (OSE + synthetic SWOT with calibrated, re-drawn errors)")
        ds = xr.open_dataset(path)[[variables]]
        nadir = xr.open_dataset(nadir_path)[[nadir_variable]]
        for d in (ds, nadir):
            assert "lat" in d.dims and "lon" in d.dims, d.dims
        ds = ds.sel(time=slice(*time_range)).sel(domain)
        nadir = nadir.sel(time=ds.time).sel(domain)
        assert np.allclose(nadir.lat, ds.lat) and np.allclose(nadir.lon, ds.lon), "nadir and DUACS grids differ"
        self.swot_mask = load_masks(masks_path)[1]
        shape = (ds.sizes["lat"], ds.sizes["lon"])
        assert self.swot_mask.shape[1:] == shape, f"mask grid {self.swot_mask.shape[1:]} != data grid {shape}"
        data = np.empty((2, ds.sizes["time"], *shape), dtype=np.float32)
        data[1] = ds[variables].values
        self.nadir_input = nadir[nadir_variable].values.astype(np.float32)
        data[0] = self.nadir_input
        self.data, self.shape = data, shape
        self.swot_dropout, self.n_pool = swot_dropout, n_pool
        self.buckets = np.array(scale_buckets_deg, dtype=float)
        lat = ds["lat"].values
        white, corr, scale, self.table = calibrated_profiles(calibration, lat, self.buckets[0], self.buckets[-1])
        self.white = white[:, None]
        # weights of the scale buckets per latitude, as in data_epoch.SwotErrorsOSSE
        w = np.stack([np.clip(1 - np.abs(scale - b) / np.diff(self.buckets).mean(), 0, None) for b in self.buckets])
        w /= w.sum(0, keepdims=True)
        w /= np.sqrt((w ** 2).sum(0, keepdims=True))
        self.corr_w = (w * corr)[:, :, None].astype(np.float32)
        self.grid_deg = float(np.abs(np.diff(lat)).mean())
        self.da = xr.DataArray(data, dims=("variable", "time", "lat", "lon"),
                               coords={"variable": list(TrainingItem._fields), "time": ds.time.values, "lat": lat, "lon": ds.lon.values})
        self.time = ds.time.values
        print(f"{len(self.time)} days, nadir input {np.isfinite(self.nadir_input).mean():.4f} of the grid")
        print("calibrated SWOT errors (band, white m, corr m, scale deg, source band):")
        for row in self.table:
            print("  {:>4} {:>4}  {:.4f}  {:.4f}  {:.2f}  {}".format(*row))

    def draw(self, seed, time_index):
        rng = np.random.default_rng(seed)
        pools = [_pool(rng, self.shape, self.n_pool, (b / 2) / self.grid_deg) for b in self.buckets]
        n_lon, n_days = self.shape[1], self.swot_mask.shape[0]
        n_dropped = 0
        self.last_draw = []  # (time index, 2024 mask day, SWOT dropped), for checks
        for t in time_index:
            day = int(rng.integers(0, n_days))
            drop = rng.random() < self.swot_dropout
            n_dropped += drop
            self.last_draw.append((t, day, drop))
            out = self.data[0, t]
            out[:] = self.nadir_input[t]
            if drop:
                continue
            sr, sc = np.nonzero(self.swot_mask[day] & ~np.isfinite(out))
            error = self.white[sr, 0] * rng.standard_normal(sr.size, dtype=np.float32)
            for k, pool in enumerate(pools):
                shift = rng.integers(n_lon)
                error += self.corr_w[k, sr, 0] * rng.choice([-1, 1]) * pool[rng.integers(self.n_pool)][sr, (sc - shift) % n_lon]
            out[sr, sc] = self.data[1, t, sr, sc] + error
        return n_dropped


class OseSwotErrorsMovingPatchDataModule(SwotErrorsMovingPatchDataModule):
    """SwotErrorsMovingPatchDataModule on OseSwotErrors (re-draws every `resample_every` epochs, validation drawn once)."""

    def __init__(self, ose_kw, domains, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, resample_every=10, val_seed=0, **kwargs):
        self.osse = OseSwotErrors(**ose_kw)
        super(SwotErrorsMovingPatchDataModule, self).__init__(self.osse.da, domains, xrds_kw, dl_kw, aug_kw=aug_kw,
                                                              norm_stats=norm_stats, **kwargs)
        self.resample_every, self.val_seed = resample_every, val_seed
        self._drawn_epoch = None
        self._index = {}
        for split in ("train", "val", "test"):
            sl = self.domains[split]["time"]
            self._index[split] = np.flatnonzero((self.osse.time >= np.datetime64(sl.start)) & (self.osse.time <= np.datetime64(sl.stop)))
        held_out = np.union1d(self._index["val"], self._index["test"])
        n = self.osse.draw([val_seed, 1], held_out)
        print(f"validation / test inputs drawn once (seed {val_seed}): {len(held_out)} days, SWOT dropped on {n}")


class Plus4dVarNetForecastPatchGPU_UNet_DuacsLosses(Plus4dVarNetForecastPatchGPU_UNet):
    """Training loss of the OSE UNet (src/models.py at 5b5c485): 50 * (MSE + coarse MSE + gradient MSE + DoG term)."""

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        coarsen = 4  # DUACS 1/4 deg -> 1 deg
        patch_dims = {"time": out.shape[1], "lat": 680 // coarsen, "lon": 1440 // coarsen}
        weights = get_forecast_wei_adaptable_per_resolution(patch_dims=patch_dims, coarsen_factor=(coarsen, coarsen),
                                                            base_crop={"lat": 4, "lon": 4})
        weights_torch = torch.tensor(weights, dtype=out.dtype, device=out.device)
        coarsen_loss = self.weighted_mse(torch.nn.AvgPool2d(coarsen)(out) - torch.nn.AvgPool2d(coarsen)(torch.nan_to_num(batch.tgt)), weights_torch)
        # as trained in August 2025: the DoG of the target is passed as the weight of weighted_mse
        DoG_loss = self.weighted_mse(dog_kornia(out, 1, 2), dog_kornia(torch.nan_to_num(batch.tgt), 1, 2))

        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_closs", coarsen_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{phase}_dogloss", DoG_loss, prog_bar=False, on_step=False, on_epoch=True)
        return 50 * loss + 50 * coarsen_loss + 50 * grad_loss + 50 * DoG_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - torch.nan_to_num(batch.tgt), self.rec_weight)
        with torch.no_grad():
            self.log(f"{phase}_mse", 50 * loss * self.norm_stats[1] ** 2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, out

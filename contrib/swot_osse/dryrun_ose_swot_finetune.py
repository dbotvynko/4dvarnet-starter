"""CPU dry run of the OSE UNet SWOT fine-tune (xp=forecast_SLA_UNet_OSE_2017_2022_SWOT_errors_finetune):
  1. norm_stats: mean / std of the DUACS target over the training period and domain, recomputed, against the std implied
     by the August 2025 metrics.csv (val_mse = 50 * val_loss * std^2) and the config value (the 2025 "Norm stats");
  2. data on a cut of the record: two draws differ on the swath pixels only, truth and nadir pixels unchanged, SWOT error
     std per latitude band against the calibration;
  3. the August 2025 checkpoint in the fine-tune model on one validation window: loss with nadir input (should be near the
     August 2025 validation level) and with nadir + synthetic SWOT.

Usage: python contrib/swot_osse/dryrun_ose_swot_finetune.py
"""
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from hydra import compose, initialize
from omegaconf import open_dict

from contrib.moving_patches.movpatch_data_tests import XrDatasetMovingPatchFastRecGPU

with initialize(config_path="../../config", version_base="1.3"):
    cfg = compose(config_name="main", overrides=["xp=forecast_SLA_UNet_OSE_2017_2022_SWOT_errors_finetune"], return_hydra_config=True)

# 1. normalisation
tr = cfg.datamodule.domains.train.time._args_
tgt = xr.open_dataset(cfg.paths.glorys12_data, chunks={"time": 50})["sla"].sel(time=slice(*tr), lat=slice(-80, 90), lon=slice(-180, 180))
mean, std = float(tgt.mean(skipna=True).compute()), float(tgt.std(skipna=True).compute())
metrics = pd.read_csv(cfg.paths.init_ckpt.split("/checkpoints/")[0] + "/metrics.csv").dropna(subset=["val_mse", "val_loss"])
implied = np.sqrt(metrics.val_mse / (50 * metrics.val_loss))
print(f"recomputed: [{mean}, {std}]  | std implied by the August 2025 metrics: {implied.median():.5f} (min {implied.min():.5f}, max {implied.max():.5f})")
mean, std = cfg.datamodule.norm_stats
print(f"config norm_stats (used below): [{mean}, {std}]")
print(f"August 2025 best val_mse {metrics.val_mse.min():.5f} (val_loss {metrics.val_loss[metrics.val_mse.idxmin()]:.5f})")

# 2. data on a cut
with open_dict(cfg):
    cfg.datamodule.ose_kw.time_range = ["2022-02-01", "2022-06-30"]
    cfg.datamodule.domains.train.time._args_ = ["2022-02-01", "2022-04-17"]
    cfg.datamodule.domains.val.time._args_ = ["2022-04-18", "2022-06-30"]
    cfg.datamodule.domains.test.time._args_ = ["2022-06-30", "2022-06-30"]
dm = hydra.utils.instantiate(cfg.datamodule)
ose = dm.osse
dm.setup("fit")
truth = ose.data[1].copy()
dm.trainer = SimpleNamespace(current_epoch=0)
dm.train_dataloader()
idx = dm._index["train"]
a = ose.data[0, idx].copy()
drop_a = np.array([d[2] for d in ose.last_draw])
dm.trainer = SimpleNamespace(current_epoch=10)
dm.train_dataloader()
b = ose.data[0, idx].copy()
nad = np.isfinite(ose.nadir_input[idx])
print("truth unchanged:", np.array_equal(truth, ose.data[1], equal_nan=True),
      "| nadir pixels = real nadir:", np.array_equal(a[nad], ose.nadir_input[idx][nad]) and np.array_equal(b[nad], ose.nadir_input[idx][nad]),
      "| draws differ:", not np.array_equal(a, b, equal_nan=True))
kept = ~drop_a
sw = np.isfinite(a) & ~nad
print(f"draw 0: SWOT dropped on {drop_a.sum()} / {len(drop_a)} days; swath pixels per kept day {sw[kept].sum() / kept.sum():.0f}"
      f" (nadir {nad.sum() / len(idx):.0f}); no swath pixel on dropped days: {sw[drop_a].sum() == 0}")
err = (a - ose.data[1, idx])[sw]
lat = np.broadcast_to(ose.da.lat.values[None, :, None], a.shape)[sw]
print("SWOT error std per band (m) vs calibration total (white^2 + corr^2)^0.5:")
for lo, hi, w, c, s, src in ose.table:
    m = (lat >= lo) & (lat < hi)
    if m.sum():
        print(f"  {lo:>4} {hi:>4}  {err[m].std():.4f}  vs {np.hypot(w, c):.4f}  (n={m.sum()})")
item = dm.train_ds[0]
print("training item:", {k: tuple(v.shape) for k, v in item._asdict().items()})

# 3. checkpoint, one validation window
model = hydra.utils.instantiate(cfg.model)
state = torch.load(cfg.paths.init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
print("load:", model.load_state_dict(state, strict=False))
model._norm_stats = (mean, std)
model.log = lambda *args, **kwargs: None
model.eval()
for label, use_swot in (("nadir only", False), ("nadir + SWOT", True)):
    if not use_swot:
        ose.data[0, dm._index["val"]] = ose.nadir_input[dm._index["val"]]
    else:
        ose.swot_dropout = 0.0
        ose.draw([0, 1], dm._index["val"])
    val = XrDatasetMovingPatchFastRecGPU(dm.input_da.sel(dm.domains["val"]), **dm.xrds_kw, postpro_fn=dm.post_fn(), rand=False)
    batch = val[len(val) // 2]
    batch = type(batch)(*[torch.as_tensor(np.asarray(x))[None] for x in batch])
    with torch.no_grad():
        total, out = model.step(model.mask_batch(batch), "val")
        mse, _ = model.base_step(model.mask_batch(batch), "val")
    print(f"{label}: val_loss {mse.item():.5f} -> val_mse {50 * mse.item() * std ** 2:.5f} | total training loss {total.item():.4f}")

"""CPU dry run of the 1-day-orbit SWOT OSSE training (xp=forecast_SLA_UNet_OSSE_SWOT_1day_orbit) on a cut of the record:
mask days follow each other day by day, SWOT is never dropped, the same strips are observed on consecutive days, two draws
differ (errors, offset) with identical truth, training items have the expected shape.

Usage: python contrib/swot_osse/dryrun_1day_orbit.py
"""
from types import SimpleNamespace

import hydra
import numpy as np
from hydra import compose, initialize
from omegaconf import open_dict

with initialize(config_path="../../config", version_base="1.3"):
    cfg = compose(config_name="main", overrides=["xp=forecast_SLA_UNet_OSSE_SWOT_1day_orbit"], return_hydra_config=True)
with open_dict(cfg):
    cfg.datamodule.osse_kw.test_cut = {"_target_": "builtins.slice", "_args_": ["2019-03-01", "2019-12-31"]}
print("masks:", cfg.paths.masks_path, "| sequence", cfg.datamodule.osse_kw.sequence, "| dropout", cfg.datamodule.osse_kw.swot_dropout,
      "| save_dir", cfg.trainer.logger.save_dir)
dm = hydra.utils.instantiate(cfg.datamodule)
osse = dm.osse
dm.setup("fit")
dm.trainer = SimpleNamespace(current_epoch=0)
dm.train_dataloader()
draw0 = list(osse.last_draw)
t = np.array([d[0] for d in draw0])
day = np.array([d[1] for d in draw0])
drop = np.array([d[2] for d in draw0])
steps = (np.diff(day) % osse.nadir.shape[0])[np.diff(t) == 1]
print(f"{len(draw0)} training days | mask-day step between consecutive days: {np.unique(steps)} | SWOT dropped on {drop.sum()} days")
a = osse.data[0, dm._index["train"]].copy()
sw_day = [osse.swot[d] for d in day[:5]]
overlap = [(sw_day[k] & sw_day[k + 1]).sum() / max(sw_day[k].sum(), 1) for k in range(4)]
print("share of SWOT pixels of day k also observed on day k+1:", np.round(overlap, 2))
dm.trainer = SimpleNamespace(current_epoch=10)
dm.train_dataloader()
b = osse.data[0, dm._index["train"]]
print("draw 0 vs 10 inputs differ:", not np.array_equal(a, b, equal_nan=True), "| offsets", day[0], osse.last_draw[0][1])
item = dm.train_ds[0]
print("training item:", {k: tuple(v.shape) for k, v in item._asdict().items()} if hasattr(item, "_asdict") else type(item))

"""Smoke test + loss-weight calibration for the velocity fine-tune: runs the starting checkpoint on a few validation windows
(1 GPU), reports the SLA loss, the two velocity losses and the velocity RMSEs, and writes the weights that make each velocity
term equal to `ratio` x the SLA loss at the start of the fine-tune.
Usage: python contrib/drifter_geos/calibrate_weights.py OUT.json [ratio=0.5] [n_batches=8]
"""
import json
import sys

import hydra
from hydra import compose, initialize_config_dir
from pathlib import Path


def main():
    out, ratio = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        cfg = compose(config_name="main", overrides=["xp=finetune_SLA_UNet_OSE_2017_2022_geos_drifter", "trainer.devices=1",
                                                     "trainer.strategy=auto", "trainer.logger=false", "trainer.callbacks=[]",
                                                     f"trainer.limit_val_batches={nb}"])
    trainer = hydra.utils.instantiate(cfg.trainer)
    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    res = trainer.validate(model, datamodule=dm)[0]
    print(res)
    sla, geos, drift = res["val_sla_total"], res["val_geos_loss"], res["val_drift_loss"]
    w = {"w_geos": ratio * sla / geos, "w_drift": ratio * sla / drift, "start": {k: float(v) for k, v in res.items()}}
    json.dump(w, open(out, "w"), indent=1)
    print("weights", w)


if __name__ == "__main__":
    main()

"""Corrected L3 MY inference configs (new files; the configs of make_l3my_inference_configs.py and their forecasts are left as they are).

Two errors of the first generator:
  1. the UNet-N and UNet-S0 pipelines come from ose_pipeline_proc_2024_SLA_UNet_Nadir_cls_nadir_swot_repro.yaml, whose entrypoint
     repeats model_config_path with the CLS Nadir+SWOT config: only the first occurrence was replaced, so these runs read the CLS
     input for all three inputs (2024: zero change; 2025: idem);
  2. the model configs keep year_ose: 2024, so the 2025 runs select no day and write nothing.
This writes config/xp/forecast_SLA_{UNet,4dvarnet}_inference_l3my_2025_<input>_v2.yaml (year_ose 2025) and
config/xp/ose_pipeline_proc_l3myv2_<year>_<model>_<input>.yaml for 2024 (unetN, unetS0) and 2025 (all four models), copies of
the first pipelines with every model_config_path set, and xp_name ..._L3MYv2<year>_inp_<input>. The list of pipelines is printed.
Usage: python scripts/slurm_swot_osse/make_l3my_inference_configs_v2.py
"""
import re
from pathlib import Path

X = Path("config/xp")
INPUTS = ["nadir", "nadir_swot", "nadir_swot_hret"]
RUNS = {"2024": ["unetN", "unetS0"], "2025": ["unetN", "unetS0", "unetSE", "fdv"]}
FAMILY = {"unetN": "UNet", "unetS0": "UNet", "unetSE": "UNet", "fdv": "4dvarnet"}


def write(path: Path, text: str, comment: str) -> None:
    assert not path.exists(), f"{path} exists: not overwritten"
    first, rest = text.split("\n", 1)
    assert first.startswith("# @package"), path
    path.write_text(f"{first}\n{comment}\n{rest}")


def main() -> None:
    for family in ("UNet", "4dvarnet"):
        for inp in INPUTS:
            src = X / f"forecast_SLA_{family}_inference_l3my_2025_{inp}.yaml"
            text, n = re.subn(r"(?m)^year_ose: 2024", "year_ose: 2025", src.read_text())
            assert n == 1, src
            write(X / f"forecast_SLA_{family}_inference_l3my_2025_{inp}_v2.yaml", text, f"# copy of {src.name} with year_ose 2025 (make_l3my_inference_configs_v2.py)")
    names = []
    for year, models in RUNS.items():
        for model in models:
            for inp in INPUTS:
                src = X / f"ose_pipeline_proc_l3my_{year}_{model}_{inp}.yaml"
                text = src.read_text()
                cfg = f"forecast_SLA_{FAMILY[model]}_inference_l3my_{year}_{inp}" + ("_v2" if year == "2025" else "")
                text, n = re.subn(r"(?m)^(\s*)model_config_path: (config/xp/\S+|\$\{paths\.model_config_path\})", rf"\1model_config_path: config/xp/{cfg}.yaml", text)
                assert n == 2, (src, n)
                text, n = re.subn(r"(?m)^xp_name: (\S+)_L3MY(\d{4})_inp_", r"xp_name: \1_L3MYv2\2_inp_", text)
                assert n == 1, src
                name = f"ose_pipeline_proc_l3myv2_{year}_{model}_{inp}"
                write(X / f"{name}.yaml", text, f"# corrected copy of {src.name}: both model_config_path set to {cfg}.yaml, xp_name L3MYv2 (make_l3my_inference_configs_v2.py)")
                names.append(name)
    print(" ".join(names))


if __name__ == "__main__":
    main()

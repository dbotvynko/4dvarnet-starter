"""Write the inference configs of the L3 MY test (2024 same-gridding bridge and independent 2025), new files only.

Inputs (oceanbench_eval/grid_l3my_karin_daily.py): /Odyssey/public/swot_traces/gridded_l3my/{nadir,nadir_swot,nadir_swot_hret}_l3my_0.25_<year>.nc,
same layout as the CLS 2024 files. Models (checkpoints of the 2024 CLS runs): UNet-N (Nadir OSSE), UNet-S0 (SWOT OSSE, perfect swaths),
UNet-SE (SWOT OSSE, calibrated re-drawn errors) and 4DVarNet (Nadir OSSE, lead-time fix). Templates: the existing 2024 CLS configs,
copied with only the input file, the xp_name, the dates (2025: January - July) and the skipped download step changed.
Output: config/xp/forecast_SLA_{UNet,4dvarnet}_inference_l3my_<year>_<input>.yaml and
        config/xp/ose_pipeline_proc_l3my_<year>_<model>_<input>.yaml; the list of pipeline names is printed.
Usage: python scripts/slurm_swot_osse/make_l3my_inference_configs.py
"""
import re
from pathlib import Path

X = Path("config/xp")
GRID = "/Odyssey/public/swot_traces/gridded_l3my/{}_l3my_0.25_{}.nc"
CLS_INPUT = "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_swot_2024_reformated.nc"
YEARS = {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-07-31")}
INPUTS = ["nadir", "nadir_swot", "nadir_swot_hret"]
FAMILIES = {"UNet": "forecast_SLA_UNet_inference_2024_cls_nadir_swot.yaml", "4dvarnet": "forecast_SLA_4dvarnet_inference_2024_cls_nadir_swot_leadfix.yaml"}
UNET_PIPE = "ose_pipeline_proc_2024_SLA_UNet_Nadir_cls_nadir_swot_repro.yaml"
MODELS = {  # model: (family, pipeline template, checkpoint or None to keep the template's, xp_name stem)
    "unetN": ("UNet", UNET_PIPE, None, "forecast_Unet_SLA_1patch_OSSE_NadirOnly"),
    "unetS0": ("UNet", UNET_PIPE, "/Odyssey/public/glorys/trainings/forecast_global_GPU_1patch_1y_SLA_UNet_17M_2016_2022_OSSESWOT/"
               "base_forecast_global_softedge_fastrec_GPU_1patch_10y_SLA_UNet/checkpoints/val_mse=0.01308-epoch=238.ckpt",
               "forecast_Unet_SLA_1patch_OSSE_NadirSwot"),
    "unetSE": ("UNet", "ose_pipeline_proc_2024_SLA_UNet_OSSE_SWOT_errors_scratch_cls_nadir_swot.yaml", None, "forecast_Unet_SLA_1patch_OSSE_SWOT_errors_scratch"),
    "fdv": ("4dvarnet", "ose_pipeline_proc_2024_SLA_4dvarnet_leadfix_cls_nadir_swot.yaml", None, "glorys4_global_1patch_SLA_one_leadtime_corrected_21_LEADFIX"),
}


def sub_once(pattern: str, repl: str, text: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    assert n == 1, pattern
    return new


def write(path: Path, text: str, comment: str) -> None:
    """Write text with the comment as its second line: the first line stays the "# @package _global_" header."""
    assert not path.exists(), f"{path} exists: not overwritten"
    first, rest = text.split("\n", 1)
    assert first.startswith("# @package"), path
    path.write_text(f"{first}\n{comment}\n{rest}")


def main() -> None:
    names = []
    for year, (first, last) in YEARS.items():
        for inp in INPUTS:
            data = GRID.format(inp, year)
            for family, template in FAMILIES.items():
                text = (X / template).read_text()
                assert text.count(CLS_INPUT) >= 1, template
                write(X / f"forecast_SLA_{family}_inference_l3my_{year}_{inp}.yaml", text.replace(CLS_INPUT, data),
                      f"# L3 MY test ({year}, input {inp}): copy of {template} with the input {data}")
            for model, (family, template, ckpt, stem) in MODELS.items():
                text = (X / template).read_text()
                text = sub_once(r"^xp_name: .*$", f"xp_name: {stem}_L3MY{year}_inp_{inp}", text)
                text = sub_once(r"^(\s*)model_config_path: config/xp/.*$", rf"\1model_config_path: config/xp/forecast_SLA_{family}_inference_l3my_{year}_{inp}.yaml", text)
                if ckpt:
                    text = sub_once(r"^(\s*)model_ckpt_path: /.*$", rf"\1model_ckpt_path: {ckpt}", text)
                text = sub_once(r"^(\s*)min_time: \d{4}-\d{2}-\d{2}", rf"\g<1>min_time: {first}", text)
                text = sub_once(r"^(\s*)max_time: \d{4}-\d{2}-\d{2}", rf"\g<1>max_time: {last}", text)
                text = sub_once(r"^(\s*skip:\s*\n\s*)data: False", r"\1data: True", text)  # inputs are read from the gridded file
                name = f"ose_pipeline_proc_l3my_{year}_{model}_{inp}"
                write(X / f"{name}.yaml", text, f"# L3 MY test: {model}, {year}, input {inp} (scripts/slurm_swot_osse/make_l3my_inference_configs.py, template {template})")
                names.append(name)
    print(" ".join(names))


if __name__ == "__main__":
    main()

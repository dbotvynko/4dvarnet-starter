#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=l3my_inference
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_l3my_inference.log
# Overnight forecasts: (1) the 20 km SWOT filtering test of UNet-N (2024 CLS input), then (2) the L3 MY test: UNet-N, UNet-S0,
# UNet-SE and 4DVarNet with the Nadir, Nadir+SWOT and Nadir+SWOT+HRET inputs gridded from DUACS L3 MY and KaRIn L3 v3.0, for 2024
# (same-gridding bridge) and January-July 2025 (independent year); configs from make_l3my_inference_configs.py.
# A run whose rec folder already holds its 7 forecast files is skipped.
# Usage: sbatch --begin=20:00 scripts/slurm_swot_osse/inference_l3my_2024_2025.sh
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
RUNS="ose_pipeline_proc_2024_SLA_UNet_Nadir_cls_nadir_swot_swotlp20 $(cat /Odyssey/private/d21botvy/oceanbench_eval/l3my_pipelines.txt)"
status_all=0
for XP in $RUNS; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -eq 7 ]; then echo "$(date) skip $XP"; continue; fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  s=$?; echo "$(date) $XP EXIT $s"; [ $s -eq 0 ] || status_all=$s
done
exit $status_all

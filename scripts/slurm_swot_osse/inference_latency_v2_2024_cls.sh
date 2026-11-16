#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=latency_v2
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_latency_v2.log
# 2-day SWOT latency forecasts (inputs /Odyssey/public/swot_traces/cls/latency/*_{A,B}.nc) of UNet-S, 4DVarNet-S and the 29-day 4DVarNet-N.
# Skip-if-complete. Usage: sbatch scripts/slurm_swot_osse/inference_latency_v2_2024_cls.sh
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
RUNS="$(cat /Odyssey/private/d21botvy/oceanbench_eval/latency_pipelines_v2.txt)"
status_all=0
for XP in $RUNS; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -eq 7 ]; then echo "$(date) skip $XP"; continue; fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  s=$?; echo "$(date) $XP EXIT $s"; [ $s -eq 0 ] || status_all=$s
done
exit $status_all

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=fdv29Nclean_cls
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_fdv29Nclean_cls.log
# 2024 forecasts of 4DVarNet-S (29-day, nadir + SWOT OSSE with calibrated re-drawn errors, val_mse=4.90858-epoch=342)
# with the CLS Nadir and Nadir+SWOT inputs; new rec folders glorys4_global_1patch_SLA_4dvarnet29S_SWOTerrors_*_eval2024. Skip-if-complete.
# Usage: sbatch scripts/slurm_swot_osse/inference_4dvarnet29S_2024_cls.sh
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
RUNS="ose_pipeline_proc_2024_SLA_4dvarnet29Nclean_cls_nadir ose_pipeline_proc_2024_SLA_4dvarnet29Nclean_cls_nadir_swot"
status_all=0
for XP in $RUNS; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -eq 7 ]; then echo "$(date) skip $XP"; continue; fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  s=$?; echo "$(date) $XP EXIT $s"; [ $s -eq 0 ] || status_all=$s
done
exit $status_all

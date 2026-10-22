#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=swoterr_scratch_inf
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_swoterr_scratch_inference_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_swoterr_scratch_inference_2024.txt
#
# 2024 SLA forecasts with CLS inputs of the UNet trained from scratch on the SWOT OSSE with calibrated, re-drawn errors
# (train_SLA_UNet_OSSE_SWOT_errors_scratch.sh, best checkpoint val_mse=0.01496-epoch=367): CLS Nadir and CLS Nadir + SWOT.
# New rec folders forecast_Unet_SLA_1patch_OSSE_SWOT_errors_scratch_inp_*_eval2024.
# Usage: sbatch scripts/slurm_swot_osse/inference_2024_cls_SWOT_errors_scratch.sh
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
for XP in ose_pipeline_proc_2024_SLA_UNet_OSSE_SWOT_errors_scratch_cls_nadir ose_pipeline_proc_2024_SLA_UNet_OSSE_SWOT_errors_scratch_cls_nadir_swot; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -eq 7 ]; then
    echo "$(date) skip $XP: 7 forecast files already in /Odyssey/public/glorys/rec/$XPNAME"; continue
  fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  status=$?
  echo "$(date) $XP EXIT $status"
  [ $status -eq 0 ] || exit $status
done

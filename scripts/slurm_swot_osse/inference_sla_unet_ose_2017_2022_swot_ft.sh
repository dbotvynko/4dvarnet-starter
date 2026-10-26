#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=inf_ose_ft
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_inference_ose_swot_ft.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_inference_ose_swot_ft.txt
#
# 2024 CLS inference of the OSE UNet 2017-2022 with its training normalisation (control, xps *_trainnorm_*) and of its SWOT
# fine-tune (xps *_SWOT_ft_*). For the fine-tune, best_for_inference.ckpt is linked to the checkpoint with the lowest
# val_mse if it does not exist. Runs whose 7 forecast files exist are skipped.
# Usage: sbatch scripts/slurm_swot_osse/inference_sla_unet_ose_2017_2022_swot_ft.sh trainnorm|SWOT_ft

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
WHICH=$1
CK=/Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSE_2017_2022_SWOT_errors_finetune/forecast_SLA_UNet_OSE_2017_2022_SWOT_errors_finetune/checkpoints
if [ "$WHICH" = SWOT_ft ] && [ ! -e $CK/best_for_inference.ckpt ]; then
  BEST=$(ls $CK/val_mse=*.ckpt | sort | head -1)
  ln -s "$(basename $BEST)" $CK/best_for_inference.ckpt
  echo "best_for_inference.ckpt -> $BEST"
fi
for XP in ose_pipeline_proc_2024_SLA_UNet_OSE_2017_2022_${WHICH}_cls_nadir ose_pipeline_proc_2024_SLA_UNet_OSE_2017_2022_${WHICH}_cls_nadir_swot; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -ge 7 ]; then
    echo "$(date) skip $XP: forecast files already in /Odyssey/public/glorys/rec/$XPNAME"; continue
  fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  status=$?
  echo "$(date) $XP EXIT $status"
  ls -la /Odyssey/public/glorys/rec/$XPNAME/*/
  [ $status -eq 0 ] || exit $status
done

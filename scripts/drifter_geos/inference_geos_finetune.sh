#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=geos_ft_inf
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_geos_finetune_inference.log
# 2023 NRT and 2024 CLS-nadir inference with the best velocity-loss fine-tune checkpoint (lowest val_total), into new rec folders
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
BEST=$(ls /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune/checkpoints/val_total=*.ckpt | sort | head -1); ln -sfn "$BEST" /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune/checkpoints/best.ckpt; echo "best: $BEST"
for XP in ose_pipeline_proc_2023_SLA_UNet_OSE_2017_2022_nrt_geosft_day13 ose_pipeline_proc_2024_SLA_UNet_OSE_2017_2022_cls_nadir_geosft_day13; do
  echo "$(date) start $XP"; HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
done

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=geos_ft_calib
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_geos_finetune_calibrate.log
# Smoke test of the velocity fine-tune + loss-weight calibration (1 GPU, 8 validation windows) -> /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune/loss_weights.json
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
mkdir -p /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune
HYDRA_FULL_ERROR=1 PYTHONPATH=$PWD python contrib/drifter_geos/calibrate_weights.py /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune/loss_weights.json 0.5 8

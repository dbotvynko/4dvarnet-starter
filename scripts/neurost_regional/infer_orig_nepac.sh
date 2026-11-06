#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=nepac_inf0
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_nepac_regional_orig.log
# Original OSE UNet checkpoint run on the regional NE Pacific patch (no training): isolates the effect of the regional crop.
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
export PYTHONPATH=$PWD
python contrib/neurost_regional/infer_regional.py /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_ResBlock_real_data_Nadir_Duacs_losses/base_forecast_global_softedge_fastrec_GPU_1patch_10y_SLA_UNet_real_data_from_sratch/checkpoints/val_mse=0.01999-epoch=151.ckpt forecast_Unet_SLA_nepac_regional_orig

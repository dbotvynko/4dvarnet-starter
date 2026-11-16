#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=denial2023_class4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_denial2023_class4.log
# 2023 daily-repeat test scored the OceanBench way (contrib/swot_osse/denial_calval_2023_class4.py), for the three UNets:
# UNet-N (default checkpoint), UNet-S (SWOT_errors_scratch), UNet-1d (SWOT_1day_orbit); best checkpoints as in the denial scripts.
# Usage: sbatch scripts/slurm_swot_osse/denial_calval_2023_class4.sh [check]   (check: 3 starts of UNet-N into a test folder)
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
R=/Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_class4_2023
if [ "$1" = "check" ]; then python contrib/swot_osse/denial_calval_2023_class4.py ${R}_check --max-starts 3; exit $?; fi
S=$(ls /Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_errors_scratch/*/checkpoints/val_mse=*.ckpt | sort -t= -k2 -g | head -1)
D=$(ls /Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_1day_orbit/*/checkpoints/val_mse=*.ckpt | sort -t= -k2 -g | head -1)
status=0
python contrib/swot_osse/denial_calval_2023_class4.py $R/unet_n || status=1
python contrib/swot_osse/denial_calval_2023_class4.py $R/unet_s --ckpt "$S" || status=1
python contrib/swot_osse/denial_calval_2023_class4.py $R/unet_1d --ckpt "$D" || status=1
exit $status

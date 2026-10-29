#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=denial_runs
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_denial_runs.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_denial_runs.txt
#
# Run-length test on the 2023 1-day orbit: number vs arrangement of SWOT input days
# (contrib/swot_osse/denial_calval_2023_runs.py). Usage: sbatch ...crossover.sh nadir | swotscratch | 1day_orbit
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
RES=/Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_runs
case $1 in
  nadir)       CKPT="" ;;
  swotscratch) CKPT=$(ls /Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_errors_scratch/*/checkpoints/val_mse=*.ckpt | sort -t= -k2 -g | head -1) ;;
  1day_orbit)  CKPT=$(ls /Odyssey/public/glorys/trainings/forecast_SLA_UNet_OSSE_SWOT_1day_orbit/*/checkpoints/val_mse=*.ckpt | sort -t= -k2 -g | head -1) ;;
esac
echo "model $1 checkpoint ${CKPT:-default}"
python contrib/swot_osse/denial_calval_2023_runs.py $RES/$1 --mask /Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_crossover/crossover_mask.npz ${CKPT:+--ckpt $CKPT} --kinds run,spaced

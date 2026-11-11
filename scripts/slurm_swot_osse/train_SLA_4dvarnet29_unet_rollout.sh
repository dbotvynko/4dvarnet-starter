#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=fdv29_unet_rollout
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=160G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_fdv29_unet_rollout.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_fdv29_unet_rollout.txt
#
# From-scratch training of the 4DVarNet UNet roll-out (contrib/ocean4dvarnet_rollout, 29-day window); first argument: the config
# forecast_SLA_4dvarnet29_unet_rollout_OSSE_Nadir or forecast_SLA_4dvarnet29_unet_rollout_OSSE_SWOT_errors.
# The 14 GB GLORYS input is staged on the node-local /SCRATCH first (the shared filesystem is slow).
# Usage: sbatch --begin=20:00 scripts/slurm_swot_osse/train_SLA_4dvarnet29_unet_rollout.sh <xp> [extra hydra overrides]

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
XP=$1; shift

SRC=/Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_4th_float32.nc
DST=/SCRATCH/d21botvy/glorys12_2010_2019_daily_sla_4th_float32.nc
mkdir -p /SCRATCH/d21botvy
if [ ! -f "$DST" ] || [ "$(stat -c %s "$DST")" != "$(stat -c %s "$SRC")" ]; then
  echo "$(date) staging $SRC to $DST"
  rsync -a "$SRC" "$DST" || { echo "staging failed, reading from Odyssey"; DST=$SRC; }
fi
echo "$(date) input $DST on $(hostname)"

HYDRA_FULL_ERROR=1 srun python main.py xp=$XP paths.glorys12_data=$DST "$@"

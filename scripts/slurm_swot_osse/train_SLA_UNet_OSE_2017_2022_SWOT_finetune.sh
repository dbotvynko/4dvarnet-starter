#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=ose_swot_ft
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=160G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_ose_swot_finetune.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_ose_swot_finetune.txt
#
# Fine-tuning of the OSE SLA UNet (2017-2022, DUACS losses) with synthetic SWOT swaths and the calibrated error model
# (config/xp/forecast_SLA_UNet_OSE_2017_2022_SWOT_errors_finetune.yaml, contrib/swot_osse/ose_swot_finetune.py).
# Inputs are read from Odyssey (only the 2017-2022 sla / sla_filtered slices are loaded, ~9 GB each).
# Usage: sbatch scripts/slurm_swot_osse/train_SLA_UNet_OSE_2017_2022_SWOT_finetune.sh datamodule.norm_stats=[mean,std] [overrides]

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
echo "$(date) on $(hostname)"
HYDRA_FULL_ERROR=1 srun python main.py xp=forecast_SLA_UNet_OSE_2017_2022_SWOT_errors_finetune "$@"

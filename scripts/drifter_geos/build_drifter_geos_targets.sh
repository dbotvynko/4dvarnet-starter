#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=drifter_geos
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=2017-2024
#SBATCH --output=/Odyssey/private/d21botvy/job_%A_%a_drifter_geos.log
# Daily geostrophic drifter-velocity targets, one array task per year (2017-2022 = training years of the OSE UNet,
# 2023-2024 kept for evaluation only). Copernicus credentials come from the submitting environment (sbatch --export=ALL).
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate oceanbench
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
python contrib/drifter_geos/build_drifter_geos_targets.py $SLURM_ARRAY_TASK_ID

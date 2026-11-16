#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=grid_allsats23
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_build_gridded_allsats_2023.log
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate oceanbench
python /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/scripts/slurm_swot_osse/build_gridded_allsats_2023.py

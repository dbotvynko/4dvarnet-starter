#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=grid_l3_1722
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_build_gridded_l3_2017_2022.log
# Rebuild the 2017-2022 gridded L3 input of the OSE UNet training (see contrib/drifter_geos/build_gridded_l3.py)
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate oceanbench
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
python contrib/drifter_geos/build_gridded_l3.py

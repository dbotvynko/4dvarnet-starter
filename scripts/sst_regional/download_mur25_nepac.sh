#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=mur25_nepac
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%A_%a_mur25_nepac.log
# MUR25 SST 2017-2024 on the NE Pacific training domain (array index 0-7 = years 2017-2024)
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate oceanbench
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
python contrib/sst_regional/download_mur25_nepac.py $((2017 + SLURM_ARRAY_TASK_ID)) --nthreads 6

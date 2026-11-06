#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=neurost_nepac
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%A_%a_neurost_nepac.log
# NeurOST SLA 2017-2022 on the NE Pacific training domain (array index 0-5 = years 2017-2022)
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate oceanbench
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
python contrib/neurost_regional/download_neurost_nepac.py $((2017 + SLURM_ARRAY_TASK_ID)) --nthreads 6 $EXTRA

#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=osse_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_osse_eval.log
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
YEAR=${1:-2023}
python contrib/swot_osse/osse_eval_glorys.py /Odyssey/private/d21botvy/oceanbench_eval/results/osse_eval_${YEAR}_tracks.json $YEAR both

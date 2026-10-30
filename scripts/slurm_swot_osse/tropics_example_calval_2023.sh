#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=tropics_example
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_tropics_example.log
export HOME=/Odyssey/private/d21botvy/
export MPLCONFIGDIR=/tmp/mpl_$USER
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
for m in nadir swotscratch 1day_orbit; do
  python contrib/swot_osse/tropics_example_calval_2023.py /Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_crossover/example_tropics --model $m $@ || exit 1
done

#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=xover_example
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/Odyssey/private/d21botvy/oceanbench_eval/logs_swot/xover_example_%j.log
#
# Figure of one crossover (contrib/swot_osse/crossover_example_calval_2023.py) for the three UNets, on CPU.
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
OUT=/Odyssey/private/d21botvy/oceanbench_eval/results/swot_denial_crossover/example
for m in nadir 1day_orbit swotscratch; do
  python -u contrib/swot_osse/crossover_example_calval_2023.py $OUT --model $m --replace || exit 1
done

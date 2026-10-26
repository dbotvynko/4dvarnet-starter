#!/bin/bash
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --job-name=dry_ose_swot
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_dryrun_ose_swot_finetune.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_dryrun_ose_swot_finetune.txt
#
# CPU dry run of the OSE UNet SWOT fine-tune (contrib/swot_osse/dryrun_ose_swot_finetune.py): norm_stats, data checks,
# checkpoint loss on one validation window.
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
PYTHONPATH=. python contrib/swot_osse/dryrun_ose_swot_finetune.py

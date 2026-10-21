#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=unet_ose1722
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_unet_ose_2017_2022_inference.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_unet_ose_2017_2022_inference.txt
#
# OSE SLA UNet trained on 2017-2022 with the DUACS losses (the model of the 2023 DriftNet forecasts,
# rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_losses_2017-2022), run with the August 2025 input loader:
#   1. 2023 NRT nadir input: reproduction of the August 2025 run (to compare with the existing 2023 files);
#   2. 2024 CLS nadir input;
#   3. 2024 CLS nadir + SWOT input.
# Usage: sbatch scripts/slurm_swot_osse/inference_sla_unet_ose_2017_2022.sh

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/

for XP in ose_pipeline_proc_2023_SLA_UNet_OSE_2017_2022_nrt_repro ose_pipeline_proc_2024_SLA_UNet_OSE_2017_2022_cls_nadir ose_pipeline_proc_2024_SLA_UNet_OSE_2017_2022_cls_nadir_swot; do
  XPNAME=$(grep -m1 "^xp_name:" config/xp/$XP.yaml | awk '{print $2}')
  if [ "$(ls /Odyssey/public/glorys/rec/$XPNAME/*/test_data_*.nc 2>/dev/null | wc -l)" -ge 7 ]; then
    echo "$(date) skip $XP: forecast files already in /Odyssey/public/glorys/rec/$XPNAME"; continue
  fi
  echo "$(date) start $XP"
  HYDRA_FULL_ERROR=1 srun python main.py xp=$XP
  status=$?
  echo "$(date) $XP EXIT $status"
  ls -la /Odyssey/public/glorys/rec/$XPNAME/*/
  [ $status -eq 0 ] || exit $status
done

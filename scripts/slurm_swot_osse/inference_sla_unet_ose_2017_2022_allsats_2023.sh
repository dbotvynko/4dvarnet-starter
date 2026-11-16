#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=unet_allsats23
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=100G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_unet_ose_2017_2022_allsats_2023_inference.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error_unet_ose_2017_2022_allsats_2023_inference.txt
# OSE 2017-2022 SLA UNet, 2023, input = all satellites (6 pipeline sats + s6a-hr, build_gridded_allsats_2023.py),
# -> /Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_losses_2017-2022_ALLSATS_2023/nrt_sla/test_data_13..20.nc
export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12/
HYDRA_FULL_ERROR=1 srun python main.py xp=ose_pipeline_proc_2023_SLA_UNet_OSE_2017_2022_allsats_day13

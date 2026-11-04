#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=geos_ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_geos_finetune.log
# Velocity fine-tune of the OSE SLA UNet on 2 GPUs (DDP), weights from the calibration job.
# Checkpoints: /Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune/checkpoints
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
read WG WD < <(python -c "import json; w=json.load(open('/Odyssey/public/glorys/trainings/forecast_global_1patch_2017_2022_SLA_UNet_geos_drifter_finetune/loss_weights.json')); print(w['w_geos'], w['w_drift'])")
echo "w_geos=$WG w_drift=$WD"
HYDRA_FULL_ERROR=1 srun python main.py xp=finetune_SLA_UNet_OSE_2017_2022_geos_drifter model.w_geos=$WG model.w_drift=$WD

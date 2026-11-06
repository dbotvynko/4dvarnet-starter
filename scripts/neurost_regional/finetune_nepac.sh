#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=nepac_ft
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_nepac_regional_ft.log
# Regional NE Pacific fine-tune (target duacs = control, or neurost), then regional 2023/2024 inference with the checkpoint
# of lowest validation drifter RMSE -> /Odyssey/public/glorys/rec/forecast_Unet_SLA_nepac_regional_ft_<run>_<year>/nrt_sla
# Usage: sbatch scripts/neurost_regional/finetune_nepac.sh duacs|neurost
RUN=$1
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
export PYTHONPATH=$PWD
TGT=null; [ "$RUN" = neurost ] && TGT='${paths.neurost}'
OUT=/Odyssey/public/glorys/trainings/forecast_nepac_2017_2022_SLA_UNet_regional_finetune/$RUN
HYDRA_FULL_ERROR=1 python main.py xp=finetune_SLA_UNet_OSE_2017_2022_nepac_regional run_name=$RUN "datamodule.input_da.tgt_files=$TGT" || exit 1
BEST=$(ls $OUT/checkpoints/drift=*.ckpt | sort | head -1); ln -sfn "$BEST" $OUT/checkpoints/best.ckpt; echo "best: $BEST"
python contrib/neurost_regional/infer_regional.py $OUT/checkpoints/best.ckpt forecast_Unet_SLA_nepac_regional_ft_$RUN

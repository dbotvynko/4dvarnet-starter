#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=nepac_ssthp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_nepac_regional_ssthp.log
# v2 (high-passed SST, sigma 1 deg). Regional NE Pacific fine-tune WITH SST input (target duacs = same as the SLA-only control run "duacs"), then regional
# 2023/2024 inference with the checkpoint of lowest validation drifter RMSE -> /Odyssey/public/glorys/rec/forecast_Unet_SLA_nepac_regional_ftssthp_<run>_<year>/nrt_sla
# Usage: sbatch scripts/sst_regional/finetune_nepac_sst.sh duacs|neurost
RUN=$1
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
export PYTHONPATH=$PWD
TGT=null; [ "$RUN" = neurost ] && TGT='${paths.neurost}'
OUT=/Odyssey/public/glorys/trainings/forecast_nepac_2017_2022_SLA_UNet_regional_finetune_ssthp/$RUN
HYDRA_FULL_ERROR=1 python main.py xp=finetune_SLA_UNet_OSE_2017_2022_nepac_regional_ssthp run_name=$RUN "datamodule.input_da.tgt_files=$TGT" || exit 1
BEST=$(ls $OUT/checkpoints/drift=*.ckpt | sort | head -1); ln -sfn "$BEST" $OUT/checkpoints/best.ckpt; echo "best: $BEST"
python contrib/sst_regional/infer_regional_sst.py $OUT/checkpoints/best.ckpt forecast_Unet_SLA_nepac_regional_ftssthp_$RUN --xp finetune_SLA_UNet_OSE_2017_2022_nepac_regional_ssthp

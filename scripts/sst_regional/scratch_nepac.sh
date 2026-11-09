#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=nepac_scratch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_nepac_scratch.log
# Regional NE Pacific UNet trained FROM SCRATCH (no pre-trained init, random SST weights), 150 epochs, lr 1e-3 cosine (as
# the original global training), same losses (SLA + geostrophic + drifters) and checkpoint selection (val drifter RMSE)
# as the regional fine-tunes. Two runs make the clean test "does SST help when the network learns with it from the start":
#   sla   : xp finetune_SLA_UNet_OSE_2017_2022_nepac_regional      (SLA only, control)
#   hpsst : xp finetune_SLA_UNet_OSE_2017_2022_nepac_regional_ssthp (SLA + high-passed MUR25 SST)
# then 2023/2024 regional inference -> /Odyssey/public/glorys/rec/forecast_Unet_SLA_nepac_regional_scratch_<run>_<year>
# Usage: sbatch scripts/sst_regional/scratch_nepac.sh sla|hpsst
RUN=$1
export HOME=/Odyssey/private/d21botvy/
source /Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh
conda activate 4dvar_v2
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
export PYTHONPATH=$PWD
if [ "$RUN" = sla ]; then XP=finetune_SLA_UNet_OSE_2017_2022_nepac_regional; BASE=forecast_nepac_2017_2022_SLA_UNet_regional_finetune
else XP=finetune_SLA_UNet_OSE_2017_2022_nepac_regional_ssthp; BASE=forecast_nepac_2017_2022_SLA_UNet_regional_finetune_ssthp; fi
OUT=/Odyssey/public/glorys/trainings/$BASE/scratch
HYDRA_FULL_ERROR=1 python main.py xp=$XP run_name=scratch model.init_ckpt=null trainer.max_epochs=150 model.opt_fn.lr=1e-3 || exit 1
BEST=$(ls $OUT/checkpoints/drift=*.ckpt | sort -t= -k2 -g | head -1); ln -sfn "$BEST" $OUT/checkpoints/best.ckpt; echo "best: $BEST"
if [ "$RUN" = sla ]; then python contrib/neurost_regional/infer_regional.py $OUT/checkpoints/best.ckpt forecast_Unet_SLA_nepac_regional_scratch_$RUN
else python contrib/sst_regional/infer_regional_sst.py $OUT/checkpoints/best.ckpt forecast_Unet_SLA_nepac_regional_scratch_$RUN --xp $XP; fi

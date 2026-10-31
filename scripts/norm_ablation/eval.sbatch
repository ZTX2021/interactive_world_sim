#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --constraint=h200
#SBATCH --job-name=iws-norm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cj2220/iws_latent_norm/logs/norm_eval_%j.out
#SBATCH --error=/scratch/cj2220/iws_latent_norm/logs/norm_eval_%j.err
#
# Inference only. Scores checkpoints on the 10 held-out T-Pushing episodes.
#   MODE=recon   STAGE=stage_1  autoencoder reconstruction
#   MODE=predict STAGE=stage_2  open-loop prediction from one frame (paper protocol)
# STEP picks the checkpoint by global step; ONLY limits the variants (default all five).
#
#   sbatch --export=ALL,MODE=recon,STAGE=stage_1,STEP=48000 scripts/norm_ablation/eval.sbatch
#   sbatch --export=ALL,MODE=recon,STAGE=stage_1,STEP=48000,ONLY="l2 layernorm" scripts/norm_ablation/eval.sbatch
set -euo pipefail
source /scratch/cj2220/iws_latent_norm/scripts/norm_ablation/common.sh
: "${MODE:?set MODE=recon|predict}" "${STAGE:?set STAGE=stage_1|stage_2}" "${STEP:?set STEP}"
VARIANTS_SEL=${ONLY:-${VARIANTS[*]}}

echo "[$(date)] eval job $SLURM_JOB_ID mode=$MODE stage=$STAGE step=$STEP variants=$VARIANTS_SEL"
for V in $VARIANTS_SEL; do
  CKPT=$(ls "$OUT_ROOT/$V/$STAGE/checkpoints/"epoch=*-step=$STEP.ckpt 2>/dev/null | head -1 || true)
  if [[ -z "$CKPT" ]]; then
    echo "skip $V: no $STAGE checkpoint at step $STEP"
    continue
  fi
  run_in_overlay "python scripts/norm_ablation/eval_video.py --mode $MODE \
    --data-dir $DATA/val --episodes 10 --videos 2 \
    --out $REPORTS/${MODE}_${V}_step${STEP}.json --ckpt '$CKPT'"
done
echo "[$(date)] done; results in $REPORTS"

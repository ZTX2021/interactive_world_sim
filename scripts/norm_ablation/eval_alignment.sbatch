#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --constraint=h200
#SBATCH --job-name=iws-align-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/cj2220/iws_latent_norm/logs/norm_align_%j.out
#SBATCH --error=/scratch/cj2220/iws_latent_norm/logs/norm_align_%j.err
#
# Inference only. Checks the action alignment of the prediction evaluation on
# the IWS authors' trained PushT stage-2 model: frame t conditioned on
# action[t] (training pairing, offset 0) versus action[t + 1] (IWS validation
# rollout, offset 1), on the 10 held-out episodes.
set -euo pipefail
source /scratch/cj2220/iws_latent_norm/scripts/norm_ablation/common.sh

CKPT=/scratch/cj2220/interactive_world_sim/outputs/pusht/stage_2_c/checkpoints/epoch=252-step=470000.ckpt
OUT=$REPORTS/alignment_check
mkdir -p "$OUT"
echo "[$(date)] alignment check $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
grep -nE "mask_prev_action|n_tokens|dyn_infer_steps|latent_norm" \
  /scratch/cj2220/interactive_world_sim/outputs/pusht/stage_2_c/.hydra/config.yaml || true
for OFF in 0 1; do
  run_in_overlay "python scripts/norm_ablation/eval_video.py --mode predict \
    --data-dir $DATA/val --episodes 10 --videos 2 --action-offset $OFF \
    --out $OUT/predict_iws_stage2c_offset$OFF.json --ckpt '$CKPT'"
done
echo "[$(date)] done; results in $OUT"

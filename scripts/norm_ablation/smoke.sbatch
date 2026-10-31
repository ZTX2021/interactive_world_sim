#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --constraint=h200
#SBATCH --job-name=iws-norm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/cj2220/iws_latent_norm/logs/norm_smoke_%j.out
#SBATCH --error=/scratch/cj2220/iws_latent_norm/logs/norm_smoke_%j.err
#
# Preflight for the normalization ablation, on the same GPU type and native
# batch as the real runs. Writes only under outputs/pusht_norm_smoke/ and
# $REPORTS/smoke/. Checks:
#   1. LatentNorm unit checks.
#   2. Every variant: 150 stage-1 steps with checkpoints, validation and the
#      max_time override, then a resume from last.ckpt to 300 steps.
#   3. Recon evaluation of every variant's checkpoint (1 episode).
#   4. l2 and layernorm: 100 stage-2 steps loading that stage-1 checkpoint,
#      then predict evaluation (1 episode). A stage-2 run whose latent_norm
#      does not match its autoencoder must fail.
set -euo pipefail
source /scratch/cj2220/iws_latent_norm/scripts/norm_ablation/common.sh

SMOKE=$REPO/outputs/pusht_norm_smoke/$SLURM_JOB_ID
OUT=$REPORTS/smoke/$SLURM_JOB_ID
mkdir -p "$SMOKE" "$OUT"
echo "[$(date)] smoke job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$REPO" log -1 --format='code %H %s'
git -C "$REPO" status --short
(cd "$REPO" && sha256sum interactive_world_sim/algorithms/latent_dynamics/latent_norm.py \
  interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py \
  configurations/algorithm/latent_world_model.yaml $(find scripts/norm_ablation -maxdepth 1 -type f | sort))

SHORT="experiment.training.checkpointing.every_n_train_steps=50 \
  experiment.validation.val_every_n_step=100 experiment.training.log_every_n_steps=10 \
  experiment.training.max_time=00:24:00:00 wandb.mode=disabled"

run_in_overlay "python -c 'import interactive_world_sim, piqa; print(interactive_world_sim.__file__, piqa.__file__)' \
  && python scripts/norm_ablation/test_latent_norm.py"

# Throughput at the real stage-1 settings (no validation or checkpoint below
# 3,000 steps). Two fresh runs; the difference removes startup time.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
timed_run() {
  local t0 t1
  t0=$(date +%s)
  run_in_overlay "$(stage1_cmd l2 "$SMOKE/throughput_$1/stage_1" wandb.mode=disabled \
    experiment.training.max_steps=$1)" >&2 || return 1
  t1=$(date +%s)
  echo $((t1 - t0))
}
# set -e does not reach inside $(...), so check each run explicitly
W100=$(timed_run 100) || { echo "FAIL: 100-step throughput run"; exit 1; }
W900=$(timed_run 900) || { echo "FAIL: 900-step throughput run"; exit 1; }
echo "THROUGHPUT gpu=$GPU_NAME wall_100=${W100}s wall_900=${W900}s steps_per_s=$(python3 -c "print(round(800/max($W900-$W100,1),3))")" | tee "$OUT/throughput.txt"

for V in "${VARIANTS[@]}"; do
  S1=$SMOKE/$V/stage_1
  echo "[$(date)] ===== $V stage 1"
  run_in_overlay "$(stage1_cmd "$V" "$S1" $SHORT experiment.training.max_steps=150)"
  run_in_overlay "$(stage1_cmd "$V" "$S1" $SHORT experiment.training.max_steps=300 load=$S1/checkpoints/last.ckpt)"
  ls "$S1/checkpoints"
  run_in_overlay "python scripts/norm_ablation/eval_video.py --mode recon --episodes 1 --videos 1 \
    --data-dir $DATA/val --out $OUT/recon_$V.json --ckpt '$S1/checkpoints/epoch=*-step=300.ckpt'"
done

for V in l2 layernorm; do
  echo "[$(date)] ===== $V stage 2"
  S2=$SMOKE/$V/stage_2
  run_in_overlay "$(stage2_cmd "$V" "$S2" "$SMOKE/$V/stage_1/checkpoints/last.ckpt" $SHORT experiment.training.max_steps=100)"
  run_in_overlay "python scripts/norm_ablation/eval_video.py --mode predict --episodes 1 --videos 1 \
    --data-dir $DATA/val --out $OUT/predict_$V.json --ckpt '$S2/checkpoints/epoch=*-step=100.ckpt'"
done

echo "[$(date)] ===== mismatched stage 2 (must fail)"
if run_in_overlay "$(stage2_cmd layernorm "$SMOKE/mismatch/stage_2" "$SMOKE/l2/stage_1/checkpoints/last.ckpt" $SHORT experiment.training.max_steps=10)"; then
  echo "FAIL: mismatched latent_norm was accepted"; exit 1
fi
echo "[$(date)] smoke passed; outputs $SMOKE, reports $OUT"

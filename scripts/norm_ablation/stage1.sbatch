#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --constraint=h200
#SBATCH --job-name=iws-norm-s1
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:45:00
#SBATCH --output=/scratch/cj2220/iws_latent_norm/logs/norm_s1_%A_%a.out
#SBATCH --error=/scratch/cj2220/iws_latent_norm/logs/norm_s1_%A_%a.err
#
# IWS stage 1 (autoencoder) for the latent-normalization ablation on T-Pushing.
# One array task per variant: 0 l2 (IWS default), 1 none, 2 layernorm,
# 3 rmsnorm, 4 groupnorm. Each trains for 24 hours of training time
# (Lightning max_time, restored on resume) on one H200, the closest Torch GPU
# to the GH200 the IWS authors used, and keeps a checkpoint every 3,000 steps.
#
# Submit:   sbatch scripts/norm_ablation/stage1.sbatch           (all five)
#           sbatch --array=2 scripts/norm_ablation/stage1.sbatch (one variant)
# Resubmitting resumes each variant from its last checkpoint.
set -euo pipefail
source /scratch/cj2220/iws_latent_norm/scripts/norm_ablation/common.sh

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
RUN_DIR=$OUT_ROOT/$VARIANT/stage_1
mkdir -p "$RUN_DIR"

# Not on B200: the iws overlay's PyTorch has no sm_100 kernels ("no kernel image
# is available", smoke 21406134, 2026-10-30). The USR1 handler below is kept for
# preemptible lanes: a preemption or the USR1 warning requeues the task; it
# resumes from last.ckpt (at most 3,000 steps lost) and Lightning restores the
# elapsed training time.
requeue_self() {
  echo "[$(date)] $1 received, requeueing $SLURM_JOB_ID"
  [[ -n "${CHILD_PID:-}" ]] && kill -TERM "$CHILD_PID" 2>/dev/null && wait "$CHILD_PID" 2>/dev/null
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
}
trap 'requeue_self USR1' USR1
trap 'echo "[$(date)] SIGTERM"; [[ -n "${CHILD_PID:-}" ]] && kill -TERM "$CHILD_PID"; wait' TERM

echo "[$(date)] job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID variant=$VARIANT on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$REPO" log -1 --format='code %H %s'
git -C "$REPO" status --short
(cd "$REPO" && sha256sum interactive_world_sim/algorithms/latent_dynamics/latent_norm.py \
  interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py \
  configurations/algorithm/latent_world_model.yaml $(find scripts/norm_ablation -maxdepth 1 -type f | sort))

RESUME=""
LATEST=$(latest_ckpt "$RUN_DIR")
if [[ -n "$LATEST" ]]; then
  RESUME="load=$LATEST"
  echo "[$(date)] resuming from $LATEST"
fi

# main.py relinks outputs/pusht_norm/latest-run at startup; stagger the tasks
sleep $((SLURM_ARRAY_TASK_ID * 30))
run_in_overlay "$(stage1_cmd "$VARIANT" "$RUN_DIR" experiment.training.max_time=00:24:00:00 $RESUME)"
echo "[$(date)] stage 1 $VARIANT exited cleanly"

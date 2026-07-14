#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-airhockey-s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/airhockey_s1_%j.out
#SBATCH --error=logs/airhockey_s1_%j.err
#SBATCH --signal=B:USR1@180
#
# Stage 1 (autoencoder) for the airhockey-1m world model on H200.
# The dataset is already on scratch ($SCRATCH/data/airhockey-1m), so unlike the
# letters scripts there is no HuggingFace download step. Stage 2 (dynamics) and
# stage 3 (autoencoder finetune) are run separately.
#
# Dataset: $SCRATCH/data/airhockey-1m   (manifest-backed HDF5 shards)
#   1000 episodes x 1000 frames, 128x128 RGB, 4-dim mallet action [p1x,p1y,p2x,p2y]
#   950 train / 50 valid episodes, split by whole game (see its README.md).
#
# Usage:
#   sbatch scripts/hpc/train_airhockey_stage1.sbatch
#
# Prereqs:
#   - $SCRATCH/iws.ext3 built (conda overlay with iws env)
#   - $SCRATCH/data/airhockey-1m present (train/ valid/ h5_manifest.json)
#   - ~/.secrets/wandb_api_key (chmod 600)

set -euo pipefail

RUN="airhockey"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

DATA_DIR="$SCRATCH/data/airhockey-1m"
S1_DIR=$REPO/outputs/$RUN/stage_1
S1_DONE=$REPO/outputs/$RUN/_stage1_done

mkdir -p logs "$S1_DIR"

if [[ ! -f "$DATA_DIR/h5_manifest.json" ]]; then
  echo "FATAL: dataset not found at $DATA_DIR (expected h5_manifest.json)"
  exit 1
fi

# ---------- signal handler ----------
requeue_self() {
  echo "[$(date)] SIGUSR1 received — requeueing job $SLURM_JOB_ID"
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill -SIGTERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_self USR1

# ---------- apptainer runner ----------
run_train() {
  local cmd="$1"
  apptainer exec --nv \
    --overlay "$OVERLAY:ro" \
    --bind /scratch/$USER:/scratch/$USER \
    "$IMAGE" \
    /bin/bash -c "
      set -e
      source /ext3/env.sh
      conda activate iws
      cd $REPO
      export HDF5_USE_FILE_LOCKING=FALSE
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export TOKENIZERS_PARALLELISM=false
      export WANDB_API_KEY=\$(cat ~/.secrets/wandb_api_key 2>/dev/null || echo '')
      $cmd
    " &
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  local rc=$?
  unset TRAIN_PID
  return $rc
}

# ---------- helpers ----------
latest_ckpt() {
  local dir="$1/checkpoints"
  if [[ -f "$dir/last.ckpt" ]]; then
    echo "$dir/last.ckpt"
  elif compgen -G "$dir/*.ckpt" > /dev/null; then
    ls -t "$dir"/*.ckpt | head -1
  fi
}

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  run=$RUN  dataset=$DATA_DIR"
echo "  S1_DIR=$S1_DIR  done=$( [[ -f $S1_DONE ]] && echo yes || echo no )"
echo "================================================================"
nvidia-smi || true

# ============================================================
# Stage 1 — Autoencoder
# ============================================================
if [[ ! -f $S1_DONE ]]; then
  RESUME_FLAG=""
  S1_LATEST=$(latest_ckpt "$S1_DIR")
  if [[ -n "$S1_LATEST" ]]; then
    RESUME_FLAG="load=$S1_LATEST"
    echo "[$(date)] Stage 1 resuming from $S1_LATEST"
  else
    echo "[$(date)] Stage 1 starting fresh"
  fi

  CMD_S1="python main.py +name=${RUN}_stage_1 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=airhockey_dataset \
    dataset.dataset_dir=$DATA_DIR \
    dataset.horizon=1 dataset.val_horizon=1 \
    dataset.obs_keys=[top_pov] \
    hydra.run.dir=$S1_DIR \
    experiment.training.batch_size=32 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    experiment.validation.batch_size=10 \
    experiment.validation.val_every_n_step=6000 \
    experiment.validation.limit_batch=1.0 \
    algorithm.latent_dim=512 \
    algorithm.action_dim=4 \
    algorithm.training_stage=1 \
    +experiment.training.checkpointing.save_last=True \
    $RESUME_FLAG"

  run_train "$CMD_S1"
  echo "[$(date)] Stage 1 finished cleanly"
  touch "$S1_DONE"
else
  echo "[$(date)] Stage 1 already done — skipping"
fi

echo "[$(date)] Stage 1 complete for run=$RUN"

#!/bin/bash
#SBATCH --account=torch_pr_37_lpinto
#SBATCH --partition=h200_courant
#SBATCH --job-name=iws-airhockey-s3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/airhockey_s3_%j.out
#SBATCH --error=logs/airhockey_s3_%j.err
#SBATCH --signal=B:USR1@180
#
# Stage 3 (autoencoder finetune) for the airhockey-1m world model on H200.
# Snapshots a stage-2 checkpoint at job start and finetunes the autoencoder
# from it. Safe to run in parallel with a still-going stage-2 job; writes to
# outputs/airhockey/stage_3/ which is disjoint from stage_2.
#
# Dataset: $SCRATCH/data/airhockey-1m   (manifest-backed HDF5 shards)
#
# Usage:
#   sbatch scripts/hpc/train_airhockey_stage3.sbatch [s2_ckpt]
#     s2_ckpt = optional explicit path to a stage-2 .ckpt file
#
# Prereqs:
#   - Stage 2 checkpoint exists at outputs/airhockey/stage_2/checkpoints/
#   - $SCRATCH/iws.ext3 built (conda overlay with iws env)
#   - ~/.secrets/wandb_api_key (chmod 600)

set -euo pipefail

RUN="airhockey"

REPO=/scratch/$USER/interactive_world_sim
OVERLAY=$SCRATCH/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif

DATA_DIR="$SCRATCH/data/airhockey-1m"
S2_DIR=$REPO/outputs/$RUN/stage_2
S3_DIR=$REPO/outputs/$RUN/stage_3
S3_DONE=$REPO/outputs/$RUN/_stage3_done

mkdir -p logs "$S3_DIR"

if [[ ! -f "$DATA_DIR/h5_manifest.json" ]]; then
  echo "FATAL: dataset not found at $DATA_DIR (expected h5_manifest.json)"
  exit 1
fi

# ---------- pick the stage-2 checkpoint (once, at job start) ----------
S2_CKPT_ARG="${1:-}"
if [[ -n "$S2_CKPT_ARG" ]]; then
  S2_CKPT="$S2_CKPT_ARG"
elif [[ -f "$S2_DIR/checkpoints/last.ckpt" ]]; then
  S2_CKPT="$S2_DIR/checkpoints/last.ckpt"
elif compgen -G "$S2_DIR/checkpoints/*.ckpt" > /dev/null; then
  S2_CKPT=$(ls -t "$S2_DIR"/checkpoints/*.ckpt | head -1)
else
  echo "FATAL: no stage-2 checkpoint found in $S2_DIR/checkpoints/"
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

# ============================================================
# Snapshot S2 ckpt + .hydra config
# latent_world_model.py resolves the load_ae config from
# dirname(dirname(ckpt))/.hydra/config.yaml, so the seed ckpt must live in a
# run-dir-shaped layout: <_s2_seed>/checkpoints/<ckpt> + <_s2_seed>/.hydra/.
# ============================================================
SNAPSHOT_DIR=$S3_DIR/_s2_seed
SNAPSHOT=$SNAPSHOT_DIR/checkpoints/$(basename "$S2_CKPT")
if [[ ! -f "$SNAPSHOT" ]]; then
  echo "[$(date)] snapshotting $S2_CKPT → $SNAPSHOT"
  mkdir -p "$SNAPSHOT_DIR/checkpoints"
  cp "$S2_CKPT" "$SNAPSHOT"
  if [[ -d "$S2_DIR/.hydra" ]]; then
    cp -r "$S2_DIR/.hydra" "$SNAPSHOT_DIR/.hydra"
  else
    echo "FATAL: expected $S2_DIR/.hydra/ to exist for load_ae config resolution"
    exit 1
  fi
fi
S2_CKPT="$SNAPSHOT"

# ---------- find latest s3 ckpt for resume ----------
RESUME_FLAG=""
S3_LATEST=$(latest_ckpt "$S3_DIR")
if [[ -n "$S3_LATEST" ]]; then
  RESUME_FLAG="load=$S3_LATEST"
  echo "[$(date)] Stage 3 resuming from $S3_LATEST"
else
  echo "[$(date)] Stage 3 starting fresh from S2 ckpt $S2_CKPT"
fi

echo "================================================================"
echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
echo "  run=$RUN  dataset=$DATA_DIR"
echo "  S2 seed: $S2_CKPT"
echo "  S3 dir : $S3_DIR"
echo "  resume : ${RESUME_FLAG:-<none>}"
echo "================================================================"
nvidia-smi || true

# ============================================================
# Stage 3 — Autoencoder finetune
# ============================================================
if [[ ! -f $S3_DONE ]]; then
  CMD_S3="python main.py +name=${RUN}_stage_3 \
    algorithm=latent_world_model \
    experiment=exp_latent_dyn \
    dataset=airhockey_dataset \
    dataset.dataset_dir=$DATA_DIR \
    dataset.horizon=1 dataset.val_horizon=200 \
    dataset.obs_keys=[top_pov] \
    hydra.run.dir=$S3_DIR \
    experiment.training.batch_size=32 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    experiment.training.data.num_workers=4 \
    experiment.validation.batch_size=2 \
    experiment.validation.val_every_n_step=30000 \
    experiment.validation.limit_batch=1.0 \
    experiment.validation.data.num_workers=4 \
    algorithm.latent_dim=512 \
    algorithm.action_dim=4 \
    algorithm.noise_scheduler.loss_weighting=uniform \
    algorithm.sampling_strategy=terminal_only \
    algorithm.load_ae=$S2_CKPT \
    algorithm.training_stage=3 \
    +experiment.training.checkpointing.save_last=True \
    $RESUME_FLAG"

  run_train "$CMD_S3"
  echo "[$(date)] Stage 3 finished cleanly"
  touch "$S3_DONE"
else
  echo "[$(date)] Stage 3 already done — skipping"
fi

echo "[$(date)] Stage 3 complete for run=$RUN"

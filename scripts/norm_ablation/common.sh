# Shared settings for the IWS latent-normalization ablation (source, don't run).
#
# Training settings are the IWS authors' defaults for 128-pixel single-camera
# tasks, read from their W&B runs `train_dyn_latent_student_v1_4_*` (2025-12 to
# 2026-03): stage 1 batch 16, LR 8e-5, fp32, horizon 1, checkpoint every 3,000
# steps keeping all, validation every 6,000; stage 2 batch 4, horizon 10,
# uniform loss weighting, terminal-only sampling. Only `algorithm.latent_norm`
# differs between variants.

# The paths default to Torch; run_local.sh sets them for other machines.
REPO=${REPO:-/scratch/cj2220/iws_latent_norm}
DATA=${DATA:-/scratch/cj2220/interactive_world_sim/data/full/pusht}
OVERLAY=/scratch/cj2220/iws.ext3
IMAGE=${CONTAINER_IMAGES_FOLDER:-/share/apps/images}/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OUT_ROOT=${OUT_ROOT:-$REPO/outputs/pusht_norm}
REPORTS=${REPORTS:-/scratch/cj2220/X/sa/reports/iws_norm_ablation_20261029}
VARIANTS=(l2 none layernorm rmsnorm groupnorm)

# run_in_overlay "<command>": run inside the iws environment, in the background,
# and wait so that a SIGUSR1/SIGTERM trap in the caller can still fire.
run_in_overlay() {
  apptainer exec --nv \
    --overlay "$OVERLAY:ro" \
    --bind /scratch/cj2220:/scratch/cj2220 \
    "$IMAGE" \
    /bin/bash -c "
      set -e
      source /ext3/env.sh
      conda activate iws
      cd $REPO
      export PYTHONPATH=/scratch/cj2220/pydeps/piqa-1.3.2\${PYTHONPATH:+:\$PYTHONPATH}
      export HDF5_USE_FILE_LOCKING=FALSE
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export TOKENIZERS_PARALLELISM=false
      export TORCH_HOME=/scratch/cj2220/.cache/torch
      export WANDB_DIR=/scratch/cj2220/wandb
      export WANDB_API_KEY=\$(cat ~/.secrets/wandb_api_key 2>/dev/null || echo '')
      $1
    " &
  CHILD_PID=$!
  wait "$CHILD_PID"
  local rc=$?
  unset CHILD_PID
  return $rc
}

latest_ckpt() {
  local dir="$1/checkpoints"
  if [[ -f "$dir/last.ckpt" ]]; then
    echo "$dir/last.ckpt"
  elif compgen -G "$dir/*.ckpt" > /dev/null; then
    ls -t "$dir"/*.ckpt | head -1
  fi
}

# stage1_cmd VARIANT RUN_DIR [hydra overrides...]
stage1_cmd() {
  local variant=$1 run_dir=$2
  shift 2
  echo "python main.py +name=pusht_norm_${variant}_stage_1 \
    algorithm=latent_world_model experiment=exp_latent_dyn dataset=real_aloha_dataset \
    dataset.dataset_dir=$DATA dataset.horizon=1 dataset.val_horizon=1 \
    dataset.obs_keys=[camera_1_color] dataset.action_mode=bimanual_push \
    hydra.run.dir=$run_dir \
    wandb.project=iws_norm_ablation \
    experiment.training.batch_size=16 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.data.num_workers=8 \
    experiment.training.checkpointing.every_n_train_steps=3000 \
    +experiment.training.checkpointing.save_top_k=-1 \
    +experiment.training.checkpointing.save_last=True \
    experiment.validation.batch_size=10 \
    experiment.validation.val_every_n_step=6000 \
    experiment.validation.limit_batch=1.0 \
    experiment.validation.data.num_workers=16 \
    algorithm.latent_dim=512 algorithm.action_dim=4 \
    algorithm.training_stage=1 \
    algorithm.latent_norm=$variant algorithm.latent_norm_rescale=True \
    $*"
}

# stage2_cmd VARIANT RUN_DIR STAGE1_CKPT [hydra overrides...]
stage2_cmd() {
  local variant=$1 run_dir=$2 ae=$3
  shift 3
  echo "python main.py +name=pusht_norm_${variant}_stage_2 \
    algorithm=latent_world_model experiment=exp_latent_dyn dataset=real_aloha_dataset \
    dataset.dataset_dir=$DATA dataset.horizon=10 dataset.val_horizon=200 \
    dataset.obs_keys=[camera_1_color] dataset.action_mode=bimanual_push \
    hydra.run.dir=$run_dir \
    wandb.project=iws_norm_ablation \
    experiment.training.batch_size=4 \
    experiment.training.max_steps=1000005 \
    experiment.training.log_every_n_steps=100 \
    experiment.training.data.num_workers=4 \
    experiment.training.checkpointing.every_n_train_steps=10000 \
    +experiment.training.checkpointing.save_top_k=-1 \
    +experiment.training.checkpointing.save_last=True \
    experiment.validation.batch_size=2 \
    experiment.validation.val_every_n_step=30000 \
    experiment.validation.limit_batch=1.0 \
    experiment.validation.data.num_workers=4 \
    algorithm.latent_dim=512 algorithm.action_dim=4 \
    algorithm.noise_scheduler.loss_weighting=uniform \
    algorithm.sampling_strategy=terminal_only \
    algorithm.load_ae=$ae \
    algorithm.training_stage=2 \
    algorithm.latent_norm=$variant algorithm.latent_norm_rescale=True \
    $*"
}

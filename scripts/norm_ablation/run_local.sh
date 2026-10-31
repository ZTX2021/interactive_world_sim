#!/bin/bash
# Run the latent-normalization ablation on a plain multi-GPU machine (no Slurm).
#
# For each variant, on one GPU: stage 1 for STAGE1_TIME of training time,
# stage 2 on that autoencoder's last checkpoint for STAGE2_TIME, then the
# image metrics of both (eval_video.py: reconstruction for stage 1, open-loop
# prediction from one frame for stage 2, 10 held-out episodes). GPUs run their
# variants in parallel; with fewer GPUs than variants a GPU runs several in
# turn. Rerunning the script resumes: finished stages (DONE marker) and
# finished evaluations (JSON) are skipped, an unfinished stage resumes from
# its last checkpoint, and Lightning restores the elapsed training time.
#
# Activate the IWS Python environment first, then from the repository root:
#   DATA=/path/to/pusht SMOKE=1 bash scripts/norm_ablation/run_local.sh   # ~15 min check
#   DATA=/path/to/pusht bash scripts/norm_ablation/run_local.sh           # the real runs
#
# Settings (environment variables):
#   DATA         PushT data with train/ and val/ (required)
#   GPUS         GPU indices, one variant per GPU at a time (default: all GPUs)
#   VARIANTS     subset of "l2 none layernorm rmsnorm groupnorm" (default: all)
#   STAGE1_TIME  default 00:15:00:00 (DD:HH:MM:SS)
#   STAGE2_TIME  default 00:08:00:00
#   OUT_ROOT     default outputs/pusht_norm (outputs/pusht_norm_smoke with SMOKE=1)
#   REPORTS      default reports/norm_ablation (reports/norm_ablation_smoke)
#   WANDB_MODE   offline (default), online or disabled
#   WANDB_ENTITY default charlesji; the project is iws_norm_ablation
#   EXTRA_S1, EXTRA_S2  extra Hydra overrides for stage 1 / stage 2
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${DATA:?set DATA to the PushT directory containing train/ and val/}"
DATA=$(cd "$DATA" && pwd)
SMOKE=${SMOKE:-0}
if [[ $SMOKE == 1 ]]; then
  OUT_ROOT=${OUT_ROOT:-$REPO/outputs/pusht_norm_smoke}
  REPORTS=${REPORTS:-$REPO/reports/norm_ablation_smoke}
  STAGE1_TIME=${STAGE1_TIME:-00:00:04:00}
  STAGE2_TIME=${STAGE2_TIME:-00:00:04:00}
  WANDB_MODE=${WANDB_MODE:-disabled}
  EPISODES=1
  SMOKE_OVERRIDES="experiment.training.checkpointing.every_n_train_steps=100 \
    experiment.validation.val_every_n_step=200 experiment.validation.limit_batch=1 \
    experiment.training.log_every_n_steps=10"
else
  OUT_ROOT=${OUT_ROOT:-$REPO/outputs/pusht_norm}
  REPORTS=${REPORTS:-$REPO/reports/norm_ablation}
  STAGE1_TIME=${STAGE1_TIME:-00:15:00:00}
  STAGE2_TIME=${STAGE2_TIME:-00:08:00:00}
  WANDB_MODE=${WANDB_MODE:-offline}
  EPISODES=10
  SMOKE_OVERRIDES=""
fi
WANDB_ENTITY=${WANDB_ENTITY:-charlesji}
export REPO DATA OUT_ROOT REPORTS
REQUESTED=${VARIANTS:-}
source "$REPO/scripts/norm_ablation/common.sh"  # sets the VARIANTS array
read -ra RUN_VARIANTS <<< "${REQUESTED:-${VARIANTS[*]}}"
for V in "${RUN_VARIANTS[@]}"; do
  [[ " ${VARIANTS[*]} " == *" $V "* ]] || { echo "unknown variant $V"; exit 1; }
done
if [[ -n "${GPUS:-}" ]]; then
  read -ra LANES <<< "$GPUS"
else
  read -ra LANES <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
(( ${#LANES[@]} > 0 )) || { echo "no GPUs found; set GPUS"; exit 1; }

export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
COMMON_OVERRIDES="wandb.mode=$WANDB_MODE wandb.entity=$WANDB_ENTITY $SMOKE_OVERRIDES"

mkdir -p "$OUT_ROOT/logs" "$REPORTS"
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*"; }

{
  log "run_local.sh smoke=$SMOKE stage1=$STAGE1_TIME stage2=$STAGE2_TIME"
  echo "variants: ${RUN_VARIANTS[*]}; GPUs: ${LANES[*]}; data: $DATA"
  echo "outputs: $OUT_ROOT; reports: $REPORTS; wandb: $WANDB_MODE $WANDB_ENTITY/iws_norm_ablation"
  git log -1 --format='code %H %s'
  git status --short
  hostname
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  python -c 'import torch, lightning; print("torch", torch.__version__, "cuda", torch.version.cuda, "lightning", lightning.__version__)'
} | tee -a "$REPORTS/run_info.txt"

# Fail fast on the environment: the unit checks, piqa, and a convolution step on
# every GPU (catches a PyTorch build without kernels for this GPU, e.g. cu126
# wheels on Blackwell).
log "preflight"
python scripts/norm_ablation/test_latent_norm.py || exit 1
python -c 'import piqa' || { echo "piqa missing: pip install piqa==1.3.2"; exit 1; }
for G in "${LANES[@]}"; do
  [[ $G == cpu ]] && continue  # GPUS=cpu: pipeline test without a GPU
  CUDA_VISIBLE_DEVICES=$G python -c '
import torch
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(4, 3, 128, 128, device="cuda", requires_grad=True)
torch.nn.Conv2d(3, 8, 3).cuda()(x).square().mean().backward()
print("GPU", torch.cuda.get_device_name(0), "ok")' || exit 1
done

# run_variant VARIANT GPU: stage 1, recon eval, stage 2, predict eval
run_variant() {
  local V=$1 G=$2 S1=$OUT_ROOT/$1/stage_1 S2=$OUT_ROOT/$1/stage_2 latest resume
  export CUDA_VISIBLE_DEVICES=${G/cpu/}
  mkdir -p "$S1" "$S2"

  if [[ ! -f $S1/DONE ]]; then
    latest=$(latest_ckpt "$S1"); resume=${latest:+load=$latest}
    log "$V stage 1 on GPU $G ${resume:-from scratch}"
    eval "$(stage1_cmd "$V" "$S1" experiment.training.max_time=$STAGE1_TIME \
      $COMMON_OVERRIDES ${EXTRA_S1:-} $resume)" || { log "$V stage 1 FAILED"; return 1; }
    [[ -f $S1/checkpoints/last.ckpt ]] || { log "$V stage 1 left no last.ckpt"; return 1; }
    date > "$S1/DONE"
  fi

  if [[ ! -f $REPORTS/recon_$V.json ]]; then
    log "$V recon eval"
    python scripts/norm_ablation/eval_video.py --mode recon --data-dir "$DATA/val" \
      --episodes $EPISODES --videos 2 --out "$REPORTS/recon_$V.json" \
      --ckpt "$S1/checkpoints/last.ckpt" || { log "$V recon eval FAILED"; return 1; }
  fi

  if [[ ! -f $S2/DONE ]]; then
    latest=$(latest_ckpt "$S2"); resume=${latest:+load=$latest}
    log "$V stage 2 on GPU $G ${resume:-from scratch}"
    eval "$(stage2_cmd "$V" "$S2" "$S1/checkpoints/last.ckpt" \
      experiment.training.max_time=$STAGE2_TIME \
      $COMMON_OVERRIDES ${EXTRA_S2:-} $resume)" || { log "$V stage 2 FAILED"; return 1; }
    [[ -f $S2/checkpoints/last.ckpt ]] || { log "$V stage 2 left no last.ckpt"; return 1; }
    date > "$S2/DONE"
  fi

  if [[ ! -f $REPORTS/predict_$V.json ]]; then
    log "$V predict eval"
    python scripts/norm_ablation/eval_video.py --mode predict --data-dir "$DATA/val" \
      --episodes $EPISODES --videos 2 --out "$REPORTS/predict_$V.json" \
      --ckpt "$S2/checkpoints/last.ckpt" || { log "$V predict eval FAILED"; return 1; }
  fi
  log "$V done"
}

# One lane per GPU; lane i runs variants i, i + #GPUs, ... in turn. main.py
# relinks OUT_ROOT/latest-run at startup, so lanes start 30 s apart.
PIDS=()
for i in "${!LANES[@]}"; do
  (
    sleep $((i * 30))
    for ((j = i; j < ${#RUN_VARIANTS[@]}; j += ${#LANES[@]})); do
      V=${RUN_VARIANTS[j]}
      run_variant "$V" "${LANES[i]}" >> "$OUT_ROOT/logs/$V.log" 2>&1
      tail -n 1 "$OUT_ROOT/logs/$V.log"
    done
  ) &
  PIDS+=($!)
done
log "started ${#PIDS[@]} GPU lanes; per-variant logs in $OUT_ROOT/logs/"
FAILED=0
for P in "${PIDS[@]}"; do wait "$P" || FAILED=1; done

python scripts/norm_ablation/summarize.py --reports "$REPORTS" --variants "${RUN_VARIANTS[@]}" \
  | tee "$REPORTS/summary.md"
for V in "${RUN_VARIANTS[@]}"; do
  [[ -f $REPORTS/predict_$V.json ]] || { echo "incomplete: $V (see $OUT_ROOT/logs/$V.log)"; FAILED=1; }
done
exit $FAILED

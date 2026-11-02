# IWS latent-normalization ablation

Replaces IWS's per-position unit-sphere latent projection (`l2`) with `none`,
`layernorm`, `rmsnorm` or `groupnorm` (`algorithm.latent_norm`, see
`interactive_world_sim/algorithms/latent_dynamics/latent_norm.py`), retrains
stage 1 and stage 2 on T-Pushing, and scores each variant's image quality.
Only `latent_norm` differs between variants; the other settings are the IWS
authors' 128-px single-camera settings (`common.sh`).

## Run on a plain GPU machine (no Slurm)

`run_local.sh` uses one GPU per variant: stage 1 for 15 h of training time,
then stage 2 on that autoencoder for 8 h, then the evaluations. With five GPUs
all five variants run at once (about 23.5 h plus data-cache build and
evaluation). With fewer GPUs, each GPU runs its variants one after another.

1. **Environment.** Install the repository as in the top-level README
   (`mamba env create -f conda_env.yaml`, `conda activate iws`,
   `uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/`,
   `pip install -e .`), then `pip install piqa==1.3.2`. The cu126 wheels have
   no Blackwell kernels (B200, RTX 50xx): there, install
   `torch==2.7.1 torchvision==0.22.1` from the `cu128` index instead. The
   preflight catches this.
2. **Data.** The T-Pushing split is 600 train and 10 validation episodes:
   ```bash
   python scripts/download_data_hf.py --repo yixuan1999/interactive-world-sim-data \
       --local_dir data/full --repo_dir pusht
   ```
   This gives `data/full/pusht/{train,val}`, about 92 GB. The first run builds
   `cache.zarr.zip` in each split (about 6 GB, roughly an hour; the
   directory must be writable). The variants wait for it under a file lock,
   and that time is not counted in the 15 h.
3. **Smoke (about 15 min).** Runs every variant for 4 min per stage on the
   real data and batch, with checkpoints, stage 2 from stage 1, and a
   one-episode evaluation. Outputs go to `outputs/pusht_norm_smoke/` and
   `reports/norm_ablation_smoke/`, separate from the real runs:
   ```bash
   DATA=data/full/pusht SMOKE=1 bash scripts/norm_ablation/run_local.sh
   ```
4. **Real runs.** Inside `tmux` or with `nohup`:
   ```bash
   DATA=data/full/pusht bash scripts/norm_ablation/run_local.sh
   ```
   Options: `GPUS="0 1 2 3 4"`, `VARIANTS="layernorm rmsnorm"`,
   `STAGE1_TIME=00:15:00:00`, `STAGE2_TIME=00:08:00:00`,
   `WANDB_MODE=offline|online|disabled` (default offline; online logs to
   `charlesji/iws_norm_ablation`, or set `WANDB_ENTITY`).

On A100, FP32 attention inputs use the math backend because Flash Attention
requires FP16/BF16. Half-precision inputs retain the Flash backend; input
precision and training settings are unchanged. Check backend selection and
finite forward/backward passes with
`python -m scripts.norm_ablation.check_attention_precision` on an idle A100.

Parallel runs publish the shared `latest-run` convenience symlink using an
atomic replacement, so one variant cannot interrupt another during startup.
This does not change checkpoint selection or training settings.

Rerunning the same command resumes after a crash or reboot: finished stages
(`DONE` files) and evaluations are skipped, an unfinished stage resumes from
its `last.ckpt`, and Lightning restores the elapsed training time. To restart
a variant from scratch, delete `outputs/pusht_norm/<variant>/` and its
`reports/norm_ablation/*_<variant>*` files.

Disk: stage 1 keeps a 383 MB checkpoint every 3,000 steps (about 20 GB per
variant in 15 h on an H200-class GPU); stage 2 keeps one every 10,000 steps.

## Outputs

- `outputs/pusht_norm/<variant>/stage_{1,2}/`: Hydra config, checkpoints,
  offline W&B logs. Per-variant logs: `outputs/pusht_norm/logs/<variant>.log`.
- `reports/norm_ablation/recon_<variant>.json`: stage-1 reconstruction of
  every frame of the 10 validation episodes.
- `reports/norm_ablation/predict_<variant>.json`: stage-2 open-loop prediction
  from the first frame with the recorded actions to the episode end
  (frame t conditioned on action[t], the training pairing).
- `..._step<N>_ep{0,1}.mp4`: GT | model videos of the first two episodes.
- `reports/norm_ablation/summary.md`: PSNR, SSIM, LPIPS (piqa, VGG) as mean ±
  sample std over all frames, with the video paths
  (`python scripts/norm_ablation/summarize.py --reports reports/norm_ablation`).
- `reports/norm_ablation/run_info.txt`: commit, host, GPUs, library versions.

## On NYU Torch (Slurm)

`smoke.sbatch`, `stage1.sbatch` (array 0–4, 24 h of training time on H200),
`eval.sbatch` and `eval_alignment.sbatch`; see the comments in each. The full
notes are in the SlotWorld repository, `sa/docs/model/iws_norm_ablation.md`.

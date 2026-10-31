"""Evaluate IWS checkpoints on held-out episodes with the SlotWorld paper protocol.

Modes:
  recon    Encode and decode every frame (stage 1 quality).
  predict  Encode the first frame, roll the dynamics out with the recorded
           actions to the end of the episode, decode every predicted frame
           (stage 2; the paper's video-prediction protocol).

PSNR, SSIM and LPIPS (VGG) are computed per frame with piqa, as in SlotWorld's
`evaluate_image_metrics.py`: `frame_mean` and `frame_std` (sample std) are over
all evaluated frames of all episodes. Prediction excludes the given first frame.

Action alignment. In the IWS dataset `action[t]` is the end-effector XY
computed from the joint positions observed at frame t, and stage-2 training
conditions frame t on `action[t]`. `--action-offset 0` (default) rolls out
with that pairing. The IWS validation rollout (`_rollout_dynamics_like_validation`)
conditions frame t on `action[t + 1]`; `--action-offset 1` reproduces it.

Example:
  python scripts/norm_ablation/eval_video.py --mode recon \
      --data-dir data/full/pusht/val --out out.json \
      --ckpt outputs/.../checkpoints/epoch=3-step=48000.ckpt
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}), replace=True)
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (  # noqa: E402
    LatentWorldModel,
)
from interactive_world_sim.datasets.latent_dynamics.real_aloha_dataset import (  # noqa: E402
    load_replay_buffer,
)


def load_model(ckpt: Path, device: str) -> tuple:
    cfg = OmegaConf.load(ckpt.parent.parent / ".hydra" / "config.yaml")
    cfg.algorithm.load_ae = None
    model = LatentWorldModel.load_from_checkpoint(
        str(ckpt), cfg=cfg.algorithm, map_location=device, weights_only=False
    )
    model.eval()
    return model, cfg


def load_episodes(data_dir: str, cfg, limit: int) -> list:
    shape_meta = OmegaConf.to_container(cfg.dataset.shape_meta, resolve=True)
    rb = load_replay_buffer(data_dir, True, shape_meta, ctrl_mode=cfg.dataset.action_mode)
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    keys = list(cfg.dataset.obs_keys)
    episodes = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if i >= limit:
            break
        obs = [np.moveaxis(rb[k][s:e], -1, 1).astype(np.float32) / 255.0 for k in keys]
        episodes.append(
            dict(index=i, obs=np.concatenate(obs, axis=1), action=rb["action"][s:e].astype(np.float32))
        )
    return episodes


class Metrics:
    def __init__(self, device: str) -> None:
        from piqa import LPIPS, PSNR, SSIM

        self.psnr = PSNR(reduction="none").to(device)
        self.ssim = SSIM(reduction="none").to(device)
        self.lpips = LPIPS(network="vgg", reduction="none").to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, chunk: int = 64) -> dict:
        out = {"psnr": [], "ssim": [], "lpips": []}
        for i in range(0, pred.shape[0], chunk):
            p = pred[i : i + chunk].float().clamp(0, 1)
            t = target[i : i + chunk].float().clamp(0, 1)
            for name in out:
                out[name].append(getattr(self, name)(p, t).cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def encode(model, obs: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    keys = model.obs_keys
    zs = []
    for i in range(0, obs.shape[0], chunk):
        x = obs[i : i + chunk]
        x = torch.cat(
            [model.normalizer[k].normalize(x[:, 3 * j : 3 * j + 3]) for j, k in enumerate(keys)], 1
        )
        zs.append(model.encoder_forward(x))
    return torch.cat(zs)


@torch.no_grad()
def rollout(model, z_gt: torch.Tensor, action: torch.Tensor, offset: int) -> torch.Tensor:
    """Predict frames 1..T-1 from frame 0; frame t is conditioned on action[t + offset].

    dynamics_forward pairs its i-th input action with frame i (frame 0 is the
    given frame). With offset 1 and the final action repeated, this is exactly
    `_rollout_dynamics_like_validation` for a single episode.
    """
    T = z_gt.shape[0]
    act = action[offset:]
    if act.shape[0] < T:
        act = torch.cat([act, act[-1:].expand(T - act.shape[0], -1)])
    return model.dynamics_forward(z_gt[None, :1], act[None, :T])[0]


def views_to_batch(x: torch.Tensor, num_views: int) -> torch.Tensor:
    """(T, 3V, H, W) -> (T*V, 3, H, W) so each view is scored as an image."""
    return x.reshape(x.shape[0] * num_views, 3, *x.shape[-2:])


@torch.no_grad()
def evaluate_episode(
    model, ep: dict, mode: str, metrics: Metrics, seed: int, device: str, action_offset: int
):
    torch.manual_seed(seed)
    obs = torch.from_numpy(ep["obs"]).to(device)
    res, num_views = obs.shape[-1], len(model.obs_keys)
    z_gt = encode(model, obs)
    if mode == "recon":
        z, target = z_gt, obs
    else:
        action = model.normalizer["action"].normalize(torch.from_numpy(ep["action"]).to(device))
        z, target = rollout(model, z_gt, action.float(), action_offset), obs[1:]
    pred = render_img_cm(model, z, res, model.normalizer, num_views=num_views)
    values = metrics(views_to_batch(pred, num_views), views_to_batch(target, num_views))
    norms = torch.linalg.vector_norm(
        z.reshape(z.shape[0], num_views, -1, *z.shape[-2:]), dim=2
    )
    return values, pred.cpu(), dict(latent_norm_mean=float(norms.mean()), latent_norm_std=float(norms.std()))


def save_video(path: Path, gt: np.ndarray, pred: torch.Tensor, offset: int) -> None:
    import imageio.v2 as imageio

    gt = gt[offset:]
    frames = np.concatenate([gt, pred.numpy()], axis=-1)  # side by side, per view stacked
    frames = np.concatenate(np.split(frames, frames.shape[1] // 3, axis=1), axis=-2)
    frames = (np.moveaxis(frames, 1, -1).clip(0, 1) * 255).astype(np.uint8)
    imageio.mimsave(path, list(frames), fps=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", nargs="+", required=True, help="checkpoint paths or globs")
    parser.add_argument("--mode", choices=["recon", "predict"], required=True)
    parser.add_argument("--data-dir", required=True, help="held-out episode directory (val)")
    parser.add_argument("--out", required=True, help="JSON output; videos go next to it")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--videos", type=int, default=2, help="episodes saved as GT|prediction mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-offset", type=int, choices=(0, 1), default=0,
        help="predict: condition frame t on action[t + offset]; 0 matches training",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted({Path(p) for c in args.ckpt for p in (glob.glob(c) or [c])})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = Metrics(device)
    episodes, results = None, []
    for ckpt in ckpts:
        start = time.time()
        model, cfg = load_model(ckpt, device)
        if args.mode == "predict" and cfg.algorithm.training_stage != 2:
            raise SystemExit(f"{ckpt} is a stage {cfg.algorithm.training_stage} checkpoint")
        if episodes is None:
            episodes = load_episodes(args.data_dir, cfg, args.episodes)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        per_episode, frames = [], {k: [] for k in ("psnr", "ssim", "lpips")}
        for ep in episodes:
            values, pred, stats = evaluate_episode(
                model, ep, args.mode, metrics, args.seed + ep["index"], device,
                args.action_offset,
            )
            row = dict(episode=ep["index"], frames=int(len(values["psnr"])), **stats)
            row.update({k: float(np.mean(v)) for k, v in values.items()})
            if ep["index"] < args.videos:
                video = out_path.with_name(
                    f"{out_path.stem}_step{state['global_step']}_ep{ep['index']}.mp4"
                )
                save_video(video, ep["obs"], pred, offset=0 if args.mode == "recon" else 1)
                row["video"] = str(video)
            per_episode.append(row)
            for k, v in values.items():
                frames[k].append(v)
        frame_mean = {k: float(np.mean(np.concatenate(v))) for k, v in frames.items()}
        frame_std = {k: float(np.std(np.concatenate(v), ddof=1)) for k, v in frames.items()}
        result = dict(
            ckpt=str(ckpt),
            global_step=int(state["global_step"]),
            epoch=int(state["epoch"]),
            mode=args.mode,
            latent_norm=cfg.algorithm.get("latent_norm", "l2"),
            latent_norm_rescale=bool(cfg.algorithm.get("latent_norm_rescale", True)),
            action_offset=args.action_offset if args.mode == "predict" else None,
            frame_mean=frame_mean,
            frame_std=frame_std,
            frames=int(sum(len(v) for v in frames["psnr"])),
            episode_mean={k: float(np.mean([r[k] for r in per_episode])) for k in frame_mean},
            per_episode=per_episode,
            seconds=round(time.time() - start, 1),
        )
        if any(not math.isfinite(v) for v in frame_mean.values()):
            result["warning"] = "nonfinite metric"
        print(json.dumps({k: result[k] for k in ("ckpt", "global_step", "frame_mean", "seconds")}))
        results.append(result)
        del model
        torch.cuda.empty_cache()
    out_path.write_text(
        json.dumps(
            dict(
                protocol=(
                    "SlotWorld evaluate_image_metrics.py: one ground-truth frame, recorded "
                    "actions to episode end; piqa PSNR/SSIM/LPIPS-VGG per frame, global mean "
                    "and sample std over all evaluated frames"
                ),
                data_dir=os.path.abspath(args.data_dir),
                episodes=args.episodes,
                seed=args.seed,
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

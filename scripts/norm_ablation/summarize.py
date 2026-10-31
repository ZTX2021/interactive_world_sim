"""Tabulate the ablation's image metrics from eval_video.py JSON files.

Reads REPORTS/recon_<variant>.json (stage 1) and REPORTS/predict_<variant>.json
(stage 2) and prints a Markdown table of frame mean ± sample std, the evaluated
checkpoint step, and the video paths.

  python scripts/norm_ablation/summarize.py --reports reports/norm_ablation
"""

import argparse
import json
from pathlib import Path

VARIANTS = ["l2", "none", "layernorm", "rmsnorm", "groupnorm"]
METRICS = [("psnr", "PSNR", 2), ("ssim", "SSIM", 4), ("lpips", "LPIPS", 4)]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["results"][-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reports", required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    args = parser.parse_args()
    reports = Path(args.reports).resolve()

    print(f"# IWS latent-normalization ablation: image metrics\n\nReports: `{reports}`\n")
    header = "| Variant | Mode | Step | Frames | " + " | ".join(m[1] for m in METRICS) + " |"
    print(header)
    print("|" + "---|" * (4 + len(METRICS)))
    videos = []
    for mode, label in (("recon", "Reconstruction"), ("predict", "Prediction")):
        for v in args.variants:
            r = load(reports / f"{mode}_{v}.json")
            if r is None:
                print(f"| `{v}` | {label} | — | — |" + " — |" * len(METRICS))
                continue
            std = r.get("frame_std")  # absent in JSON from older eval_video.py
            cells = [
                f"{r['frame_mean'][k]:.{d}f}" + (f" ± {std[k]:.{d}f}" if std else "")
                for k, _, d in METRICS
            ]
            note = " ⚠ nonfinite" if "warning" in r else ""
            print(f"| `{v}` | {label} | {r['global_step']} | {r.get('frames', '—')} | " + " | ".join(cells) + f" |{note}")
            videos += [(v, label, e["video"]) for e in r["per_episode"] if "video" in e]
    print(
        "\nPrediction: encode the first frame of each held-out episode, roll out with the "
        "recorded actions to the episode end (frame t on action[t]), decode; the given frame "
        "is excluded. Reconstruction: encode and decode every frame. piqa PSNR/SSIM/LPIPS-VGG "
        "per frame; mean ± sample std over all frames.\n"
    )
    print("Videos (GT | model):\n")
    for v, label, path in videos:
        print(f"- `{v}` {label.lower()}: `{path}`")


if __name__ == "__main__":
    main()

"""Checks for LatentNorm. Run: python scripts/norm_ablation/test_latent_norm.py"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from interactive_world_sim.algorithms.latent_dynamics.latent_norm import (  # noqa: E402
    LATENT_NORM_MODES,
    LatentNorm,
)


def original_l2(z: torch.Tensor, num_views: int) -> torch.Tensor:
    """The projection in IWS `encoder_forward` before this change."""
    z = z.clone()
    c_per_v = z.shape[1] // num_views
    for i in range(num_views):
        chunk = z[:, i * c_per_v : (i + 1) * c_per_v].clone()
        z[:, i * c_per_v : (i + 1) * c_per_v] = chunk / (
            torch.norm(chunk, dim=1, keepdim=True) + 1e-8
        )
    return z


def well_conditioned(z: torch.Tensor, views: int, mode: str) -> torch.Tensor:
    """Channel mask of positions whose statistic is far above the norm's eps.

    LayerNorm and RMSNorm divide by sqrt(stat + eps), so an output has exactly
    unit statistic only where stat >> eps. Where the channels of a position are
    nearly equal (stat near eps = 1e-5), normalizing again rescales the vector,
    as with nn.LayerNorm; exact checks skip those positions.
    """
    x = z.reshape(z.shape[0], views, 4, *z.shape[-2:])
    if mode == "layernorm":
        stat = x.var(dim=2, unbiased=False, keepdim=True)
    elif mode == "rmsnorm":
        stat = x.pow(2).mean(dim=2, keepdim=True)
    else:
        return torch.ones_like(z, dtype=torch.bool)
    return (stat > 0.5).expand_as(x).reshape(z.shape)


# eps shifts a re-normalized latent by about 2 * eps (relative), 2e-5 here; a
# logic error, e.g. not undoing the affine transform, gives errors of order 0.1.
ATOL = 1e-3


def close(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, label: str) -> bool:
    err = (a[mask] - b[mask]).abs().max().item()
    print(f"{label}: max |project(y) - y| on well-conditioned positions = {err:.2e}")
    return err < ATOL


def main() -> None:
    torch.manual_seed(0)
    for views in (1, 2):
        z = torch.randn(6, 4 * views, 32, 32) * 3 + 0.5
        for mode in LATENT_NORM_MODES:
            ok = well_conditioned(z, views, mode)
            assert ok.float().mean() > 0.95, (mode, ok.float().mean())
            norm = LatentNorm(mode, views, 4, rescale=True)
            y = norm(z)
            assert y.shape == z.shape
            # projecting an encoder output must not change it
            assert close(norm.project(y), y, ok, f"{mode} views={views}"), mode
            # with random affine parameters as well
            if norm.learned:
                with torch.no_grad():
                    norm.weight.uniform_(0.5, 2.0)
                    if hasattr(norm, "bias"):
                        norm.bias.normal_()
                y2 = norm(z)
                assert close(norm.project(y2), y2, ok, f"{mode} views={views} affine"), mode
                # projection works on (T, B, C, H, W) too
                assert close(norm.project(y2[None])[0], y2, ok, f"{mode} views={views} 5-D"), mode
            params = {k for k, _ in norm.named_parameters()}
            expected = {
                "l2": set(), "none": set(), "rmsnorm": {"weight"},
                "layernorm": {"weight", "bias"}, "groupnorm": {"weight", "bias"},
            }[mode]
            assert params == expected, (mode, params)
        assert torch.equal(LatentNorm("l2", views, 4)(z), original_l2(z, views))
        # at init, rescaled per-position norms have norm 1 (well-conditioned positions)
        for mode in ("layernorm", "rmsnorm"):
            y = LatentNorm(mode, views, 4, rescale=True)(z)
            n = y.reshape(6, views, 4, 32, 32).norm(dim=2)
            ok = well_conditioned(z, views, mode).reshape(6, views, 4, 32, 32)[:, :, 0]
            assert torch.allclose(n[ok], torch.ones_like(n[ok]), atol=1e-3), (mode, n.mean())
        # groupnorm: mean squared norm per position is 1
        y = LatentNorm("groupnorm", views, 4, rescale=True)(z)
        assert abs(y.pow(2).sum(1).mean() / views - 1) < 1e-3
        y = LatentNorm("layernorm", views, 4, rescale=False)(z)
        n = y.reshape(6, views, 4, 32, 32).norm(dim=2)
        assert abs(n.mean() - 2) < 1e-3
    print("latent_norm checks passed")


if __name__ == "__main__":
    main()

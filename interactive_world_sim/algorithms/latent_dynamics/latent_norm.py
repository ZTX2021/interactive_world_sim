"""Latent normalization variants for the normalization ablation.

IWS projects every spatial position of each view's latent onto the unit sphere
over its channels (``l2``). The other modes replace that projection:

- ``none``: no normalization.
- ``layernorm``: per position, zero mean and unit variance over the channels.
- ``rmsnorm``: per position, unit RMS over the channels.
- ``groupnorm``: one group per view, statistics over channels and positions.

The learned modes have per-channel affine parameters, as in ``nn.LayerNorm``,
``nn.RMSNorm`` and ``nn.GroupNorm`` (RMSNorm has no bias). With ``rescale``
their normalized output is multiplied by ``1 / sqrt(C)``, so that at
initialization each position has norm 1 on average, as in ``l2``.

``project`` is the counterpart of the projection IWS applies to dynamics
output. It maps a latent back onto the set the encoder can produce: the affine
transform is undone, the latent is normalized and the transform is reapplied.
Like the ``l2`` projection it is idempotent.
"""

import math

import torch
import torch.nn as nn

LATENT_NORM_MODES = ("l2", "none", "layernorm", "rmsnorm", "groupnorm")


class LatentNorm(nn.Module):
    """Normalize latents of shape (..., num_views * C, H, W) view by view."""

    def __init__(
        self, mode: str, num_views: int, channels_per_view: int, rescale: bool = True
    ) -> None:
        super().__init__()
        if mode not in LATENT_NORM_MODES:
            raise ValueError(f"latent_norm must be one of {LATENT_NORM_MODES}, got {mode!r}")
        self.mode = mode
        self.num_views = num_views
        self.channels_per_view = channels_per_view
        self.learned = mode in ("layernorm", "rmsnorm", "groupnorm")
        self.scale = 1.0 / math.sqrt(channels_per_view) if self.learned and rescale else 1.0
        # nn.LayerNorm / nn.GroupNorm use 1e-5; nn.RMSNorm defaults to finfo eps
        self.eps = torch.finfo(torch.float32).eps if mode == "rmsnorm" else 1e-5
        channels = num_views * channels_per_view
        if self.learned:
            self.weight = nn.Parameter(torch.ones(channels))
        if mode in ("layernorm", "groupnorm"):
            self.bias = nn.Parameter(torch.zeros(channels))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-3]
        x = x.reshape(-1, self.num_views, self.channels_per_view, *x.shape[-2:])
        if self.mode == "l2":
            x = x / (torch.norm(x, dim=2, keepdim=True) + 1e-8)
        elif self.mode == "layernorm":
            mean = x.mean(dim=2, keepdim=True)
            var = x.var(dim=2, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
        elif self.mode == "rmsnorm":
            x = x * torch.rsqrt(x.pow(2).mean(dim=2, keepdim=True) + self.eps)
        elif self.mode == "groupnorm":
            mean = x.mean(dim=(2, 3, 4), keepdim=True)
            var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x.reshape(*lead, -1, *x.shape[-2:])

    def _channel(self, p: torch.Tensor) -> torch.Tensor:
        return p.view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize encoder output."""
        if self.mode == "none":
            return x
        x = self._normalize(x)
        if self.learned:
            x = x * (self.scale * self._channel(self.weight))
            if hasattr(self, "bias"):
                x = x + self._channel(self.bias)
        return x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project a predicted latent onto the encoder's output set."""
        if not self.learned:
            return self.forward(x)
        if hasattr(self, "bias"):
            x = x - self._channel(self.bias)
        w = self.weight
        w = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6) * torch.sign(w + 1e-12), w)
        return self.forward(x / self._channel(w))

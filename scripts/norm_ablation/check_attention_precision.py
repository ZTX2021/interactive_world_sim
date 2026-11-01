"""Check A100 attention backend selection and finite forward/backward passes."""
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend

from interactive_world_sim.algorithms.models import attention


def main():
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    properties = torch.cuda.get_device_properties(0)
    assert (properties.major, properties.minor) == (8, 0), "Run this check on A100"
    torch.manual_seed(42)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for causal in (False, True):
            model = attention.Attention(64, heads=2, dim_head=32).cuda().to(dtype)
            x = torch.randn(2, 10, 64, device="cuda", dtype=dtype, requires_grad=True)
            with patch.object(attention, "sdpa_kernel", wraps=attention.sdpa_kernel) as backend:
                y = model(x, is_causal=causal)
            expected = SDPBackend.MATH if dtype == torch.float32 else SDPBackend.FLASH_ATTENTION
            assert backend.call_args.kwargs["backends"] == [expected]
            assert y.dtype == dtype and torch.isfinite(y).all()
            y.float().square().mean().backward()
            assert torch.isfinite(x.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            if dtype == torch.float32:
                reference = attention.Attention(64, heads=2, dim_head=32).cuda()
                reference.load_state_dict(model.state_dict())
                reference.cuda_backends = [SDPBackend.MATH]
                torch.testing.assert_close(y, reference(x.detach(), is_causal=causal))
            print(f"PASS dtype={dtype} causal={causal} backend={expected}")


if __name__ == "__main__":
    main()

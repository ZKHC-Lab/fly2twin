"""Optional inference bridge to FlashAttention 1.0.9 built for the worker ABI.

The extension is deliberately separate from torch and the model checkpoint.
See docs/ATTENTION_BACKENDS.md for the pinned build and compatibility boundary.
"""
from functools import lru_cache
import importlib
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def extension():
    name = "drone3d_flash_attn_v109"
    if importlib.util.find_spec(name) is not None:
        return importlib.import_module(name)
    # Workers intentionally strip supervisor PYTHONPATH. Keep the optional ABI-
    # specific library beside this model, never in the system Python environment.
    path = Path(__file__).resolve().parents[3] / ".native/flash_attn_v109" / (name + ".so")
    if not path.is_file():
        raise RuntimeError(
            "flash_attn_v1 is not built for this worker; see docs/ATTENTION_BACKENDS.md"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attention(q, k, v):
    # LayerNorm/RoPE can yield FP32 inside BF16 autocast. Match SDPA's effective
    # dtype, but leave genuinely FP32 camera-head attention unchanged.
    dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        return F.scaled_dot_product_attention(q, k, v)
    if torch.is_grad_enabled():
        raise RuntimeError("flash_attn_v1 is inference-only")
    batch, heads, nq, dim = q.shape
    nk = k.shape[-2]
    qf = q.to(dtype).transpose(1, 2).reshape(batch * nq, heads, dim).contiguous()
    kf = k.to(dtype).transpose(1, 2).reshape(batch * nk, heads, dim).contiguous()
    vf = v.to(dtype).transpose(1, 2).reshape(batch * nk, heads, dim).contiguous()
    out = torch.empty_like(qf)
    cuq = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * nq
    cuk = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * nk
    extension().fwd(qf, kf, vf, out, cuq, cuk, nq, nk, 0., 1. / math.sqrt(dim),
                    False, False, False, 0, None)
    return out.reshape(batch, nq, heads, dim).transpose(1, 2)

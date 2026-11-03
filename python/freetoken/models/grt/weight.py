from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.models.loader import iter_weight_files


def quantize_symmetric(
    tensor: torch.Tensor, bits: int = 8, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel/per-row weight quantization (`bits` in `{8, 4}`)."""
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")
    qmax = 127.0 if bits == 8 else 7.0
    t_f32 = tensor.to(torch.float32)
    scale = t_f32.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(t_f32 / scale).clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        return q, scale
    if q.shape[dim] % 2 != 0:
        raise ValueError(
            f"INT4 packing requires even size along dim={dim}, got {q.shape[dim]}"
        )
    q_u8 = (q + 8).to(torch.uint8)
    idx0 = torch.arange(0, q_u8.shape[dim], 2, device=q.device)
    idx1 = idx0 + 1
    low = q_u8.index_select(dim, idx0)
    high = q_u8.index_select(dim, idx1)
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed, scale


def dequantize_symmetric(
    qweight: torch.Tensor, scale: torch.Tensor, bits: int = 8, dim: int = -1
) -> torch.Tensor:
    """Dequantize a weight tensor produced by `quantize_symmetric`."""
    if bits == 8:
        return qweight.to(torch.float32) * scale.to(torch.float32)
    if bits == 4:
        low = (qweight & 0x0F).to(torch.int8) - 8
        high = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
        d = dim if dim >= 0 else qweight.ndim + dim
        unpacked = torch.stack([low, high], dim=d + 1).flatten(d, d + 1)
        return unpacked.to(torch.float32) * scale.to(torch.float32)
    raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # the butterfly permutations are stored as private buffers (`_attn_perms`); the
                # model keeps them as state, which drops names with a leading underscore
                prefix, _, leaf = name.rpartition(".")
                yield f"{prefix}.{leaf.lstrip('_')}", f.get_tensor(name)


__all__ = ["iter_weights", "quantize_symmetric", "dequantize_symmetric"]

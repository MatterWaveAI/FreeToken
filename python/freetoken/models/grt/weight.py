from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.models.loader import iter_weight_files


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


__all__ = ["iter_weights"]

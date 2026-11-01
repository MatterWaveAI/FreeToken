"""A tiny GRT checkpoint laid out as transformers' GRTForCausalLM.save_pretrained writes it, the
butterfly permutation buffers included."""

from __future__ import annotations

import json
import math
import os

import torch
from safetensors.torch import save_file

TINY_CONFIG = {
    "architectures": ["GRTForCausalLM"],
    "model_type": "grt",
    "vocab_size": 97,
    "hidden_size": 128,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 256,
    "n_qkv_rotations": 3,
    "ffn_n_rotations": 4,
    "n_mixer_rotations": 2,
    "max_position_embeddings": 256,
    "residual_gate_mode": "bounded_depth",
    "bos_token_id": 2,
    "eos_token_id": 3,
}


def butterfly(n_stages: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    half = dim // 2
    n_levels = max(1, int(math.log2(half))) if half > 1 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        order = []
        for start in range(0, dim, 2 * stride):
            for i in range(stride):
                order += [start + i, start + i + stride]
        perms.append(order)
    perm = torch.tensor(perms, dtype=torch.long)
    return perm, torch.argsort(perm, dim=-1)


def tiny_state(cfg: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, mean=0.0, std=1.0):
        return torch.randn(shape, generator=g) * std + mean

    d, heads, vocab = cfg["hidden_size"], cfg["num_attention_heads"], cfg["vocab_size"]
    head_dim, width = d // heads, cfg["intermediate_size"]
    kq, kf, km = cfg["n_qkv_rotations"], cfg["ffn_n_rotations"], cfg["n_mixer_rotations"]
    group = next(s for s in (16, 8, 4, 2) if (d // 2) % s == 0)
    sd = {
        "model.embedding.phase_angles": rnd(vocab, d // 2),
        "model.embedding.log_magnitude": rnd(vocab, std=0.2),
        "model.pos_encoding.log_freqs": -torch.arange(d // 2) / (d // 2) * math.log(10000.0),
        "model.final_norm.scale": rnd(d, mean=1.0, std=0.1),
        "lm_head.output_scale": torch.tensor(2.0),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.blocks.{i}."
        for w in ("q", "k", "v", "o"):
            sd[p + f"attention.{w}_angles"] = rnd(kq, heads, head_dim // 2, std=0.5)
            sd[p + f"attention.{w}_scales"] = rnd(kq, heads, head_dim, mean=0.5, std=0.1)
            sd[p + f"attention.{w}_coupling"] = rnd(kq, heads, 4 * head_dim // 2, head_dim // 2, std=0.3)
        sd[p + "attention._attn_perms"], sd[p + "attention._attn_inv_perms"] = butterfly(kq, head_dim)
        sd[p + "mixer.angles"] = rnd(km, d // 2, std=0.5)
        sd[p + "mixer.scales"] = rnd(km, d, mean=0.5, std=0.1)
        sd[p + "mixer.coupling"] = rnd(km, d // 2 // group, 4 * group, group, std=0.3)
        sd[p + "mixer._perms_stack"], sd[p + "mixer._inv_perms_stack"] = butterfly(km, d)
        for w in ("expand", "contract"):
            sd[p + f"ffn.{w}_angles"] = rnd(kf, width // 2, std=0.5)
            sd[p + f"ffn.{w}_scales"] = rnd(kf, width, mean=0.5, std=0.1)
        sd[p + "ffn._ffn_perms"], sd[p + "ffn._ffn_inv_perms"] = butterfly(kf, width)
        for name in ("norm1", "norm_mixer", "norm_ffn"):
            sd[p + f"{name}.scale"] = rnd(d, mean=1.0, std=0.1)
        for gate in ("attn", "mixer", "ffn"):
            sd[p + f"res_gate_{gate}"] = rnd(std=0.5).reshape(())
    return {k: v.contiguous() for k, v in sd.items()}


def write_tiny_checkpoint(path: str, cfg: dict = TINY_CONFIG, seed: int = 0) -> dict[str, torch.Tensor]:
    os.makedirs(path, exist_ok=True)
    state = tiny_state(cfg, seed)
    save_file(state, os.path.join(path, "model.safetensors"))
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return state

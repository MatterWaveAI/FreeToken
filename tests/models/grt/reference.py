"""GRT forward transcribed from the transformers reference implementation (modeling_grt.py,
Apache-2.0, MatterWave AI) as plain torch over the checkpoint's own tensors: fp32, no cache,
every position recomputed, one projection and one permutation at a time."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _features(x0: torch.Tensor, x1: torch.Tensor, mixer: bool) -> torch.Tensor:
    x0n = torch.cat([x0[..., 1:], x0[..., :1]], dim=-1)
    x1n = torch.cat([x1[..., 1:], x1[..., :1]], dim=-1)
    bivector = x0 * x0n - x1 * x1n if mixer else x0 * x1n - x1 * x0n
    return torch.cat([x0 * x0 + x1 * x1, x0, x1, bivector], dim=-1)


def _rotate(x0, x1, cos, sin, shape):
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).view(shape)


def _coupled(x, angles, coupling, raw_scales, perms, inv_perms):
    scales = F.softplus(raw_scales)
    for k in range(angles.shape[0]):
        if perms is not None:
            x = x[..., perms[k]]
        pairs = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        f = _features(x0, x1, False)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        theta = angles[k] + torch.tanh(torch.einsum("nhj,hji->nhi", f, coupling[k]))
        x = _rotate(x0, x1, torch.cos(theta), torch.sin(theta), x.shape) * scales[k]
        if inv_perms is not None:
            x = x[..., inv_perms[k]]
    return x


def _mixer(x, sd, p):
    d = x.shape[-1]
    angles, raw_scales, coupling = sd[p + "angles"], sd[p + "scales"], sd[p + "coupling"]
    groups, group = coupling.shape[1], coupling.shape[-1]
    for k in range(angles.shape[0]):
        x = x[..., sd[p + "_perms_stack"][k]]
        pairs = x.view(-1, groups, group, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        f = _features(x0, x1, True)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-2)
        theta = angles[k].view(groups, group) + torch.tanh(torch.einsum("ngj,gji->ngi", f, coupling[k]))
        y = _rotate(x0, x1, torch.cos(theta), torch.sin(theta), (x.shape[0], groups, 2 * group))
        x = (y * F.softplus(raw_scales[k]).view(groups, 2 * group)).view(-1, d)
        x = x[..., sd[p + "_inv_perms_stack"][k]]
    return x


def _plain(x, angles, raw_scales, perms, inv_perms):
    scales = F.softplus(raw_scales)
    for k in range(angles.shape[0]):
        if perms is not None:
            x = x[:, perms[k]]
        pairs = x.view(x.shape[0], -1, 2)
        x = _rotate(pairs[..., 0], pairs[..., 1], torch.cos(angles[k]), torch.sin(angles[k]), x.shape) * scales[k]
        if inv_perms is not None:
            x = x[:, inv_perms[k]]
    return x


def _norm(x, scale):
    pairs = x.view(*x.shape[:-1], -1, 2)
    rms = torch.sqrt((pairs**2).sum(dim=-1).mean(dim=-1, keepdim=True) + 1e-8)
    return x / rms * scale


def _gate(sd, name, mode):
    raw = sd.get(name)
    if raw is None or mode == "none":
        return 1.0
    if mode == "bounded":
        return 1.0 + torch.tanh(raw) * 0.5
    if mode == "bounded_depth":
        return 0.55 + torch.tanh(raw) * 0.45
    return raw


@torch.no_grad()
def logits(sd: dict[str, torch.Tensor], cfg: dict, ids: list[int]) -> torch.Tensor:
    """``[len(ids), vocab]`` logits of one sequence, from the HF-named state dict and config.json."""
    d, heads = cfg["d_model"], cfg["n_heads"]
    head_dim = d // heads
    mode = cfg.get("residual_gate_mode", "bounded_depth")
    d_ff = cfg.get("d_ff") or cfg.get("ffn_mult", 4) * d
    device = sd["model.output_scale"].device

    ids_t = torch.tensor(ids, device=device)
    angles = sd["model.embedding.phase_angles"][ids_t]
    x = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(len(ids), d)
    x = x * torch.exp(sd["model.embedding.log_magnitude"][ids_t].clamp(-5.0, 5.0)).unsqueeze(-1)
    pos = torch.arange(len(ids), device=device, dtype=torch.float32).unsqueeze(-1)
    pos_angles = pos * torch.exp(sd["model.pos_encoding.log_freqs"])
    pairs = x.view(len(ids), d // 2, 2)
    x = _rotate(pairs[..., 0], pairs[..., 1], torch.cos(pos_angles), torch.sin(pos_angles), x.shape)

    for i in range(cfg["n_blocks"]):
        p = f"model.blocks.{i}."
        a = p + "attention."
        h = _norm(x, sd[p + "norm1.scale"]).view(len(ids), heads, head_dim)
        perms, inv = sd.get(a + "_attn_perms"), sd.get(a + "_attn_inv_perms")
        q, k, v = (
            _coupled(h, sd[a + f"{w}_angles"], sd[a + f"{w}_coupling"], sd[a + f"{w}_scales"], perms, inv)
            for w in ("q", "k", "v")
        )
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        o = _coupled(o.transpose(0, 1), sd[a + "o_angles"], sd[a + "o_coupling"], sd[a + "o_scales"], perms, inv)
        x = x + _gate(sd, p + "res_gate_attn", mode) * o.reshape(len(ids), d)
        if p + "mixer.angles" in sd:
            x = x + _gate(sd, p + "res_gate_mixer", mode) * _mixer(_norm(x, sd[p + "norm_mixer.scale"]), sd, p + "mixer.")
        f = F.pad(_norm(x, sd[p + "norm_ffn.scale"]), (0, d_ff - d))
        fp, fi = sd.get(p + "ffn._ffn_perms"), sd.get(p + "ffn._ffn_inv_perms")
        f = _plain(f, sd[p + "ffn.expand_angles"], sd[p + "ffn.expand_scales"], fp, fi)
        fpairs = f.view(len(ids), -1, 2)
        radius = torch.norm(fpairs, dim=-1, keepdim=True)
        f = (fpairs * (F.gelu(radius) / (radius + 1e-8))).view(len(ids), d_ff)
        f = _plain(f, sd[p + "ffn.contract_angles"], sd[p + "ffn.contract_scales"], fp, fi)
        x = x + _gate(sd, p + "res_gate_ffn", mode) * f[:, :d]

    x = _norm(x, sd["model.final_norm.scale"])
    table_angles = sd["model.embedding.phase_angles"]
    table = torch.stack([torch.cos(table_angles), torch.sin(table_angles)], dim=-1).reshape(-1, d)
    table = table * torch.exp(sd["model.embedding.log_magnitude"].clamp(-5.0, 5.0)).unsqueeze(-1)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    table = table / (table.norm(dim=-1, keepdim=True) + 1e-6)
    return (x @ table.T) * F.softplus(sd["model.output_scale"])

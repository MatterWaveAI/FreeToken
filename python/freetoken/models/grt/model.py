"""GRT (Geometric Rotational Transformer): every projection is a chain of butterfly-permuted Givens
rotations, the embedding stores phase angles, and the head is a cosine against the embedding table.

The model runs in fp32, as the checkpoints are trained and stored; only q/k/v cross into the
engine dtype for the paged KV cache and the attention backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, OPList
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

    from .config import GRTArgs

_F32 = torch.float32


def _param(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=_F32)


def _perms(n_stages: int, dim: int) -> torch.Tensor:
    return torch.empty((n_stages, dim), dtype=torch.long)


def _chain_gathers(perms: torch.Tensor | None, inv_perms: torch.Tensor | None) -> torch.Tensor | None:
    """One gather per stage boundary: stage k's inverse permutation composed with stage k+1's permutation."""
    if perms is None:
        return None
    steps = [perms[0]]
    steps += [inv_perms[k - 1][perms[k]] for k in range(1, perms.shape[0])]
    steps.append(inv_perms[-1])
    return torch.stack(steps)


def _clifford_features(x0: torch.Tensor, x1: torch.Tensor, mixer_bivector: bool) -> torch.Tensor:
    """Scalar energy, both vector parts and the bivector with the next pair: ``[S0, V1, V2, B12]``."""
    x0_next = torch.roll(x0, -1, dims=-1)
    x1_next = torch.roll(x1, -1, dims=-1)
    # the mixer was trained with the other bivector contraction
    bivector = x0 * x0_next - x1 * x1_next if mixer_bivector else x0 * x1_next - x1 * x0_next
    return torch.cat([x0 * x0 + x1 * x1, x0, x1, bivector], dim=-1)


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)


def _coupled_chain(
    x: torch.Tensor,
    groups: int,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    scales: torch.Tensor,
    gathers: torch.Tensor | None,
    feature_eps: float,
    mixer_bivector: bool = False,
) -> torch.Tensor:
    """``x`` is ``[T, H, D]``; gathers permute D, the pair math runs on ``groups`` groups of ``H*D/groups``.

    angles ``[K, groups, P]``, coupling ``[K, groups, 4P, P]``, scales ``[K, groups, 2P]`` (already positive).
    """
    t, h, d = x.shape
    for k in range(angles.shape[0]):
        if gathers is not None:
            x = x[..., gathers[k]]
        g = x.reshape(t, groups, -1)
        x0, x1 = g[..., 0::2], g[..., 1::2]
        features = _clifford_features(x0, x1, mixer_bivector)
        features = features / (features.norm(dim=-1, keepdim=True) + feature_eps)
        theta = angles[k] + torch.tanh(torch.einsum("tgj,gji->tgi", features, coupling[k]))
        x = (_rotate_pairs(g, torch.cos(theta), torch.sin(theta)) * scales[k]).view(t, h, d)
    if gathers is not None:
        x = x[..., gathers[-1]]
    return x


def _plain_chain(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, scales: torch.Tensor, gathers: torch.Tensor | None) -> torch.Tensor:
    for k in range(cos.shape[0]):
        if gathers is not None:
            x = x[..., gathers[k]]
        x = _rotate_pairs(x, cos[k], sin[k]) * scales[k]
    if gathers is not None:
        x = x[..., gathers[-1]]
    return x


class GRTPearlNorm(BaseOP):
    """RMS over pair radii: rescales every pair's magnitude and leaves its phase untouched."""

    def __init__(self, size: int):
        self.scale = _param(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius_sq = x[..., 0::2] ** 2 + x[..., 1::2] ** 2
        return x * torch.rsqrt(radius_sq.mean(dim=-1, keepdim=True) + 1e-8) * self.scale


class GRTEmbedding(BaseOP):
    def __init__(self, vocab_size: int, hidden_size: int):
        self.phase_angles = _param(vocab_size, hidden_size // 2)
        self.log_magnitude = _param(vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        angles = F.embedding(ids, self.phase_angles)
        pairs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).flatten(-2)
        magnitude = F.embedding(ids, self.log_magnitude.unsqueeze(-1)).clamp(-5.0, 5.0).exp()
        return pairs * magnitude


class GRTPositionalRotation(BaseOP):
    """Rotates each pair of the whole stream by ``position * exp(log_freqs)``, once, at the input."""

    def __init__(self, hidden_size: int):
        self.log_freqs = _param(hidden_size // 2)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # fp32 angles: a bf16 position is not exact above 256
        angles = positions.to(_F32).unsqueeze(-1) * self.log_freqs.exp()
        return _rotate_pairs(x, torch.cos(angles), torch.sin(angles))


class GRTAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: GRTArgs = config.grt_args
        self.layer_id = layer_id
        self._heads = config.num_qo_heads
        self._head_dim = config.head_dim
        self._kv_dtype = torch.get_default_dtype()
        k, h, p = args.n_qkv_rotations, self._heads, self._head_dim // 2
        for name in ("q", "k", "v", "o"):
            setattr(self, f"{name}_angles", _param(k, h, p))
            setattr(self, f"{name}_scales", _param(k, h, 2 * p))
            setattr(self, f"{name}_coupling", _param(k, h, 4 * p, p))
        # a single-stage chain has no butterfly and the checkpoint stores no permutations for it
        self.attn_perms = _perms(k, self._head_dim) if k > 1 else None
        self.attn_inv_perms = _perms(k, self._head_dim) if k > 1 else None
        self._qkv: tuple[torch.Tensor, ...] | None = None
        self._o: tuple[torch.Tensor, ...] | None = None
        self._gathers: torch.Tensor | None = None

    def prepare_for_runtime(self) -> None:
        # q, k and v read the same input and share the permutations, so they run as one chain of 3*H heads
        self._qkv = (
            torch.cat([self.q_angles, self.k_angles, self.v_angles], dim=1),
            torch.cat([self.q_coupling, self.k_coupling, self.v_coupling], dim=1),
            F.softplus(torch.cat([self.q_scales, self.k_scales, self.v_scales], dim=1)),
        )
        self._o = (self.o_angles, self.o_coupling, F.softplus(self.o_scales))
        self._gathers = _chain_gathers(self.attn_perms, self.attn_inv_perms)

    def _chain(self, x: torch.Tensor, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        angles, coupling, scales = params
        return _coupled_chain(x, x.shape[1], angles, coupling, scales, self._gathers, 1e-6)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._qkv is not None, "prepare_for_runtime() was not run"
        ctx = get_global_ctx()
        t, h, d = x.shape[0], self._heads, self._head_dim
        qkv = self._chain(x.view(t, h, d).repeat(1, 3, 1), self._qkv)
        q, k, v = (part.to(self._kv_dtype).contiguous() for part in qkv.split(h, dim=1))
        o = ctx.attn_backend.forward(q, k.view(t, h * d), v.view(t, h * d), self.layer_id, ctx.batch)
        return self._chain(o.view(t, h, d).to(_F32), self._o).view(t, h * d)


class GRTMixer(BaseOP):
    """Mixes across heads at full width; the coupling is block-diagonal over groups of pairs."""

    def __init__(self, config: ModelConfig):
        args: GRTArgs = config.grt_args
        d = config.hidden_size
        n_pairs = d // 2
        k = args.n_mixer_rotations
        self._group = next((g for g in (16, 8, 4, 2) if n_pairs % g == 0), 1)
        self._n_groups = n_pairs // self._group
        self.angles = _param(k, n_pairs)
        self.scales = _param(k, d)
        self.coupling = _param(k, self._n_groups, 4 * self._group, self._group)
        self.perms_stack = _perms(k, d)
        self.inv_perms_stack = _perms(k, d)
        self._params: tuple[torch.Tensor, ...] | None = None
        self._gathers: torch.Tensor | None = None

    def prepare_for_runtime(self) -> None:
        k = self.angles.shape[0]
        self._params = (
            self.angles.view(k, self._n_groups, self._group),
            self.coupling,
            F.softplus(self.scales).view(k, self._n_groups, 2 * self._group),
        )
        self._gathers = _chain_gathers(self.perms_stack, self.inv_perms_stack)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles, coupling, scales = self._params
        t, d = x.shape
        y = _coupled_chain(x.view(t, 1, d), self._n_groups, angles, coupling, scales, self._gathers, 1e-2, mixer_bivector=True)
        return y.view(t, d)


class GRTFeedForward(BaseOP):
    """Zero-pad to intermediate_size, rotate, radial GELU on pair magnitudes, rotate, truncate back to
    hidden_size."""

    def __init__(self, config: ModelConfig):
        args: GRTArgs = config.grt_args
        k, width = args.ffn_n_rotations, config.intermediate_size
        self._hidden_size = config.hidden_size
        self._width = width
        self.expand_angles = _param(k, width // 2)
        self.expand_scales = _param(k, width)
        self.contract_angles = _param(k, width // 2)
        self.contract_scales = _param(k, width)
        self.ffn_perms = _perms(k, width) if k > 1 else None
        self.ffn_inv_perms = _perms(k, width) if k > 1 else None
        self._expand: tuple[torch.Tensor, ...] | None = None
        self._contract: tuple[torch.Tensor, ...] | None = None
        self._gathers: torch.Tensor | None = None

    def prepare_for_runtime(self) -> None:
        self._expand = (self.expand_angles.cos(), self.expand_angles.sin(), F.softplus(self.expand_scales))
        self._contract = (self.contract_angles.cos(), self.contract_angles.sin(), F.softplus(self.contract_scales))
        self._gathers = _chain_gathers(self.ffn_perms, self.ffn_inv_perms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _plain_chain(F.pad(x, (0, self._width - self._hidden_size)), *self._expand, self._gathers)
        pairs = x.view(*x.shape[:-1], -1, 2)
        radius = pairs.norm(dim=-1, keepdim=True)
        x = (pairs * (F.gelu(radius) / (radius + 1e-8))).flatten(-2)
        x = _plain_chain(x, *self._contract, self._gathers)
        return x[..., : self._hidden_size]


class GRTDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: GRTArgs = config.grt_args
        d = config.hidden_size
        self._gate_mode = args.gate_mode
        gated = args.gate_mode != "none"
        # the mixer gate exists whenever gates do, mixer or not, as in the reference checkpoints
        self.res_gate_attn = _param() if gated else None
        self.res_gate_mixer = _param() if gated else None
        self.res_gate_ffn = _param() if gated else None
        self.norm1 = GRTPearlNorm(d)
        self.attention = GRTAttention(config, layer_id)
        has_mixer = args.n_mixer_rotations > 0
        self.norm_mixer = GRTPearlNorm(d) if has_mixer else None
        self.mixer = GRTMixer(config) if has_mixer else None
        self.norm_ffn = GRTPearlNorm(d)
        self.ffn = GRTFeedForward(config)
        self.post_norm_attn = GRTPearlNorm(d) if args.peri_ln else None
        self.post_norm_ffn = GRTPearlNorm(d) if args.peri_ln else None
        self.res_norm_block = GRTPearlNorm(d) if args.residual_pearlnorm else None
        self._gates = (1.0, 1.0, 1.0)
        self._layer_id = layer_id

    def _gate(self, raw: torch.Tensor | None) -> float:
        if raw is None:
            return 1.0
        value = raw.item()
        if self._gate_mode == "bounded":
            return 1.0 + 0.5 * math.tanh(value)
        if self._gate_mode == "bounded_depth":
            return 0.55 + 0.45 * math.tanh(value)
        return value

    def prepare_for_runtime(self) -> None:
        self._gates = tuple(self._gate(g) for g in (self.res_gate_attn, self.res_gate_mixer, self.res_gate_ffn))
        self.attention.prepare_for_runtime()
        if self.mixer is not None:
            self.mixer.prepare_for_runtime()
        self.ffn.prepare_for_runtime()

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_attn, gate_mixer, gate_ffn = self._gates
        y = self.attention.forward(self.norm1.forward(x))
        if self.post_norm_attn is not None:
            y = self.post_norm_attn.forward(y)
        x = x + gate_attn * y
        if self.mixer is not None:
            x = x + gate_mixer * self.mixer.forward(self.norm_mixer.forward(x))
        y = self.ffn.forward(self.norm_ffn.forward(x))
        if self.post_norm_ffn is not None:
            y = self.post_norm_ffn.forward(y)
        x = x + gate_ffn * y
        if self.res_norm_block is not None:
            x = self.res_norm_block.forward(x)
        return x


class GRTModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embedding = GRTEmbedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = GRTPositionalRotation(config.hidden_size)
        self.blocks = OPList([GRTDecoderLayer(config, i) for i in range(config.num_layers)])
        self.final_norm = GRTPearlNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding.forward(self.embedding.forward(input_ids), positions)
        for block in self.blocks.op_list:
            x = block.forward(x)
        return self.final_norm.forward(x)


class GRTCosineHead(BaseOP):
    """Cosine against every token's embedding times one learned temperature. The table is the tied
    input embedding; table and temperature are constants, so ``fold`` combines them once."""

    def __init__(self):
        self.output_scale = _param()
        self._head: torch.Tensor | None = None

    def fold(self, table: torch.Tensor) -> None:
        table = table / (table.norm(dim=-1, keepdim=True) + 1e-6)
        self._head = table * F.softplus(self.output_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._head is not None, "prepare_for_runtime() was not run"
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x @ self._head.T


class GRTForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        if get_tp_info().size != 1:
            raise ValueError("GRT has no tensor-parallel layout; serve it with --tp-size 1")
        self.model = GRTModel(config)
        self.lm_head = GRTCosineHead()
        self._vocab_size = config.vocab_size
        super().__init__()

    def prepare_for_runtime(self) -> None:
        for block in self.model.blocks.op_list:
            block.prepare_for_runtime()
        ids = torch.arange(self._vocab_size, device=self.lm_head.output_scale.device)
        self.lm_head.fold(self.model.embedding.forward(ids))

    @nvtx_annotate("LMHead")
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)]
        return self.lm_head.forward(x)

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self._logits(self.model.forward(batch.input_ids, batch.positions))


__all__ = ["GRTForCausalLM"]

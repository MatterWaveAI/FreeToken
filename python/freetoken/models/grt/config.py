from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig

_GATE_MODES = ("free", "bounded", "bounded_depth", "none")


@dataclass(frozen=True)
class GRTArgs:
    n_qkv_rotations: int
    ffn_n_rotations: int
    n_mixer_rotations: int
    gate_mode: str
    peri_ln: bool
    residual_pearlnorm: bool
    quantize_bits: int | None = None


def _resolve_quantize_bits(hf_config: Any) -> int | None:
    bits = getattr(hf_config, "quantize_bits", None)
    if bits in (4, 8):
        return int(bits)
    qcfg = getattr(hf_config, "quantization_config", None)
    if isinstance(qcfg, dict):
        b = qcfg.get("bits") or qcfg.get("num_bits")
        if b in (4, 8):
            return int(b)
        if qcfg.get("load_in_4bit"):
            return 4
        if qcfg.get("load_in_8bit"):
            return 8
    return None


def parse_config(hf_config: Any) -> ModelConfig:
    hidden_size = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    if hidden_size % 2 or hidden_size % n_heads or (hidden_size // n_heads) % 2:
        raise ValueError(
            f"GRT needs an even hidden_size and head_dim; got hidden_size={hidden_size}, num_attention_heads={n_heads}"
        )

    gate_mode = getattr(hf_config, "residual_gate_mode", "bounded_depth")
    if not getattr(hf_config, "learnable_residual_gates", True):
        gate_mode = "none"
    if gate_mode not in _GATE_MODES:
        raise ValueError(f"Unknown GRT residual_gate_mode {gate_mode!r}")
    residual_norm = getattr(hf_config, "residual_norm_mode", "none")
    if residual_norm not in ("none", "pearlnorm"):
        raise ValueError(f"Unknown GRT residual_norm_mode {residual_norm!r}")

    head_dim = hidden_size // n_heads
    max_position = getattr(hf_config, "max_position_embeddings", 2048)
    args = GRTArgs(
        n_qkv_rotations=getattr(hf_config, "n_qkv_rotations", 11),
        ffn_n_rotations=getattr(hf_config, "ffn_n_rotations", 21),
        n_mixer_rotations=getattr(hf_config, "n_mixer_rotations", 1),
        gate_mode=gate_mode,
        peri_ln=bool(getattr(hf_config, "peri_ln", False)),
        residual_pearlnorm=residual_norm == "pearlnorm",
        quantize_bits=_resolve_quantize_bits(hf_config),
    )
    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=n_heads,
        num_kv_heads=n_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=getattr(hf_config, "intermediate_size", None) or 4 * hidden_size,
        rms_norm_eps=1e-8,
        # position is one rotation of the whole stream at the input, not a per-head rope;
        # base 0 keeps the attention layers from building a rope table
        rotary_config=RotaryConfig(
            head_dim=head_dim, rotary_dim=head_dim, max_position=max_position, base=0.0, scaling=None
        ),
        hidden_act="gelu",
        # the cosine head reads the input embedding, so --embed-weights host must refuse it
        tie_word_embeddings=True,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type=getattr(hf_config, "model_type", "grt"),
        architectures=getattr(hf_config, "architectures", ["GRTForCausalLM"]),
        grt_args=args,
    )


__all__ = ["GRTArgs", "parse_config"]

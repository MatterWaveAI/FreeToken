from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig

_GATE_MODES = ("free", "bounded", "bounded_depth", "none")


@dataclass(frozen=True)
class GRTArgs:
    d_ff: int
    n_qkv_rotations: int
    ffn_n_rotations: int
    n_mixer_rotations: int
    gate_mode: str
    peri_ln: bool
    residual_pearlnorm: bool


def parse_config(hf_config: Any) -> ModelConfig:
    d_model = hf_config.d_model
    n_heads = hf_config.n_heads
    if d_model % 2 or d_model % n_heads or (d_model // n_heads) % 2:
        raise ValueError(f"GRT needs an even d_model and head_dim; got d_model={d_model}, n_heads={n_heads}")

    # every released checkpoint, and the original implementation, uses the 4 Clifford features
    if getattr(hf_config, "coupling_features", "clifford") != "clifford":
        raise ValueError("GRT supports only coupling_features='clifford'")
    # Later passes re-run the stack over positions whose earlier-pass keys a paged cache never holds.
    if getattr(hf_config, "ouroboros_passes", 1) != 1:
        raise ValueError("GRT with ouroboros_passes != 1 cannot be served from a KV cache")
    gate_mode = getattr(hf_config, "residual_gate_mode", "bounded_depth")
    if not getattr(hf_config, "learnable_residual_gates", True):
        gate_mode = "none"
    if gate_mode not in _GATE_MODES:
        raise ValueError(f"Unknown GRT residual_gate_mode {gate_mode!r}")
    residual_norm = getattr(hf_config, "residual_norm_mode", "none")
    if residual_norm not in ("none", "pearlnorm"):
        raise ValueError(f"Unknown GRT residual_norm_mode {residual_norm!r}")

    d_ff = getattr(hf_config, "d_ff", None) or getattr(hf_config, "ffn_mult", 4) * d_model
    head_dim = d_model // n_heads
    max_seq_len = getattr(hf_config, "max_seq_len", 1024)
    args = GRTArgs(
        d_ff=d_ff,
        n_qkv_rotations=getattr(hf_config, "n_qkv_rotations", 11),
        ffn_n_rotations=getattr(hf_config, "ffn_n_rotations", 21),
        n_mixer_rotations=getattr(hf_config, "n_mixer_rotations", 1),
        gate_mode=gate_mode,
        peri_ln=bool(getattr(hf_config, "peri_ln", False)),
        residual_pearlnorm=residual_norm == "pearlnorm",
    )
    return ModelConfig(
        num_layers=hf_config.n_blocks,
        num_qo_heads=n_heads,
        num_kv_heads=n_heads,
        head_dim=head_dim,
        hidden_size=d_model,
        vocab_size=hf_config.vocab_size,
        intermediate_size=d_ff,
        rms_norm_eps=1e-8,
        # position is one rotation of the whole stream at the input, not a per-head rope;
        # base 0 keeps the attention layers from building a rope table
        rotary_config=RotaryConfig(
            head_dim=head_dim, rotary_dim=head_dim, max_position=max_seq_len, base=0.0, scaling=None
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

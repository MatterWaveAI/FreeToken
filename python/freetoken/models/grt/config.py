from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig

_GATE_MODES = ("free", "bounded", "bounded_depth", "none")


@dataclass(frozen=True)
class GRTVisionArgs:
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    spatial_merge_size: int = 2
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 6
    num_heads: int = 4
    patch_n_rotations: int = 11
    merger_n_rotations: int = 21
    ffn_n_rotations: int = 21
    n_qkv_rotations: int = 11
    n_mixer_rotations: int = 1
    gate_mode: str = "bounded_depth"


@dataclass(frozen=True)
class GRTArgs:
    n_qkv_rotations: int
    ffn_n_rotations: int
    n_mixer_rotations: int
    gate_mode: str
    peri_ln: bool
    residual_pearlnorm: bool
    quantize_bits: int | None = None
    vision_args: GRTVisionArgs | None = None
    image_token_id: int = 151655


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
    text_cfg = getattr(hf_config, "text_config", None) or hf_config
    vision_cfg = getattr(hf_config, "vision_config", None)
    hidden_size = text_cfg.hidden_size
    n_heads = text_cfg.num_attention_heads
    if hidden_size % 2 or hidden_size % n_heads or (hidden_size // n_heads) % 2:
        raise ValueError(
            f"GRT needs an even hidden_size and head_dim; got hidden_size={hidden_size}, num_attention_heads={n_heads}"
        )

    gate_mode = getattr(text_cfg, "residual_gate_mode", "bounded_depth")
    if not getattr(text_cfg, "learnable_residual_gates", True):
        gate_mode = "none"
    if gate_mode not in _GATE_MODES:
        raise ValueError(f"Unknown GRT residual_gate_mode {gate_mode!r}")
    residual_norm = getattr(text_cfg, "residual_norm_mode", "none")
    if residual_norm not in ("none", "pearlnorm"):
        raise ValueError(f"Unknown GRT residual_norm_mode {residual_norm!r}")

    vision_args = None
    if vision_cfg is not None:
        v_hidden = getattr(vision_cfg, "hidden_size", 256)
        v_gate = getattr(vision_cfg, "residual_gate_mode", "bounded_depth")
        if not getattr(vision_cfg, "learnable_residual_gates", True):
            v_gate = "none"
        vision_args = GRTVisionArgs(
            image_size=getattr(vision_cfg, "image_size", 224),
            patch_size=getattr(vision_cfg, "patch_size", 14),
            num_channels=getattr(vision_cfg, "num_channels", 3),
            spatial_merge_size=getattr(vision_cfg, "spatial_merge_size", 2),
            hidden_size=v_hidden,
            intermediate_size=getattr(vision_cfg, "intermediate_size", None) or 4 * v_hidden,
            num_layers=getattr(vision_cfg, "num_hidden_layers", 6),
            num_heads=getattr(vision_cfg, "num_attention_heads", 4),
            patch_n_rotations=getattr(vision_cfg, "patch_n_rotations", 11),
            merger_n_rotations=getattr(vision_cfg, "merger_n_rotations", 21),
            ffn_n_rotations=getattr(vision_cfg, "ffn_n_rotations", 21),
            n_qkv_rotations=getattr(vision_cfg, "n_qkv_rotations", 11),
            n_mixer_rotations=getattr(vision_cfg, "n_mixer_rotations", 1),
            gate_mode=v_gate,
        )

    head_dim = hidden_size // n_heads
    max_position = getattr(text_cfg, "max_position_embeddings", 2048)
    args = GRTArgs(
        n_qkv_rotations=getattr(text_cfg, "n_qkv_rotations", 11),
        ffn_n_rotations=getattr(text_cfg, "ffn_n_rotations", 21),
        n_mixer_rotations=getattr(text_cfg, "n_mixer_rotations", 1),
        gate_mode=gate_mode,
        peri_ln=bool(getattr(text_cfg, "peri_ln", False)),
        residual_pearlnorm=residual_norm == "pearlnorm",
        quantize_bits=_resolve_quantize_bits(hf_config) or _resolve_quantize_bits(text_cfg),
        vision_args=vision_args,
        image_token_id=getattr(hf_config, "image_token_id", 151655),
    )
    return ModelConfig(
        num_layers=text_cfg.num_hidden_layers,
        num_qo_heads=n_heads,
        num_kv_heads=n_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=text_cfg.vocab_size,
        intermediate_size=getattr(text_cfg, "intermediate_size", None) or 4 * hidden_size,
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


__all__ = ["GRTArgs", "GRTVisionArgs", "parse_config"]

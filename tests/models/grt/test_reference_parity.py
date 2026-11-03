"""GRT through the paged KV cache and the triton backend against the cache-free reference: ragged
prefill batches, chunked prefill and greedy decode, on the fp32 KV pool GRT is served with. Covers
the composed permutations, q/k/v run as one chain, and the positions of cached rows."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest
import torch

import freetoken.core as core
from freetoken.core import Batch, Context, Req
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.engine import _materialize_loaded_weight_state_dict
from freetoken.layers.quantization import NoQuantConfig
from freetoken.models import create_model
from freetoken.models.grt import iter_weights, parse_config, quantize_model_weights
from freetoken.utils.hf import cached_load_hf_config
from freetoken.utils.torch_utils import torch_dtype

from . import reference
from .common import write_tiny_checkpoint

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

WIDTH = 512
ATOL = RTOL = 1e-3


class TinyGRT:
    def __init__(self, checkpoint: str, monkeypatch, max_reqs: int = 2):
        from freetoken.attention.triton import TritonAttentionBackend
        from freetoken.kvcache.mha_pool import MHAKVCache

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)
        device = torch.device("cuda")
        config = replace(parse_config(cached_load_hf_config(checkpoint)), quant=NoQuantConfig())
        self.ctx = Context(page_size=1)
        self.ctx.page_table = torch.arange(max_reqs * WIDTH, dtype=torch.int32, device=device).view(max_reqs, WIDTH)
        self.ctx.kv_cache = MHAKVCache(
            config.num_kv_heads, config.num_layers, config.head_dim,
            num_pages=max_reqs * WIDTH, page_size=1, dtype=torch.float32, device=device,
        )
        monkeypatch.setattr(core, "_GLOBAL_CTX", self.ctx)
        self.ctx.attn_backend = TritonAttentionBackend(config)
        with torch.device("meta"), torch_dtype(torch.float32):
            self.model = create_model(config)
        weights = iter_weights(checkpoint, device, include_moe_experts=False, include_non_moe=True)
        self.model.load_state_dict(_materialize_loaded_weight_state_dict(self.model.state_dict(), weights, device=device))
        self.model.prepare_for_runtime()

    @torch.no_grad()
    def forward(self, reqs: list[Req], phase: str) -> torch.Tensor:
        batch = Batch(reqs=reqs, phase=phase)
        batch.padded_reqs = reqs
        batch.input_ids = torch.cat([r.input_ids[r.cached_len : r.device_len] for r in reqs]).cuda()
        batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in reqs]).cuda()
        batch.out_loc = torch.cat([self.ctx.page_table[r.table_idx, r.cached_len : r.device_len] for r in reqs])
        self.ctx.attn_backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch):
            return self.model.forward().float()


def _req(table_idx: int, ids: list[int], cached_len: int = 0, output_len: int = 16) -> Req:
    return Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx, cached_len=cached_len,
               output_len=output_len, uid=table_idx, sampling_params=None, cache_handle=None)


def _prompts(vocab: int, lengths: tuple[int, ...], seed: int = 1) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("grt_tiny"))
    state = write_tiny_checkpoint(path)
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return path, {k: v.to(dev) for k, v in state.items()}, cfg


@CUDA_ONLY
def test_ragged_prefill_matches_reference(tiny, monkeypatch):
    path, state, cfg = tiny
    engine = TinyGRT(path, monkeypatch)
    prompts = _prompts(cfg["vocab_size"], (7, 150))
    got = engine.forward([_req(i, p) for i, p in enumerate(prompts)], "prefill")
    want = torch.stack([reference.logits(state, cfg, p)[-1] for p in prompts])
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


@CUDA_ONLY
def test_chunked_prefill_matches_reference(tiny, monkeypatch):
    """The second chunk's rows sit at positions 130.. and attend to the first chunk through the pool."""
    path, state, cfg = tiny
    engine = TinyGRT(path, monkeypatch)
    (prompt,) = _prompts(cfg["vocab_size"], (170,), seed=2)
    engine.forward([_req(0, prompt[:130])], "prefill")
    got = engine.forward([_req(0, prompt, cached_len=130)], "prefill")
    torch.testing.assert_close(got[0], reference.logits(state, cfg, prompt)[-1], atol=ATOL, rtol=RTOL)


@CUDA_ONLY
def test_batched_greedy_decode_matches_reference(tiny, monkeypatch):
    path, state, cfg = tiny
    engine = TinyGRT(path, monkeypatch)
    prompts = _prompts(cfg["vocab_size"], (5, 33), seed=3)
    reqs = [_req(i, p) for i, p in enumerate(prompts)]
    logits = engine.forward(reqs, "prefill")
    for step in range(8):
        want = torch.stack([reference.logits(state, cfg, r.input_ids.tolist())[-1] for r in reqs])
        torch.testing.assert_close(logits, want, atol=ATOL, rtol=RTOL)
        if step == 7:
            break
        for req, token in zip(reqs, want.argmax(dim=-1).tolist()):
            req.complete_one()
            req.append_host(torch.tensor([token], dtype=torch.int32))
        logits = engine.forward(reqs, "decode")


class _SDPATestBackend:
    def forward(self, q, k, v, layer_id, batch):
        t, h, d = q.shape
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = k.view(t, h, d).transpose(0, 1).unsqueeze(0)
        vh = v.view(t, h, d).transpose(0, 1).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        return out.squeeze(0).transpose(0, 1).contiguous()


@pytest.mark.parametrize("bits,atol", [(8, 0.15), (4, 0.65)])
def test_weight_quantization_matches_fp32_reference(tiny, monkeypatch, bits: int, atol: float):
    from contextlib import nullcontext

    path, state, cfg = tiny
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = replace(parse_config(cached_load_hf_config(path)), quant=NoQuantConfig())
    ctx = Context(page_size=1)
    ctx.attn_backend = _SDPATestBackend()
    ctx._batch = object()
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)

    with torch.device("meta"), torch_dtype(torch.float32):
        model = create_model(config)
    weights = iter_weights(path, device, include_moe_experts=False, include_non_moe=True)
    model.load_state_dict(_materialize_loaded_weight_state_dict(model.state_dict(), weights, device=device))
    model.prepare_for_runtime()
    quantize_model_weights(model, bits=bits)

    expected_dtype = torch.int8 if bits == 8 else torch.uint8
    assert model.model.embedding._table_q.dtype == expected_dtype
    assert model.lm_head._head.dtype == expected_dtype
    for block in model.model.blocks.op_list:
        assert block.attention._qkv[0].dtype == torch.float32  # angles
        assert block.attention._qkv[1].dtype == expected_dtype  # coupling
        assert block.attention._qkv[2].dtype == torch.float32  # scales
        assert block.attention._o[0].dtype == torch.float32
        assert block.attention._o[1].dtype == expected_dtype
        assert block.attention._o[2].dtype == torch.float32
        assert block.mixer.angles.dtype == torch.float32
        assert block.mixer.scales.dtype == torch.float32
        assert block.mixer.coupling.dtype == expected_dtype

    (prompt,) = _prompts(cfg["vocab_size"], (12,), seed=5)
    ids_t = torch.tensor(prompt, dtype=torch.long, device=device)
    pos_t = torch.arange(len(prompt), dtype=torch.long, device=device)
    with torch.no_grad():
        got = model.lm_head.forward(model.model.forward(ids_t, pos_t))
        want = reference.logits(state, cfg, prompt)
    torch.testing.assert_close(got, want, atol=atol, rtol=0.1)


@CUDA_ONLY
@pytest.mark.needs_weights
def test_released_checkpoint_greedy_matches_reference(monkeypatch):
    path = os.environ.get("FREETOKEN_GRT_MODEL")
    if not path:
        pytest.skip("set FREETOKEN_GRT_MODEL to a local GRT checkpoint (e.g. matterwaveai/grt-30m)")
    from freetoken.models.loader import iter_weight_files
    from safetensors.torch import load_file

    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    state = {}
    for file in iter_weight_files(path):
        state.update(load_file(file, device="cuda"))
    engine = TinyGRT(path, monkeypatch, max_reqs=1)
    (prompt,) = _prompts(cfg["vocab_size"], (12,), seed=4)
    req = _req(0, prompt, output_len=24)
    logits = engine.forward([req], "prefill")
    for step in range(16):
        want = reference.logits(state, cfg, req.input_ids.tolist())[-1]
        torch.testing.assert_close(logits[0], want, atol=ATOL, rtol=RTOL)
        if step == 15:
            break
        req.complete_one()
        req.append_host(want.argmax().reshape(1).to(torch.int32).cpu())
        logits = engine.forward([req], "decode")

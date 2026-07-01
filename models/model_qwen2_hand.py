"""
Qwen2-1.5B 手写架构 - 可直接加载 HuggingFace 预训练权重

Qwen2 架构:
  Embedding -> Qwen2Block x 28
     RMSNorm -> GQA Self-Attention (Q=12 heads, KV=2 heads) + RoPE -> Residual
     RMSNorm -> SwiGLU FFN (gate/up/down) -> Residual
  -> RMSNorm -> Linear LM Head

GQA: Grouped Query Attention - KV 头只有 2 个, Q 头 12 个, 每个 KV 头被 6 个 Q 头共享.
推理时 KV-Cache 减小 6 倍.
"""

from __future__ import annotations
import math, json
import torch, torch.nn as nn, torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    def __init__(self, d_k: int, max_len: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.d_k, self.max_len, self.base = d_k, max_len, base
        theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        freqs = torch.outer(torch.arange(max_len).float(), theta)
        self.register_buffer("cos_table", freqs.cos())
        self.register_buffer("sin_table", freqs.sin())

    def _get_rotations(self, offset, seq_len, device, dtype):
        max_needed = offset + seq_len
        if max_needed <= self.cos_table.size(0):
            cos = self.cos_table[offset:max_needed, :].to(dtype)
            sin = self.sin_table[offset:max_needed, :].to(dtype)
        else:
            theta = 1.0 / (self.base ** (torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
            freqs = torch.outer(torch.arange(offset, max_needed, device=device).float(), theta)
            cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def forward(self, q, k, offset=0):
        cos, sin = self._get_rotations(offset, q.size(2), q.device, q.dtype)
        return _rotate_half(q, cos, sin), _rotate_half(k, cos, sin)


def _rotate_half(x, cos, sin):
    d = x.size(-1)
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)


class GQAttention(nn.Module):
    """Grouped Query Attention - Qwen2 style."""
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0, max_len=32768, rope_base=1000000.0):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.scale = math.sqrt(self.d_k)
        self.q_proj = nn.Linear(d_model, n_heads*self.d_k, bias=True)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.d_k, bias=True)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.d_k, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_k, max_len, base=rope_base)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _reshape_q(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_k).permute(0,2,1,3).contiguous()

    def _reshape_kv(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.n_kv_heads, self.d_k).permute(0,2,1,3).contiguous()

    def _repeat_kv(self, k):
        if self.n_rep == 1: return k
        B, n, S, d = k.shape
        return k.unsqueeze(2).expand(B,n,self.n_rep,S,d).reshape(B,n*self.n_rep,S,d)

    def forward(self, x, mask=None, past_kv=None, use_cache=False, cache_offset=0):
        Q = self._reshape_q(self.q_proj(x))
        K = self._reshape_kv(self.k_proj(x))
        V = self._reshape_kv(self.v_proj(x))
        Q, K = self.rope(Q, K, offset=cache_offset)
        if past_kv is not None:
            pk, pv = past_kv
            K = torch.cat([pk, K], dim=2).contiguous()
            V = torch.cat([pv, V], dim=2).contiguous()
        present_kv = (K.detach(), V.detach()) if use_cache else None
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)
        attn = torch.matmul(Q, K.transpose(-2,-1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, V)
        out = out.permute(0,2,1,3).contiguous().view(out.size(0), -1, self.n_heads*self.d_k)
        return (self.o_proj(out), present_kv) if use_cache else self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Qwen2Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.0, max_len=32768, rope_base=1000000.0):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model)
        self.self_attn = GQAttention(d_model, n_heads, n_kv_heads, dropout, max_len, rope_base)
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff, dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False, cache_offset=0):
        r = self.self_attn(self.input_layernorm(x), mask, past_kv, use_cache, cache_offset)
        h, nkv = r if use_cache else (r, None)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return (x, nkv) if use_cache else x


class Qwen2Hand(nn.Module):
    """Qwen2-1.5B hand-written model, can load HuggingFace pretrained weights."""
    def __init__(self, d_model=1536, n_heads=12, n_kv_heads=2, n_layers=28, d_ff=8960,
                 vocab_size=151936, max_len=32768, dropout=0.0, rope_base=1000000.0):
        super().__init__()
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.n_layers, self.max_len = n_layers, max_len
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Qwen2Block(d_model, n_heads, n_kv_heads, d_ff, dropout, max_len, rope_base)
            for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self._dev = torch.device("cpu")

    @property
    def device(self): return self._dev
    def to(self, dev):
        super().to(dev)
        self._dev = torch.device(dev) if isinstance(dev, str) else dev
        return self

    @staticmethod
    def make_causal_mask(S, dev):
        return torch.triu(torch.ones(S,S,device=dev,dtype=torch.bool), diagonal=1)

    def _build_mask(self, x):
        return self.make_causal_mask(x.size(1), self._dev).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        h = self.embed_tokens(x)
        mask = self._build_mask(x)
        for L in self.layers: h = L(h, mask)
        return self.lm_head(self.norm(h))

    def encode_prompt(self, prompt):
        h, mask = self.embed_tokens(prompt), self._build_mask(prompt)
        cache = []
        for L in self.layers:
            h, kv = L(h, mask, use_cache=True, cache_offset=0)
            cache.append(kv)
        return self.norm(h), cache

    def decode_step(self, token, past_cache, cache_offset):
        h = self.embed_tokens(token)
        new_cache = []
        for i, L in enumerate(self.layers):
            h, kv = L(h, mask=None, past_kv=past_cache[i], use_cache=True, cache_offset=cache_offset)
            new_cache.append(kv)
        return self.lm_head(self.norm(h)), new_cache

    def load_qwen2_weights(self, state_dict):
        stripped = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
        own = self.state_dict()
        ld, sk = 0, 0
        for k, v in stripped.items():
            if k == "lm_head.weight": sk += 1; continue
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); ld += 1
            else: sk += 1
        t = len(stripped)-1
        print(f"[Qwen2] weights: {ld}/{t} loaded, {sk} skipped")
        if ld >= t*0.99: print("[Qwen2] pretrained weights loaded successfully")

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen2-1.5B-Instruct", device="cpu"):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        print(f"[Qwen2] loading: {model_name}")
        cp = hf_hub_download(model_name, "config.json")
        cfg = json.load(open(cp))
        try:
            wp = hf_hub_download(model_name, "model.safetensors")
            sd = load_file(wp, device="cpu")
        except Exception:
            wp = hf_hub_download(model_name, "pytorch_model.bin")
            sd = torch.load(wp, map_location="cpu", weights_only=True)
        m = cls(d_model=cfg["hidden_size"], n_heads=cfg["num_attention_heads"],
                n_kv_heads=cfg["num_key_value_heads"], n_layers=cfg["num_hidden_layers"],
                d_ff=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
                max_len=cfg.get("max_position_embeddings",32768),
                dropout=cfg.get("attention_dropout",0.0),
                rope_base=cfg.get("rope_parameters",{}).get("rope_theta",1000000.0))
        m.load_qwen2_weights(sd)
        m.to(device).eval()
        from transformers import AutoTokenizer
        m.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        t = sum(p.numel() for p in m.parameters())
        print(f"[Qwen2] params: {t:,} (d={cfg['hidden_size']}, L={cfg['num_hidden_layers']}, GQA {cfg['num_attention_heads']}/{cfg['num_key_value_heads']})")
        return m


if __name__ == "__main__":
    m = Qwen2Hand(d_model=256, n_heads=4, n_kv_heads=2, n_layers=4, d_ff=1024, vocab_size=5000, max_len=512, dropout=0.1)
    print(f"test params: {sum(p.numel() for p in m.parameters()):,}")
    B, S = 4, 50
    x = torch.randint(0, 5000, (B, S))
    logits = m(x)
    assert logits.shape == (B, S, 5000), f"bad shape {logits.shape}"
    print(f"train: {x.shape} -> {logits.shape} OK")
    prompt = torch.randint(0, 5000, (1, 10))
    h, cache = m.encode_prompt(prompt)
    assert h.shape == (1, 10, 256)
    print(f"encode: {h.shape}, cache={len(cache)} OK")
    tok = torch.randint(0, 5000, (1, 1))
    l2, c2 = m.decode_step(tok, cache, 10)
    assert l2.shape == (1, 1, 5000)
    l3, _ = m.decode_step(tok, c2, 11)
    assert l3.shape == (1, 1, 5000)
    print(f"decode: {l2.shape} OK")
    print("\n[OK] Qwen2Hand architecture test passed!")

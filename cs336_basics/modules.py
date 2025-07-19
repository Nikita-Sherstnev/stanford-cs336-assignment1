from __future__ import annotations

from jaxtyping import Float, Int
from einops import rearrange

import numpy.typing as npt
import torch
import torch.nn as nn

from torch import Tensor


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = 2 / (in_features+out_features)
        var = std ** 0.5
        W = torch.nn.init.trunc_normal_(W, std=std, a=-3 * var, b=3 * var)
        self.weight = torch.nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('ij,... j->... i', self.weight, x)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    ln = Linear(d_in, d_out, dtype=torch.float32)
    ln.load_state_dict({"weight": weights})
    return ln(in_features)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        embs = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        embs = torch.nn.init.trunc_normal_(embs, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(embs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    emb = Embedding(vocab_size, d_model)
    emb.load_state_dict({"weight": weights})
    return emb(token_ids)


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        result = x / rms * self.weight
        return result.to(in_dtype)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.load_state_dict({"weight": weights})
    return rmsnorm(in_features)



def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return silu(in_features)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    swiglu = SwiGLU(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    swiglu.load_state_dict({"w1.weight": w1_weight, "w2.weight": w2_weight, "w3.weight": w3_weight})
    return swiglu(in_features)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        pos = torch.arange(max_seq_len, device=device).float()

        dim_idx = torch.arange(0, d_k, 2, device=device).float()

        freqs: Float[Tensor, "d_k//2"] = 1.0 / (theta ** (dim_idx / d_k))

        freqs: Float[Tensor, "max_seq_len d_k//2"] = torch.outer(pos, freqs)

        _cos: Float[Tensor, "max_seq_len d_k"] = freqs.cos().repeat_interleave(2, dim=-1)
        _sin: Float[Tensor, "max_seq_len d_k"] = freqs.sin().repeat_interleave(2, dim=-1)

        self.register_buffer("_cos", _cos, persistent=False)
        self.register_buffer("_sin", _sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self._cos[token_positions].to(x.device)
        sin = self._sin[token_positions].to(x.device)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated_x1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        rotated_x2 = x2 * cos[..., 1::2] + x1 * sin[..., 1::2]

        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated_x = rotated_x.flatten(-2, -1)

        return rotated_x


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    maxes, inds = torch.max(in_features, dim=dim, keepdim=True)
    exps = torch.exp(in_features - maxes)
    return exps / torch.sum(exps, dim=dim, keepdim=True)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return softmax(in_features, dim)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    qk = torch.einsum(" ... q d , ... k d -> ... q k ", Q, K)
    qk = qk / Q.shape[-1] ** 0.5
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-inf"))

    attn = softmax(qk, dim=-1)
    return torch.einsum("... q v , ... v d -> ... q d", attn, V)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        if self.rope and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)

        attn_v = run_scaled_dot_product_attention(Q, K, V, mask)
        attn_v = rearrange(attn_v, " b h s d -> b s (h d)")
        return self.output_proj(attn_v)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    mha = MultiHeadSelfAttention(d_model, num_heads)
    mha.load_state_dict({'q_proj.weight': q_proj_weight, 'k_proj.weight': k_proj_weight,
                         'v_proj.weight': v_proj_weight, 'output_proj.weight': o_proj_weight})
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len, device=in_features.device)
    mha = MultiHeadSelfAttention(d_model, num_heads, rope)
    mha.load_state_dict({'q_proj.weight': q_proj_weight, 'k_proj.weight': k_proj_weight,
                         'v_proj.weight': v_proj_weight, 'output_proj.weight': o_proj_weight})
    return mha(in_features, token_positions)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        y = y + self.ffn(self.ln2(y))
        return y


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len)
    trans_block = TransformerBlock(d_model, num_heads, d_ff, rope)

    token_positions = torch.arange(in_features.shape[-2])
    trans_block.load_state_dict(weights)
    return trans_block(in_features, token_positions)


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float,
                 vocab_size: int, context_length: int, num_layers: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        rope = RotaryPositionalEmbedding(rope_theta, d_model//num_heads, context_length)

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope, device=device, dtype=dtype)
             for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x, token_positions)

        return self.lm_head(self.ln_final(x))


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    trans = Transformer(d_model, num_heads, d_ff, rope_theta,
                        vocab_size, context_length, num_layers)
    trans.load_state_dict(weights)
    token_positions = torch.arange(in_indices.shape[-1])
    return trans(in_indices, token_positions)


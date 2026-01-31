import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model_registry import ModelConfig, register_model


@dataclass
class ModernTransformerConfig(ModelConfig):
    """Configuration for ModernTransformer with GQA and RoPE."""

    name: str = "modern"
    vocab_size: int = 50257
    n_embed: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_group: int = 4
    max_seq_len: int = 1024


class RotaryGQA(nn.Module):
    """Grouped Query Attention (GQA) with RoPE embeddings.

    GQA reduces KV cache memory by using fewer key-value heads than query heads.
    Multiple query heads share a single KV head, interpolating between:
    - MHA (n_group=1): Each query head has its own KV head
    - MQA (n_group=n_head): All query heads share one KV head
    """

    def __init__(self, config: ModernTransformerConfig):
        super().__init__()

        self.head_dim = config.n_embed // config.n_head
        self.num_kv_head = config.n_head // config.n_group
        self.n_group = config.n_group

        self.w_q = nn.Linear(config.n_embed, config.n_embed)
        self.w_k = nn.Linear(config.n_embed, self.num_kv_head * self.head_dim)
        self.w_v = nn.Linear(config.n_embed, self.num_kv_head * self.head_dim)
        self.w_o = nn.Linear(config.n_embed, config.n_embed)
        self.w_o.NANOGPT_SCALE_INIT = 1  # type: ignore

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.causal_mask: torch.Tensor
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
            ),
        )

        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, config.max_seq_len)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos, sin):
        batch, num_heads, seq_len, head_dim = q.shape
        num_kv_heads = k.size(1)

        # Reshape to expose pairs: [B, n_head, T, head_dim / 2, 2]]
        q_pairs = q.view(batch, num_heads, seq_len, head_dim // 2, 2)
        k_pairs = k.view(batch, num_kv_heads, seq_len, head_dim // 2, 2)

        # Extract the two elements of each pair
        q_even = q_pairs[..., 0]  # [batch, heads, seq, head_dim/2]
        q_odd = q_pairs[..., 1]
        k_even = k_pairs[..., 0]
        k_odd = k_pairs[..., 1]

        # Broadcast cos/sin: [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply 2x2 rotation matrix to each pair:
        # [cos θ  -sin θ] [x_even]
        # [sin θ   cos θ] [x_odd ]
        q_even_rot = q_even * cos - q_odd * sin
        q_odd_rot = q_even * sin + q_odd * cos
        k_even_rot = k_even * cos - k_odd * sin
        k_odd_rot = k_even * sin + k_odd * cos

        # Reassemble pairs back to original shape
        q_rot = torch.stack([q_even_rot, q_odd_rot], dim=-1)  # [..., head_dim/2, 2]
        k_rot = torch.stack([k_even_rot, k_odd_rot], dim=-1)

        q_rot = q_rot.view(batch, num_heads, seq_len, head_dim)
        k_rot = k_rot.view(batch, num_kv_heads, seq_len, head_dim)

        return q_rot, k_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # q: [B, T, C], k/v: [B, T, num_kv_head * head_dim]
        q: torch.Tensor = self.w_q(x)
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        # Now we split into proper [B, n_heads, T, head_dim] for multi head attention.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # K,V are [B, T, n_kv_head, head_dim]
        k = k.view(B, T, self.num_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_head, self.head_dim).transpose(1, 2)

        # we rotate them according to RoPE angles
        cos, sin = self.rotary_emb(x, T)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Then we replicate them for [B, T, n_head, head_dim] for GQA
        k = k.repeat_interleave(self.n_group, dim=1)
        v = v.repeat_interleave(self.n_group, dim=1)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(k.size(-1))

        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Attn is [B, n_head, T, T], v is [B, n_head, T, head_dim]. Out is [B, n_head, T, head_dim]
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.w_o(out)

        return out


class MLP(nn.Module):
    def __init__(self, config: ModernTransformerConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embed, config.n_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: ModernTransformerConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attention = RotaryGQA(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm, allowing for a gradient highway.
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute rotary embeddings up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )

        # [seq_len] * [dim / 2] -> [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq)
        # [seq_len, dim]
        self.cos_freqs: torch.Tensor
        self.sin_freqs: torch.Tensor

        self.register_buffer("cos_freqs", freqs.cos(), persistent=False)
        self.register_buffer("sin_freqs", freqs.sin(), persistent=False)
        
        self.max_seq_len = seq_len

    def forward(self, x, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_freqs[:seq_len].to(x.dtype),
            self.sin_freqs[:seq_len].to(x.dtype),
        )


class ModernTransformer(nn.Module):
    """Decoder-only transformer with Grouped Query Attention (GQA) and RoPE.

    Architecture inspired by LLaMA: uses RoPE for positional encoding and GQA
    to reduce KV cache memory while maintaining model quality.
    """

    def __init__(self, config: ModernTransformerConfig):
        super().__init__()
        self._validate_config(config)

        self.config = config

        # Word to embeddings layer (no positional embeddings - RoPE handles position)
        self.wte: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embed)
        # Hidden layers in the middle, the transformer blocks
        self.h: nn.ModuleList = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        # Layer normalization at the very end
        self.ln_f: nn.LayerNorm = nn.LayerNorm(config.n_embed)
        # Project back from embedding representation to tokens.
        self.lm_head: nn.Linear = nn.Linear(
            config.n_embed, config.vocab_size, bias=False
        )

        self.apply(self._init_weights)

    def _validate_config(self, config: ModernTransformerConfig):
        if config.n_embed % config.n_head != 0:
            raise ValueError(
                f"n_embed ({config.n_embed}) must be divisible by n_head ({config.n_head})"
            )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            weight_std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                weight_std /= math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=weight_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def tie_weights(self) -> None:
        """Tie input embeddings to output projection weights.

        Weight tying (from "Using the Output Embedding to Improve Language Models",
        Press & Wolf 2016) shares parameters between the input embedding matrix and
        the output projection layer. This works because both layers operate in the
        same semantic space: wte maps tokens -> embeddings, while lm_head maps
        embeddings -> token logits. Sharing these weights:
          1. Reduces parameters by ~vocab_size * n_embed (significant for large vocabs)
          2. Acts as regularization, improving generalization
          3. Often improves performance, especially on smaller models

        Call this after model.to(device) to avoid MPS/CUDA storage issues.
        """
        self.lm_head.weight = self.wte.weight

    def forward(self, input_token_ids):
        B, T = input_token_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # [B, T, n_embed] - position is encoded via RoPE in attention layers
        x = self.wte(input_token_ids)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        # [B, T, vocab_size] (logits for every token in the sentence)
        per_token_logits = self.lm_head(x)

        return per_token_logits


@register_model("modern")
def create_modern(model_config):
    return ModernTransformer(model_config)

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model_registry import ModelConfig, register_model


@dataclass
class GPT2Config(ModelConfig):
    """GPT-2 specific configuration"""

    name: str = "gpt2"  # Default name
    vocab_size: int = 50257
    n_embed: int = 768
    n_layer: int = 12
    n_head: int = 12
    block_size: int = 1024


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.qvk = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.w_o = nn.Linear(config.n_embed, config.n_embed)
        self.w_o.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # [B, T, 3 * C] tensor with the tensors
        qkv: torch.Tensor = self.qvk(x)

        q, k, v = qkv.split(self.n_embed, dim=2)

        # Now we split into proper [B, n_heads, T, head_dim] for multi head attention.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(k.size(-1))

        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Attn is [B, n_head, T, T], v is [B, n_head, T, C]. Out is back to [B, n_head, T, C]
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.w_o(out)

        return out


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
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

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT2(nn.Module):
    """
    GPT-2 code as written by Karpathy in his youtube series.
    The baseline for this exercise, everything else will be likely a variation of this.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self._validate_config(config)

        self.config = config

        # Word to embeddings layer, the entry point of the whole thing :).
        self.wte: nn.Embedding = nn.Embedding(config.vocab_size, config.n_embed)
        # The positional embeddings (wpe = word position embeddings).
        self.wpe: nn.Embedding = nn.Embedding(config.block_size, config.n_embed)
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

    def _validate_config(self, config: GPT2Config):
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

    def forward(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_token_ids.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        # B, T, n_embed
        embeddings = self.wte(input_token_ids)
        # T
        pos = torch.arange(0, T, dtype=torch.long, device=input_token_ids.device)
        # [T, n_embed]
        pos_embeddings = self.wpe(pos)

        # Becomes [B, T, n_embed] due to torch broadcastin
        x = embeddings + pos_embeddings

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        # [B, T, vocab_size] (logits for every token in the sentence)
        per_token_logits = self.lm_head(x)

        return per_token_logits


@register_model("gpt2")
def create_gpt2(model_config):
    return GPT2(model_config)

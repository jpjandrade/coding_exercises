from dataclasses import dataclass
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# Set device based on what's available.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Torch backend device: {device}")

TOP_K = 50

GPT2_CONFIG_ARGS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),  # 1558M params
}


@dataclass
class GPTConfig:
    block_size: int = 1024  # max seq length
    vocab_size: int = (
        50257  # number of tokens: 50k BPE merges + 256 bytes tokens + 1 <|endoftext|> tokens.
    )
    n_layer: int = 12  # number of attn layers
    n_head: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections for all heads, concatenated.
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Called "bias" for parity with OpenAI paper, but it's the mask that stops
        # the model from seeing the future.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # batch size, number of tokens and channel size (i.e., embedding layers, should be n_embed).
        B, T, C = x.size()

        qkv = self.c_attn(x)

        # This is how we do effectively, we just split now.
        q, k, v = qkv.split(self.n_embed, dim=2)

        # All tensors are (B, nh, T, hs), because nh * hs = C, this is just a reshape.
        # These are just n_head parallel attention modules which happen in parallel.
        # They are in the same tensor for effiency.

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # [B, nh, T, hs]

        # Attention matrix (B, nh, T, T), what's the weight of each token to each other.
        # (tranpose swaps the second to last with the last, then we do mat mul between
        # [B, nh, T, hs] and [B, nh, hs, T]). So we treat both B and nh as batches.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Autoregressive mask, makes the attention only care about past tokens.
        # We want the softmax "probabilities" of the future tokens to be zero,
        # so we set the tokens to -inf before the softmax.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # normalize attn to 1.
        att = F.softmax(att, dim=-1)

        # Weighted sum we found "interesting" for every single token.
        # It's [B, nh, T, T] @ [B, nh, T, hs] -> [B, nh, T, hs]
        y = att @ v
        # Concatenation operation to make it back to T, C per batch.
        # Essentially coalesces [nh, hs] back to C
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # Both steps sum to x to implement a clean residual pathway,
        # so (half) the gradients can flow unaltered to the input layer.
        # 1. Attn can be imagined as a "reduce", it's the point where all the tokens communicate with each other.
        x = x + self.attn(self.ln_1(x))
        # 2. MLP can be imagined as a "map", where each token gets altered by itself.
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # word token embeddings (classic one hot encoding to embeddings).
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                # word positional embeddings (aka positional encoding).
                wpe=nn.Embedding(config.block_size, config.n_embed),
                # h for hidden, the hidden transformer layers, aka the main thing.
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                # Layer normalization to the final output of the whole thing.
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

        # Project back from n_embed (i.e. embedding dimensions) to the vocab_size (i.e., the tokens).
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx is [B, T] (like, literally the entry sentences in token format)
        B, T = idx.size()
        assert T <= self.config.block_size
        # Forward token and pos embeddings.
        # The device = idx.device guarantees that there's no costly mismatch, because every single
        # new tensor comes from pos or idx.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx)  # token embeddings, [B, T, n_embed]
        # Becomes [B, T, n_embed] due to broadcasting, start matching the last dim and keep going.
        x = tok_emb + pos_emb

        # Forward the blocks of the transformer. Attention is all you need and all :)
        for i, block in enumerate(self.transformer.h):
            x = block(x)

        # Forward the final layernorm and the classifier.
        x = self.transformer.ln_f(x)

        # (B, T, vocab_size) -> per every token (in the entire sentence) we're calculating the next token.
        # It's a bit silly, we'll get there.
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = GPT2_CONFIG_ARGS[model_type]

        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"Found {sd_hf[k].shape} and {sd[k].shape} for {k}, should be equal."
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
# Uncomment this to load from hugging face directly to test correctness.
# (And take the first element of logits below)
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# Put model in evaluation mode. Normally means we disable dropout, batchnorm and etc.
model.eval()

# Set device in the top, could be whatever you want.
model.to(device)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
print("Tokens: ", tokens)
tokens = torch.tensor(tokens, dtype=torch.long)

tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

x = tokens.to(device)
model.forward(x)

torch.manual_seed(42)
torch.mps.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # Throw away all the other logits, we only care about the last location.
        # (because we're predicting T + 1).
        # This is wasteful of course, but we're not exactly OpenAI / Deepmind here.
        logits = logits[:, -1, :]  # (B, vocab_size)

        # Transform logits into probabilities.
        probs = F.softmax(logits, dim=-1)

        # Select the top-k more probable tokens.
        # Both (B, TOP_K)
        topk_probs, topk_indices = torch.topk(probs, TOP_K, dim=-1)

        # Select a token from the top-k probabilities.
        # (B, 1), hopefully.
        ix = torch.multinomial(topk_probs, 1)

        # Get the selected indices above from the top_k.
        xcol = torch.gather(topk_indices, -1, ix)

        # Append to the sequence we have so far
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

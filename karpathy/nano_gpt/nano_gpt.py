# code followed from https://www.youtube.com/watch?v=kCc8FmEb1nY

import os
import torch
import torch.nn as nn
from torch.nn import functional as tf

TRAINING_DATA = 'input.txt'
MODEL_CHECKPOINT_NAME = 'nano_gpt.pt'

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 10
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embeddings = 384
n_heads = 6
n_layer = 6
dropout = 0.2
new_tokens_to_generate = 10000
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

assert (n_embeddings % n_heads == 0)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size)
                                   for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B, T, head_size
        q = self.query(x)  # B, T, head_size
        wei = q @ k.transpose(-2, -1) / C ** 0.5  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # T, T
        wei = tf.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,  4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        head_size = dim // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(dim)
        self.norm_sa = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, x):
        att_res = x + self.sa(self.norm_sa(x))
        out = att_res + self.ffwd(self.norm_ff(att_res))
        return out


class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.blocks = nn.Sequential(
            *[Block(n_embeddings, n_heads=n_heads) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # B, T, E
        pos_embd = self.position_embedding_table(
            torch.arange(T, device=device))  # T, C
        x = tok_emb + pos_embd
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)  # B, T, vocab_size
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = tf.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # only have embedding pos for block_size
            idx_cropped = idx[:, -block_size:]
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]

            probs = tf.softmax(logits, dim=-1)  # B, C

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)  # B, T + 1

        return idx


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
            out[split] = losses.mean()

    model.train()
    return out


# read training file
if not os.path.isfile(TRAINING_DATA):
    raise FileNotFoundError("Training data not found!")

with open(TRAINING_DATA, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [char_to_int[c] for c in s]
# decoder: take a list of integers, output a string
def decode(let): return ''.join([int_to_char[i] for i in let])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model = NanoGPT(vocab_size)

if os.path.isfile(MODEL_CHECKPOINT_NAME):
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_NAME))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % eval_interval == 0:
        torch.save(model.state_dict(), MODEL_CHECKPOINT_NAME)
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=new_tokens_to_generate)[0].tolist()))

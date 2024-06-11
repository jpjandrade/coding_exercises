# code followed from https://www.youtube.com/watch?v=kCc8FmEb1nY

import os
import torch
import torch.nn as nn
from torch.nn import functional as tf

TRAINING_DATA = 'input.txt'
MODEL_CHECKPOINT_NAME = 'bygram.pt'

batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 100
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embeddings = 32
tokens_to_generate = 10000
device = 'mps' if torch.backends.mps.is_available() else 'cpu'


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # B, T, E
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))  # T, C
        x = tok_emb + pos_embd
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
            logits, loss = self(idx)

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

model = BigramLanguageModel(vocab_size)

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
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

torch.save(model.state_dict(), MODEL_CHECKPOINT_NAME)
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=tokens_to_generate)[0].tolist()))

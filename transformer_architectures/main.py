import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from dataloader import DataLoaderLite
from model_registry import LoadModel
from gqa import GQATransformerConfig

# Set device based on what's available.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Torch backend device: {device}")

torch.manual_seed(1337)
torch.mps.manual_seed(1337)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    n_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 8
    sequence_length: int = 1024
    warmup_steps: int = 100
    gradient_clip: float = 1.0


training_config = TrainingConfig(
    n_steps=100, learning_rate=1e-3, batch_size=8, sequence_length=1024
)

data = DataLoaderLite(training_config.batch_size, training_config.sequence_length)

model_config = GQATransformerConfig()
model = LoadModel(model_config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Set device in the top, could be whatever you want.
model.to(device)
# Weight sharing after moving to device to avoid MPS issues
if hasattr(model, "tie_weights"):
    model.tie_weights()  # type: ignore

for i in range(training_config.n_steps):
    t_start = time.time()
    optimizer.zero_grad()
    x, y = data.next_batch()
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    loss.backward()
    optimizer.step()

    t_end = time.time()
    dt = (t_end - t_start) * 1000  # time in ms
    tokens_per_second = (data.B * data.T) / (t_end - t_start)

    print(
        f"Step {i}: loss: {loss.item()}, time: {dt:.2f}, tokens/s: {tokens_per_second:.2f}"
    )

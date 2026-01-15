from dataclasses import dataclass, field
from typing import Callable, Dict, Any
import torch.nn


@dataclass
class ModelConfig:
    """Configuration for loading a model."""

    name: str

    # Common model configurations.
    # If needed a model may ignore these.
    # Embedding dimension
    n_embed: int = 768
    n_layer: int = 12
    # Number of attention heads per attention block.
    n_head: int = 6
    vocab_size: int = 50257

    # Model specific kwargs.
    kwargs: Dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, Callable[[ModelConfig], torch.nn.Module]] = {}


def register_model(name: str):
    """Decorator to register model factory functions"""

    def decorator(fn: Callable[[ModelConfig], torch.nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def LoadModel(config: ModelConfig) -> torch.nn.Module:
    if isinstance(config, str):
        config = ModelConfig(name=config)
    if config.name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model {config.name}. Available: {available}")
    return MODEL_REGISTRY[config.name](config)

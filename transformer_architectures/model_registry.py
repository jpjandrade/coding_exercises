from dataclasses import dataclass, field
from typing import Callable, Dict, Any
import torch.nn

MODEL_REGISTRY: Dict[str, Callable[[], torch.nn.Module]] = {}


def register_model(name: str):
    """Decorator to register model factory functions"""

    def decorator(fn: Callable[[], torch.nn.Module]):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def LoadModel(config) -> torch.nn.Module:
    if isinstance(config, str):
        config = ModelConfig(name=config)
    if config.name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model {config.name}. Available: {available}")
    return MODEL_REGISTRY[config.name]()


@dataclass
class ModelConfig:
    """Configuration for loading a model."""

    name: str

    # Common model configurations.
    # If needed a model may ignore these.
    hidden_size: int = 768
    num_layers: int = 12
    # Number of attention heads per
    num_heads: int = 6

    # Model specific kwargs.
    kwargs: Dict[str, Any] = field(default_factory=dict)

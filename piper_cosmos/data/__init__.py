"""Dataset utilities for Piper/Cosmos experiments."""

__all__ = ["PiperDualDataset"]


def __getattr__(name: str):
    if name == "PiperDualDataset":
        from .piper_dual_dataset import PiperDualDataset

        return PiperDualDataset
    raise AttributeError(name)

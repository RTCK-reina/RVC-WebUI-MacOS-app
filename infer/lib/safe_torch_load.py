"""Explicit, fail-closed torch checkpoint loading helpers."""

from __future__ import annotations

from typing import Any

import torch


def load_weights(path: str, *, map_location: Any = "cpu") -> Any:
    """Load a checkpoint without allowing arbitrary pickle execution.

    PyTorch 2.6+ supports ``weights_only=True``. If the runtime is too old to
    accept that parameter, fail closed instead of silently falling back to
    unsafe pickle deserialization for user-supplied model files.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" in str(exc):
            raise RuntimeError(
                "This PyTorch runtime cannot safely load user checkpoints; "
                "install PyTorch with weights_only support."
            ) from exc
        raise

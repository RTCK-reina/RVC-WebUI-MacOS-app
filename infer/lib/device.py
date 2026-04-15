"""Device-related helpers for the macOS-focused build.

Only MPS (Apple Silicon) and CPU are supported. CUDA / XPU / DirectML branches
have been removed because they cannot run on macOS.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def empty_device_cache() -> None:
    """Free cached memory on the active accelerator, if any.

    Calls ``torch.mps.empty_cache()`` when running on Apple Silicon with MPS
    available, and is a no-op on CPU-only systems.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

"""Device-related helpers for the macOS-focused build.

Only MPS (Apple Silicon) and CPU are supported. CUDA / XPU / DirectML branches
have been removed because they cannot run on macOS.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Fraction of the Metal recommended working-set size above which
# ``empty_device_cache_if_needed`` actually flushes the allocator pool.
_PRESSURE_THRESHOLD = float(os.environ.get("RVC_MPS_CACHE_PRESSURE", "0.5"))


def empty_device_cache() -> None:
    """Free cached memory on the active accelerator, if any.

    Calls ``torch.mps.empty_cache()`` when running on Apple Silicon with MPS
    available, and is a no-op on CPU-only systems.
    """
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def empty_device_cache_if_needed() -> bool:
    """Flush the MPS allocator pool only under memory pressure.

    ``torch.mps.empty_cache()`` releases the whole allocator pool and implies
    a device synchronization, so calling it after every pipeline run makes
    batch inference re-allocate every buffer per file. This variant is cheap
    to call unconditionally: it flushes only when the driver's allocated
    bytes exceed ``RVC_MPS_CACHE_PRESSURE`` (default 0.5) of the recommended
    maximum working-set size. Returns True when a flush happened.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        allocated = torch.mps.driver_allocated_memory()
        budget = torch.mps.recommended_max_memory()
    except (AttributeError, RuntimeError):
        # Older torch without the introspection API - keep legacy behavior.
        torch.mps.empty_cache()
        return True
    if budget > 0 and allocated / budget >= _PRESSURE_THRESHOLD:
        torch.mps.empty_cache()
        return True
    return False

"""Compatibility shim for PyTorch 2.6+ ``torch.load`` default changes.

PyTorch 2.6 flipped ``torch.load``'s default from ``weights_only=False`` to
``weights_only=True``. That breaks third-party checkpoints that pickle
non-tensor objects — notably the HuBERT base checkpoint bundled with fairseq,
whose state contains a ``fairseq.data.dictionary.Dictionary`` instance.

Design choice: we expose a ``legacy_load`` context manager that scopes the
relaxed ``weights_only`` default to a single call site, rather than the
previous module-import side effect of a global monkey-patch. The scoped
approach keeps the rest of the app on PyTorch 2.6's safer default while still
letting fairseq's ``checkpoint_utils.load_model_ensemble_and_task`` succeed.

Usage::

    from infer.lib.torch_compat import legacy_load

    with legacy_load():
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path], suffix=""
        )

Historical note: earlier revisions of this module installed a global patch at
import time. That worked but widened the pickle-deserialization surface to
every ``torch.load`` call in the process — including user-supplied checkpoints
loaded by ``infer/lib/train/process_ckpt.py``. The context manager narrows
that surface to the specific fairseq loader.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterator

import torch

# Module-level lock and reference count so that concurrent legacy_load()
# blocks (from worker threads in rpc_server) patch torch.load exactly once
# and restore it deterministically when the last caller exits.
_legacy_load_lock = threading.Lock()
_legacy_load_count = 0
_legacy_load_original: object = None


def _patched_load(*args: object, **kwargs: object) -> object:
    kwargs.setdefault("weights_only", False)  # type: ignore[attr-defined]
    return _legacy_load_original(*args, **kwargs)  # type: ignore[misc]


@contextlib.contextmanager
def legacy_load() -> Iterator[None]:
    """Temporarily restore ``torch.load``'s pre-2.6 default of
    ``weights_only=False`` for the duration of the ``with`` block.

    Thread-safe: concurrent calls from different threads share a single patch
    (installed on first enter, restored when the last caller exits). A
    module-level lock guards the patch/restore transitions so that worker
    threads in rpc_server cannot corrupt each other's torch.load reference.

    Any explicit ``weights_only=`` kwarg passed to ``torch.load`` by the
    caller wins over the patched default.

    Callers should wrap only the narrowest possible region (ideally a single
    ``load_checkpoint_to_cpu`` / ``load_model_ensemble_and_task`` call) to
    avoid widening the pickle deserialization surface.
    """
    global _legacy_load_count, _legacy_load_original

    with _legacy_load_lock:
        if _legacy_load_count == 0:
            _legacy_load_original = torch.load
            torch.load = _patched_load  # type: ignore[assignment]
        _legacy_load_count += 1
    try:
        yield
    finally:
        with _legacy_load_lock:
            _legacy_load_count -= 1
            if _legacy_load_count == 0:
                torch.load = _legacy_load_original  # type: ignore[assignment]
                _legacy_load_original = None

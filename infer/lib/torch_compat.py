"""Compatibility shim for PyTorch 2.6+ ``torch.load`` default changes.

PyTorch 2.6 flipped ``torch.load``'s default from ``weights_only=False`` to
``weights_only=True``. That breaks third-party checkpoints that pickle
non-tensor objects — notably the HuBERT base checkpoint bundled with fairseq,
whose state contains a ``fairseq.data.dictionary.Dictionary`` instance.

Design choice: we expose a ``legacy_load`` context manager that scopes the
relaxed ``weights_only`` default to a single call site, rather than the
previous module-import side effect of an unconditional global monkey-patch.

**Thread safety:** ``rpc_server`` dispatches blocking RPC methods on a
dedicated worker thread while non-blocking methods run on the main
dispatcher thread, so a naive ``torch.load = patched; ...; torch.load =
original`` would race — a concurrent ``torch.load`` call on another thread
could observe either the patched or the original function depending on
timing. We therefore install the wrapper **once, at import time**, and
control its behaviour through ``threading.local`` so each thread makes its
own decision about whether to relax ``weights_only``.

Usage::

    from infer.lib.torch_compat import legacy_load

    with legacy_load():
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [hubert_path], suffix=""
        )

    # Outside the `with`, torch.load behaves exactly as PyTorch ships it
    # (weights_only=True by default on 2.6+). Other threads are unaffected.

Historical note: earlier revisions of this module installed a global patch
that flipped the default globally — that widened the pickle-deserialization
surface to every ``torch.load`` call in the process, including
user-supplied checkpoints loaded by ``infer/lib/train/process_ckpt.py``.
A later revision replaced that with a non-thread-safe scoped patch. This
revision keeps the scoping but moves the flag to thread-local storage so
the wrapper is always installed and its behaviour is deterministic under
concurrency.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterator

import torch

# Per-thread toggle: when the current thread is inside a `legacy_load()`
# block, `_flag.enabled` is True and the wrapper injects
# ``weights_only=False`` as the default. For every other thread the flag is
# unset (getattr fallback to False) and `torch.load` behaves stock.
_flag = threading.local()

# Capture the original ``torch.load`` exactly once, before installing the
# wrapper. We keep a reference here so nested re-imports of this module
# don't install the wrapper on top of itself.
_original_load = torch.load
_PATCH_MARK = "_rvc_torch_compat_wrapper"


def _patched_load(*args, **kwargs):  # type: ignore[no-untyped-def]
    if getattr(_flag, "enabled", False):
        kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


# Install the wrapper once. Subsequent imports of this module (e.g. from
# test fixtures that reload() it) will see the marker and skip reinstall.
if not getattr(torch.load, _PATCH_MARK, False):
    setattr(_patched_load, _PATCH_MARK, True)
    torch.load = _patched_load  # type: ignore[assignment]


@contextlib.contextmanager
def legacy_load() -> Iterator[None]:
    """Enable the pre-2.6 ``weights_only=False`` default for torch.load
    **on the calling thread only**, for the duration of the ``with`` block.

    Nested ``with legacy_load()`` calls on the same thread are safe: the
    flag is saved and restored so the inner scope does not prematurely
    disable the outer one.

    Callers outside any active ``legacy_load()`` block see the stock
    PyTorch 2.6+ behaviour with ``weights_only=True`` defaulted in.
    """
    previous = getattr(_flag, "enabled", False)
    _flag.enabled = True
    try:
        yield
    finally:
        _flag.enabled = previous

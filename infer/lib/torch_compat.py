"""Compatibility shim for PyTorch 2.6+ ``torch.load`` default changes.

PyTorch 2.6 flipped ``torch.load``'s default from ``weights_only=False`` to
``weights_only=True``. That breaks third-party checkpoints that pickle
non-tensor objects — notably the HuBERT base checkpoint bundled with fairseq,
whose state contains a ``fairseq.data.dictionary.Dictionary`` instance.

We control every checkpoint this app loads (they ship inside the .app bundle
or come from the user's own training runs), so disabling the new safety
default is fine. Importing this module monkey-patches ``torch.load`` so every
caller — including fairseq's internal ``checkpoint_utils.load_checkpoint_to_cpu``
— gets the pre-2.6 behavior unless it opts into weights-only explicitly.

Import this module **before** importing ``fairseq`` (or anything else that
loads a checkpoint on import) for the patch to take effect.
"""

from __future__ import annotations

import torch

_PATCH_FLAG = "_rvc_weights_only_shim"


def _install() -> None:
    if getattr(torch.load, _PATCH_FLAG, False):
        return

    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    setattr(_patched_load, _PATCH_FLAG, True)
    torch.load = _patched_load  # type: ignore[assignment]


_install()

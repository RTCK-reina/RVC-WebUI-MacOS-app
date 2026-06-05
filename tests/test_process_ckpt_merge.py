"""Regression tests for ``infer.lib.train.process_ckpt.merge``.

Covers two pre-existing bugs in ``merge()``:

  1. The inner ``extract()`` helper used to return ``{"weight": {...}}``
     instead of a flat state dict. Raw training checkpoints (``G_*.pth``)
     therefore blew up with ``AttributeError: 'dict' object has no
     attribute 'float'`` inside the merge loop.

  2. ``authors()`` was called **after** ``ckpt1``/``ckpt2`` had been
     reassigned to tensor-only dicts. It read ``c.get("author", "")`` on
     dicts whose values were tensors, silently returned ``""``, and the
     merged output lost its author attribution.

conftest.py stubs both ``torch`` and ``infer.lib.train.process_ckpt``.
This module pops the stubs, imports the real modules from the ``rvc``
conda env, and restores state on teardown so subsequent test files
(which rely on the stubs) continue to work.
"""
from __future__ import annotations

import importlib
import sys

import pytest


_SWAPPED_KEYS = ("torch", "infer.lib.train.process_ckpt")


def _swap_in_real_modules():
    """Replace the conftest torch/process_ckpt stubs with real imports.

    Returns the saved sys.modules entries for later restoration.
    """
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key in _SWAPPED_KEYS or key.startswith("torch."):
            saved[key] = sys.modules[key]
    for key in list(saved):
        sys.modules.pop(key, None)

    try:
        import torch  # real torch from rvc conda env

        if not hasattr(torch, "zeros") or not callable(getattr(torch, "zeros", None)):
            raise RuntimeError(
                "real torch required for this test — got a stub. "
                "Run pytest inside the rvc conda env."
            )
        # The stub advertised version "2.0.0+mock" — guard against that sneaking in.
        if str(getattr(torch, "__version__", "")).endswith("+mock"):
            raise RuntimeError("real torch required; got +mock stub")

        mod = importlib.import_module("infer.lib.train.process_ckpt")
        # Avoid pulling real infer.modules.vc (heavy). These are set after import
        # so the merge() body uses them via module attribute lookup.
        mod.model_hash_ckpt = lambda _opt: "hash-stub"
        mod.hash_id = lambda _h: "id-stub"
        return saved, mod, torch
    except Exception:
        _restore_modules(saved)
        raise


def _restore_modules(saved: dict[str, object]):
    for key in list(sys.modules):
        if key in _SWAPPED_KEYS or key.startswith("torch."):
            sys.modules.pop(key, None)
    sys.modules.update(saved)


@pytest.fixture(scope="module")
def real_process_ckpt():
    # Module-scoped: torch's native extension cannot be unloaded and
    # re-imported safely (RuntimeError: function already has a docstring),
    # so we do the stub swap exactly once per test module.
    try:
        saved, mod, torch = _swap_in_real_modules()
    except ModuleNotFoundError as e:
        if e.name == "torch":
            pytest.skip("real torch is not installed in this Python environment")
        raise
    except RuntimeError as e:
        if "real torch required" in str(e):
            pytest.skip(str(e))
        raise
    try:
        yield mod, torch
    finally:
        _restore_modules(saved)


def _inference_ckpt(torch, author: str):
    # Two small tensors with matching keys so the merge loop runs cleanly.
    return {
        "weight": {
            "layer.weight": torch.zeros(2, 2),
            "layer.bias": torch.zeros(2),
        },
        "config": [1, 2, 3],
        "author": author,
    }


def _raw_training_ckpt(torch, author: str):
    # Raw G_*.pth shape — state dict lives under "model". A real checkpoint
    # from utils.save_checkpoint has NO "config" key (only model/iteration/
    # optimizer/learning_rate), so we deliberately omit it: merge() must derive
    # config from the sr/version args instead of indexing ckpt1["config"].
    return {
        "model": {
            "layer.weight": torch.zeros(2, 2),
            "layer.bias": torch.zeros(2),
        },
        "iteration": 0,
        "author": author,
    }


def test_merge_preserves_author_from_inference_ckpts(
    tmp_path, monkeypatch, real_process_ckpt
):
    """Regression guard for bug 2 (author attribution dropped)."""
    mod, torch = real_process_ckpt
    monkeypatch.setenv("weight_root", str(tmp_path))

    p1 = tmp_path / "a.pth"
    p2 = tmp_path / "b.pth"
    torch.save(_inference_ckpt(torch, "Alice"), str(p1))
    torch.save(_inference_ckpt(torch, "Bob"), str(p2))

    result = mod.merge(
        str(p1), str(p2), 0.5, "40k", 1, "info", "merged_authors", "v2"
    )
    assert result == "Success.", result

    out = tmp_path / "merged_authors.pth"
    assert out.exists()
    loaded = torch.load(str(out), map_location="cpu")
    assert loaded["author"] == "Alice & Bob"


def test_merge_extracts_raw_training_ckpt(
    tmp_path, monkeypatch, real_process_ckpt
):
    """Regression guard for bug 1 (extract() shape mismatch).

    Without the extract() fix, merge() catches AttributeError and returns
    a traceback string instead of "Success.".
    """
    mod, torch = real_process_ckpt
    monkeypatch.setenv("weight_root", str(tmp_path))

    p1 = tmp_path / "raw.pth"
    p2 = tmp_path / "infer.pth"
    torch.save(_raw_training_ckpt(torch, "Carol"), str(p1))
    torch.save(_inference_ckpt(torch, "Dave"), str(p2))

    result = mod.merge(
        str(p1), str(p2), 0.5, "40k", 1, "info", "merged_raw", "v2"
    )
    assert result == "Success.", result

    out = tmp_path / "merged_raw.pth"
    assert out.exists()
    loaded = torch.load(str(out), map_location="cpu")
    assert loaded["author"] == "Carol & Dave"


def test_merge_raw_ckpt_without_config_stamps_synth_config(
    tmp_path, monkeypatch, real_process_ckpt
):
    """Regression guard for bug 3 (KeyError on a configless raw G_*.pth).

    merge() used to read ``ckpt1["config"]`` unconditionally; a real raw
    checkpoint has no such key, so it raised KeyError that the bare except
    swallowed into a returned traceback. It must now derive the config from
    the requested sr/version, stamping the canonical 40k synthesizer config.
    """
    mod, torch = real_process_ckpt
    monkeypatch.setenv("weight_root", str(tmp_path))

    # BOTH inputs are raw checkpoints with no embedded "config".
    p1 = tmp_path / "raw1.pth"
    p2 = tmp_path / "raw2.pth"
    torch.save(_raw_training_ckpt(torch, "Eve"), str(p1))
    torch.save(_raw_training_ckpt(torch, "Frank"), str(p2))

    result = mod.merge(
        str(p1), str(p2), 0.5, "40k", 1, "info", "merged_cfg", "v2"
    )
    assert result == "Success.", result

    loaded = torch.load(str(tmp_path / "merged_cfg.pth"), map_location="cpu")
    # The derived config must match the canonical 40k table (18 fields, with
    # the trailing sampling rate 40000), NOT a KeyError-swallowed failure.
    assert loaded["config"] == mod._synth_config("40k", "v2")
    assert loaded["config"][-1] == 40000
    assert loaded["sr"] == "40k"  # string key, not int 40000

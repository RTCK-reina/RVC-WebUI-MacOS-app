"""tests/test_device.py

infer/lib/device.py のユニットテスト。

テスト対象:
- empty_device_cache: MPS 利用可否に応じた動作
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from infer.lib import device as device_module  # noqa: E402


class TestEmptyDeviceCache:
    def test_calls_mps_empty_cache_when_mps_available(self):
        """MPS が利用可能な場合は torch.mps.empty_cache() を呼ぶ。"""
        with patch.object(
            sys.modules["torch"].backends.mps,
            "is_available",
            return_value=True,
        ):
            with patch.object(
                sys.modules["torch"].mps,
                "empty_cache",
            ) as mock_empty:
                device_module.empty_device_cache()
                mock_empty.assert_called_once()

    def test_noop_when_mps_not_available(self):
        """MPS が利用不可の場合は何も呼ばない（CPU-only）。"""
        with patch.object(
            sys.modules["torch"].backends.mps,
            "is_available",
            return_value=False,
        ):
            with patch.object(
                sys.modules["torch"].mps,
                "empty_cache",
            ) as mock_empty:
                device_module.empty_device_cache()
                mock_empty.assert_not_called()

    @pytest.mark.skipif(
        not __import__("sys")
        .modules.get("torch", MagicMock())
        .backends.mps.is_available(),
        reason="MPS not available on this machine",
    )
    def test_real_mps_empty_cache(self):
        """実際の MPS が存在する場合の結合テスト（Apple Silicon のみ実行）。"""
        import torch

        # 例外が出なければ OK
        torch.mps.empty_cache()

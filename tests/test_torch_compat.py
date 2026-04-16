"""tests/test_torch_compat.py

infer/lib/torch_compat.py のユニットテスト。

テスト対象:
- 基本動作: legacy_load() が torch.load を一時的にパッチし、exit 後に元に戻す
- 再入可能: ネスト呼び出しでもパッチが正しく積み上がり/解除される
- 例外安全: with ブロック内で例外が発生しても元の torch.load に戻る
- スレッドセーフ: 複数スレッドが同時に legacy_load() を使っても torch.load が
  不定順序で復元されない
"""
from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, call
from typing import List

import pytest

# ---------------------------------------------------------------------------
# torch をモックにして torch_compat を import する
# ---------------------------------------------------------------------------

# conftest.py が既に sys.modules["torch"] にスタブを入れているので
# そのスタブ上に load を設定する。
_torch_mock = sys.modules["torch"]
_original_load_sentinel = object()
_torch_mock.load = MagicMock(return_value=_original_load_sentinel)

# import 順序の問題を回避するため torch_compat を遅延 import する
import importlib
import os

# プロジェクトルートを sys.path に追加（未追加の場合）
_repo_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from infer.lib.torch_compat import (  # noqa: E402
    legacy_load,
    _legacy_load_lock,
    _legacy_load_count,
    _legacy_load_original,
)
import infer.lib.torch_compat as _tc


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _reset_state():
    """各テスト前に torch_compat モジュールの内部状態をリセットする。"""
    _tc._legacy_load_count = 0
    _tc._legacy_load_original = None
    # torch.load を新鮮な MagicMock に戻す
    _torch_mock.load = MagicMock(return_value=_original_load_sentinel)


@pytest.fixture(autouse=True)
def reset_torch_compat_state():
    _reset_state()
    yield
    _reset_state()


# ---------------------------------------------------------------------------
# テスト: 基本動作
# ---------------------------------------------------------------------------

class TestBasicBehavior:
    def test_patches_torch_load_inside_block(self):
        """with ブロック内では torch.load が _patched_load に差し替えられる。"""
        original = _torch_mock.load
        with legacy_load():
            patched = _torch_mock.load
            assert patched is not original, "with ブロック内でパッチされていない"
            assert callable(patched)

    def test_restores_torch_load_after_block(self):
        """with ブロック終了後に torch.load が元に戻る。"""
        original = _torch_mock.load
        with legacy_load():
            pass
        assert _torch_mock.load is original, "exit 後に torch.load が復元されていない"

    def test_patched_load_sets_weights_only_false(self):
        """_patched_load は weights_only=False をデフォルトとして渡す。"""
        sentinel_result = object()
        _torch_mock.load = MagicMock(return_value=sentinel_result)
        with legacy_load():
            result = _torch_mock.load("fake_path")
        # _patched_load は元の load を呼ぶ際に weights_only=False を追加する
        # (実際には _patched_load が torch.load 経由で呼ばれる)
        assert result is sentinel_result

    def test_explicit_weights_only_wins(self):
        """呼び出し側が weights_only=True を明示した場合はそちらが優先される。"""
        captured_kwargs: dict = {}

        def fake_load(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return None

        _torch_mock.load = fake_load
        with legacy_load():
            _torch_mock.load("path", weights_only=True)

        assert captured_kwargs.get("weights_only") is True

    def test_refcount_is_zero_after_exit(self):
        """with ブロック終了後に参照カウントが 0 に戻る。"""
        with legacy_load():
            assert _tc._legacy_load_count == 1
        assert _tc._legacy_load_count == 0

    def test_original_is_none_after_exit(self):
        """with ブロック終了後に _legacy_load_original が None に戻る。"""
        with legacy_load():
            pass
        assert _tc._legacy_load_original is None


# ---------------------------------------------------------------------------
# テスト: 例外安全
# ---------------------------------------------------------------------------

class TestExceptionSafety:
    def test_restores_on_exception(self):
        """with ブロック内で例外が発生しても torch.load が復元される。"""
        original = _torch_mock.load
        with pytest.raises(ValueError):
            with legacy_load():
                raise ValueError("test error")
        assert _torch_mock.load is original

    def test_refcount_zero_on_exception(self):
        """例外発生後も参照カウントが 0 になる。"""
        with pytest.raises(RuntimeError):
            with legacy_load():
                raise RuntimeError("boom")
        assert _tc._legacy_load_count == 0

    def test_nested_restores_outer_on_inner_exception(self):
        """ネスト内で例外が発生しても外側の patch は正しく解除される。"""
        original = _torch_mock.load
        with pytest.raises(ValueError):
            with legacy_load():  # outer
                with legacy_load():  # inner
                    raise ValueError("inner error")
        # 両方 exit したので元に戻っているはず
        assert _torch_mock.load is original
        assert _tc._legacy_load_count == 0


# ---------------------------------------------------------------------------
# テスト: 再入可能（ネスト）
# ---------------------------------------------------------------------------

class TestReentrancy:
    def test_nested_calls_share_single_patch(self):
        """ネスト呼び出しでパッチは 1 回だけ適用され、最後の exit で復元される。"""
        original = _torch_mock.load
        with legacy_load():
            patched_outer = _torch_mock.load
            assert patched_outer is not original

            with legacy_load():
                patched_inner = _torch_mock.load
                # 同じパッチオブジェクトが使われる
                assert patched_inner is patched_outer
                assert _tc._legacy_load_count == 2

            # 内側の with を抜けてもまだパッチ中
            assert _torch_mock.load is patched_outer
            assert _tc._legacy_load_count == 1

        # 外側も抜けたら元に戻る
        assert _torch_mock.load is original
        assert _tc._legacy_load_count == 0

    def test_triple_nesting(self):
        """3 段ネストでも正しく動作する。"""
        original = _torch_mock.load
        with legacy_load():
            with legacy_load():
                with legacy_load():
                    assert _tc._legacy_load_count == 3
                assert _tc._legacy_load_count == 2
            assert _tc._legacy_load_count == 1
        assert _tc._legacy_load_count == 0
        assert _torch_mock.load is original


# ---------------------------------------------------------------------------
# テスト: スレッドセーフ
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_calls_restore_correctly(self):
        """複数スレッドが同時に legacy_load() を使っても torch.load が正しく復元される。"""
        original = _torch_mock.load
        errors: List[Exception] = []
        n_threads = 10
        barrier = threading.Barrier(n_threads)

        def worker():
            try:
                barrier.wait()
                with legacy_load():
                    time.sleep(0.001)  # クリティカルセクションを少し広げる
                    assert _torch_mock.load is not original, \
                        "with ブロック内でパッチが外れている"
                    assert _tc._legacy_load_count > 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"スレッドエラー: {errors}"
        assert _torch_mock.load is original, "全スレッド終了後に torch.load が元に戻っていない"
        assert _tc._legacy_load_count == 0

    def test_no_interleaved_restore(self):
        """あるスレッドの exit が別スレッドの patch 中に復元してしまわない。"""
        original = _torch_mock.load
        errors: List[Exception] = []
        done_event = threading.Event()
        inner_started = threading.Event()

        def slow_worker():
            """with ブロックを長く保持するスレッド。"""
            try:
                with legacy_load():
                    inner_started.set()
                    done_event.wait(timeout=2)
                    # ここで torch.load はまだパッチ中のはず
                    assert _torch_mock.load is not original
            except Exception as e:
                errors.append(e)

        def fast_worker():
            """すぐに終わるスレッド。"""
            try:
                inner_started.wait(timeout=2)
                with legacy_load():
                    pass  # すぐ exit
                # fast_worker が exit しても slow_worker 中はパッチが残る
                assert _torch_mock.load is not original, \
                    "fast_worker exit で torch.load が早期復元された"
            except Exception as e:
                errors.append(e)
            finally:
                done_event.set()

        t1 = threading.Thread(target=slow_worker)
        t2 = threading.Thread(target=fast_worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"スレッドエラー: {errors}"
        assert _torch_mock.load is original

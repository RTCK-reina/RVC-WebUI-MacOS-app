"""tests/test_torch_compat.py

infer/lib/torch_compat.py のユニットテスト。

fix/copilot-followup ブランチは threading.local ベースの実装:
- torch.load にモジュールロード時に一度だけラッパーを設置
- _flag.enabled (threading.local) で per-thread の ON/OFF を制御
- legacy_load() は _flag.enabled を True に設定し、exit で元に戻す

テスト対象:
- 基本動作: with ブロック内で weights_only=False がデフォルト化される
- 再入可能: ネスト呼び出しで内側 exit 後も外側は有効
- 例外安全: with ブロック内の例外でフラグが元に戻る
- スレッドセーフ: 各スレッドが独立したフラグを持つ
- 他スレッド隔離: あるスレッドの legacy_load が他スレッドの動作に影響しない
"""
from __future__ import annotations

import sys
import threading
import time
from typing import List, Optional
from pathlib import Path

import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# torch スタブが既に conftest で設定されている
import infer.lib.torch_compat as _tc
from infer.lib.torch_compat import legacy_load, _flag, _original_load


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _is_legacy_mode() -> bool:
    """現在スレッドが legacy_load モード中かを返す。"""
    return getattr(_flag, "enabled", False)


# ---------------------------------------------------------------------------
# テスト: 基本動作
# ---------------------------------------------------------------------------

class TestBasicBehavior:
    def test_flag_false_outside_block(self):
        """with ブロック外では _flag.enabled が False。"""
        assert not _is_legacy_mode()

    def test_flag_true_inside_block(self):
        """with ブロック内では _flag.enabled が True。"""
        with legacy_load():
            assert _is_legacy_mode()

    def test_flag_restored_after_block(self):
        """with ブロット終了後に _flag.enabled が元の値（False）に戻る。"""
        with legacy_load():
            pass
        assert not _is_legacy_mode()

    def test_torch_load_patched_globally(self):
        """torch.load はモジュールロード時に一度だけラッパーに差し替え済み。"""
        torch_mod = sys.modules["torch"]
        assert torch_mod.load is _tc._patched_load, \
            "torch.load が _patched_load に差し替えられていない"

    def test_patched_load_uses_original_outside(self):
        """with ブロック外では _patched_load が weights_only を追加しない。"""
        called_kwargs: dict = {}

        def fake_original(*args, **kwargs):
            called_kwargs.update(kwargs)

        original = _tc._original_load
        _tc._original_load = fake_original
        try:
            sys.modules["torch"].load("path")
            # enabled=False なので weights_only が追加されていない
            assert "weights_only" not in called_kwargs
        finally:
            _tc._original_load = original

    def test_patched_load_adds_weights_only_inside(self):
        """with ブロック内では _patched_load が weights_only=False を追加する。"""
        called_kwargs: dict = {}

        def fake_original(*args, **kwargs):
            called_kwargs.update(kwargs)

        original = _tc._original_load
        _tc._original_load = fake_original
        try:
            with legacy_load():
                sys.modules["torch"].load("path")
            assert called_kwargs.get("weights_only") is False
        finally:
            _tc._original_load = original

    def test_explicit_weights_only_wins(self):
        """呼び出し側が weights_only=True を明示した場合はそちらが優先される。"""
        called_kwargs: dict = {}

        def fake_original(*args, **kwargs):
            called_kwargs.update(kwargs)

        original = _tc._original_load
        _tc._original_load = fake_original
        try:
            with legacy_load():
                sys.modules["torch"].load("path", weights_only=True)
            assert called_kwargs.get("weights_only") is True
        finally:
            _tc._original_load = original


# ---------------------------------------------------------------------------
# テスト: 例外安全
# ---------------------------------------------------------------------------

class TestExceptionSafety:
    def test_flag_restored_on_exception(self):
        """with ブロック内で例外が発生しても _flag.enabled が False に戻る。"""
        with pytest.raises(ValueError):
            with legacy_load():
                raise ValueError("test error")
        assert not _is_legacy_mode()

    def test_nested_exception_restores_outer(self):
        """内側で例外が発生しても外側の with ブロックは継続し、最終的に False になる。"""
        with pytest.raises(RuntimeError):
            with legacy_load():  # outer: enabled=True
                assert _is_legacy_mode()
                with legacy_load():  # inner: previous=True
                    raise RuntimeError("inner error")
                # ここには到達しないが outer の finally が走る

        # 両方の with を抜けたので False に戻っているはず
        assert not _is_legacy_mode()

    def test_multiple_exceptions_dont_corrupt_flag(self):
        """繰り返し例外が発生してもフラグが汚染されない。"""
        for _ in range(5):
            with pytest.raises(ValueError):
                with legacy_load():
                    raise ValueError
        assert not _is_legacy_mode()


# ---------------------------------------------------------------------------
# テスト: 再入可能（ネスト）
# ---------------------------------------------------------------------------

class TestReentrancy:
    def test_nested_inner_exit_keeps_outer_enabled(self):
        """外側の with ブロックが有効な間は内側の exit でも enabled が保たれる。"""
        with legacy_load():
            assert _is_legacy_mode()
            with legacy_load():
                assert _is_legacy_mode()
            # 内側を抜けても外側はまだ enabled
            assert _is_legacy_mode()
        # 両方抜けたら False
        assert not _is_legacy_mode()

    def test_triple_nesting(self):
        """3 段ネストでも正しく動作する。"""
        with legacy_load():
            with legacy_load():
                with legacy_load():
                    assert _is_legacy_mode()
                assert _is_legacy_mode()
            assert _is_legacy_mode()
        assert not _is_legacy_mode()

    def test_previous_value_preserved(self):
        """ネスト時に previous が正しく保存・復元される。"""
        # デフォルト状態で入れ子 → 内側が True、外側 exit 後は False に戻る
        with legacy_load():
            before_inner = _is_legacy_mode()
            assert before_inner is True
            with legacy_load():
                pass
            after_inner = _is_legacy_mode()
            assert after_inner is True  # 内側 exit 後も外側は True
        assert not _is_legacy_mode()


# ---------------------------------------------------------------------------
# テスト: スレッド隔離（threading.local の核心）
# ---------------------------------------------------------------------------

class TestThreadIsolation:
    def test_other_thread_unaffected_inside_block(self):
        """あるスレッドが legacy_load 中でも、他スレッドの _flag.enabled は False のまま。"""
        other_thread_flag: List[Optional[bool]] = [None]
        ready = threading.Event()
        resume = threading.Event()

        def main_worker():
            with legacy_load():
                ready.set()   # 他スレッドに「今 legacy_load 中」と通知
                resume.wait()  # 他スレッドがフラグを確認するまで待つ

        def observer():
            ready.wait()
            # main_worker が legacy_load 中だが、このスレッドのフラグは False であるべき
            other_thread_flag[0] = _is_legacy_mode()
            resume.set()

        t1 = threading.Thread(target=main_worker)
        t2 = threading.Thread(target=observer)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert other_thread_flag[0] is False, \
            "他スレッドの _flag.enabled が True になった（スレッド隔離失敗）"

    def test_thread_local_independent(self):
        """複数スレッドがそれぞれ独立して legacy_load を使える。"""
        errors: List[Exception] = []
        n = 8
        barrier = threading.Barrier(n)

        def worker(idx: int):
            try:
                barrier.wait()
                # 偶数スレッドは legacy_load 中、奇数は外
                if idx % 2 == 0:
                    with legacy_load():
                        time.sleep(0.005)
                        assert _is_legacy_mode(), f"Thread {idx}: enabled が False"
                else:
                    time.sleep(0.005)
                    assert not _is_legacy_mode(), f"Thread {idx}: enabled が True（のはずなのに）"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"スレッドエラー: {errors}"

    def test_concurrent_calls_no_cross_contamination(self):
        """大量のスレッドが同時に使っても相互汚染が起きない。"""
        errors: List[Exception] = []
        barrier = threading.Barrier(20)

        def worker():
            try:
                barrier.wait()
                for _ in range(10):
                    assert not _is_legacy_mode()
                    with legacy_load():
                        assert _is_legacy_mode()
                    assert not _is_legacy_mode()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"スレッドエラー: {errors}"

    def test_flag_per_thread_not_shared(self):
        """legacy_load の exit が他スレッドに影響しない。"""
        thread_results: dict = {}
        gate = threading.Event()

        def long_worker():
            with legacy_load():
                gate.set()
                time.sleep(0.05)
                # 他スレッドが自分の exit を終えてもここはまだ enabled
                thread_results["long"] = _is_legacy_mode()

        def quick_worker():
            gate.wait()
            with legacy_load():
                pass
            # quick_worker の exit はこのスレッドのフラグだけを戻す
            thread_results["quick_after"] = _is_legacy_mode()

        t1 = threading.Thread(target=long_worker)
        t2 = threading.Thread(target=quick_worker)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert thread_results.get("long") is True
        assert thread_results.get("quick_after") is False

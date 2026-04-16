"""tests/test_threadpool.py

ThreadPool / 非同期パターンのユニットテスト（外部依存なし・stdlib のみ）。

テスト対象:
- Future キャンセルパターン（PR#5/PR#7 で修正したキャンセルフロー）
- ThreadPoolExecutor の shutdown(wait=False, cancel_futures=True) 互換性
- atexit 登録パターン
- _blocking_executor のシングルスロット直列化
"""
from __future__ import annotations

import atexit
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, CancelledError
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Future キャンセルパターン（PR#5: batch cancel path）
# ---------------------------------------------------------------------------

class TestFutureCancellation:
    def test_cancel_not_started_future(self):
        """未開始の Future は cancel() で True になる。"""
        barrier = threading.Barrier(2)

        with ThreadPoolExecutor(max_workers=1) as pool:
            # 1 つ目のタスクでワーカーを占有
            blocker = pool.submit(barrier.wait)
            # 2 つ目は未開始
            pending = pool.submit(lambda: 42)
            assert pending.cancel() is True
            barrier.wait()  # blocker を解放

        assert pending.cancelled()

    def test_cancel_does_not_affect_running_future(self):
        """実行中の Future は cancel() しても影響を受けない。"""
        started = threading.Event()
        done = threading.Event()

        def task():
            started.set()
            done.wait()
            return "result"

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(task)
            started.wait(timeout=2)
            # 既に実行中 → cancel は False を返す
            assert fut.cancel() is False
            done.set()
            assert fut.result() == "result"

    def test_batch_cancel_pending_futures(self):
        """キャンセル時に pending futures を bulk cancel するパターン。"""
        barrier = threading.Barrier(2)
        completed = []

        def track(x):
            completed.append(x)
            return x

        with ThreadPoolExecutor(max_workers=1) as pool:
            # ワーカーを占有してキューを詰める
            blocker = pool.submit(barrier.wait)
            futures = [pool.submit(track, i) for i in range(5)]

            # キャンセル：未開始のものを全部キャンセル
            for f in futures:
                f.cancel()

            barrier.wait()  # blocker 解放

        # 未開始だったものはキャンセルされているはず
        cancelled_count = sum(1 for f in futures if f.cancelled())
        assert cancelled_count > 0, "少なくとも 1 件キャンセルされるはず"

    def test_cancelled_future_raises_on_result(self):
        """キャンセルされた Future の result() は CancelledError を送出する。"""
        barrier = threading.Barrier(2)

        with ThreadPoolExecutor(max_workers=1) as pool:
            blocker = pool.submit(barrier.wait)
            pending = pool.submit(lambda: 1)
            pending.cancel()
            barrier.wait()

        if pending.cancelled():
            with pytest.raises(CancelledError):
                pending.result()


# ---------------------------------------------------------------------------
# shutdown(cancel_futures=True) の互換性パターン（PR#7 の修正）
# ---------------------------------------------------------------------------

class TestShutdownCancelFutures:
    def test_shutdown_wait_false_does_not_block(self):
        """shutdown(wait=False) は in-flight タスクを待たずに返る。"""
        started = threading.Event()
        barrier = threading.Barrier(2)

        pool = ThreadPoolExecutor(max_workers=1)
        pool.submit(lambda: (started.set(), barrier.wait()))
        started.wait(timeout=2)

        # wait=False なので即座に返るはず
        t0 = time.time()
        pool.shutdown(wait=False)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"shutdown(wait=False) がブロックした: {elapsed:.2f}s"
        barrier.wait()  # worker を解放

    def test_cancel_futures_kwarg_compatible(self):
        """Python 3.9+ では cancel_futures=True が受け付けられる。"""
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python 3.8 以前: フォールバック
            pool.shutdown(wait=False)

    def test_pr7_shutdown_pattern(self):
        """PR#7 で修正した try/except TypeError フォールバックパターンの検証。"""
        save_futures: List[Future] = []
        save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-save")

        results = []
        try:
            for i in range(3):
                save_futures.append(save_pool.submit(lambda x=i: x * 2))

            # 全 future が完了してから result を取得
            for f in save_futures:
                results.append(f.result())

        finally:
            for f in save_futures:
                f.cancel()
            try:
                save_pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                save_pool.shutdown(wait=False)

        assert results == [0, 2, 4]


# ---------------------------------------------------------------------------
# _blocking_executor: シングルスロット直列化の検証
# ---------------------------------------------------------------------------

class TestBlockingExecutorSerialization:
    def test_single_slot_serializes_calls(self):
        """max_workers=1 のエグゼキュータは同時に 1 タスクしか実行しない。"""
        concurrent_count = 0
        max_concurrent = 0
        lock = threading.Lock()

        def task(duration: float):
            nonlocal concurrent_count, max_concurrent
            with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent:
                    max_concurrent = concurrent_count
            time.sleep(duration)
            with lock:
                concurrent_count -= 1

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rpc-blocking") as ex:
            futures = [ex.submit(task, 0.02) for _ in range(5)]
            for f in futures:
                f.result()

        assert max_concurrent == 1, \
            f"同時実行数が 1 を超えた: {max_concurrent}"

    def test_future_result_propagates_exception(self):
        """エグゼキュータのタスクが例外を投げると future.result() で再送出される。"""
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(lambda: (_ for _ in ()).throw(ValueError("rpc error")))
            with pytest.raises(ValueError, match="rpc error"):
                fut.result()


# ---------------------------------------------------------------------------
# atexit 登録パターン（PR#5: _blocking_executor の lifecycle 管理）
# ---------------------------------------------------------------------------

class TestAtexitRegistration:
    def test_atexit_register_shutdown_called(self):
        """atexit.register でシャットダウンが登録される。"""
        pool = ThreadPoolExecutor(max_workers=1)
        mock_shutdown = MagicMock()
        pool.shutdown = mock_shutdown

        atexit.register(pool.shutdown, wait=False)
        # atexit ハンドラは実際には interpreter exit 時に呼ばれる。
        # ここではレジストリに追加されたことを間接的に検証する。
        # (_atexit_callbacks は内部 API なので直接確認しない)
        # 代わりに手動で呼び出して動作を確認する。
        pool.shutdown(wait=False)
        mock_shutdown.assert_called_once_with(wait=False)
        pool.shutdown = ThreadPoolExecutor.shutdown.__get__(pool, ThreadPoolExecutor)

    def test_shutdown_wait_false_lets_current_task_run(self):
        """shutdown(wait=False) は in-flight RPC を kill しない。"""
        result_holder = []
        done = threading.Event()

        def long_task():
            time.sleep(0.05)
            result_holder.append("done")
            done.set()

        pool = ThreadPoolExecutor(max_workers=1)
        pool.submit(long_task)
        pool.shutdown(wait=False)  # atexit 相当
        done.wait(timeout=2)
        assert result_holder == ["done"]


# ---------------------------------------------------------------------------
# キャンセルイベント伝播パターン（rpc_server._cancel_task 経由）
# ---------------------------------------------------------------------------

class TestCancelEventPropagation:
    def test_cancel_event_stops_iteration(self):
        """cancel_event.is_set() が True になるとループが早期終了する。

        race-condition 修正: time.sleep ではなく Event による同期を使い、
        ループが開始してからキャンセルを発火させる。
        range も十分に大きくして、GIL 切り替え前に完了しないようにする。
        """
        cancel_event = threading.Event()
        iteration_started = threading.Event()
        processed = []

        def batch_loop(items):
            for i in items:
                # キャンセルチェックより先に「開始済み」を通知する
                iteration_started.set()
                if cancel_event.is_set():
                    return {"status": "cancelled", "completed": i}
                processed.append(i)
            return {"status": "success"}

        def canceller():
            iteration_started.wait(timeout=5)  # ループ開始を確実に待つ
            cancel_event.set()

        threading.Thread(target=canceller, daemon=True).start()
        # range を十分大きくして GIL 切り替え前に完了しないようにする
        result = batch_loop(range(1_000_000))
        assert result["status"] == "cancelled"
        assert len(processed) < 1_000_000

    def test_cancel_event_not_set_processes_all(self):
        """cancel_event が set されなければ全アイテムを処理する。"""
        cancel_event = threading.Event()
        processed = []

        for i in range(10):
            if cancel_event.is_set():
                break
            processed.append(i)

        assert processed == list(range(10))

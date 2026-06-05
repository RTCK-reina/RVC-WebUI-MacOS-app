"""tests/test_rpc_runtime.py

rpc_server.py の新しいランタイム挙動（純粋ロジック）の回帰テスト。

カバー対象:
- _send / send_notification のバウンド付き write queue とドロップ方針
  （realtime_metrics / resource_stats / progress は満杯時に破棄、
   レスポンス・エラー・一回限りの通知は温存）。

子プロセス回収（_reap_child_processes）と graceful shutdown / SIGTERM ハンドラは
実 psutil が必要で conftest スタブ下では検証できないため、実サーバを起動する
tests/test_rpc_integration.py 側で end-to-end に検証している。

conftest.py が torch をスタブするので real torch は不要。
"""

from __future__ import annotations

import json

import pytest

# conftest stub の後に import（rpc_server は module-level で重い import を行う）。
import rpc_server  # noqa: E402


class TestWriteQueueDropPolicy:
    @pytest.fixture
    def bounded_queue(self, monkeypatch):
        from queue import Queue

        q = Queue(maxsize=2)
        monkeypatch.setattr(rpc_server, "_write_queue", q)
        return q

    def test_disposable_notification_dropped_when_full(self, bounded_queue):
        q = bounded_queue
        # 1 件目（disposable）は空きがあるので入る。
        rpc_server.send_notification("realtime_metrics", {"n": 1})
        assert q.qsize() == 1
        # lossless なレスポンスも入って満杯に。
        rpc_server.send_response(7, {"ok": True})
        assert q.qsize() == 2
        # 満杯時、disposable は無言で破棄される（ブロックも例外もなし）。
        rpc_server.send_notification("resource_stats", {"n": 2})
        rpc_server.send_notification("progress", {"task_id": "t", "percent": 50})
        assert q.qsize() == 2

        # 中身と順序を確認: 落ちたのは disposable のみ、レスポンスは温存。
        first = json.loads(q.get())
        second = json.loads(q.get())
        assert first["method"] == "realtime_metrics"
        assert second["result"] == {"ok": True}

    def test_disposable_set_matches_methods(self):
        assert rpc_server._DROPPABLE_NOTIFICATIONS == {
            "realtime_metrics",
            "resource_stats",
            "progress",
        }

    def test_non_disposable_notification_is_lossless(self, bounded_queue):
        q = bounded_queue
        # status / realtime_event 等はドロップ対象外。空きがある限り必ず入る。
        rpc_server.send_notification("status", {"status": "idle"})
        rpc_server.send_notification("realtime_event", {"kind": "started"})
        assert q.qsize() == 2
        methods = {json.loads(q.get())["method"] for _ in range(2)}
        assert methods == {"status", "realtime_event"}

    def test_queue_is_bounded(self):
        assert rpc_server._write_queue.maxsize == 4096

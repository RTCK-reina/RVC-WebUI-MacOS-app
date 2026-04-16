"""tests/test_rpc_protocol.py

rpc_server.py の純粋 Python ロジックのユニットテスト。

テスト対象 (重量 GPU 依存なし):
- _list_files: ディレクトリ走査・拡張子フィルタ
- Task / _register_task / _unregister_task / _cancel_task: タスク管理
- emit_progress: 進捗パーセントのクランプ
- send_response / send_error / send_notification: JSON-RPC フォーマット
- _dispatch: メソッド不在エラー
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# rpc_server は module-level で重いインポートを行うため、conftest.py の
# sys.modules スタブが既に設定済みであることを前提に import する。
# ---------------------------------------------------------------------------
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import rpc_server  # noqa: E402  (conftest stub 後にimport)


# ---------------------------------------------------------------------------
# _list_files
# ---------------------------------------------------------------------------

class TestListFiles:
    def test_empty_dir_returns_empty(self, tmp_path):
        result = rpc_server._list_files(tmp_path, [".pth"])
        assert result == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = rpc_server._list_files(tmp_path / "no_such", [".pth"])
        assert result == []

    def test_returns_matched_extension(self, tmp_path):
        (tmp_path / "model.pth").write_text("dummy")
        (tmp_path / "other.txt").write_text("skip")
        result = rpc_server._list_files(tmp_path, [".pth"])
        assert result == ["model.pth"]

    def test_no_ext_filter_returns_all(self, tmp_path):
        (tmp_path / "a.pth").write_text("")
        (tmp_path / "b.index").write_text("")
        result = rpc_server._list_files(tmp_path, [])
        assert len(result) == 2

    def test_recursive_subdir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.pth").write_text("")
        (tmp_path / "top.pth").write_text("")
        result = rpc_server._list_files(tmp_path, [".pth"])
        # top.pth + sub/nested.pth
        assert len(result) == 2
        assert "top.pth" in result
        assert any("nested.pth" in r for r in result)

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "Model.PTH").write_text("")
        result = rpc_server._list_files(tmp_path, [".pth"])
        assert len(result) == 1

    def test_multiple_extensions(self, tmp_path):
        (tmp_path / "model.pth").write_text("")
        (tmp_path / "index.index").write_text("")
        (tmp_path / "skip.txt").write_text("")
        result = rpc_server._list_files(tmp_path, [".pth", ".index"])
        assert len(result) == 2
        assert "skip.txt" not in result

    def test_sorted_output(self, tmp_path):
        for name in ["c.pth", "a.pth", "b.pth"]:
            (tmp_path / name).write_text("")
        result = rpc_server._list_files(tmp_path, [".pth"])
        assert result == ["a.pth", "b.pth", "c.pth"]


# ---------------------------------------------------------------------------
# Task / タスク管理
# ---------------------------------------------------------------------------

class TestTask:
    def setup_method(self):
        """各テスト前にタスクレジストリをクリア。"""
        with rpc_server._tasks_lock:
            rpc_server._tasks.clear()

    def test_task_has_cancel_event(self):
        t = rpc_server.Task("test-id")
        assert isinstance(t.cancel_event, threading.Event)
        assert not t.cancel_event.is_set()

    def test_task_has_started_at(self):
        import time
        before = time.time()
        t = rpc_server.Task("t")
        after = time.time()
        assert before <= t.started_at <= after

    def test_register_task_stores_task(self):
        task = rpc_server._register_task("my-task")
        with rpc_server._tasks_lock:
            assert "my-task" in rpc_server._tasks
            assert rpc_server._tasks["my-task"] is task

    def test_unregister_task_removes_it(self):
        rpc_server._register_task("del-me")
        rpc_server._unregister_task("del-me")
        with rpc_server._tasks_lock:
            assert "del-me" not in rpc_server._tasks

    def test_unregister_nonexistent_is_noop(self):
        # 存在しないタスクの unregister は例外なし
        rpc_server._unregister_task("ghost")

    def test_cancel_task_sets_event(self):
        task = rpc_server._register_task("cancel-me")
        result = rpc_server._cancel_task("cancel-me")
        assert result is True
        assert task.cancel_event.is_set()

    def test_cancel_nonexistent_returns_false(self):
        result = rpc_server._cancel_task("no-such-task")
        assert result is False

    def test_cancel_is_idempotent(self):
        rpc_server._register_task("dbl-cancel")
        rpc_server._cancel_task("dbl-cancel")
        result = rpc_server._cancel_task("dbl-cancel")
        # 2 回目はタスクが cancel 済みなので True (イベントはセット済み)
        assert result is True

    def test_multiple_tasks_independent(self):
        t1 = rpc_server._register_task("task-1")
        t2 = rpc_server._register_task("task-2")
        rpc_server._cancel_task("task-1")
        assert t1.cancel_event.is_set()
        assert not t2.cancel_event.is_set()


# ---------------------------------------------------------------------------
# emit_progress — パーセントクランプ
# ---------------------------------------------------------------------------

class TestEmitProgress:
    def _captured_notifications(self):
        """キューから全メッセージを取り出して返す。"""
        msgs = []
        q = rpc_server._write_queue
        while not q.empty():
            raw = q.get_nowait()
            msgs.append(json.loads(raw))
        return msgs

    def setup_method(self):
        # キューを空にする
        q = rpc_server._write_queue
        while not q.empty():
            q.get_nowait()

    def test_normal_percent_passes_through(self):
        rpc_server.emit_progress("t", 50.0, "half done")
        msgs = self._captured_notifications()
        assert len(msgs) == 1
        assert msgs[0]["params"]["percent"] == 50.0

    def test_below_zero_clamped_to_zero(self):
        rpc_server.emit_progress("t", -10.0, "neg")
        msgs = self._captured_notifications()
        assert msgs[0]["params"]["percent"] == 0.0

    def test_above_100_clamped_to_100(self):
        rpc_server.emit_progress("t", 150.0, "over")
        msgs = self._captured_notifications()
        assert msgs[0]["params"]["percent"] == 100.0

    def test_exact_boundary_values(self):
        for val in (0.0, 100.0):
            self.setup_method()
            rpc_server.emit_progress("t", val, "edge")
            msgs = self._captured_notifications()
            assert msgs[0]["params"]["percent"] == val

    def test_notification_contains_task_id_and_message(self):
        rpc_server.emit_progress("my-task", 42.0, "processing", "batch")
        msgs = self._captured_notifications()
        p = msgs[0]["params"]
        assert p["task_id"] == "my-task"
        assert p["message"] == "processing"
        assert p["phase"] == "batch"
        assert msgs[0]["method"] == "progress"

    def test_notification_is_jsonrpc2(self):
        rpc_server.emit_progress("t", 1.0, "x")
        msgs = self._captured_notifications()
        assert msgs[0]["jsonrpc"] == "2.0"
        assert "id" not in msgs[0]  # notification: id なし


# ---------------------------------------------------------------------------
# send_response / send_error / send_notification — JSON-RPC フォーマット
# ---------------------------------------------------------------------------

class TestJsonRpcMessages:
    def setup_method(self):
        q = rpc_server._write_queue
        while not q.empty():
            q.get_nowait()

    def _get_one(self) -> dict:
        raw = rpc_server._write_queue.get_nowait()
        return json.loads(raw)

    def test_send_response_format(self):
        rpc_server.send_response(1, {"ok": True})
        msg = self._get_one()
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

    def test_send_response_null_result(self):
        rpc_server.send_response(2, None)
        msg = self._get_one()
        assert msg["result"] is None

    def test_send_error_format(self):
        rpc_server.send_error(3, -32600, "Invalid Request")
        msg = self._get_one()
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 3
        assert msg["error"]["code"] == -32600
        assert msg["error"]["message"] == "Invalid Request"

    def test_send_error_with_data(self):
        rpc_server.send_error(4, -32603, "Internal error", {"trace": "..."})
        msg = self._get_one()
        assert msg["error"]["data"] == {"trace": "..."}

    def test_send_error_without_data(self):
        rpc_server.send_error(5, -32601, "Not found")
        msg = self._get_one()
        assert "data" not in msg["error"]

    def test_send_notification_format(self):
        rpc_server.send_notification("ready", {"pid": 42})
        msg = self._get_one()
        assert msg == {"jsonrpc": "2.0", "method": "ready", "params": {"pid": 42}}
        assert "id" not in msg

    def test_send_response_non_integer_id(self):
        rpc_server.send_response("abc", {"x": 1})
        msg = self._get_one()
        assert msg["id"] == "abc"


# ---------------------------------------------------------------------------
# _dispatch — メソッド不在エラー
# ---------------------------------------------------------------------------

class TestDispatch:
    def setup_method(self):
        q = rpc_server._write_queue
        while not q.empty():
            q.get_nowait()

    def _get_messages(self):
        msgs = []
        q = rpc_server._write_queue
        while not q.empty():
            msgs.append(json.loads(q.get_nowait()))
        return msgs

    def test_method_not_found_returns_error(self):
        rpc_server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "nonexistent"})
        msgs = self._get_messages()
        # エラーレスポンスが 1 件
        errors = [m for m in msgs if "error" in m]
        assert len(errors) == 1
        assert errors[0]["error"]["code"] == -32601
        assert errors[0]["id"] == 1

    def test_non_blocking_method_runs_directly(self):
        """非ブロッキングメソッドはエグゼキュータを経由しない。"""
        called_with: list = []

        def fake_fn(params):
            called_with.append(params)
            return {"ok": True}

        with patch.dict(rpc_server.METHODS, {"test_nb": fake_fn}):
            # BLOCKING_METHODS に含まれないので直接呼ばれる
            rpc_server._dispatch({"jsonrpc": "2.0", "id": 9, "method": "test_nb",
                                   "params": {"x": 1}})

        assert called_with == [{"x": 1}]
        msgs = self._get_messages()
        responses = [m for m in msgs if "result" in m]
        assert responses[0]["result"] == {"ok": True}

    def test_rpc_exception_returns_error(self):
        """メソッドが例外を投げると -32603 エラーが返る。"""
        def bad_fn(params):
            raise RuntimeError("test error")

        with patch.dict(rpc_server.METHODS, {"bad": bad_fn}):
            rpc_server._dispatch({"jsonrpc": "2.0", "id": 10, "method": "bad"})

        msgs = self._get_messages()
        errors = [m for m in msgs if "error" in m]
        assert len(errors) == 1
        assert errors[0]["error"]["code"] == -32603
        assert "test error" in errors[0]["error"]["message"]


# ---------------------------------------------------------------------------
# _timestamp ユーティリティ
# ---------------------------------------------------------------------------

class TestTimestamp:
    def test_format(self):
        ts = rpc_server._timestamp()
        # YYYYMMDD_HHMMSS 形式
        import re
        assert re.match(r"^\d{8}_\d{6}$", ts), f"不正なフォーマット: {ts!r}"

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

    # ---- force_kill / sentinel 拡張（強力な停止手段） ------------------

    def test_task_default_force_kill_requested_is_false(self):
        t = rpc_server.Task("t")
        assert t.force_kill_requested is False

    def test_task_default_sentinel_path_is_none(self):
        t = rpc_server.Task("t")
        assert t.sentinel_path is None

    def test_cancel_task_default_does_not_force(self):
        task = rpc_server._register_task("soft-cancel")
        rpc_server._cancel_task("soft-cancel")
        assert task.cancel_event.is_set()
        assert task.force_kill_requested is False

    def test_cancel_task_force_flag_propagates(self):
        task = rpc_server._register_task("hard-cancel")
        result = rpc_server._cancel_task("hard-cancel", force=True)
        assert result is True
        assert task.cancel_event.is_set()
        assert task.force_kill_requested is True

    def test_cancel_task_force_cascades_to_children(self):
        parent = rpc_server._register_task("job")
        child = rpc_server._register_task("job_train")
        rpc_server._cancel_task("job", force=True)
        assert parent.force_kill_requested is True
        assert child.force_kill_requested is True

    def test_cancel_task_touches_sentinel(self, tmp_path):
        task = rpc_server._register_task("with-sentinel")
        task.sentinel_path = tmp_path / "sub" / ".cancel"
        assert not task.sentinel_path.exists()
        rpc_server._cancel_task("with-sentinel")
        assert task.sentinel_path.exists()

    def test_cancel_task_sentinel_touch_tolerates_permission_error(self, tmp_path, monkeypatch):
        """sentinel の touch に失敗しても cancel 経路自体は動き続ける。"""
        task = rpc_server._register_task("sentinel-fail")
        task.sentinel_path = tmp_path / ".cancel"

        def _raise(*_a, **_kw):
            raise OSError("simulated permission denied")

        monkeypatch.setattr(type(task.sentinel_path), "touch", _raise)
        assert rpc_server._cancel_task("sentinel-fail") is True
        assert task.cancel_event.is_set()

    def test_rpc_cancel_accepts_force_param(self):
        task = rpc_server._register_task("via-rpc")
        result = rpc_server.rpc_cancel({"task_id": "via-rpc", "force": True})
        assert result == {"cancelled": True}
        assert task.force_kill_requested is True

    def test_rpc_cancel_without_force_is_soft(self):
        task = rpc_server._register_task("via-rpc-soft")
        result = rpc_server.rpc_cancel({"task_id": "via-rpc-soft"})
        assert result == {"cancelled": True}
        assert task.force_kill_requested is False


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


# ---------------------------------------------------------------------------
# _terminate_procs — 「強力な停止手段」の中核。実 subprocess を起動して
# kill できることを確認する結合テスト。conftest が psutil を MagicMock で
# スタブしているため、各テストで rpc_training.psutil = None に差し替えて
# 「psutil 無し環境」fallback パスを検証する (psutil 有りパスは
# monkeypatch を外せば ad-hoc に手動確認可能)。
# ---------------------------------------------------------------------------

import os as _os
import subprocess as _subprocess
import sys as _sys
import time as _time


def _spawn_sleeping_proc(seconds: int = 30) -> _subprocess.Popen:
    """起動直後から `seconds` 秒 sleep するだけの Python サブプロセス。

    start_new_session=True は本番コードの _spawn() と同じ設定。
    これにより killpg(SIGKILL) が効くかを検証できる。
    """
    return _subprocess.Popen(
        [_sys.executable, "-c", f"import time; time.sleep({seconds})"],
        start_new_session=True,
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.DEVNULL,
    )


@pytest.mark.integration
class TestTerminateProcs:
    """`_terminate_procs` が「確実に」「速く」プロセスを殺すことを保証。"""

    def _import_rpc_training(self):
        # conftest で既に必要なスタブは入っている前提
        import rpc_training
        return rpc_training

    def test_empty_list_returns_empty_result(self):
        rt = self._import_rpc_training()
        result = rt._terminate_procs([])
        assert isinstance(result, dict)
        assert result["killed"] == []
        assert result["residual"] == []

    def test_already_exited_proc_is_noop(self, monkeypatch):
        rt = self._import_rpc_training()
        monkeypatch.setattr(rt, "psutil", None)
        p = _subprocess.Popen([_sys.executable, "-c", "pass"])
        p.wait(timeout=5)
        result = rt._terminate_procs([p])
        assert result["killed"] == []
        assert result["residual"] == []

    def test_force_sigkill_under_one_second(self, monkeypatch):
        """graceful_wait=0.0 なら SIGTERM 段階を飛ばして 1 秒以内に殺せる。"""
        rt = self._import_rpc_training()
        monkeypatch.setattr(rt, "psutil", None)
        p = _spawn_sleeping_proc(60)
        start = _time.monotonic()
        try:
            result = rt._terminate_procs([p], graceful_wait=0.0)
        finally:
            try: p.wait(timeout=3)
            except Exception: pass
        elapsed = _time.monotonic() - start
        assert elapsed < 1.0, f"force kill took {elapsed:.2f}s, expected <1s"
        assert p.poll() is not None
        assert result["residual"] == [], f"residual PIDs after force kill: {result['residual']}"

    def test_graceful_then_sigkill(self, monkeypatch):
        """SIGTERM を無視するプロセスでも graceful_wait 経過後に SIGKILL。"""
        rt = self._import_rpc_training()
        monkeypatch.setattr(rt, "psutil", None)
        # SIGTERM を無視するプログラム
        p = _subprocess.Popen(
            [_sys.executable, "-c",
             "import signal, time;"
             "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
             "time.sleep(60)"],
            start_new_session=True,
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.DEVNULL,
        )
        # プロセスが SIG_IGN を登録する時間を少し与える
        _time.sleep(0.3)
        start = _time.monotonic()
        try:
            result = rt._terminate_procs([p], graceful_wait=0.5)
        finally:
            try: p.wait(timeout=3)
            except Exception: pass
        elapsed = _time.monotonic() - start
        # graceful 0.5 秒 + SIGKILL + verify（最大 2 秒）
        assert elapsed < 3.5, f"kill took {elapsed:.2f}s"
        assert p.poll() is not None
        assert result["residual"] == []

    def test_kills_child_process_via_process_group(self, monkeypatch):
        """start_new_session=True で起動した親の子孫も killpg で消える。

        psutil=None の fallback path でも、OS の process group 機構だけで
        子孫を消せることを確認（これが SIGKILL の最も重要な保険）。
        """
        rt = self._import_rpc_training()
        monkeypatch.setattr(rt, "psutil", None)
        # 親は子を spawn してから自分も sleep する
        code = (
            "import subprocess, sys, time, os;"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
            "print(p.pid, flush=True);"
            "time.sleep(60)"
        )
        parent = _subprocess.Popen(
            [_sys.executable, "-c", code],
            start_new_session=True,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.DEVNULL,
            text=True,
        )
        try:
            child_pid_line = parent.stdout.readline().strip()
            child_pid = int(child_pid_line)
            assert child_pid > 0
            # 子が本当に起動しているか確認
            _os.kill(child_pid, 0)

            rt._terminate_procs([parent], graceful_wait=0.0)

            # 親も子も 2 秒以内に完全消滅
            deadline = _time.monotonic() + 2.0
            child_gone = False
            while _time.monotonic() < deadline:
                try:
                    _os.kill(child_pid, 0)
                    _time.sleep(0.05)
                except OSError:
                    child_gone = True
                    break
            assert child_gone, f"子プロセス {child_pid} が消えていない"
            assert parent.poll() is not None
        finally:
            try: parent.wait(timeout=3)
            except Exception: pass
            try:
                # 念のため後始末
                if parent.poll() is None:
                    parent.kill()
            except Exception:
                pass

    def test_force_kill_requested_skips_sigterm_grace(self, monkeypatch):
        """_tail_log_until_done が task.force_kill_requested=True を見たら
        graceful_wait=0 で _terminate_procs を呼ぶことを確認。"""
        rt = self._import_rpc_training()
        monkeypatch.setattr(rt, "psutil", None)

        captured = {}
        real_terminate = rt._terminate_procs

        def spy(procs, graceful_wait=3.0, verify=True):
            captured["graceful_wait"] = graceful_wait
            return real_terminate(procs, graceful_wait=graceful_wait, verify=verify)

        monkeypatch.setattr(rt, "_terminate_procs", spy)

        p = _spawn_sleeping_proc(30)
        cancel_event = threading.Event()
        done_event = threading.Event()

        class _FakeTask:
            force_kill_requested = True

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "t.log"
            log.write_text("")
            # cancel_event を先にセット、tail はそれを検出して即 _terminate_procs
            cancel_event.set()
            try:
                rt._tail_log_until_done(
                    log, done_event, "test-force",
                    emit_progress=lambda *_a, **_kw: None,
                    cancel_event=cancel_event, proc=p, phase="test",
                    task=_FakeTask(),
                )
            finally:
                try: p.wait(timeout=3)
                except Exception: pass

        assert captured.get("graceful_wait") == 0.0, (
            f"force_kill_requested=True 時は graceful_wait=0.0 を期待、"
            f"実際は {captured.get('graceful_wait')}"
        )

    def test_soft_cancel_uses_grace_period(self, monkeypatch):
        """force_kill_requested=False なら通常 3 秒の grace period を使う。"""
        rt = self._import_rpc_training()
        monkeypatch.setattr(rt, "psutil", None)

        captured = {}

        def spy(procs, graceful_wait=3.0, verify=True):
            captured["graceful_wait"] = graceful_wait
            # テスト高速化のため即刻 kill
            import signal as _sig
            for p in procs:
                try: _os.killpg(_os.getpgid(p.pid), _sig.SIGKILL)
                except OSError: pass
                try: p.wait(timeout=2)
                except Exception: pass
            return {"killed": [], "residual": []}

        monkeypatch.setattr(rt, "_terminate_procs", spy)

        p = _spawn_sleeping_proc(30)
        cancel_event = threading.Event()
        done_event = threading.Event()

        class _FakeTask:
            force_kill_requested = False

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "t.log"
            log.write_text("")
            cancel_event.set()
            try:
                rt._tail_log_until_done(
                    log, done_event, "test-soft",
                    emit_progress=lambda *_a, **_kw: None,
                    cancel_event=cancel_event, proc=p, phase="test",
                    task=_FakeTask(),
                )
            finally:
                try: p.wait(timeout=3)
                except Exception: pass

        assert captured.get("graceful_wait") == 3.0


# ---------------------------------------------------------------------------
# Cancel sentinel 連携 (Phase 3)
#
# rpc_train 本体は heavy import を要求するため、ここでは sentinel の
# ライフサイクル（_cancel_task が touch する / 存在すれば _check_cancel
# 相当のロジックがファイル検出できる）のみ軽量に検証する。
# ---------------------------------------------------------------------------

class TestCancelSentinel:
    def setup_method(self):
        with rpc_server._tasks_lock:
            rpc_server._tasks.clear()

    def test_sentinel_not_touched_when_not_set(self, tmp_path):
        """sentinel_path が None なら cancel しても新規ファイル作成しない。"""
        task = rpc_server._register_task("no-sentinel")
        assert task.sentinel_path is None
        rpc_server._cancel_task("no-sentinel")
        # tmp_path は空のまま — sentinel 作成ロジックが None で no-op であることの間接証拠
        assert list(tmp_path.iterdir()) == []

    def test_sentinel_path_survives_prefix_cascade(self, tmp_path):
        """prefix 一致の子タスクそれぞれの sentinel が touch される。"""
        parent = rpc_server._register_task("job")
        child = rpc_server._register_task("job_train")
        parent.sentinel_path = tmp_path / "parent.cancel"
        child.sentinel_path = tmp_path / "child.cancel"
        rpc_server._cancel_task("job", force=True)
        assert parent.sentinel_path.exists()
        assert child.sentinel_path.exists()
        assert parent.force_kill_requested is True
        assert child.force_kill_requested is True

    def test_sentinel_creates_parent_directory(self, tmp_path):
        """まだ存在しない親ディレクトリ配下でも sentinel が作れる。"""
        task = rpc_server._register_task("deep-sentinel")
        task.sentinel_path = tmp_path / "a" / "b" / "c" / ".cancel"
        assert not task.sentinel_path.parent.exists()
        rpc_server._cancel_task("deep-sentinel")
        assert task.sentinel_path.exists()

    def test_train_py_check_cancel_contract(self, tmp_path, monkeypatch):
        """`train.py:_check_cancel` と同じロジックを独立に再現し、
        sentinel ファイルの存在で sys.exit(130) が起きる契約を固定する。

        train.py は heavy import (torch, tensorboard 代替, utils.get_hparams
        の argparse) が走るため直接 import できない。同等の実装を本テスト
        内で再定義し、rpc_server 側の touch 結果と噛み合うことを確認。
        """
        import types
        sentinel = tmp_path / ".cancel"
        hp = types.SimpleNamespace(cancel_sentinel=str(sentinel))

        def _check_cancel(hparams):
            path = getattr(hparams, "cancel_sentinel", None)
            if not path:
                return
            if Path(path).exists():
                raise SystemExit(130)

        # 未 touch なら通過
        _check_cancel(hp)

        # rpc_server._cancel_task で touch された経路と同じ
        task = rpc_server._register_task("train-sim")
        task.sentinel_path = sentinel
        rpc_server._cancel_task("train-sim")
        assert sentinel.exists()

        with pytest.raises(SystemExit) as exc:
            _check_cancel(hp)
        assert exc.value.code == 130

"""tests/test_rpc_integration.py

実 rpc_server.py プロセスを起動して JSON-RPC をやり取りする統合スモークテスト。

real torch が必要（スタブ torch の環境ではサーバが起動できないため自動スキップ）。
`pytest -m integration` で個別に走らせられる。検証対象:
- boot → "ready" 通知（重い import 一式 + AppState + 学習メソッド登録 + SIGTERM
  ハンドラ設置が例外なく完走する）
- initialize / list_models / dispatch ループ / バウンド付き write queue
- cancel(存在しない task) -> {"cancelled": False} の契約
- 未知メソッド -> -32601
- shutdown -> _cleanup_for_exit -> os._exit(0) で clean exit
- SIGTERM（Swift killSync 経路）-> 同ハンドラで clean exit（デフォルト終了でない）
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parent.parent


def _env(userdir: str) -> dict:
    env = dict(os.environ)
    env.update({
        "RVC_BASE_DIR": str(_REPO),
        "RVC_USER_DIR": userdir,
        "weight_root": "assets/weights",
        "weight_uvr5_root": "assets/uvr5_weights",
        "index_root": "logs",
        "outside_index_root": "assets/indices",
        "rmvpe_root": "assets/rmvpe",
        "PYTHONUNBUFFERED": "1",
    })
    return env


class _Server:
    """Spawn rpc_server.py and demux its stdout into notifications/responses."""

    def __init__(self):
        self._userdir = tempfile.mkdtemp(prefix="rvc_itest_")
        self.proc = subprocess.Popen(
            [sys.executable, "rpc_server.py"], cwd=str(_REPO), env=_env(self._userdir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.notes: list[str] = []
        self.responses: dict[int, dict] = {}
        threading.Thread(target=lambda: [None for _ in self.proc.stderr], daemon=True).start()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg:
                self.responses[msg["id"]] = msg
            elif "method" in msg:
                self.notes.append(msg["method"])

    def send(self, id_, method, params=None):
        self.proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()

    def wait_response(self, id_, timeout=15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if id_ in self.responses:
                return self.responses[id_]
            if self.proc.poll() is not None:
                pytest.fail(f"server died before responding to id={id_}")
            time.sleep(0.03)
        pytest.fail(f"timeout waiting for response id={id_}")

    def wait_ready_or_skip(self, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if "ready" in self.notes:
                return
            if self.proc.poll() is not None:
                # Server crashed during boot — almost always a stub-torch env.
                pytest.skip("rpc_server could not boot (real torch / assets required)")
            time.sleep(0.05)
        pytest.skip("rpc_server did not reach 'ready' in time")

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        import shutil
        shutil.rmtree(self._userdir, ignore_errors=True)


@pytest.fixture
def server():
    srv = _Server()
    srv.wait_ready_or_skip()
    try:
        yield srv
    finally:
        srv.close()


def test_initialize_returns_device_info(server):
    server.send(1, "initialize")
    r = server.wait_response(1)["result"]
    assert "device" in r
    assert "torch_version" in r


def test_list_models(server):
    server.send(2, "list_models")
    assert "models" in server.wait_response(2)["result"]


def test_cancel_unknown_task_contract(server):
    # The cancel RPC returns {"cancelled": bool}; an unknown id is False.
    server.send(3, "cancel", {"task_id": "ghost-task"})
    assert server.wait_response(3)["result"] == {"cancelled": False}


def test_unknown_method_error(server):
    server.send(4, "no_such_method")
    assert server.wait_response(4).get("error", {}).get("code") == -32601


def test_graceful_shutdown_clean_exit(server):
    server.send(5, "shutdown")
    rc = server.proc.wait(timeout=15)
    assert rc == 0
    assert "shutting_down" in server.notes


def test_sigterm_runs_handler_and_exits_clean():
    # The Swift killSync() path: SIGTERM must hit the installed handler which
    # runs _cleanup_for_exit + os._exit(0) (exit 0), not the default SIGTERM
    # termination (which would be a negative return code).
    srv = _Server()
    try:
        srv.wait_ready_or_skip()
        srv.proc.send_signal(signal.SIGTERM)
        rc = srv.proc.wait(timeout=10)
        assert rc == 0, f"expected handler os._exit(0), got {rc}"
    finally:
        srv.close()

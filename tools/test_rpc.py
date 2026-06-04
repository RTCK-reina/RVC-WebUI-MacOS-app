#!/usr/bin/env python3
"""Minimal smoke test for rpc_server.py.

Spawns the server, sends `initialize`, and prints the result. Also waits a few
seconds to observe `resource_stats` notifications.

Usage:
    python tools/test_rpc.py            # uses current python
    RVC_PYTHON=/path/to/python3 python tools/test_rpc.py
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    python = os.environ.get("RVC_PYTHON") or sys.executable
    server = HERE / "rpc_server.py"
    base_dir = os.environ.get("RVC_BASE_DIR") or str(HERE)
    user_dir = os.environ.get("RVC_USER_DIR") or str(HERE / ".rpc_test_user")

    Path(user_dir).mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            python,
            str(server),
            "--base-dir",
            base_dir,
            "--user-dir",
            user_dir,
            "--nocheck",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    # Reader thread — prints notifications as they arrive.
    notifications = []
    responses = {}

    def reader():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[non-json] {line}")
                continue
            if "id" in msg and msg["id"] is not None:
                responses[msg["id"]] = msg
                print(
                    f"<= RESP id={msg['id']}: {json.dumps(msg.get('result') or msg.get('error'), indent=2)[:400]}"
                )
            else:
                notifications.append(msg)
                if msg.get("method") not in ("resource_stats",):
                    print(
                        f"<= NOTIF {msg['method']}: {json.dumps(msg.get('params'))[:200]}"
                    )

    threading.Thread(target=reader, daemon=True).start()

    # Collect stderr in the background so the user sees import errors etc.
    def err_reader():
        for raw in iter(proc.stderr.readline, b""):
            sys.stderr.write("[py] " + raw.decode("utf-8", errors="replace"))

    threading.Thread(target=err_reader, daemon=True).start()

    def send(method, params=None, id_=None):
        req = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if id_ is not None:
            req["id"] = id_
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()

    # Wait for ready.
    print("=> waiting for ready...")
    for _ in range(100):
        if any(n.get("method") == "ready" for n in notifications):
            break
        time.sleep(0.3)
    else:
        print("!! server never became ready, aborting")
        proc.terminate()
        sys.exit(1)

    print("=> sending initialize")
    send("initialize", {}, id_=1)
    time.sleep(1.0)

    print("=> sending list_models")
    send("list_models", {}, id_=2)
    time.sleep(0.5)

    print("=> sending list_audio_devices")
    send("list_audio_devices", {}, id_=3)
    time.sleep(1.0)

    # Let the resource monitor emit a couple of samples.
    print("=> observing resource_stats for 3s...")
    time.sleep(3.0)
    res_count = sum(1 for n in notifications if n.get("method") == "resource_stats")
    print(f"   received {res_count} resource_stats notifications")

    print("=> sending shutdown")
    send("shutdown", {}, id_=99)
    time.sleep(1.0)

    proc.wait(timeout=5)
    print(f"=> exit code: {proc.returncode}")


if __name__ == "__main__":
    main()

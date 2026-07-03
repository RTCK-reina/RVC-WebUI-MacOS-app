#!/usr/bin/env python3
"""JSON-RPC 2.0 server for the SwiftUI RVC app.

Reads line-delimited JSON-RPC requests from stdin and writes responses /
notifications to stdout. stderr is reserved for logging diagnostics consumed
by the Swift side.

No Gradio, no HTTP server, no network calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Queue, Full
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------------
# CRITICAL: force stdout to line-buffered mode *before* any output.
#
# When Python's stdout is a pipe (non-tty), the default is block buffering.
# Even explicit flush() calls can interact badly with Swift's Pipe reader on
# macOS, causing the .app to time out before the "ready" notification is
# received. reconfigure() guarantees every newline-terminated write is
# flushed to the kernel pipe buffer immediately.
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass
try:
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Early CLI parsing -- we need --base-dir / --user-dir *before* importing
# Config, because Config() resolves asset paths on construction.
# ---------------------------------------------------------------------------

_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--base-dir", type=str, default=None)
_early_parser.add_argument("--user-dir", type=str, default=None)
_early_parser.add_argument("--nocheck", action="store_true")
_early_args, _rest = _early_parser.parse_known_args()

if _early_args.base_dir:
    os.environ["RVC_BASE_DIR"] = str(Path(_early_args.base_dir).expanduser().resolve())
if _early_args.user_dir:
    os.environ["RVC_USER_DIR"] = str(Path(_early_args.user_dir).expanduser().resolve())


def _configure_early_cache_dirs() -> None:
    """Set writable cache env vars before importing librosa/numba users.

    The heavy VC imports below pull in modules that lazily import librosa and
    numba. If those modules are imported before Config() has populated cache
    paths, numba can later fail with "cannot cache function ... no locator
    available" when model info or inference first touches librosa.feature.
    """
    raw_user_dir = os.environ.get("RVC_USER_DIR")
    if not raw_user_dir:
        return
    user_dir = Path(raw_user_dir).expanduser().resolve()
    cache_dir = user_dir / "cache"
    for path in (
        cache_dir,
        cache_dir / "fontconfig",
        cache_dir / "matplotlib",
        cache_dir / "numba",
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))


_configure_early_cache_dirs()

# Ensure we can import the repo modules whether invoked from the repo root or
# from inside the .app bundle.
_base = Path(os.environ.get("RVC_BASE_DIR", os.getcwd())).resolve()
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

# Change cwd to base_dir so that any cwd-relative paths that sneak through
# (i18n, training subprocesses, etc.) still resolve. Without this, launching
# the .app via Finder leaves cwd at "/" which breaks relative opens.
try:
    os.chdir(str(_base))
except Exception:
    pass

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load env defaults (sha256 checksums, etc). These files ship inside the bundle.
try:
    from dotenv import load_dotenv

    env_file = _base / ".env"
    sha_file = _base / "sha256.env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
    if sha_file.exists():
        load_dotenv(sha_file, override=False)
except Exception:
    pass

# Configure logging to stderr (stdout is reserved for JSON-RPC traffic).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("rpc_server")

# ---------------------------------------------------------------------------
# Heavy imports (after env setup).
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from configs import Config  # noqa: E402
from infer.lib.audio import save_audio  # noqa: E402
from infer.lib.path_safety import (  # noqa: E402
    PathValidationError,
    optional_dir_under,
    optional_file_under,
    require_env_root,
    resolve_under,
    safe_format,
    safe_leaf_name,
    safe_stem,
)
from infer.modules.vc import VC, show_info, hash_similarity  # noqa: E402

# librosa is already pulled in transitively by the VC import above; bind it at
# module scope so the real-time audio callback doesn't pay an import-system
# lookup + import-lock on every block.
import librosa  # noqa: E402
from infer.modules.uvr5.modules import uvr  # noqa: E402
from infer.lib.train.process_ckpt import (  # noqa: E402
    change_info,
    extract_small_model,
    merge,
)

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # Resource monitor degrades gracefully if unavailable.


def _empty_device_cache() -> None:
    """Return freed blocks to the CUDA/MPS caching allocator.

    Dropping a Python reference to a model frees the objects but the caching
    allocator keeps the device blocks cached, so memory is not actually
    returned until this is called. Mirrors infer/modules/vc/modules.py.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stdout writer -- every line is a single JSON object, flushed immediately.
# Thread-safe via a background writer thread (multiple workers may emit
# progress notifications concurrently).
# ---------------------------------------------------------------------------

# Bounded so a stalled/slow Swift stdout reader cannot grow this process's RSS
# without limit. High-frequency disposable notifications (10Hz realtime_metrics,
# 1Hz resource_stats, progress) are dropped when the queue is full — the next
# sample supersedes a lost one — while responses/errors and one-shot
# notifications use a blocking put and are never lost.
_WRITE_QUEUE_MAX = 4096
_write_queue: "Queue[str]" = Queue(maxsize=_WRITE_QUEUE_MAX)

# Notification methods whose payloads are best-effort and safe to drop under
# backpressure.
_DROPPABLE_NOTIFICATIONS = {"realtime_metrics", "resource_stats", "progress"}


def _writer_thread():
    while True:
        line = _write_queue.get()
        if line is None:
            break
        try:
            sys.stdout.write(line)
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            logger.exception("stdout write failed")


def _send(obj: dict, droppable: bool = False) -> None:
    line = json.dumps(obj, ensure_ascii=False, default=str)
    if droppable:
        try:
            _write_queue.put_nowait(line)
        except Full:
            pass  # reader is behind — drop this disposable sample
    else:
        _write_queue.put(line)


def send_notification(method: str, params: dict) -> None:
    _send(
        {"jsonrpc": "2.0", "method": method, "params": params},
        droppable=(method in _DROPPABLE_NOTIFICATIONS),
    )


def send_response(id_: Any, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": id_, "result": result})


def send_error(id_: Any, code: int, message: str, data: Any = None) -> None:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    _send({"jsonrpc": "2.0", "id": id_, "error": err})


# ---------------------------------------------------------------------------
# Cancellation tokens and task registry.
# ---------------------------------------------------------------------------


class Task:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.cancel_event = threading.Event()
        self.started_at = time.time()
        # Set by _cancel_task(force=True); consumed by _tail_log_until_done
        # to skip the SIGTERM grace period and go straight to SIGKILL.
        self.force_kill_requested: bool = False
        # Optional filesystem sentinel. When cancel fires we touch this
        # path so long-running subprocesses (e.g. train.py) can poll for
        # it and self-exit at the next checkpoint — covers the case
        # where SIGTERM is swallowed by a PyTorch C extension.
        self.sentinel_path: Optional[Path] = None


_tasks: Dict[str, Task] = {}
_tasks_lock = threading.Lock()


def _register_task(task_id: str) -> Task:
    t = Task(task_id)
    with _tasks_lock:
        _tasks[task_id] = t
    return t


def _unregister_task(task_id: str) -> None:
    with _tasks_lock:
        _tasks.pop(task_id, None)


def _cancel_task(task_id: str, force: bool = False) -> bool:
    """Cancel a task and all its children (prefix match).

    train_all registers sub-stages as "{parent_id}_preprocess", etc.
    Cancelling the parent ID cascades to all sub-stages.

    The sentinel file at ``task.sentinel_path`` (when set) is touched
    on **every** cancel — both soft and force — because the sentinel is
    an orthogonal, always-useful self-exit signal for a co-operating
    subprocess: there's no harm in offering a clean shutdown path even
    when we haven't escalated to SIGKILL yet.

    When ``force`` is True, we additionally set ``force_kill_requested``
    on every matching task so the tail-log loop skips its SIGTERM grace
    period and goes straight to SIGKILL — used by the UI's "strong stop"
    path when a plain cancel doesn't take effect within a couple of
    seconds.
    """
    cancelled_any = False
    with _tasks_lock:
        for tid, t in _tasks.items():
            if tid == task_id or tid.startswith(task_id + "_"):
                if force:
                    t.force_kill_requested = True
                t.cancel_event.set()
                sentinel = getattr(t, "sentinel_path", None)
                if sentinel is not None:
                    try:
                        sentinel.parent.mkdir(parents=True, exist_ok=True)
                        sentinel.touch(exist_ok=True)
                    except OSError:
                        pass
                cancelled_any = True
    return cancelled_any


def emit_progress(task_id: str, percent: float, message: str, phase: str = "") -> None:
    send_notification(
        "progress",
        {
            "task_id": task_id,
            "percent": float(max(0.0, min(100.0, percent))),
            "message": message,
            "phase": phase,
            "timestamp": time.time(),
        },
    )


# ---------------------------------------------------------------------------
# Subprocess-isolated worker runner.
# ---------------------------------------------------------------------------

import multiprocessing as _mp


def _run_in_subprocess(
    target: callable,
    args: tuple,
    cancel_event: threading.Event,
    task_id: str,
    poll_interval: float = 0.3,
    task: Optional["Task"] = None,
) -> dict:
    """Run *target* in a child process; kill it if cancel_event fires.

    *target(queue, *args)* must put its result as a dict onto the queue.
    Progress dicts ``{"progress": (percent, message, phase)}`` are forwarded
    via emit_progress.  The final dict is expected to have a "status" key.

    When *task* has ``force_kill_requested`` set (the UI's "strong stop"),
    skip the SIGTERM grace and go straight to SIGKILL — matching the documented
    force contract that the training path already honours.
    """
    ctx = _mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(queue, *args), daemon=True)
    proc.start()
    result: dict = {"status": "error", "error": "worker did not respond"}
    try:
        while proc.is_alive():
            if cancel_event.is_set():
                force = bool(
                    task is not None and getattr(task, "force_kill_requested", False)
                )
                if force:
                    proc.kill()  # skip the SIGTERM grace on a strong stop
                    proc.join(timeout=2)
                else:
                    proc.terminate()
                    proc.join(timeout=3)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
                return {"status": "cancelled"}
            # Drain the queue.
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except Exception:
                    break
                if isinstance(item, dict):
                    if "progress" in item:
                        pct, msg, phase = item["progress"]
                        emit_progress(task_id, pct, msg, phase)
                    else:
                        result = item
            proc.join(timeout=poll_interval)
        # Drain remaining items after process exits.
        while not queue.empty():
            try:
                item = queue.get_nowait()
            except Exception:
                break
            if isinstance(item, dict) and "progress" not in item:
                result = item
            elif isinstance(item, dict) and "progress" in item:
                pct, msg, phase = item["progress"]
                emit_progress(task_id, pct, msg, phase)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        # Release the Queue's pipe FDs deterministically on every path
        # (success / error / cancel); otherwise they leak until GC for the
        # life of this long-running server.
        try:
            queue.cancel_join_thread()
            queue.close()
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Application state.
# ---------------------------------------------------------------------------


class AppState:
    def __init__(self):
        self.config: Config = Config()
        # Force --nocheck behaviour: the .app ships its own assets and never
        # performs network downloads.
        self.config.nocheck = True
        self.vc = VC(self.config)
        self.realtime = None  # Filled in during realtime_start.
        self.active_status = (
            "idle"  # idle / inferring / training / separating / realtime
        )

    def status(self, s: str) -> None:
        self.active_status = s
        send_notification("status", {"status": s})


app_state: Optional[AppState] = None


# ---------------------------------------------------------------------------
# Resource monitor thread.
# ---------------------------------------------------------------------------


def _resource_monitor(stop_event: threading.Event):
    # psutil's cpu_percent needs a priming call.
    if psutil:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    # Emit every 1s while a task is active; back off to ~3s when idle so we
    # don't do the psutil/torch sampling + JSON encode + pipe write + SwiftUI
    # publish every second for an app that's just sitting there. The 1s tick
    # cadence is kept so a task starting mid-idle is picked up promptly.
    idle_skip = 0
    while not stop_event.is_set():
        active = bool(app_state and app_state.active_status != "idle")
        if not active:
            idle_skip += 1
            if idle_skip < 3:
                stop_event.wait(1.0)
                continue
        idle_skip = 0
        stats: Dict[str, Any] = {
            "timestamp": time.time(),
            "status": app_state.active_status if app_state else "starting",
            "device": (app_state.config.device if app_state else "unknown"),
            "is_half": bool(app_state.config.is_half) if app_state else False,
        }

        if psutil:
            try:
                stats["cpu_percent"] = float(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                stats["memory_used_gb"] = round(vm.used / 1024**3, 2)
                stats["memory_total_gb"] = round(vm.total / 1024**3, 2)
                stats["memory_percent"] = float(vm.percent)
                proc = psutil.Process(os.getpid())
                stats["process_memory_gb"] = round(proc.memory_info().rss / 1024**3, 2)
            except Exception:
                pass

        # MPS / CUDA memory.
        try:
            if torch.backends.mps.is_available():
                used = torch.mps.current_allocated_memory()
                stats["gpu_memory_used_mb"] = int(used / (1024 * 1024))
                try:
                    driver = torch.mps.driver_allocated_memory()
                    stats["gpu_memory_driver_mb"] = int(driver / (1024 * 1024))
                except Exception:
                    driver = used
                # Apple Silicon: recommend_max or physical memory as ceiling.
                try:
                    total = torch.mps.recommended_max_memory()
                    stats["gpu_memory_total_mb"] = int(total / (1024 * 1024))
                    stats["gpu_percent"] = (
                        round(driver / total * 100, 1) if total > 0 else 0.0
                    )
                except Exception:
                    # Fallback: use system memory as upper bound for unified memory.
                    if psutil:
                        total = psutil.virtual_memory().total
                        stats["gpu_memory_total_mb"] = int(total / (1024 * 1024))
                        stats["gpu_percent"] = (
                            round(driver / total * 100, 1) if total > 0 else 0.0
                        )
                stats["gpu_backend"] = "mps"
            elif torch.cuda.is_available():
                used = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_mem
                stats["gpu_memory_used_mb"] = int(used / (1024 * 1024))
                stats["gpu_memory_reserved_mb"] = int(
                    torch.cuda.memory_reserved() / (1024 * 1024)
                )
                stats["gpu_memory_total_mb"] = int(total / (1024 * 1024))
                stats["gpu_percent"] = (
                    round(used / total * 100, 1) if total > 0 else 0.0
                )
                stats["gpu_backend"] = "cuda"
            else:
                stats["gpu_backend"] = "cpu"
        except Exception:
            pass

        send_notification("resource_stats", stats)
        stop_event.wait(1.0)


# ---------------------------------------------------------------------------
# RPC method implementations.
# ---------------------------------------------------------------------------


def _list_files(dir_path: Path, exts: list) -> list:
    if not dir_path.is_dir():
        return []
    results = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and (not exts or p.suffix.lower() in exts):
            results.append(p.name)
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and (not exts or sub.suffix.lower() in exts):
                    results.append(str(sub.relative_to(dir_path)))
    return results


def rpc_initialize(params: dict) -> dict:
    """Return device info and directory layout. Called once on app start."""
    assert app_state is not None
    c = app_state.config
    return {
        "base_dir": str(c.base_dir),
        "user_dir": str(c.user_dir),
        "device": c.device,
        "is_half": bool(c.is_half),
        "gpu_name": c.gpu_name,
        "gpu_mem_gb": c.gpu_mem,
        "n_cpu": c.n_cpu,
        "torch_version": torch.__version__,
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            getattr(c, "mps_available", torch.backends.mps.is_available())
        ),
        "mps_error": getattr(c, "mps_error", None),
        "cuda_available": torch.cuda.is_available(),
        "gpu_training_available": c.device == "mps" or str(c.device).startswith("cuda"),
        "python_version": sys.version.split()[0],
        "paths": {
            "weight_root": os.environ.get("weight_root"),
            "weight_uvr5_root": os.environ.get("weight_uvr5_root"),
            "index_root": os.environ.get("index_root"),
            "outside_index_root": os.environ.get("outside_index_root"),
            "rmvpe_root": os.environ.get("rmvpe_root"),
            "output_root": os.environ.get("output_root"),
            "input_root": os.environ.get("input_root"),
            "temp": os.environ.get("TEMP"),
        },
    }


def rpc_list_models(params: dict) -> dict:
    root = Path(os.environ.get("weight_root") or "")
    return {"models": _list_files(root, [".pth"])}


def rpc_list_indices(params: dict) -> dict:
    roots = [os.environ.get("index_root"), os.environ.get("outside_index_root")]
    out = []
    seen = set()
    for r in roots:
        if not r:
            continue
        for p in Path(r).rglob("*.index"):
            if "trained" in p.name:
                continue
            rel = str(p)
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
    return {"indices": out}


def rpc_list_uvr5_models(params: dict) -> dict:
    root = Path(os.environ.get("weight_uvr5_root") or "")
    models = []
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.suffix.lower() in (".pth", ".onnx"):
                models.append(p.stem)
    return {"models": models}


def rpc_load_model(params: dict) -> dict:
    assert app_state is not None
    sid = params.get("sid", "") or ""
    return app_state.vc.load_vc(sid)


def rpc_unload_model(params: dict) -> dict:
    assert app_state is not None
    app_state.vc.load_vc("")
    return {"unloaded": True}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_AUDIO_OUTPUT_FORMATS = ("flac", "wav", "mp3", "m4a")


def _output_root() -> Path:
    return (
        Path(
            os.environ.get("output_root")
            or (Path.home() / "Documents" / "RVC-WebUI" / "output")
        )
        .expanduser()
        .resolve()
    )


def _index_roots() -> list[Path]:
    return [
        Path(r).expanduser().resolve()
        for r in (os.environ.get("outside_index_root"), os.environ.get("index_root"))
        if r
    ]


def _resolve_under_any(roots: list[Path], value: object, field: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    errors = []
    for root in roots:
        try:
            return str(resolve_under(root, raw, field))
        except PathValidationError as exc:
            errors.append(str(exc))
    allowed = ", ".join(str(r) for r in roots) or "<none>"
    raise PathValidationError(f"{field} must stay under one of: {allowed}")


def _require_existing_file(value: object, field: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise PathValidationError(f"{field} is required")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise PathValidationError(f"{field} does not exist or is not a file: {raw}")
    return str(path)


def _default_output_path(
    input_path: str, model_sid: str, fmt: str, subdir: str
) -> Path:
    out_root = _output_root()
    out_dir = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(input_path, "input_audio_path", fallback="audio")
    model_stem = safe_stem(model_sid, "sid", fallback="model")
    return out_dir / f"{stem}_{model_stem}_{_timestamp()}.{fmt}"


def rpc_vc_single(params: dict) -> dict:
    assert app_state is not None
    sid = params["sid"]
    input_path = _require_existing_file(params["input_audio_path"], "input_audio_path")
    fmt = safe_format(params.get("format", "flac"), _AUDIO_OUTPUT_FORMATS)
    file_index = _resolve_under_any(
        _index_roots(), params.get("file_index", ""), "file_index"
    )
    file_index2 = _resolve_under_any(
        _index_roots(), params.get("file_index2", ""), "file_index2"
    )
    default_output = _default_output_path(input_path, sid, fmt, "inference")
    output_path = optional_file_under(
        _output_root(), params.get("output_path"), "output_path", default=default_output
    )

    task_id = params.get("task_id", f"vc_single_{int(time.time()*1000)}")
    _register_task(task_id)  # registered for status visibility, no cancel honored
    app_state.status("inferring")
    try:
        emit_progress(task_id, 5, "Loading audio", "inference")

        # Run inference in a worker thread purely so the RPC dispatcher stays
        # responsive (resource_stats notifications keep flowing). Cancellation
        # is intentionally NOT honored here: a single-file PyTorch inference
        # cannot be safely interrupted, so the UI does not expose a cancel
        # button for this operation.
        result_box: dict = {}
        err_box: dict = {}

        def runner():
            try:
                result_box["result"] = app_state.vc.vc_single(
                    params.get("sid_index", 0),
                    input_path,
                    int(params.get("f0_up_key", 0)),
                    None,
                    params.get("f0_method", "rmvpe"),
                    file_index,
                    file_index2,
                    float(params.get("index_rate", 0.75)),
                    int(params.get("filter_radius", 3)),
                    int(params.get("resample_sr", 0)),
                    float(params.get("rms_mix_rate", 0.25)),
                    float(params.get("protect", 0.33)),
                )
            except BaseException as e:  # noqa: BLE001
                err_box["error"] = e

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        tick = 0
        while t.is_alive():
            t.join(timeout=0.25)
            tick += 1
            if tick % 4 == 0:  # ~1s
                emit_progress(task_id, 40, "推論中…", "inference")
        if "error" in err_box:
            raise err_box["error"]
        info, opt = result_box["result"]
        if opt is None:
            return {"status": "error", "info": info}

        emit_progress(task_id, 90, "Saving output", "inference")
        tgt_sr, audio_opt = opt
        # audio_opt is already int16-scale (infer/modules/vc/modules.py: VC.vc_single
        # applies .astype(np.int16)).
        # f32=True would cast int16 values into a float32 IEEE WAV where ±1.0 is full
        # scale — the ±32440 values then explode into 32440× overgain and clip hard
        # in every downstream player / FLAC re-encode. Keep f32=False for int16 PCM.
        save_audio(str(output_path), audio_opt, tgt_sr, f32=False, format=fmt)
        emit_progress(task_id, 100, "Done", "inference")

        return {
            "status": "success",
            "info": info,
            "output_path": str(output_path),
            "sample_rate": int(tgt_sr),
        }
    finally:
        _unregister_task(task_id)
        app_state.status("idle")


def rpc_vc_multi(params: dict) -> dict:
    assert app_state is not None
    sid = params["sid"]
    dir_path = params.get("dir_path", "") or ""
    paths = params.get("paths", []) or []
    fmt = safe_format(params.get("format", "flac"), _AUDIO_OUTPUT_FORMATS)
    file_index = _resolve_under_any(
        _index_roots(), params.get("file_index", ""), "file_index"
    )
    file_index2 = _resolve_under_any(
        _index_roots(), params.get("file_index2", ""), "file_index2"
    )
    out_root = optional_dir_under(
        _output_root(),
        params.get("output_dir"),
        "output_dir",
        default=_output_root() / "batch" / _timestamp(),
    )

    task_id = params.get("task_id", f"vc_multi_{int(time.time()*1000)}")
    task = _register_task(task_id)
    app_state.status("inferring")
    try:
        # Collect files.
        if dir_path:
            input_dir = Path(dir_path).expanduser().resolve()
            if not input_dir.is_dir():
                raise PathValidationError(
                    f"dir_path does not exist or is not a directory: {dir_path}"
                )
            all_paths = [str(p) for p in input_dir.iterdir() if p.is_file()]
        else:
            all_paths = [
                _require_existing_file(p, f"paths[{i}]") for i, p in enumerate(paths)
            ]
        total = len(all_paths)
        if total == 0:
            raise PathValidationError("No input files")

        results = []
        for i, path in enumerate(all_paths):
            if task.cancel_event.is_set():
                return {"status": "cancelled", "completed": i, "total": total}
            emit_progress(
                task_id,
                (i / total) * 100.0,
                f"({i + 1}/{total}) {Path(path).name}",
                "batch",
            )
            info, opt = app_state.vc.vc_single(
                params.get("sid_index", 0),
                path,
                int(params.get("f0_up_key", 0)),
                None,
                params.get("f0_method", "rmvpe"),
                file_index,
                file_index2,
                float(params.get("index_rate", 0.75)),
                int(params.get("filter_radius", 3)),
                int(params.get("resample_sr", 0)),
                float(params.get("rms_mix_rate", 0.25)),
                float(params.get("protect", 0.33)),
            )
            if opt is not None:
                tgt_sr, audio_opt = opt
                model_stem = Path(sid).stem if sid else "model"
                out_name = f"{safe_stem(path, 'input path', fallback='audio')}_{safe_stem(model_stem, 'sid', fallback='model')}.{fmt}"
                out_path = out_root / out_name
                # See rpc_vc_single above: audio_opt is int16-scale, so f32=False.
                save_audio(str(out_path), audio_opt, tgt_sr, f32=False, format=fmt)
                results.append({"input": path, "output": str(out_path), "info": info})
            else:
                results.append({"input": path, "output": None, "info": info})
        emit_progress(task_id, 100, f"Completed {total} files", "batch")
        return {"status": "success", "results": results, "output_dir": str(out_root)}
    finally:
        _unregister_task(task_id)
        app_state.status("idle")


def _uvr5_worker(
    queue,
    model_name,
    inp_root,
    save_root_vocal,
    input_paths,
    save_root_ins,
    agg,
    fmt,
    polish_model,
    scratch_dir,
):
    """Subprocess entry point for UVR5 separation.

    Runs the full separation (+ optional polish) and posts results/progress
    dicts onto *queue*.  Runs in a completely separate process so the parent
    can terminate() it at any point for instant cancellation.

    *scratch_dir* is a session-unique path supplied by the parent so the parent
    can reliably clean it up even when this worker is terminate()d mid-polish
    (its own ``finally`` below is skipped on SIGTERM/SIGKILL).
    """
    import os, traceback, shutil
    from pathlib import Path

    class _Faux:
        def __init__(self, name):
            self.name = name

    try:
        from infer.modules.uvr5.modules import uvr, _is_audio_file

        total = max(1, len(input_paths))
        pass1_scale = 50.0 if polish_model else 100.0
        messages = []

        # --- Pass 1 -------------------------------------------------------
        faux_paths = [_Faux(p) for p in input_paths]
        done = 0
        for msg in uvr(
            model_name, inp_root, save_root_vocal, faux_paths, save_root_ins, agg, fmt
        ):
            messages.append(msg)
            done = min(total, msg.count("->"))
            queue.put(
                {
                    "progress": (
                        (done / total) * pass1_scale,
                        f"({done}/{total}) 分離中…",
                        "separation",
                    )
                }
            )

        # --- Pass 2 (polish) ----------------------------------------------
        polished_dir = None
        if polish_model:
            queue.put({"progress": (pass1_scale, "仕上げ準備中…", "separation")})
            vocal_files = []
            seen = set()
            for d in (save_root_vocal, save_root_ins):
                dp = Path(d)
                if not dp.is_dir():
                    continue
                for p in dp.iterdir():
                    if not p.is_file() or not p.name.startswith("vocal_"):
                        continue
                    if not _is_audio_file(p.name, strict=False):
                        continue
                    full = str(p.resolve())
                    if full not in seen:
                        seen.add(full)
                        vocal_files.append(full)

            if not vocal_files:
                messages.append("Polish skipped: no vocal_* outputs found.")
            else:
                polished_dir = str(Path(save_root_vocal) / "polished")
                Path(polished_dir).mkdir(parents=True, exist_ok=True)
                residue_dir = str(Path(save_root_vocal) / "polished" / "residue")
                Path(residue_dir).mkdir(parents=True, exist_ok=True)

                scratch_dir = Path(scratch_dir)
                scratch_dir.mkdir(parents=True, exist_ok=True)
                scratch_paths = []
                for vf in vocal_files:
                    dst = scratch_dir / os.path.basename(vf)
                    try:
                        shutil.copy2(vf, dst)
                        scratch_paths.append(str(dst))
                    except Exception:
                        pass

                polish_faux = [_Faux(p) for p in scratch_paths]
                p_total = max(1, len(scratch_paths))
                try:
                    p_done = 0
                    for msg in uvr(
                        polish_model,
                        "",
                        polished_dir,
                        polish_faux,
                        residue_dir,
                        agg,
                        fmt,
                    ):
                        messages.append(msg)
                        p_done = min(p_total, msg.count("->"))
                        queue.put(
                            {
                                "progress": (
                                    pass1_scale
                                    + (p_done / p_total) * (100 - pass1_scale),
                                    f"仕上げ中 ({p_done}/{p_total})",
                                    "separation",
                                )
                            }
                        )
                finally:
                    shutil.rmtree(scratch_dir, ignore_errors=True)

        result = {
            "status": "success",
            "messages": messages,
            "output_vocal": save_root_vocal,
            "output_accompaniment": save_root_ins,
        }
        if polished_dir:
            result["polished_dir"] = polished_dir
        queue.put(result)
    except Exception:
        queue.put({"status": "error", "error": traceback.format_exc()})


def rpc_uvr5(params: dict) -> dict:
    """Run UVR5 vocal/instrumental separation in a subprocess.

    The heavy model inference runs in a completely separate process so that
    cancellation can terminate() it immediately — even mid-file.
    """
    assert app_state is not None
    model_name = safe_leaf_name(params["model_name"], "model_name")
    inp_root = params.get("input_dir", "") or ""
    paths = params.get("paths", []) or []
    base_vocal = optional_dir_under(
        _output_root(),
        params.get("output_vocal"),
        "output_vocal",
        default=_output_root() / "separation" / "vocals",
    )
    base_ins = optional_dir_under(
        _output_root(),
        params.get("output_accompaniment"),
        "output_accompaniment",
        default=_output_root() / "separation" / "accompaniment",
    )
    from datetime import datetime as _dt

    session_stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    save_root_vocal = str(base_vocal / session_stamp)
    save_root_ins = str(base_ins / session_stamp)
    Path(save_root_vocal).mkdir(parents=True, exist_ok=True)
    Path(save_root_ins).mkdir(parents=True, exist_ok=True)
    # Session-unique scratch dir so concurrent/cancelled polish runs never
    # collide, and so the parent can sweep it even if the worker is killed
    # before its own cleanup runs.
    scratch_dir = str(
        Path(os.environ.get("TEMP", "/tmp")) / f"polish_worker_{session_stamp}"
    )
    agg = int(params.get("agg", 10))
    fmt = safe_format(params.get("format", "flac"), _AUDIO_OUTPUT_FORMATS)
    polish_model = (params.get("polish_model") or "").strip()
    if polish_model:
        polish_model = safe_leaf_name(polish_model, "polish_model")

    from infer.modules.uvr5.modules import _is_audio_file

    if inp_root:
        inp_dir = Path(inp_root).expanduser().resolve()
        if not inp_dir.is_dir():
            return {
                "status": "error",
                "error": f"Input directory not found: {inp_root}",
            }
        inp_root = str(inp_dir)
        input_paths = [
            str(p)
            for p in inp_dir.iterdir()
            if p.is_file() and _is_audio_file(p.name, strict=True)
        ]
    else:
        raw = [p if isinstance(p, str) else p.get("name") for p in paths]
        input_paths = [
            _require_existing_file(p, f"paths[{i}]")
            for i, p in enumerate(raw)
            if p and _is_audio_file(p, strict=False)
        ]
    if not input_paths:
        raise PathValidationError("No input audio files")

    task_id = params.get("task_id", f"uvr5_{int(time.time()*1000)}")
    task = _register_task(task_id)
    app_state.status("separating")

    try:
        result = _run_in_subprocess(
            _uvr5_worker,
            (
                model_name,
                inp_root,
                save_root_vocal,
                input_paths,
                save_root_ins,
                agg,
                fmt,
                polish_model,
                scratch_dir,
            ),
            cancel_event=task.cancel_event,
            task_id=task_id,
            task=task,
        )
        if result.get("status") == "success":
            # uvr() yields per-file errors as message strings but the worker
            # still returns success even when the model failed to load or every
            # file failed — leaving 0 outputs. Verify at least one vocal/
            # instrument file was actually produced before claiming success,
            # mirroring the training-stage artifact checks.
            def _has_output(d):
                p = Path(d)
                return p.is_dir() and any(c.is_file() for c in p.iterdir())

            if not (_has_output(save_root_vocal) or _has_output(save_root_ins)):
                msgs = result.get("messages") or []
                detail = next(
                    (m for m in reversed(msgs) if "Traceback" in m or "->" in m), ""
                )
                return {
                    "status": "error",
                    "error": "分離は出力を生成しませんでした（モデルが見つからないか、全ファイルが失敗した可能性）",
                    "messages": msgs,
                    "detail": detail[:500],
                }
            emit_progress(task_id, 100, "完了", "separation")
        return result
    finally:
        # The worker's own scratch cleanup is skipped when we terminate()/kill()
        # it mid-polish, so sweep the session-unique scratch dir here regardless
        # of outcome, and drop the pre-created output dirs if a cancel/error
        # left them empty.
        shutil.rmtree(scratch_dir, ignore_errors=True)
        for d in (save_root_vocal, save_root_ins):
            try:
                p = Path(d)
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            except OSError:
                pass
        _unregister_task(task_id)
        app_state.status("idle")


def rpc_model_info(params: dict) -> dict:
    path = params["path"]
    info = show_info(path)
    status = "error" if info.lstrip().startswith("Traceback") else "success"
    return {
        "status": status,
        "info": info,
        "error": info if status == "error" else None,
    }


def rpc_model_change_info(params: dict) -> dict:
    path = params["path"]
    info = params.get("info", "")
    name = params.get("name", "")
    return _require_success({"result": change_info(path, info, name)}, "情報の更新")


def rpc_model_compare(params: dict) -> dict:
    a = params["id_a"]
    b = params["id_b"]
    sim = hash_similarity(a, b)
    if not isinstance(sim, str):
        sim = "%.2f%%" % (sim * 100)
    return {"similarity": sim}


def _run_with_progress(task_id: str, phase: str, fn, *args, **kwargs):
    """Execute a blocking Python function while emitting periodic progress.

    Runs `fn` in a worker thread so the RPC dispatcher can keep serving fast
    methods (resource_stats notifications, list_models, etc.) while the heavy
    op runs. There is intentionally NO cancellation: PyTorch C extensions
    can't be safely interrupted, and exposing a cancel that doesn't really
    cancel is misleading.
    """
    assert app_state is not None
    task = _register_task(task_id)
    app_state.status("processing")
    result_box: dict = {}
    err_box: dict = {}

    def runner():
        try:
            result_box["result"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            err_box["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    emit_progress(task_id, 5, f"{phase} 開始", phase)
    try:
        tick = 0
        while t.is_alive():
            t.join(timeout=0.25)
            tick += 1
            if tick % 4 == 0:  # ~1s
                emit_progress(task_id, 50, f"{phase} 処理中…", phase)
        if "error" in err_box:
            raise err_box["error"]
        emit_progress(task_id, 100, f"{phase} 完了", phase)
        return {"result": result_box.get("result")}
    finally:
        _unregister_task(task_id)
        app_state.status("idle")


def _require_success(result: dict, op: str) -> dict:
    """Route a string-returning op's failure to a real RPC error.

    process_ckpt.merge / extract_small_model / change_info signal failure by
    *returning* a traceback (or error) string instead of raising, so a failed
    run otherwise reaches the Swift UI as a normal "result" string (shown in
    the success/result area, never flagged red). Detect the non-"Success."
    sentinel and raise so the dispatcher emits a JSON-RPC error and the UI's
    catch-path surfaces it correctly. The success contract ({"result":
    "Success."}) is unchanged.
    """
    s = result.get("result") if isinstance(result, dict) else result
    if isinstance(s, str) and s.strip() != "Success.":
        raise RuntimeError("%s に失敗しました: %s" % (op, s.strip()))
    return result


def _sr_key(raw_sr) -> str:
    """Normalize a sample rate to the string key ("40k"/"48k"/"32k").

    process_ckpt.merge / extract_small_model store this string verbatim as the
    model's ``sr`` metadata and index their config tables by it. Accept either
    an int (Hz) or an already-formatted string from the caller.
    """
    if isinstance(raw_sr, str):
        return raw_sr
    _sr_map = {40000: "40k", 48000: "48k", 32000: "32k"}
    return _sr_map.get(int(raw_sr), "40k")


def rpc_model_merge(params: dict) -> dict:
    task_id = params.get("task_id", f"model_merge_{int(time.time()*1000)}")
    return _require_success(
        _run_with_progress(
            task_id,
            "merge",
            merge,
            params["path_a"],
            params["path_b"],
            float(params.get("alpha", 0.5)),
            # merge() stores sr verbatim as the model's metadata and (for raw
            # G_*.pth inputs) indexes its config table by it, so it must be the
            # string key — not the int 40000 — to stay consistent with every
            # other model producer.
            _sr_key(params.get("sr", "40k")),
            int(params.get("if_f0", 1)),
            params.get("info", ""),
            params.get("name", ""),
            params.get("version", "v2"),
        ),
        "モデルのマージ",
    )


def rpc_model_extract(params: dict) -> dict:
    task_id = params.get("task_id", f"model_extract_{int(time.time()*1000)}")
    # extract_small_model expects sr as a string key ("40k" / "48k" / "32k")
    # because it indexes into hard-coded config tables keyed by that string.
    # Accept either int (Hz) or already-formatted string from the caller.
    sr_key = _sr_key(params.get("sr", "40k"))
    return _require_success(
        _run_with_progress(
            task_id,
            "extract",
            extract_small_model,
            params["ckpt_path"],
            params.get("name", ""),
            params.get("author", ""),
            sr_key,
            int(params.get("if_f0", 1)),
            params.get("info", ""),
            params.get("version", "v2"),
        ),
        "モデルの抽出",
    )


def rpc_export_onnx(params: dict) -> dict:
    from rvc.onnx import export_onnx  # lazy import (heavy)

    ckpt_path = params["ckpt_path"]
    default_output = (
        _output_root()
        / "onnx"
        / f"{safe_stem(ckpt_path, 'ckpt_path', fallback='model')}.onnx"
    )
    output_path = optional_file_under(
        _output_root(),
        params.get("output_path"),
        "output_path",
        default=default_output,
    )
    task_id = params.get("task_id", f"export_onnx_{int(time.time()*1000)}")
    result = _run_with_progress(
        task_id,
        "onnx",
        export_onnx,
        ckpt_path,
        str(output_path),
    )
    if result.get("status") == "cancelled":
        return result
    return {"status": "success", "output_path": str(output_path)}


def rpc_list_audio_devices(params: dict) -> dict:
    try:
        import sounddevice as sd
    except Exception as e:
        return {"error": str(e), "input": [], "output": []}
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    return {
        "host_apis": [h["name"] for h in hostapis],
        "input": [
            {
                "index": i,
                "name": d["name"],
                "hostapi": hostapis[d["hostapi"]]["name"],
                "max_channels": d["max_input_channels"],
                "default_sr": d.get("default_samplerate"),
            }
            for i, d in enumerate(devs)
            if d["max_input_channels"] > 0
        ],
        "output": [
            {
                "index": i,
                "name": d["name"],
                "hostapi": hostapis[d["hostapi"]]["name"],
                "max_channels": d["max_output_channels"],
                "default_sr": d.get("default_samplerate"),
            }
            for i, d in enumerate(devs)
            if d["max_output_channels"] > 0
        ],
    }


# ---------------------------------------------------------------------------
# Realtime VC — streams audio through RVC using sounddevice.
# ---------------------------------------------------------------------------


class _RealtimeVC:
    """Manages the sounddevice stream and per-block RVC inference."""

    def __init__(
        self,
        rvc_obj,
        samplerate: int,
        channels,
        block_time: float,
        threshold: float,
        f0method: str,
        protect: float,
        device_str: str,
        input_device=None,
        output_device=None,
    ):
        import sounddevice as sd
        import torchaudio.transforms as tat

        self.rvc = rvc_obj
        self.samplerate = samplerate
        self.channels = channels
        self.out_channels = channels[1] if isinstance(channels, tuple) else channels
        self.threshold = threshold
        self.f0method = f0method
        self.protect = protect
        self.device_str = device_str
        self.stream = None
        self._error_count = 0
        self._error_notified = False

        # --- Diagnostics state (consumed by the Swift UI via realtime_metrics).
        # Peaks accumulate between emits; RMS/latency are block-local snapshots.
        self._max_input_peak = 0.0
        self._max_output_peak = 0.0
        self._last_inference_ms = 0.0
        self._last_block_ms = 0.0
        self._last_input_rms_db = -90.0
        self._last_output_rms_db = -90.0
        self._last_is_gated = False
        self._last_emit_time = time.time()
        self._cb_input_underflow = 0
        self._cb_input_overflow = 0
        self._cb_output_underflow = 0
        self._cb_output_overflow = 0
        self._cb_priming_output = 0
        self._input_device_name = "unknown"
        self._output_device_name = "unknown"
        self._input_device_index = (
            int(input_device)
            if isinstance(input_device, int) and input_device >= 0
            else -1
        )
        self._output_device_index = (
            int(output_device)
            if isinstance(output_device, int) and output_device >= 0
            else -1
        )

        zc = samplerate // 100
        self.zc = zc
        self.block_frame = int(np.round(block_time * samplerate / zc)) * zc
        self.block_frame_16k = 160 * self.block_frame // zc
        crossfade_time = 0.05
        self.crossfade_frame = int(np.round(crossfade_time * samplerate / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        extra_time = 2.5
        self.extra_frame = int(np.round(extra_time * samplerate / zc)) * zc
        total_len = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        self.input_wav = torch.zeros(total_len, device=device_str, dtype=torch.float32)
        self.input_wav_res = torch.zeros(
            160 * total_len // zc, device=device_str, dtype=torch.float32
        )
        self.rms_buffer = np.zeros(4 * zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=device_str, dtype=torch.float32
        )
        # Pre-allocated constant kernel for the SOLA crossfade denominator.
        # sola_buffer_frame is fixed for the session, so allocating this once
        # avoids a per-block device allocation on the real-time audio thread.
        self.sola_ones = torch.ones(
            1, 1, self.sola_buffer_frame, device=device_str, dtype=torch.float32
        )
        self.skip_head = self.extra_frame // zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=device_str,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=samplerate, new_freq=16000, dtype=torch.float32
        ).to(device_str)
        if rvc_obj.tgt_sr != samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=rvc_obj.tgt_sr, new_freq=samplerate, dtype=torch.float32
            ).to(device_str)
        else:
            self.resampler2 = None

        self.stream = sd.Stream(
            callback=self._audio_callback,
            blocksize=self.block_frame,
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            device=(input_device, output_device),
        )
        self.stream.start()

        # Resolve the device indices sounddevice actually bound to. When the
        # caller passes None we need this round-trip to tell the UI which
        # device the host API picked — otherwise "自動" is opaque.
        try:
            stream_dev = self.stream.device
            if isinstance(stream_dev, (tuple, list)) and len(stream_dev) >= 2:
                in_idx, out_idx = stream_dev[0], stream_dev[1]
            else:
                in_idx, out_idx = None, None
            if in_idx is not None and in_idx >= 0:
                self._input_device_name = sd.query_devices(in_idx).get(
                    "name", "unknown"
                )
                self._input_device_index = int(in_idx)
            if out_idx is not None and out_idx >= 0:
                self._output_device_name = sd.query_devices(out_idx).get(
                    "name", "unknown"
                )
                self._output_device_index = int(out_idx)
        except Exception:
            logger.exception("failed to resolve active device names")

        try:
            send_notification(
                "realtime_event",
                {
                    "timestamp": time.time(),
                    "level": "info",
                    "kind": "started",
                    "message": (
                        f"started — in:{self._input_device_name} "
                        f"out:{self._output_device_name} "
                        f"sr:{samplerate} block:{block_time:.3f}s "
                        f"thresh:{threshold}dB f0:{f0method}"
                    ),
                },
            )
        except Exception:
            pass

    def _audio_callback(self, indata, outdata, frames, times, status):
        _cb_start = time.perf_counter()

        # CallbackFlags: bit-set sounddevice passes when the host API detects
        # an under/overrun. Cheap to read and invaluable for diagnosing "why
        # is the output choppy" — we surface cumulative counts to the UI.
        if status:
            try:
                if status.input_underflow:
                    self._cb_input_underflow += 1
                if status.input_overflow:
                    self._cb_input_overflow += 1
                if status.output_underflow:
                    self._cb_output_underflow += 1
                if status.output_overflow:
                    self._cb_output_overflow += 1
                if getattr(status, "priming_output", False):
                    self._cb_priming_output += 1
            except Exception:
                pass

        # Pre-gate input peak/RMS — reflects what the microphone actually delivered.
        try:
            in_peak = float(np.abs(indata).max()) if indata.size else 0.0
            if in_peak > self._max_input_peak:
                self._max_input_peak = in_peak
            in_rms_lin = float(np.sqrt(np.mean(indata * indata) + 1e-12))
            self._last_input_rms_db = 20.0 * float(np.log10(in_rms_lin + 1e-7))
        except Exception:
            pass

        try:
            indata_mono = librosa.to_mono(indata.T)
            # Noise gate
            gated_all = False
            if self.threshold > -60:
                indata_mono = np.append(self.rms_buffer, indata_mono)
                rms = librosa.feature.rms(
                    y=indata_mono, frame_length=4 * self.zc, hop_length=self.zc
                )[:, 2:]
                self.rms_buffer[:] = indata_mono[-4 * self.zc :]
                indata_mono = indata_mono[2 * self.zc - self.zc // 2 :]
                db_thresh = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.threshold
                for i in range(db_thresh.shape[0]):
                    if db_thresh[i]:
                        indata_mono[i * self.zc : (i + 1) * self.zc] = 0
                indata_mono = indata_mono[self.zc // 2 :]
                # True only when the gate zeroed every sub-block — lets the UI
                # distinguish "ゲート作動で意図的に無音" from "マイクが何も拾ってない".
                gated_all = bool(db_thresh.all())
            self._last_is_gated = gated_all

            # Shift input buffer
            self.input_wav[: -self.block_frame] = self.input_wav[
                self.block_frame :
            ].clone()
            self.input_wav[-indata_mono.shape[0] :] = torch.from_numpy(indata_mono).to(
                self.device_str
            )

            # Resample to 16k
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                self.block_frame_16k :
            ].clone()
            self.input_wav_res[-160 * (indata_mono.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata_mono.shape[0] - 2 * self.zc :])[
                    160:
                ]
            )

            # RVC inference
            _infer_start = time.perf_counter()
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.f0method,
                self.protect,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
            self._last_inference_ms = (time.perf_counter() - _infer_start) * 1000.0

            # SOLA crossfade
            conv_input = infer_wav[
                None, None, : self.sola_buffer_frame + self.sola_search_frame
            ]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_ones) + 1e-8)
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()
            infer_wav = infer_wav[sola_offset:]
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
            self.sola_buffer[:] = infer_wav[
                self.block_frame : self.block_frame + self.sola_buffer_frame
            ]
            outdata[:] = (
                infer_wav[: self.block_frame]
                .repeat(self.out_channels, 1)
                .t()
                .cpu()
                .numpy()
            )
            self._error_count = 0  # Reset on success.

            # Post-render output peak/RMS.
            try:
                out_peak = float(np.abs(outdata).max()) if outdata.size else 0.0
                if out_peak > self._max_output_peak:
                    self._max_output_peak = out_peak
                out_rms_lin = float(np.sqrt(np.mean(outdata * outdata) + 1e-12))
                self._last_output_rms_db = 20.0 * float(np.log10(out_rms_lin + 1e-7))
            except Exception:
                pass
        except Exception as exc:
            outdata[:] = 0
            self._last_output_rms_db = -90.0
            self._error_count += 1
            if self._error_count <= 3:
                logger.exception("realtime callback error #%d", self._error_count)
            if self._error_count >= 10 and not self._error_notified:
                self._error_notified = True
                try:
                    send_notification(
                        "realtime_error",
                        {
                            "error": str(exc),
                            "count": self._error_count,
                        },
                    )
                    send_notification(
                        "realtime_event",
                        {
                            "timestamp": time.time(),
                            "level": "error",
                            "kind": "callback_error",
                            "message": f"推論コールバック例外が連続発生: {exc}",
                        },
                    )
                except Exception:
                    pass

        self._last_block_ms = (time.perf_counter() - _cb_start) * 1000.0

        # Throttled emit (~10Hz) — send_notification hands off to a writer
        # thread so the real-time callback just enqueues a dict.
        now = time.time()
        if (now - self._last_emit_time) >= 0.1:
            self._emit_metrics(now)
            self._last_emit_time = now

    def _emit_metrics(self, now: float) -> None:
        """Publish the latest diagnostics snapshot to the UI.

        Called from the audio callback — keep it cheap. send_notification
        just does Queue.put so no blocking I/O on the real-time thread.
        Peak accumulators reset here so the UI sees per-emit max, not lifetime max.
        """

        def _safe_db(peak_lin: float) -> float:
            return 20.0 * float(np.log10(max(peak_lin, 1e-7)))

        params = {
            "timestamp": now,
            "input_rms_db": round(self._last_input_rms_db, 1),
            "output_rms_db": round(self._last_output_rms_db, 1),
            "input_peak_db": round(_safe_db(self._max_input_peak), 1),
            "output_peak_db": round(_safe_db(self._max_output_peak), 1),
            "input_clipped": bool(self._max_input_peak >= 0.999),
            "output_clipped": bool(self._max_output_peak >= 0.999),
            "is_gated": bool(self._last_is_gated),
            "inference_ms": round(self._last_inference_ms, 2),
            "block_ms": round(self._last_block_ms, 2),
            "block_time_budget_ms": round(
                self.block_frame / self.samplerate * 1000.0, 2
            ),
            "error_count": int(self._error_count),
            "callback_flags": {
                "input_underflow": int(self._cb_input_underflow),
                "input_overflow": int(self._cb_input_overflow),
                "output_underflow": int(self._cb_output_underflow),
                "output_overflow": int(self._cb_output_overflow),
                "priming_output": int(self._cb_priming_output),
            },
            "input_device_index": int(self._input_device_index),
            "output_device_index": int(self._output_device_index),
            "input_device_name": str(self._input_device_name),
            "output_device_name": str(self._output_device_name),
            "samplerate": int(self.samplerate),
            "threshold_db": float(self.threshold),
        }
        try:
            send_notification("realtime_metrics", params)
        except Exception:
            pass
        self._max_input_peak = 0.0
        self._max_output_peak = 0.0

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
            finally:
                self.stream = None
        # Release the loaded RVC model (HuBERT + net_g + RMVPE + index +
        # persistent device tensors) and return its blocks to the allocator.
        # Without this, every start/stop or model-switch cycle leaks GPU/MPS
        # memory because the caching allocator keeps the freed blocks cached.
        self.rvc = None
        _empty_device_cache()
        try:
            send_notification(
                "realtime_event",
                {
                    "timestamp": time.time(),
                    "level": "info",
                    "kind": "stopped",
                    "message": "session stopped",
                },
            )
        except Exception:
            pass


def rpc_realtime_start(params: dict) -> dict:
    global app_state
    import sounddevice as sd

    # Stop any existing session.
    if app_state.realtime is not None:
        try:
            app_state.realtime.stop()
        except Exception:
            pass
        app_state.realtime = None

    weight_root = require_env_root("weight_root")
    pth_path = _resolve_under_any([weight_root], params.get("pth_path", ""), "pth_path")
    index_path = _resolve_under_any(
        _index_roots(), params.get("index_path", ""), "index_path"
    )
    pitch = params.get("pitch", 0)
    formant = params.get("formant", 0)
    index_rate = params.get("index_rate", 0)
    threshold = params.get("threshold", -60)
    block_time = params.get("block_time", 0.25)
    sample_rate = int(params.get("sample_rate", 48000))
    f0method = params.get("f0_method", "fcpe")
    protect = params.get("protect", 0.33)
    input_device = params.get("input_device")
    output_device = params.get("output_device")

    if input_device is not None:
        input_device = int(input_device)
    if output_device is not None:
        output_device = int(output_device)

    try:
        visible_devices = sd.query_devices()
    except Exception as e:
        return {"status": "error", "error": f"Audio device query failed: {e}"}

    has_input = any(d["max_input_channels"] > 0 for d in visible_devices)
    has_output = any(d["max_output_channels"] > 0 for d in visible_devices)
    if input_device is None and not has_input:
        return {
            "status": "error",
            "error": "No Core Audio input device is visible to the Python backend.",
        }
    if output_device is None and not has_output:
        return {
            "status": "error",
            "error": "No Core Audio output device is visible to the Python backend.",
        }
    if input_device is not None:
        try:
            input_info = sd.query_devices(input_device)
            if input_info.get("max_input_channels", 0) <= 0:
                return {
                    "status": "error",
                    "error": f"Selected input device has no input channels: {input_device}",
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Invalid input device {input_device}: {e}",
            }
    if output_device is not None:
        try:
            output_info = sd.query_devices(output_device)
            if output_info.get("max_output_channels", 0) <= 0:
                return {
                    "status": "error",
                    "error": f"Selected output device has no output channels: {output_device}",
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Invalid output device {output_device}: {e}",
            }

    device = app_state.config.device
    is_half = app_state.config.is_half

    try:
        from infer.lib.rtrvc import RVC

        rvc_obj = RVC(
            key=pitch,
            formant=formant,
            pth_path=pth_path,
            index_path=index_path,
            index_rate=index_rate,
            n_cpu=min(os.cpu_count() or 4, 4),
            device=device,
            is_half=is_half,
        )
    except Exception as e:
        _empty_device_cache()  # reclaim any partially-loaded model tensors
        return {"status": "error", "error": f"Model load failed: {e}"}

    # Input is always mono (we downmix). Output channels from the device.
    out_channels = 2
    if output_device is not None:
        try:
            dev_info = sd.query_devices(output_device)
            out_channels = min(dev_info.get("max_output_channels", 2), 2)
        except Exception:
            out_channels = 2
    channels = (1, out_channels)

    try:
        rt = _RealtimeVC(
            rvc_obj=rvc_obj,
            samplerate=sample_rate,
            channels=channels,
            block_time=block_time,
            threshold=threshold,
            f0method=f0method,
            protect=protect,
            device_str=device,
            input_device=input_device,
            output_device=output_device,
        )
    except Exception as e:
        # The RVC model was fully built before the stream failed — drop it and
        # reclaim its device memory instead of leaking it on every failed start.
        try:
            del rvc_obj
        except Exception:
            pass
        _empty_device_cache()
        return {"status": "error", "error": f"Stream start failed: {e}"}

    app_state.realtime = rt
    app_state.status("realtime")
    return {
        "status": "success",
        "sample_rate": sample_rate,
        "model_sr": rvc_obj.tgt_sr,
        "input_device_index": rt._input_device_index,
        "output_device_index": rt._output_device_index,
        "input_device_name": rt._input_device_name,
        "output_device_name": rt._output_device_name,
        "block_time_ms": round(rt.block_frame / sample_rate * 1000.0, 2),
    }


def rpc_realtime_stop(params: dict) -> dict:
    global app_state
    if app_state.realtime is not None:
        app_state.realtime.stop()
        app_state.realtime = None
    app_state.status("idle")
    return {"status": "stopped"}


def rpc_realtime_update_params(params: dict) -> dict:
    global app_state
    rt = app_state.realtime
    # rt.rvc is dropped to None by _RealtimeVC.stop(); guard against a param
    # update that races a concurrent stop (update runs on the dispatch thread,
    # stop on the blocking worker) so we return cleanly instead of AttributeError.
    if rt is None or getattr(rt, "rvc", None) is None:
        return {"status": "not_running"}
    updated = {}
    if "pitch" in params:
        rt.rvc.set_key(params["pitch"])
        updated["pitch"] = params["pitch"]
    if "formant" in params:
        rt.rvc.set_formant(params["formant"])
        updated["formant"] = params["formant"]
    if "index_rate" in params:
        rt.rvc.set_index_rate(params["index_rate"])
        updated["index_rate"] = params["index_rate"]
    if updated:
        try:
            send_notification(
                "realtime_event",
                {
                    "timestamp": time.time(),
                    "level": "info",
                    "kind": "params_updated",
                    "message": "updated: "
                    + ", ".join(f"{k}={v}" for k, v in updated.items()),
                },
            )
        except Exception:
            pass
    return {"status": "updated"}


def rpc_cancel(params: dict) -> dict:
    tid = params.get("task_id", "")
    force = bool(params.get("force", False))
    return {"cancelled": _cancel_task(tid, force=force)}


def _reap_child_processes(force: bool = False, timeout: float = 1.5) -> None:
    """Terminate every descendant process so training/UVR5 workers don't orphan.

    Heavy workers are spawned with ``start_new_session=True`` (training) or as
    multiprocessing 'spawn' children (UVR5), so they leave the backend's process
    group — a plain parent-pid kill from Swift never reaches them. They do
    remain children in the process tree, so psutil finds them here. ``force``
    SIGKILLs immediately (used on the SIGTERM fast path); otherwise SIGTERM then
    SIGKILL any survivors after a grace window.
    """
    if psutil is None:
        return
    try:
        children = psutil.Process().children(recursive=True)
    except Exception:
        return
    if not children:
        return
    for c in children:
        try:
            c.kill() if force else c.terminate()
        except Exception:
            pass
    if force:
        return
    try:
        _, alive = psutil.wait_procs(children, timeout=timeout)
    except Exception:
        alive = children
    for c in alive:
        try:
            c.kill()
        except Exception:
            pass


def _cleanup_for_exit(force: bool = False) -> None:
    """Best-effort teardown before the process exits: stop the realtime audio
    stream (so Core Audio releases the device cleanly) and reap child workers."""
    try:
        if app_state is not None and app_state.realtime is not None:
            app_state.realtime.stop()
            app_state.realtime = None
    except Exception:
        pass
    _reap_child_processes(force=force)


def _on_sigterm(signum, frame):
    # Swift's killSync() sends SIGTERM (then SIGKILL after a short grace). Reap
    # our children — which that signal can't reach — and stop the audio stream
    # before exiting. force=True so children die within the grace window.
    try:
        _cleanup_for_exit(force=True)
    finally:
        os._exit(0)


def rpc_shutdown(params: dict) -> dict:
    send_notification("shutting_down", {})

    def _do_shutdown():
        _cleanup_for_exit()
        os._exit(0)

    threading.Timer(0.2, _do_shutdown).start()
    return {"ok": True}


# Training endpoints are stubs for Phase 6; the Swift side can detect
# unsupported methods via a clear error code until they land.
def rpc_not_implemented(method: str):
    def _impl(params: dict) -> dict:
        raise RuntimeError(f"Method '{method}' is not implemented yet")

    return _impl


METHODS: Dict[str, Callable[[dict], Any]] = {
    "initialize": rpc_initialize,
    "list_models": rpc_list_models,
    "list_indices": rpc_list_indices,
    "list_uvr5_models": rpc_list_uvr5_models,
    "load_model": rpc_load_model,
    "unload_model": rpc_unload_model,
    "vc_single": rpc_vc_single,
    "vc_multi": rpc_vc_multi,
    "uvr5": rpc_uvr5,
    "model_info": rpc_model_info,
    "model_change_info": rpc_model_change_info,
    "model_compare": rpc_model_compare,
    "model_merge": rpc_model_merge,
    "model_extract": rpc_model_extract,
    "export_onnx": rpc_export_onnx,
    "list_audio_devices": rpc_list_audio_devices,
    "realtime_start": rpc_realtime_start,
    "realtime_stop": rpc_realtime_stop,
    "realtime_update_params": rpc_realtime_update_params,
    "cancel": rpc_cancel,
    "shutdown": rpc_shutdown,
}


def _install_training_methods() -> None:
    """Late-import to avoid paying for training module imports on startup."""
    try:
        from rpc_training import build_methods as _build_training
    except Exception as e:
        logger.error("failed to import rpc_training: %s", e)
        return

    ctx = {
        "config": app_state.config,
        "emit_progress": emit_progress,
        "register_task": _register_task,
        "unregister_task": _unregister_task,
        "status": lambda s: app_state.status(s),
    }
    methods = _build_training(ctx)
    # Override placeholder methods with the real implementations.
    for name, fn in methods.items():
        METHODS[name] = fn
        # Training methods are blocking too.
        BLOCKING_METHODS.add(name)


# ---------------------------------------------------------------------------
# Dispatch loop. Requests that may block (inference, separation) are dispatched
# to a dedicated worker thread so fast methods like list_models and cancel keep
# responding.
# ---------------------------------------------------------------------------

BLOCKING_METHODS = {
    "vc_single",
    "vc_multi",
    "uvr5",
    "model_merge",
    "model_extract",
    "export_onnx",
    "realtime_start",
    # realtime_stop is blocking so it serializes behind realtime_start on
    # _blocking_worker. Otherwise a Stop racing a still-loading Start runs on
    # the dispatch thread, observes app_state.realtime is still None, and does
    # nothing — leaving the just-started audio stream orphaned with no UI
    # handle. Serializing makes Stop wait until Start has published, then tear
    # it down. (realtime_start runs on its own worker thread, so the stdin
    # dispatch loop itself is never blocked.)
    "realtime_stop",
    "preprocess",
    "extract_f0",
    "train",
    "train_index",
    "train_all",
}

_blocking_worker = threading.Lock()


def _dispatch(request: dict) -> None:
    method = request.get("method", "")
    params = request.get("params") or {}
    req_id = request.get("id")
    fn = METHODS.get(method)
    if fn is None:
        send_error(req_id, -32601, f"Method not found: {method}")
        return
    try:
        if method in BLOCKING_METHODS:
            # Serialize blocking ops so we don't load two models at once.
            with _blocking_worker:
                result = fn(params)
        else:
            result = fn(params)
        send_response(req_id, result)
    except PathValidationError as e:
        # Domain-level validation failure: invalid params, not a server bug.
        # No traceback — the message alone is what the UI should surface.
        logger.error("RPC validation error in %s: %s", method, e)
        send_error(req_id, -32602, str(e))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("RPC error in %s: %s\n%s", method, e, tb)
        send_error(req_id, -32603, str(e), {"traceback": tb})


def _handle_request_async(request: dict) -> None:
    method = request.get("method", "")
    if method in BLOCKING_METHODS:
        threading.Thread(target=_dispatch, args=(request,), daemon=True).start()
    else:
        _dispatch(request)


def main():
    global app_state

    # Reap child workers + stop the audio stream when Swift terminates us via
    # SIGTERM (its killSync path), so training/UVR5 workers don't orphan and
    # the audio device is released cleanly.
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass  # not on the main thread / unsupported platform

    # Start stdout writer thread before anything that might emit notifications.
    threading.Thread(target=_writer_thread, daemon=True).start()

    # Emit an "alive" notification IMMEDIATELY so Swift knows the process
    # started. Heavy imports above may have taken several seconds; without
    # this early signal, Swift's waitForReady() could time out on cold boot.
    send_notification(
        "alive",
        {"pid": os.getpid(), "python_version": sys.version.split()[0]},
    )
    # Force another flush for good measure.
    try:
        sys.stdout.flush()
    except Exception:
        pass

    logger.info("rpc_server starting (pid=%d)", os.getpid())
    app_state = AppState()
    _install_training_methods()
    send_notification(
        "ready",
        {
            "pid": os.getpid(),
            "device": app_state.config.device,
            "base_dir": str(app_state.config.base_dir),
            "user_dir": str(app_state.config.user_dir),
        },
    )
    try:
        sys.stdout.flush()
    except Exception:
        pass

    # Start resource monitor.
    stop_evt = threading.Event()
    threading.Thread(target=_resource_monitor, args=(stop_evt,), daemon=True).start()

    # Read loop: each line is a JSON-RPC request or notification.
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as e:
                send_error(None, -32700, f"Parse error: {e}")
                continue
            _handle_request_async(req)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        logger.info("rpc_server exiting")


if __name__ == "__main__":
    main()

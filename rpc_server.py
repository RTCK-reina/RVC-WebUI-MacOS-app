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
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
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

from configs import Config  # noqa: E402
from infer.lib.audio import save_audio  # noqa: E402
from infer.modules.vc import VC, show_info, hash_similarity  # noqa: E402
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

# ---------------------------------------------------------------------------
# Stdout writer -- every line is a single JSON object, flushed immediately.
# Thread-safe via a background writer thread (multiple workers may emit
# progress notifications concurrently).
# ---------------------------------------------------------------------------

_write_queue: "Queue[str]" = Queue()


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


def _send(obj: dict) -> None:
    _write_queue.put(json.dumps(obj, ensure_ascii=False, default=str))


def send_notification(method: str, params: dict) -> None:
    _send({"jsonrpc": "2.0", "method": method, "params": params})


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


def _cancel_task(task_id: str) -> bool:
    with _tasks_lock:
        t = _tasks.get(task_id)
    if t is None:
        return False
    t.cancel_event.set()
    return True


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
        self.active_status = "idle"  # idle / inferring / training / separating / realtime

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

    while not stop_event.is_set():
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
                stats["memory_used_gb"] = round(vm.used / 1024 ** 3, 2)
                stats["memory_total_gb"] = round(vm.total / 1024 ** 3, 2)
                stats["memory_percent"] = float(vm.percent)
                proc = psutil.Process(os.getpid())
                stats["process_memory_gb"] = round(
                    proc.memory_info().rss / 1024 ** 3, 2
                )
            except Exception:
                pass

        # MPS / CUDA memory.
        try:
            if torch.backends.mps.is_available():
                stats["gpu_memory_used_mb"] = int(
                    torch.mps.current_allocated_memory() / (1024 * 1024)
                )
                try:
                    stats["gpu_memory_driver_mb"] = int(
                        torch.mps.driver_allocated_memory() / (1024 * 1024)
                    )
                except Exception:
                    pass
                stats["gpu_backend"] = "mps"
            elif torch.cuda.is_available():
                stats["gpu_memory_used_mb"] = int(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                )
                stats["gpu_memory_reserved_mb"] = int(
                    torch.cuda.memory_reserved() / (1024 * 1024)
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
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
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


def _default_output_path(input_path: str, model_sid: str, fmt: str, subdir: str) -> str:
    out_root = Path(os.environ.get("output_root") or (Path.home() / "Documents" / "RVC-WebUI" / "output"))
    out_dir = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    model_stem = Path(model_sid).stem if model_sid else "model"
    return str(out_dir / f"{stem}_{model_stem}_{_timestamp()}.{fmt}")


def rpc_vc_single(params: dict) -> dict:
    assert app_state is not None
    sid = params["sid"]
    input_path = params["input_audio_path"]
    fmt = params.get("format", "flac")
    output_path = params.get("output_path") or _default_output_path(
        input_path, sid, fmt, "inference"
    )

    task_id = params.get("task_id", f"vc_single_{int(time.time()*1000)}")
    _register_task(task_id)  # registered for status visibility, no cancel honored
    app_state.status("inferring")
    try:
        # B-2: the previous implementation ran vc_single inside a fresh
        # daemon Thread and busy-waited on `while t.is_alive(): t.join(0.25)`
        # solely so it could emit a "推論中…" progress tick every second.
        # Now that the RPC is already dispatched from _blocking_executor
        # (on its own worker thread), spinning up another thread here is
        # pure overhead — the dispatcher thread is free to block on
        # vc_single directly. We keep the up-front "Loading audio" and
        # "推論中…" progress notifications so the UI bar still moves before
        # the synchronous call starts.
        emit_progress(task_id, 5, "Loading audio", "inference")
        emit_progress(task_id, 40, "推論中…", "inference")

        # Cancellation is intentionally NOT honored for single-file inference:
        # a single PyTorch forward pass cannot be safely interrupted, and the
        # UI does not expose a cancel button for this operation.
        info, opt = app_state.vc.vc_single(
            params.get("sid_index", 0),
            input_path,
            int(params.get("f0_up_key", 0)),
            None,
            params.get("f0_method", "rmvpe"),
            params.get("file_index", ""),
            params.get("file_index2", ""),
            float(params.get("index_rate", 0.75)),
            int(params.get("filter_radius", 3)),
            int(params.get("resample_sr", 0)),
            float(params.get("rms_mix_rate", 0.25)),
            float(params.get("protect", 0.33)),
        )
        if opt is None:
            return {"status": "error", "info": info}

        emit_progress(task_id, 90, "Saving output", "inference")
        tgt_sr, audio_opt = opt
        save_audio(output_path, audio_opt, tgt_sr, f32=True, format=fmt)
        emit_progress(task_id, 100, "Done", "inference")

        return {
            "status": "success",
            "info": info,
            "output_path": output_path,
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
    fmt = params.get("format", "flac")
    out_root = params.get("output_dir") or str(
        Path(os.environ.get("output_root") or (Path.home() / "Documents" / "RVC-WebUI" / "output")) / "batch" / _timestamp()
    )
    Path(out_root).mkdir(parents=True, exist_ok=True)

    task_id = params.get("task_id", f"vc_multi_{int(time.time()*1000)}")
    task = _register_task(task_id)
    app_state.status("inferring")
    try:
        # Collect files.
        if dir_path:
            all_paths = [str(p) for p in Path(dir_path).iterdir() if p.is_file()]
        else:
            all_paths = list(paths)
        total = len(all_paths)
        if total == 0:
            return {"status": "error", "info": "No input files"}

        # B-3: the VC inference itself is kept strictly sequential because
        # net_g / Pipeline / GPU allocator are shared state and not
        # thread-safe. However the post-inference `save_audio` work
        # (resampling + encoding + disk write) is independent between
        # files and can happily run on I/O threads while the next file is
        # being inferred. A small ThreadPoolExecutor (2 workers) is enough
        # to overlap the encode of file N with the forward pass of file
        # N+1 — more workers would only help for very large outputs, and
        # risks contending for the disk.
        save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vc-multi-save")
        save_futures: list = []
        results = []
        try:
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
                    params.get("file_index", ""),
                    params.get("file_index2", ""),
                    float(params.get("index_rate", 0.75)),
                    int(params.get("filter_radius", 3)),
                    int(params.get("resample_sr", 0)),
                    float(params.get("rms_mix_rate", 0.25)),
                    float(params.get("protect", 0.33)),
                )
                if opt is not None:
                    tgt_sr, audio_opt = opt
                    model_stem = Path(sid).stem if sid else "model"
                    out_name = f"{Path(path).stem}_{model_stem}.{fmt}"
                    out_path = str(Path(out_root) / out_name)
                    # save_audio mutates audio_opt in place during encoding
                    # on some formats; copy so the async write doesn't race
                    # with the next iteration's numpy buffers. The copy cost
                    # is negligible next to encode+disk.
                    save_futures.append(
                        save_pool.submit(
                            save_audio, out_path, audio_opt.copy(), tgt_sr,
                            True, fmt,  # f32=True, format=fmt
                        )
                    )
                    results.append({"input": path, "output": out_path, "info": info})
                else:
                    results.append({"input": path, "output": None, "info": info})
            # Wait for pending encodes before returning; surface the first
            # encoding error if any.
            for fut in save_futures:
                fut.result()
        finally:
            save_pool.shutdown(wait=True)
        emit_progress(task_id, 100, f"Completed {total} files", "batch")
        return {"status": "success", "results": results, "output_dir": out_root}
    finally:
        _unregister_task(task_id)
        app_state.status("idle")


def rpc_uvr5(params: dict) -> dict:
    """Run UVR5 vocal/instrumental separation.

    Optional second-pass "polish": set `polish_model` to another UVR5 model
    (typically a DeEcho/DeReverb model) and the vocal output of the first
    pass is fed through it, with the polished result written to
    {output_vocal}/polished/.
    """
    assert app_state is not None
    model_name = params["model_name"]
    inp_root = params.get("input_dir", "") or ""
    paths = params.get("paths", []) or []
    save_root_vocal = params.get("output_vocal") or str(
        Path(os.environ.get("output_root") or "") / "separation" / "vocals"
    )
    save_root_ins = params.get("output_accompaniment") or str(
        Path(os.environ.get("output_root") or "") / "separation" / "accompaniment"
    )
    Path(save_root_vocal).mkdir(parents=True, exist_ok=True)
    Path(save_root_ins).mkdir(parents=True, exist_ok=True)
    agg = int(params.get("agg", 10))
    fmt = params.get("format", "flac")
    polish_model = (params.get("polish_model") or "").strip()

    task_id = params.get("task_id", f"uvr5_{int(time.time()*1000)}")
    task = _register_task(task_id)
    app_state.status("separating")

    # UVR5 does not natively report progress; we estimate via file count.
    # Defer file filtering to the uvr() function, which has stricter rules,
    # but still pre-filter to produce an accurate total for progress.
    from infer.modules.uvr5.modules import _is_audio_file  # lazy import
    if inp_root:
        # Directory scan: strict filtering (skip our own .reformatted.wav).
        input_paths = [
            str(p) for p in Path(inp_root).iterdir()
            if p.is_file() and _is_audio_file(p.name, strict=True)
        ]
    else:
        # User-selected individual files: lenient filtering.
        raw = [p if isinstance(p, str) else p.get("name") for p in paths]
        input_paths = [p for p in raw if p and _is_audio_file(p, strict=False)]

    class _Faux:
        def __init__(self, name):
            self.name = name

    total = max(1, len(input_paths))
    # Reserve half the progress bar for polish pass if enabled.
    pass1_scale = 50.0 if polish_model else 100.0
    messages = []
    polished_dir = None
    try:
        # --- Pass 1: primary separation -----------------------------------
        faux_paths = [_Faux(p) for p in input_paths]
        done = 0
        for msg in uvr(
            model_name, inp_root, save_root_vocal, faux_paths,
            save_root_ins, agg, fmt,
        ):
            if task.cancel_event.is_set():
                return {"status": "cancelled", "messages": messages}
            messages.append(msg)
            done = min(total, msg.count("->"))
            emit_progress(
                task_id,
                (done / total) * pass1_scale,
                f"({done}/{total}) 分離中…",
                "separation",
            )

        # --- Pass 2: polish pass (optional) -------------------------------
        if polish_model:
            emit_progress(task_id, pass1_scale, "仕上げ準備中…", "separation")

            # Collect vocal outputs produced by pass 1. Reverse-direction
            # models (HP3, DeEcho*) write "vocal_*" into save_root_ins while
            # non-reverse models write it into save_root_vocal — scan both.
            vocal_files: list[str] = []
            seen: set[str] = set()
            for d in (save_root_vocal, save_root_ins):
                dp = Path(d)
                if not dp.is_dir():
                    continue
                for p in dp.iterdir():
                    if not p.is_file():
                        continue
                    if not p.name.startswith("vocal_"):
                        continue
                    if not _is_audio_file(p.name, strict=False):
                        continue
                    full = str(p.resolve())
                    if full in seen:
                        continue
                    seen.add(full)
                    vocal_files.append(full)

            if not vocal_files:
                messages.append("Polish skipped: no vocal_* outputs found.")
            else:
                import shutil as _shutil
                polished_dir = str(Path(save_root_vocal) / "polished")
                Path(polished_dir).mkdir(parents=True, exist_ok=True)
                residue_dir = str(Path(save_root_vocal) / "polished" / "residue")
                Path(residue_dir).mkdir(parents=True, exist_ok=True)

                # IMPORTANT: uvr() may reformat input and DELETE the original
                # file when converting to 44100/stereo. Copy pass-1 vocals to
                # a scratch dir first so the user's output files survive.
                scratch_dir = Path(os.environ.get("TEMP") or "/tmp") / f"polish_{task_id}"
                scratch_dir.mkdir(parents=True, exist_ok=True)
                scratch_paths: list[str] = []
                for vf in vocal_files:
                    dst = scratch_dir / os.path.basename(vf)
                    try:
                        _shutil.copy2(vf, dst)
                        scratch_paths.append(str(dst))
                    except Exception as e:
                        logger.warning("polish scratch copy failed for %s: %s", vf, e)

                polish_paths = [_Faux(p) for p in scratch_paths]
                p_total = max(1, len(scratch_paths))
                p_done = 0
                try:
                    for msg in uvr(
                        polish_model, "", polished_dir,
                        polish_paths, residue_dir, agg, fmt,
                    ):
                        if task.cancel_event.is_set():
                            return {
                                "status": "cancelled",
                                "messages": messages,
                                "polished_dir": polished_dir,
                            }
                        messages.append(msg)
                        p_done = min(p_total, msg.count("->"))
                        emit_progress(
                            task_id,
                            pass1_scale + (p_done / p_total) * (100 - pass1_scale),
                            f"仕上げ中 ({p_done}/{p_total})",
                            "separation",
                        )
                finally:
                    # Clean up scratch dir regardless of success/cancel.
                    _shutil.rmtree(scratch_dir, ignore_errors=True)

        emit_progress(task_id, 100, "完了", "separation")
        result = {
            "status": "success",
            "messages": messages,
            "output_vocal": save_root_vocal,
            "output_accompaniment": save_root_ins,
        }
        if polished_dir:
            result["polished_dir"] = polished_dir
        return result
    finally:
        _unregister_task(task_id)
        app_state.status("idle")


def rpc_model_info(params: dict) -> dict:
    path = params["path"]
    return {"info": show_info(path)}


def rpc_model_change_info(params: dict) -> dict:
    path = params["path"]
    info = params.get("info", "")
    name = params.get("name", "")
    return {"result": change_info(path, info, name)}


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


def rpc_model_merge(params: dict) -> dict:
    task_id = params.get("task_id", f"model_merge_{int(time.time()*1000)}")
    return _run_with_progress(
        task_id, "merge",
        merge,
        params["path_a"],
        params["path_b"],
        float(params.get("alpha", 0.5)),
        int(params.get("sr", 40000)),
        int(params.get("if_f0", 1)),
        params.get("info", ""),
        params.get("name", ""),
        params.get("version", "v2"),
    )


def rpc_model_extract(params: dict) -> dict:
    task_id = params.get("task_id", f"model_extract_{int(time.time()*1000)}")
    return _run_with_progress(
        task_id, "extract",
        extract_small_model,
        params["ckpt_path"],
        params.get("name", ""),
        int(params.get("sr", 40000)),
        int(params.get("if_f0", 1)),
        params.get("info", ""),
        params.get("version", "v2"),
    )


def rpc_export_onnx(params: dict) -> dict:
    from rvc.onnx import export_onnx  # lazy import (heavy)

    ckpt_path = params["ckpt_path"]
    output_path = params.get("output_path") or str(
        Path(os.environ.get("output_root") or "")
        / "onnx"
        / (Path(ckpt_path).stem + ".onnx")
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    task_id = params.get("task_id", f"export_onnx_{int(time.time()*1000)}")
    result = _run_with_progress(
        task_id, "onnx",
        export_onnx, ckpt_path, output_path,
    )
    if result.get("status") == "cancelled":
        return result
    return {"status": "success", "output_path": output_path}


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


def rpc_cancel(params: dict) -> dict:
    tid = params.get("task_id", "")
    return {"cancelled": _cancel_task(tid)}


def rpc_shutdown(params: dict) -> dict:
    send_notification("shutting_down", {})
    threading.Timer(0.2, lambda: os._exit(0)).start()
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
    "preprocess",
    "extract_f0",
    "train",
    "train_index",
    "train_all",
}

# B-1: single-slot ThreadPoolExecutor replaces the previous
# `_blocking_worker = Lock()` + per-dispatch `threading.Thread(...).start()`
# pattern. Behaviour is equivalent (strict serialization — model and GPU
# state are not thread-safe) but the executor reuses a single worker thread
# instead of spawning a fresh OS thread per RPC, and gives us a proper
# Future handle so the dispatch side can wait on `future.result()` rather
# than spinning on `while thread.is_alive(): thread.join(0.25)` (B-2).
_blocking_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="rpc-blocking"
)


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
            # Submit to the dedicated single-slot executor. max_workers=1 is
            # deliberate: the VC model and GPU allocator are shared state
            # that has never been audited for concurrent access. Future
            # work can raise the cap after adding model-scoped locks
            # (plan file PR #5 main_concerns #1).
            future = _blocking_executor.submit(fn, params)
            result = future.result()
        else:
            result = fn(params)
        send_response(req_id, result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("RPC error in %s: %s\n%s", method, e, tb)
        send_error(req_id, -32603, str(e), {"traceback": tb})


def _handle_request_async(request: dict) -> None:
    # The dispatch wrapper still runs on a background thread so that
    # non-blocking calls (list_models, cancel, resource_stats observers) can
    # proceed while a BLOCKING op is queued on the executor — but the
    # heavy work itself runs in the pool, not in freshly-spawned threads.
    method = request.get("method", "")
    if method in BLOCKING_METHODS:
        threading.Thread(target=_dispatch, args=(request,), daemon=True).start()
    else:
        _dispatch(request)


def main():
    global app_state

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

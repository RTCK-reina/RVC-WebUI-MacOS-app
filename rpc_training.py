"""Training-related RPC methods for rpc_server.py.

Kept in its own file because these flows spawn subprocesses, poll log files,
and are significantly more involved than the inference path. rpc_server.py
imports `register_methods` below to wire them in.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path
from random import shuffle
from typing import Callable, Dict, Optional

import numpy as np

from infer.lib.path_safety import PathValidationError, safe_format, safe_leaf_name

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # Fallback: killpg + kill(pid, 0) only.

logger = logging.getLogger("rpc_training")

# sr name -> Hz (mirrors web.py sr_dict)
_SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}
_VERSIONS = ("v1", "v2")
_F0_METHODS = ("pm", "harvest", "dio", "rmvpe", "rmvpe_gpu", "crepe", "fcpe")


# ---------------------------------------------------------------------------
# Helpers shared across the training steps.
# ---------------------------------------------------------------------------


def _bundle_python_cmd(config) -> str:
    """Pick the python interpreter to launch training subprocesses with.

    In the bundled .app we want to call the bundled python binary
    (Resources/python/bin/python3), not /usr/bin/env python3.
    """
    bundled = Path(config.base_dir).parent / "python" / "bin" / "python3"
    if bundled.exists():
        return str(bundled)
    return config.python_cmd or "python3"


def _exp_dir(config, exp_name: str) -> Path:
    """Experiment directory under the user dir (writable)."""
    safe_name = safe_leaf_name(exp_name, "exp_name")
    p = Path(config.user_dir) / "logs" / safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _require_existing_dir(value: str, field: str) -> str:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise PathValidationError(
            f"{field} does not exist or is not a directory: {value}"
        )
    return str(path.resolve())


def _safe_sr_name(value: object) -> str:
    return safe_format(value or "40k", _SR_DICT.keys(), "sr")


def _safe_version(value: object) -> str:
    return safe_format(value or "v2", _VERSIONS, "version")


def _safe_f0_method(value: object) -> str:
    return safe_format(value or "rmvpe", _F0_METHODS, "f0_method")


def _spawn(
    cmd: list,
    cwd: str,
    log_path: Path,
    env_extra: Optional[dict] = None,
) -> subprocess.Popen:
    """Start a subprocess, redirecting stdout+stderr to log_path.

    The log file descriptor is closed in the parent process right after
    Popen inherits it. The kernel keeps the underlying file open for the
    child via descriptor duplication, so the child can still write until it
    exits; the parent no longer holds a reference that could leak across
    many _spawn() invocations (observed with repeated preprocess/train
    cycles).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("exec: %s (log=%s)", " ".join(cmd), log_path)
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    with open(log_path, "a", encoding="utf-8") as log_f:
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            # Create a new process group so we can kill the entire tree
            # (train.py spawns mp.Process children that would otherwise
            # survive a plain p.terminate()).
            start_new_session=True,
        )


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a non-zombie live process.

    Zombies are treated as dead — after SIGKILL, a direct child sits as
    a zombie until the parent ``wait()``s, and we don't want the verify
    loop spinning on that. Without psutil we try ``waitpid(WNOHANG)``
    first to reap our own zombie children (ECHILD when not our child —
    we then fall back to ``kill(pid, 0)`` existence check). Never raises.
    """
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return False
    # psutil-free path.
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            # It was our child and we just reaped its zombie.
            return False
    except (ChildProcessError, OSError):
        # Not our child, or already reaped. Fall through to kill-0 probe.
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _collect_descendant_pids(pids: list) -> list:
    """Return *pids* plus every descendant PID we can observe via psutil.

    Without psutil we have no cross-platform way to walk the tree, so we
    return the input unchanged — the caller still gets the process-group
    kill path via ``os.killpg``. With psutil we pick up grandchildren that
    escaped the original session (e.g. a helper script that itself called
    ``start_new_session=True``), which ``killpg`` alone would miss.
    Order is preserved and duplicates removed.
    """
    out = list(pids)
    if psutil is None:
        return list(dict.fromkeys(out))
    for pid in pids:
        try:
            for child in psutil.Process(pid).children(recursive=True):
                out.append(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
    return list(dict.fromkeys(out))


def _terminate_procs(
    procs: list,
    graceful_wait: float = 3.0,
    verify: bool = True,
) -> dict:
    """Kill process groups and every descendant we can reach.

    Layered so that "strong kill" requirements are met even when the
    primary target ignores SIGTERM:

    1. When ``graceful_wait > 0``, SIGTERM the process group and wait up
       to ``graceful_wait`` seconds for clean exit.
    2. SIGKILL the process group (covers children spawned in the same
       session) AND every PID returned by
       ``psutil.Process.children(recursive=True)`` (covers grandchildren
       that ``setsid`` into a new session — e.g. internal
       ``subprocess.Popen(..., start_new_session=True)`` inside a training
       helper).
    3. When ``verify=True``, loop for up to 2 seconds re-issuing SIGKILL
       to any PID that stubbornly survives, so callers can distinguish
       "killed cleanly" from "stuck in uninterruptible kernel wait".

    Pass ``graceful_wait=0.0`` to skip SIGTERM entirely — used by the
    ``force`` cancel path (see ``_cancel_task(force=True)``).

    Returns ``{"killed": [pid, ...], "residual": [pid, ...]}``. Existing
    call sites ignore the return value, so this is backwards-compatible.
    Never raises.
    """
    import signal as _signal
    import time as _time

    alive_procs = [p for p in procs if p is not None and p.poll() is None]
    if not alive_procs:
        return {"killed": [], "residual": []}

    ancestor_pids = [p.pid for p in alive_procs]
    all_pids = _collect_descendant_pids(ancestor_pids)

    if graceful_wait > 0:
        for p in alive_procs:
            try:
                os.killpg(os.getpgid(p.pid), _signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    p.terminate()
                except Exception:
                    pass
        deadline = _time.time() + graceful_wait
        while _time.time() < deadline:
            if all(p.poll() is not None for p in alive_procs):
                # Grandchildren may still be around even if the direct
                # Popen child exited; fall through to the SIGKILL sweep.
                break
            _time.sleep(0.1)

    # SIGKILL the process groups first (cheap, catches most children).
    for p in alive_procs:
        try:
            os.killpg(os.getpgid(p.pid), _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.kill()
            except Exception:
                pass
    # Belt-and-suspenders: also SIGKILL every descendant we resolved via
    # psutil. Catches grandchildren that re-sessioned away from the group.
    if psutil is not None:
        for pid in all_pids:
            try:
                psutil.Process(pid).kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

    # Reap Popen returncodes BEFORE the verify loop: without this, a
    # freshly-killed direct child sits as a zombie until wait()
    # acknowledges it, and the psutil-less `os.kill(pid, 0)` fallback
    # still reports the zombie as "alive" — making the verify loop spin
    # for its full 2s window even though the process is fully dead.
    for p in alive_procs:
        try:
            p.wait(timeout=0.5)
        except Exception:
            pass

    residual: list = []
    if verify:
        deadline = _time.time() + 2.0
        while _time.time() < deadline:
            residual = [pid for pid in all_pids if _pid_alive(pid)]
            if not residual:
                break
            for pid in residual:
                try:
                    os.kill(pid, _signal.SIGKILL)
                except OSError:
                    pass
            _time.sleep(0.1)

    if residual:
        logger.warning(
            "_terminate_procs: %d PIDs still alive after SIGKILL: %r",
            len(residual),
            residual,
        )
    return {"killed": all_pids, "residual": residual}


def _tail_log_until_done(
    log_path: Path,
    proc_done_event: threading.Event,
    task_id: str,
    emit_progress: Callable[[str, float, str, str], None],
    cancel_event: Optional[threading.Event] = None,
    proc: Optional[subprocess.Popen] = None,
    procs: Optional[list] = None,
    phase: str = "training",
    percent_from_log: Optional[Callable[[str], Optional[float]]] = None,
    task: Optional[object] = None,
) -> str:
    """Stream log_path growth while the subprocess runs.

    Emits a `progress` notification roughly every second with the tail of
    the log. On cancellation, kill every known subprocess (and its
    descendants) — if ``task.force_kill_requested`` is set, skip the
    SIGTERM grace period entirely and go straight to SIGKILL so "stop"
    feels instantaneous. Returns the full final log contents.

    `proc` accepts a single subprocess.Popen; `procs` accepts a list (for
    multi-process stages like F0 feature extraction). Passing ``task``
    enables the force-kill path and surfaces residual-PID warnings via
    ``emit_progress``.
    """
    last_size = 0
    last_tail = ""
    percent = 0.0
    # Loop every 0.2s so cancel_event detection is at most ~200ms late,
    # but only actually re-read/emit once per second to avoid log churn.
    POLL = 0.2
    EMIT_EVERY = 1.0
    last_emit = 0.0

    while not proc_done_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            target_procs = list(procs) if procs else []
            if proc is not None:
                target_procs.append(proc)
            grace = (
                0.0
                if (task is not None and getattr(task, "force_kill_requested", False))
                else 3.0
            )
            result = _terminate_procs(target_procs, graceful_wait=grace)
            if isinstance(result, dict) and result.get("residual"):
                # Surface "SIGKILL could not reap everything" to the UI so
                # the operator knows a manual cleanup may be required.
                try:
                    emit_progress(
                        task_id,
                        percent,
                        f"stop: residual PIDs {result['residual']}",
                        phase,
                    )
                except Exception:
                    pass
            break
        now = time.monotonic()
        if now - last_emit >= EMIT_EVERY:
            try:
                if log_path.exists():
                    size = log_path.stat().st_size
                    if size != last_size:
                        with open(
                            log_path, "r", encoding="utf-8", errors="replace"
                        ) as f:
                            content = f.read()
                        last_size = size
                        # Show the last ~500 chars of log as progress message.
                        last_tail = content[-500:].strip().replace("\n", " | ")
                        if percent_from_log:
                            derived = percent_from_log(content)
                            if derived is not None:
                                percent = derived
                emit_progress(task_id, percent, last_tail or "running...", phase)
            except Exception as e:
                logger.warning("log tail error: %s", e)
            last_emit = now
        # Use Event.wait instead of sleep so cancel wakes us immediately.
        if cancel_event is not None:
            cancel_event.wait(POLL)
        else:
            time.sleep(POLL)

    # Final read.
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _wait_process(proc: subprocess.Popen, done_event: threading.Event) -> None:
    """Background helper that flips `done_event` once `proc` exits."""
    proc.wait()
    done_event.set()


def _wait_multi_processes(procs: list, done_event: threading.Event) -> None:
    for p in procs:
        p.wait()
    done_event.set()


# ---------------------------------------------------------------------------
# Percent estimators from log tails.
# ---------------------------------------------------------------------------


def _train_percent_from_log(content: str) -> Optional[float]:
    """Very best-effort estimator for the training log.

    infer/modules/train/train.py prints lines like 'Epoch: 3 [123/456]' — we
    scrape the latest epoch number and compare against the user-provided
    total. Anything unparseable returns None.
    """
    total = getattr(_train_percent_from_log, "_total", None)
    if not total:
        return None
    # Match e.g. "Epoch: 3, " or "Epoch: 3 [".
    import re

    matches = list(re.finditer(r"[Ee]poch[: ]+(\d+)", content))
    if not matches:
        return None
    last = int(matches[-1].group(1))
    return max(0.0, min(100.0, (last / total) * 100.0))


def _gpu_backend(config) -> Optional[str]:
    device = str(getattr(config, "device", "") or "")
    if device == "mps":
        return "mps"
    if device.startswith("cuda"):
        return "cuda"
    return None


def _require_gpu(params: dict, config) -> Optional[dict]:
    if not bool(params.get("require_gpu", False)):
        return None
    backend = _gpu_backend(config)
    if backend is not None:
        return None
    details = getattr(config, "mps_error", None)
    error = "GPU training was requested, but no supported GPU backend is available."
    if details:
        error += f" MPS: {details}"
    return {"status": "error", "error": error}


def _count_percent(content: str, marker: str, total: int) -> Optional[float]:
    if total <= 0:
        return None
    done = content.count(marker)
    return max(0.0, min(100.0, (done / total) * 100.0))


# ---------------------------------------------------------------------------
# RPC handlers.
# ---------------------------------------------------------------------------


def rpc_preprocess(params: dict, ctx):
    """Run infer/modules/train/preprocess.py as a subprocess and stream logs."""
    config = ctx["config"]
    emit_progress = ctx["emit_progress"]
    register_task = ctx["register_task"]
    unregister_task = ctx["unregister_task"]
    status = ctx["status"]

    exp_name = safe_leaf_name(params["exp_name"], "exp_name")
    trainset_dir = _require_existing_dir(params["trainset_dir"], "trainset_dir")
    sr_name = _safe_sr_name(params.get("sr", "40k"))
    n_p = int(params.get("n_p", max(1, config.n_cpu)))
    task_id = params.get("task_id", f"preprocess_{int(time.time()*1000)}")

    sr = _SR_DICT.get(sr_name, 40000)
    exp_dir = _exp_dir(config, exp_name)
    log_path = exp_dir / "preprocess.log"
    log_path.write_text("")

    task = register_task(task_id)
    status("training")
    try:
        cmd = [
            _bundle_python_cmd(config),
            str(
                Path(config.base_dir) / "infer" / "modules" / "train" / "preprocess.py"
            ),
            trainset_dir,
            str(sr),
            str(n_p),
            str(exp_dir),
            str(config.noparallel),
            "%.1f" % config.preprocess_per,
        ]
        proc = _spawn(cmd, cwd=str(config.base_dir), log_path=log_path)
        done_event = threading.Event()
        threading.Thread(
            target=_wait_process, args=(proc, done_event), daemon=True
        ).start()
        log = _tail_log_until_done(
            log_path,
            done_event,
            task_id,
            emit_progress,
            cancel_event=task.cancel_event,
            proc=proc,
            phase="preprocessing",
            task=task,
        )
        # Cancellation is authoritative over returncode: after _terminate_procs
        # sends SIGTERM + SIGKILL, `proc.returncode` may still be None if the
        # grace window elapsed before the child actually exited, and
        # `None not in (0, None)` is False — so a plain returncode check would
        # fall through into the `success` branch and the UI would think a
        # cancelled job completed normally. Check the cancel flag first.
        if task.cancel_event.is_set():
            return {"status": "cancelled", "log": log, "exp_dir": str(exp_dir)}
        emit_progress(task_id, 100.0, "preprocess done", "preprocessing")
        # A non-zero exit from preprocess.py means the stage failed (missing
        # inputs, permission error, load_audio traceback, etc.). Report that
        # faithfully instead of masking the failure as success — otherwise
        # train_all happily proceeds through stages that never produced
        # output and the downstream traceback is misleading.
        if proc.returncode not in (0, None):
            return {
                "status": "error",
                "error": "preprocess.py exited with code %d" % proc.returncode,
                "log": log,
                "exp_dir": str(exp_dir),
            }
        # preprocess.py swallows per-file and whole-run exceptions and always
        # exits 0, so a returncode of 0 does NOT prove it produced anything.
        # Verify the slice output is non-empty; otherwise report a clear error
        # instead of letting train_all advance to extract_f0/train on an empty
        # dataset and surface a confusing downstream traceback.
        gt_wavs = exp_dir / "0_gt_wavs"
        if not gt_wavs.is_dir() or not any(gt_wavs.glob("*.wav")):
            return {
                "status": "error",
                "error": "preprocess produced no audio — check the trainset path and that it contains readable audio files",
                "log": log,
                "exp_dir": str(exp_dir),
            }
        return {"status": "success", "log": log, "exp_dir": str(exp_dir)}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        unregister_task(task_id)
        status("idle")


def rpc_extract_f0(params: dict, ctx):
    """Run extract_f0_print.py + extract_feature_print.py and stream logs."""
    config = ctx["config"]
    emit_progress = ctx["emit_progress"]
    register_task = ctx["register_task"]
    unregister_task = ctx["unregister_task"]
    status = ctx["status"]

    exp_name = safe_leaf_name(params["exp_name"], "exp_name")
    f0_method = _safe_f0_method(params.get("f0_method", "rmvpe"))
    if_f0 = bool(params.get("if_f0", True))
    version = _safe_version(params.get("version", "v2"))
    # For rpc usage we assume single GPU (or MPS / CPU). Comma-separated gpus
    # may still be supplied for CUDA boxes.
    gpus = params.get("gpus", "0").strip()
    gpus_list = [g for g in gpus.split("-") if g.strip()]
    n_p = int(params.get("n_p", max(1, config.n_cpu)))
    task_id = params.get("task_id", f"extract_f0_{int(time.time()*1000)}")

    exp_dir = _exp_dir(config, exp_name)
    log_path = exp_dir / "extract_f0_feature.log"
    log_path.write_text("")

    if (gpu_error := _require_gpu(params, config)) is not None:
        return gpu_error

    task = register_task(task_id)
    status("training")
    try:
        base = str(config.base_dir)
        # Step 1: F0 extraction.
        if if_f0 and f0_method != "rmvpe_gpu":
            cmd = [
                _bundle_python_cmd(config),
                str(Path(base) / "infer" / "modules" / "train" / "extract_f0_print.py"),
                str(exp_dir),
                str(n_p),
                f0_method,
                str(config.device),
                str(config.is_half),
            ]
            proc = _spawn(cmd, cwd=base, log_path=log_path)
            done_event = threading.Event()
            threading.Thread(
                target=_wait_process, args=(proc, done_event), daemon=True
            ).start()
            _tail_log_until_done(
                log_path,
                done_event,
                task_id,
                emit_progress,
                cancel_event=task.cancel_event,
                proc=proc,
                phase="f0",
                task=task,
            )
            # Same cancellation-over-returncode ordering as rpc_preprocess:
            # after SIGTERM the returncode may still be None on a racy exit.
            if task.cancel_event.is_set():
                return {"status": "cancelled"}
            if proc.returncode not in (0, None):
                return {
                    "status": "error",
                    "error": "extract_f0_print.py exited with code %d"
                    % proc.returncode,
                }
            # extract_f0_print.py catches per-file failures and exits 0 even
            # when nothing was produced (e.g. RMVPE weights missing). A clean
            # exit is not proof of output — verify both F0 dirs are non-empty.
            f0_dir = exp_dir / "2a_f0"
            f0nsf_dir = exp_dir / "2b-f0nsf"
            if not (f0_dir.is_dir() and any(f0_dir.glob("*.npy"))) or not (
                f0nsf_dir.is_dir() and any(f0nsf_dir.glob("*.npy"))
            ):
                return {
                    "status": "error",
                    "error": "F0 extraction produced no output — the RMVPE weights may be missing or every clip failed",
                }

        # Step 2: Feature extraction (may parallelize across gpus).
        procs = []
        leng = max(1, len(gpus_list) or 1)
        for idx, g in enumerate(gpus_list or ["0"]):
            if task.cancel_event.is_set():
                _terminate_procs(procs)
                return {"status": "cancelled"}
            cmd = [
                _bundle_python_cmd(config),
                str(
                    Path(base)
                    / "infer"
                    / "modules"
                    / "train"
                    / "extract_feature_print.py"
                ),
                str(config.device),
                str(leng),
                str(idx),
                str(g),
                str(exp_dir),
                version,
                str(config.is_half),
            ]
            procs.append(_spawn(cmd, cwd=base, log_path=log_path))

        done_event = threading.Event()
        threading.Thread(
            target=_wait_multi_processes, args=(procs, done_event), daemon=True
        ).start()
        log = _tail_log_until_done(
            log_path,
            done_event,
            task_id,
            emit_progress,
            cancel_event=task.cancel_event,
            procs=procs,
            phase="features",
            task=task,
        )
        # Cancellation-over-returncode: if we were cancelled, one or more
        # workers may still have returncode=None (SIGTERM grace elapsed
        # before exit). Report cancelled before claiming anything else.
        if task.cancel_event.is_set():
            return {"status": "cancelled", "log": log}
        emit_progress(task_id, 100.0, "feature extraction done", "features")
        # Check exit codes of all feature-extraction workers; any non-zero
        # exit means that GPU shard failed (e.g. fairseq import error,
        # missing HuBERT weights, etc.) and the rest of the pipeline has
        # incomplete feature output to train on.
        bad = [p.returncode for p in procs if p.returncode not in (0, None)]
        if bad:
            return {
                "status": "error",
                "error": "extract_feature_print.py worker(s) exited with codes %r"
                % bad,
                "log": log,
            }
        # A worker can still exit 0 after silently dropping every file. Verify
        # the feature directory actually has output so train.py is not started
        # against an empty 3_featureNNN dir (which yields an opaque crash far
        # from the real cause).
        feat_dir = exp_dir / ("3_feature768" if version == "v2" else "3_feature256")
        if not (feat_dir.is_dir() and any(feat_dir.glob("*.npy"))):
            return {
                "status": "error",
                "error": "feature extraction produced no output — the HuBERT weights may be missing or every clip failed",
                "log": log,
            }
        return {"status": "success", "log": log}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        unregister_task(task_id)
        status("idle")


def rpc_train(params: dict, ctx):
    """Launch infer/modules/train/train.py and stream its log."""
    config = ctx["config"]
    emit_progress = ctx["emit_progress"]
    register_task = ctx["register_task"]
    unregister_task = ctx["unregister_task"]
    status = ctx["status"]

    exp_name = safe_leaf_name(params["exp_name"], "exp_name")
    sr_name = _safe_sr_name(params.get("sr", "40k"))
    if_f0 = bool(params.get("if_f0", True))
    spk_id = int(params.get("spk_id", 0))
    save_epoch = int(params.get("save_epoch", 5))
    total_epoch = int(params.get("total_epoch", 200))
    batch_size = int(params.get("batch_size", 4))
    if_save_latest = bool(params.get("if_save_latest", True))
    pretrained_G = params.get("pretrained_G", "")
    pretrained_D = params.get("pretrained_D", "")
    gpus = params.get("gpus", "")
    if_cache_gpu = bool(params.get("if_cache_gpu", False))
    if_save_every_weights = bool(params.get("if_save_every_weights", True))
    version = _safe_version(params.get("version", "v2"))
    author = params.get("author", "")
    task_id = params.get("task_id", f"train_{int(time.time()*1000)}")

    exp_dir = _exp_dir(config, exp_name)
    log_path = exp_dir / "train.log"
    log_path.write_text("")

    if (gpu_error := _require_gpu(params, config)) is not None:
        return gpu_error

    # Build filelist.txt from the preprocess/feature extraction outputs.
    try:
        _write_filelist(config, exp_dir, sr_name, if_f0, spk_id, version)
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

    # Seed the training config.json under the experiment dir (mirrors web.py).
    if version == "v1" or sr_name == "40k":
        config_key = "v1/%s.json" % sr_name
    else:
        config_key = "v2/%s.json" % sr_name
    config_save_path = exp_dir / "config.json"
    if not config_save_path.exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config.get(config_key, {}),
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )

    task = register_task(task_id)
    # Wire a cancel sentinel: rpc_server._cancel_task() touches this file
    # when a force-cancel arrives, and train.py polls it at epoch/batch
    # boundaries. Second channel, orthogonal to SIGTERM/SIGKILL — lets
    # training self-exit even if the signal is stuck behind a PyTorch
    # C extension. Unlink any stale sentinel from a prior run so we don't
    # exit immediately on startup.
    sentinel = exp_dir / ".cancel_sentinel"
    try:
        if sentinel.exists():
            sentinel.unlink()
    except OSError:
        pass
    task.sentinel_path = sentinel
    status("training")
    try:
        base = str(config.base_dir)
        cmd = [
            _bundle_python_cmd(config),
            str(Path(base) / "infer" / "modules" / "train" / "train.py"),
            "-e",
            exp_name,
            "-sr",
            sr_name,
            "-f0",
            "1" if if_f0 else "0",
            "-bs",
            str(batch_size),
            "-te",
            str(total_epoch),
            "-se",
            str(save_epoch),
            "-l",
            "1" if if_save_latest else "0",
            "-c",
            "1" if if_cache_gpu else "0",
            "-sw",
            "1" if if_save_every_weights else "0",
            "-v",
            version,
            "-a",
            author,
            "--cancel-sentinel",
            str(sentinel),
        ]
        if pretrained_G:
            cmd += ["-pg", pretrained_G]
        if pretrained_D:
            cmd += ["-pd", pretrained_D]
        if gpus:
            cmd += ["-g", gpus]
        # Training scripts resolve paths relative to cwd (logs/<exp>/...) — run
        # from the user_dir so logs/ points to the writable user logs dir.
        proc = _spawn(
            cmd,
            cwd=str(config.user_dir),
            log_path=log_path,
            env_extra={"RVC_TRAIN_DEVICE": str(config.device)},
        )

        # Configure the percent estimator with the total epoch for this run.
        _train_percent_from_log._total = total_epoch  # type: ignore

        done_event = threading.Event()
        threading.Thread(
            target=_wait_process, args=(proc, done_event), daemon=True
        ).start()
        log = _tail_log_until_done(
            log_path,
            done_event,
            task_id,
            emit_progress,
            cancel_event=task.cancel_event,
            proc=proc,
            phase="training",
            percent_from_log=_train_percent_from_log,
            task=task,
        )
        # Cancellation-over-returncode: see rpc_preprocess for the rationale.
        if task.cancel_event.is_set():
            return {"status": "cancelled", "log": log, "exp_dir": str(exp_dir)}
        emit_progress(task_id, 100.0, "training done", "training")
        # A non-zero train.py exit means training crashed (missing package,
        # empty filelist, MPS op error, ...). Propagate that so rpc_train_all
        # stops instead of advancing to train_index with no checkpoint.
        if proc.returncode not in (0, None):
            return {
                "status": "error",
                "error": "train.py exited with code %d" % proc.returncode,
                "log": log,
                "exp_dir": str(exp_dir),
            }
        return {"status": "success", "log": log, "exp_dir": str(exp_dir)}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        # Remove the cancel sentinel now that the run (and any cancel) is fully
        # handled. _cancel_task touches it on every cancel, and the only other
        # cleanup is the start-of-run unlink — so without this it would linger
        # on disk until the same experiment is trained again.
        try:
            sentinel.unlink(missing_ok=True)
        except OSError:
            pass
        unregister_task(task_id)
        status("idle")


def _write_filelist(
    config, exp_dir: Path, sr_name: str, if_f0: bool, spk_id: int, version: str
):
    """Write the filelist.txt consumed by infer/modules/train/train.py.

    Normalize stems across directories that use different naming conventions:
      * 0_gt_wavs/0_1.wav           -> stem "0_1"
      * 3_featureNNN/0_1.npy        -> stem "0_1"
      * 2a_f0/0_1.wav.npy           -> Path.stem gives "0_1.wav" (strip trailing .wav)
      * 2b-f0nsf/0_1.wav.npy        -> same
    Without the trailing-".wav" strip the f0/f0nsf stems never matched
    gt_wavs / feature stems, so the set intersection was always empty and
    filelist.txt ended up zero-byte (train.py then had nothing to train on).
    """

    def _normalized_stem(p: Path) -> str:
        s = p.stem  # strips one extension ".npy" or ".wav"
        # For "foo.wav.npy" Path.stem is "foo.wav"; peel the extra ".wav" off
        # so the set intersection lines up across directories.
        if s.endswith(".wav"):
            s = s[:-4]
        return s

    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = (
            {_normalized_stem(p) for p in gt_wavs_dir.iterdir()}
            & {_normalized_stem(p) for p in feature_dir.iterdir()}
            & {_normalized_stem(p) for p in f0_dir.iterdir()}
            & {_normalized_stem(p) for p in f0nsf_dir.iterdir()}
        )
    else:
        names = {_normalized_stem(p) for p in gt_wavs_dir.iterdir()} & {
            _normalized_stem(p) for p in feature_dir.iterdir()
        }

    opt = []
    for name in names:
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%d"
                % (
                    gt_wavs_dir,
                    name,
                    feature_dir,
                    name,
                    f0_dir,
                    name,
                    f0nsf_dir,
                    name,
                    spk_id,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%d"
                % (gt_wavs_dir, name, feature_dir, name, spk_id)
            )

    fea_dim = 256 if version == "v1" else 768
    mute_dir = Path(config.base_dir) / "logs" / "mute"
    if mute_dir.exists():
        for _ in range(2):
            if if_f0:
                opt.append(
                    "%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|%d"
                    % (mute_dir, sr_name, mute_dir, fea_dim, mute_dir, mute_dir, spk_id)
                )
            else:
                opt.append(
                    "%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|%d"
                    % (mute_dir, sr_name, mute_dir, fea_dim, spk_id)
                )
    shuffle(opt)
    (exp_dir / "filelist.txt").write_text("\n".join(opt))


def rpc_train_index(params: dict, ctx):
    """Build a FAISS index from the extracted features."""
    config = ctx["config"]
    emit_progress = ctx["emit_progress"]
    register_task = ctx["register_task"]
    unregister_task = ctx["unregister_task"]
    status = ctx["status"]

    exp_name = safe_leaf_name(params["exp_name"], "exp_name")
    version = _safe_version(params.get("version", "v2"))
    task_id = params.get("task_id", f"train_index_{int(time.time()*1000)}")

    import faiss  # heavy, imported lazily
    from sklearn.cluster import MiniBatchKMeans

    exp_dir = _exp_dir(config, exp_name)
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.exists():
        return {
            "status": "error",
            "error": "feature_dir does not exist (run extract_f0 first)",
        }

    entries = sorted(feature_dir.iterdir())
    if not entries:
        return {"status": "error", "error": "feature_dir is empty"}

    task = register_task(task_id)
    status("training")
    infos = []

    def _check_cancel():
        if task.cancel_event.is_set():
            return {"status": "cancelled", "messages": infos}
        return None

    try:
        emit_progress(task_id, 5, "loading features", "index")
        npys = []
        for i, entry in enumerate(entries):
            if r := _check_cancel():
                return r
            npys.append(np.load(entry))
            if i % 20 == 0:
                emit_progress(
                    task_id,
                    5 + (i / len(entries)) * 15,
                    f"loading features ({i+1}/{len(entries)})",
                    "index",
                )
        if r := _check_cancel():
            return r
        big_npy = np.concatenate(npys, 0)
        idx = np.arange(big_npy.shape[0])
        np.random.shuffle(idx)
        big_npy = big_npy[idx]
        if big_npy.shape[0] > 2e5:
            emit_progress(task_id, 25, "kmeans to 10k centers", "index")
            if r := _check_cancel():
                return r
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=False,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        if r := _check_cancel():
            return r

        np.save(exp_dir / "total_fea.npy", big_npy)
        # n_ivf must be >= 1; with very small datasets (< 39 features) the
        # original `min(int(16*sqrt(N)), N//39)` collapses to 0 and FAISS
        # rejects "IVF0,Flat". max(1, ...) keeps tiny datasets buildable
        # (degenerates to a single-cluster IVF, effectively equivalent to Flat).
        n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
        emit_progress(task_id, 50, "training IVF index", "index")
        index = faiss.index_factory(
            256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf
        )
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)
        if r := _check_cancel():
            return r
        trained_path = exp_dir / (
            "trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (n_ivf, index_ivf.nprobe, exp_name, version)
        )
        faiss.write_index(index, str(trained_path))

        emit_progress(task_id, 80, "adding vectors", "index")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            if r := _check_cancel():
                return r
            index.add(big_npy[i : i + batch_size_add])
        index_save_path = exp_dir / (
            "added_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (n_ivf, index_ivf.nprobe, exp_name, version)
        )
        faiss.write_index(index, str(index_save_path))
        infos.append(f"built: {index_save_path}")

        # Symlink / copy into outside_index_root (user-visible).
        outside_root = Path(os.environ.get("outside_index_root") or "")
        if outside_root:
            outside_root.mkdir(parents=True, exist_ok=True)
            link_target = outside_root / (
                "%s_IVF%s_Flat_nprobe_%s_%s_%s.index"
                % (exp_name, n_ivf, index_ivf.nprobe, exp_name, version)
            )
            try:
                if link_target.exists():
                    link_target.unlink()
                if platform.system() == "Windows":
                    shutil.copy(index_save_path, link_target)
                else:
                    os.symlink(index_save_path, link_target)
                infos.append(f"linked: {link_target}")
            except Exception as e:
                infos.append(f"link failed: {e}")

        emit_progress(task_id, 100, "index done", "index")
        return {
            "status": "success",
            "messages": infos,
            "index_path": str(index_save_path),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        unregister_task(task_id)
        status("idle")


def rpc_train_all(params: dict, ctx):
    """One-click training: preprocess -> extract_f0 -> train -> train_index."""
    emit_progress = ctx["emit_progress"]
    register_task = ctx["register_task"]
    unregister_task = ctx["unregister_task"]
    task_id = params.get("task_id", f"train_all_{int(time.time()*1000)}")

    # Register the parent task so _cancel_task(parent_id) can set its event.
    # The prefix-match in _cancel_task also cascades to child stages.
    parent_task = register_task(task_id)

    stages = [
        ("preprocess", rpc_preprocess),
        ("extract_f0", rpc_extract_f0),
        ("train", rpc_train),
        ("train_index", rpc_train_index),
    ]
    results = {}
    try:
        for name, fn in stages:
            # Check parent cancel before launching next stage.
            if parent_task.cancel_event.is_set():
                return {
                    "status": "cancelled",
                    "cancelled_stage": name,
                    "results": results,
                }
            stage_params = dict(params)
            stage_params["task_id"] = f"{task_id}_{name}"
            emit_progress(
                task_id,
                (len(results) / len(stages)) * 100.0,
                f"stage: {name}",
                "pipeline",
            )
            r = fn(stage_params, ctx)
            results[name] = r
            if r.get("status") == "cancelled":
                return {
                    "status": "cancelled",
                    "cancelled_stage": name,
                    "results": results,
                }
            if r.get("status") != "success":
                # Surface a top-level "error" string so the UI shows the real
                # cause (which stage failed and why) instead of a generic
                # "学習ステージ失敗: train_all". Reuse the sub-stage's own
                # error message when it has one.
                sub_error = r.get("error")
                return {
                    "status": "error",
                    "failed_stage": name,
                    "error": (
                        "%s ステージが失敗しました: %s" % (name, sub_error)
                        if sub_error
                        else "%s ステージが失敗しました" % name
                    ),
                    "results": results,
                }
        emit_progress(task_id, 100.0, "all stages done", "pipeline")
        return {"status": "success", "results": results}
    finally:
        unregister_task(task_id)


# ---------------------------------------------------------------------------
# Entry used by rpc_server.py to wire the methods into the dispatcher.
# ---------------------------------------------------------------------------


def build_methods(ctx) -> Dict[str, Callable[[dict], dict]]:
    """Return a dict of RPC method name -> callable taking `params`.

    `ctx` bundles references to runtime state / helpers from rpc_server:
      - config
      - emit_progress(task_id, percent, message, phase)
      - register_task / unregister_task
      - status(str)
    """
    return {
        "preprocess": lambda p: rpc_preprocess(p, ctx),
        "extract_f0": lambda p: rpc_extract_f0(p, ctx),
        "train": lambda p: rpc_train(p, ctx),
        "train_index": lambda p: rpc_train_index(p, ctx),
        "train_all": lambda p: rpc_train_all(p, ctx),
    }

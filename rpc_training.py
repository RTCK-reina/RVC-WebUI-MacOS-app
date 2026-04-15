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

logger = logging.getLogger("rpc_training")

# sr name -> Hz (mirrors web.py sr_dict)
_SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}


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
    p = Path(config.user_dir) / "logs" / exp_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _spawn(cmd: list, cwd: str, log_path: Path) -> subprocess.Popen:
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
    with open(log_path, "a", encoding="utf-8") as log_f:
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )


def _terminate_procs(procs: list, graceful_wait: float = 3.0) -> None:
    """Politely ask procs to exit, then SIGKILL any stragglers.

    Sends SIGTERM, waits up to `graceful_wait` seconds, then sends SIGKILL
    to anything still running. Never raises.
    """
    alive = [p for p in procs if p is not None and p.poll() is None]
    for p in alive:
        try:
            p.terminate()
        except Exception:
            pass
    # Give them a chance to clean up.
    import time as _time
    deadline = _time.time() + graceful_wait
    while _time.time() < deadline:
        alive = [p for p in alive if p.poll() is None]
        if not alive:
            return
        _time.sleep(0.1)
    # Still alive: force kill.
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass


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
) -> str:
    """Stream log_path growth while the subprocess runs.

    Emits a `progress` notification every second with the tail of the log.
    On cancellation, SIGTERM all known subprocesses and fall back to SIGKILL
    after a short grace period. Returns the full final log contents.

    `proc` accepts a single subprocess.Popen; `procs` accepts a list (for
    multi-process stages like F0 feature extraction).
    """
    last_size = 0
    last_tail = ""
    percent = 0.0

    while not proc_done_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            target_procs = list(procs) if procs else []
            if proc is not None:
                target_procs.append(proc)
            _terminate_procs(target_procs, graceful_wait=3.0)
            break
        try:
            if log_path.exists():
                size = log_path.stat().st_size
                if size != last_size:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
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
        time.sleep(1.0)

    # Final read.
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _wait_process(proc: subprocess.Popen, done_event: threading.Event) -> None:
    """Background helper that flips `done_event` once `proc` exits."""
    proc.wait()
    done_event.set()


def _wait_multi_processes(
    procs: list, done_event: threading.Event
) -> None:
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

    exp_name = params["exp_name"]
    trainset_dir = params["trainset_dir"]
    sr_name = params.get("sr", "40k")
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
            str(Path(config.base_dir) / "infer" / "modules" / "train" / "preprocess.py"),
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
            log_path, done_event, task_id, emit_progress,
            cancel_event=task.cancel_event, proc=proc, phase="preprocessing",
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

    exp_name = params["exp_name"]
    f0_method = params.get("f0_method", "rmvpe")
    if_f0 = bool(params.get("if_f0", True))
    version = params.get("version", "v2")
    # For rpc usage we assume single GPU (or MPS / CPU). Comma-separated gpus
    # may still be supplied for CUDA boxes.
    gpus = params.get("gpus", "0").strip()
    gpus_list = [g for g in gpus.split("-") if g.strip()]
    n_p = int(params.get("n_p", max(1, config.n_cpu)))
    task_id = params.get("task_id", f"extract_f0_{int(time.time()*1000)}")

    exp_dir = _exp_dir(config, exp_name)
    log_path = exp_dir / "extract_f0_feature.log"
    log_path.write_text("")

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
                log_path, done_event, task_id, emit_progress,
                cancel_event=task.cancel_event, proc=proc, phase="f0",
            )
            # Same cancellation-over-returncode ordering as rpc_preprocess:
            # after SIGTERM the returncode may still be None on a racy exit.
            if task.cancel_event.is_set():
                return {"status": "cancelled"}
            if proc.returncode not in (0, None):
                return {
                    "status": "error",
                    "error": "extract_f0_print.py exited with code %d" % proc.returncode,
                }

        # Step 2: Feature extraction (may parallelize across gpus).
        procs = []
        leng = max(1, len(gpus_list) or 1)
        for idx, g in enumerate(gpus_list or ["0"]):
            cmd = [
                _bundle_python_cmd(config),
                str(Path(base) / "infer" / "modules" / "train" / "extract_feature_print.py"),
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
            log_path, done_event, task_id, emit_progress,
            cancel_event=task.cancel_event, procs=procs, phase="features",
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
                "error": "extract_feature_print.py worker(s) exited with codes %r" % bad,
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

    exp_name = params["exp_name"]
    sr_name = params.get("sr", "40k")
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
    version = params.get("version", "v2")
    author = params.get("author", "")
    task_id = params.get("task_id", f"train_{int(time.time()*1000)}")

    exp_dir = _exp_dir(config, exp_name)
    log_path = exp_dir / "train.log"
    log_path.write_text("")

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
    status("training")
    try:
        base = str(config.base_dir)
        cmd = [
            _bundle_python_cmd(config),
            str(Path(base) / "infer" / "modules" / "train" / "train.py"),
            "-e", exp_name,
            "-sr", sr_name,
            "-f0", "1" if if_f0 else "0",
            "-bs", str(batch_size),
            "-te", str(total_epoch),
            "-se", str(save_epoch),
            "-l", "1" if if_save_latest else "0",
            "-c", "1" if if_cache_gpu else "0",
            "-sw", "1" if if_save_every_weights else "0",
            "-v", version,
            "-a", author,
        ]
        if pretrained_G:
            cmd += ["-pg", pretrained_G]
        if pretrained_D:
            cmd += ["-pd", pretrained_D]
        if gpus:
            cmd += ["-g", gpus]
        # Training scripts resolve paths relative to cwd (logs/<exp>/...) — run
        # from the user_dir so logs/ points to the writable user logs dir.
        proc = _spawn(cmd, cwd=str(config.user_dir), log_path=log_path)

        # Configure the percent estimator with the total epoch for this run.
        _train_percent_from_log._total = total_epoch  # type: ignore

        done_event = threading.Event()
        threading.Thread(
            target=_wait_process, args=(proc, done_event), daemon=True
        ).start()
        log = _tail_log_until_done(
            log_path, done_event, task_id, emit_progress,
            cancel_event=task.cancel_event, proc=proc, phase="training",
            percent_from_log=_train_percent_from_log,
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
        unregister_task(task_id)
        status("idle")


def _write_filelist(config, exp_dir: Path, sr_name: str, if_f0: bool, spk_id: int, version: str):
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
        names = (
            {_normalized_stem(p) for p in gt_wavs_dir.iterdir()}
            & {_normalized_stem(p) for p in feature_dir.iterdir()}
        )

    opt = []
    for name in names:
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%d"
                % (gt_wavs_dir, name, feature_dir, name, f0_dir, name, f0nsf_dir, name, spk_id)
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

    exp_name = params["exp_name"]
    version = params.get("version", "v2")
    task_id = params.get("task_id", f"train_index_{int(time.time()*1000)}")

    import faiss  # heavy, imported lazily
    from sklearn.cluster import MiniBatchKMeans

    exp_dir = _exp_dir(config, exp_name)
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.exists():
        return {"status": "error", "error": "feature_dir does not exist (run extract_f0 first)"}

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
        # C-1: previously this block appended every .npy to a Python list and
        # then called `np.concatenate(npys, 0)`, which peaks at 2× the total
        # feature tensor size in RAM (the input list plus the concatenated
        # output). On Mac Unified Memory, large feature sets can push against
        # the allocator and stall.
        #
        # Two-pass streaming: first pass reads each .npy only to collect
        # shape (mmap-based, no full copy), computes the total row count and
        # the feature dimension, allocates the destination once, then the
        # second pass memcpys each file straight into its slice. Peak memory
        # is ~1× instead of ~2×.
        total_rows = 0
        feat_dim = None
        for entry in entries:
            if (r := _check_cancel()): return r
            # memmap avoids reading the full array just to learn the shape.
            arr = np.load(entry, mmap_mode="r")
            total_rows += arr.shape[0]
            if feat_dim is None:
                feat_dim = arr.shape[1]
            elif arr.shape[1] != feat_dim:
                return {
                    "status": "error",
                    "error": f"feature dim mismatch: {entry} has {arr.shape[1]} "
                             f"vs expected {feat_dim}",
                }
            del arr
        if feat_dim is None:
            return {"status": "error", "error": "no features to load"}

        # Try float32 first (matches what training consumed); fall back to the
        # dtype of the first .npy if it turns out to be something else.
        dtype = np.load(entries[0], mmap_mode="r").dtype
        big_npy = np.empty((total_rows, feat_dim), dtype=dtype)
        write_at = 0
        for i, entry in enumerate(entries):
            if (r := _check_cancel()): return r
            arr = np.load(entry)  # full load, written to the preallocated buffer
            big_npy[write_at : write_at + arr.shape[0]] = arr
            write_at += arr.shape[0]
            del arr
            if i % 20 == 0:
                emit_progress(task_id, 5 + (i / len(entries)) * 15,
                              f"loading features ({i+1}/{len(entries)})", "index")

        if (r := _check_cancel()): return r
        # Shuffle in place (np.random.permutation allocates a fresh array;
        # shuffling the index array and applying it is what the upstream did,
        # but we can save one copy by shuffling big_npy directly).
        rng = np.random.default_rng()
        rng.shuffle(big_npy, axis=0)
        if big_npy.shape[0] > 2e5:
            emit_progress(task_id, 25, "kmeans to 10k centers", "index")
            if (r := _check_cancel()): return r
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
        if (r := _check_cancel()): return r

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
        if (r := _check_cancel()): return r
        trained_path = exp_dir / (
            "trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (n_ivf, index_ivf.nprobe, exp_name, version)
        )
        faiss.write_index(index, str(trained_path))

        emit_progress(task_id, 80, "adding vectors", "index")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            if (r := _check_cancel()): return r
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
        return {"status": "success", "messages": infos, "index_path": str(index_save_path)}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    finally:
        unregister_task(task_id)
        status("idle")


def rpc_train_all(params: dict, ctx):
    """One-click training: preprocess -> extract_f0 -> train -> train_index."""
    # Chain the existing handlers. They already emit their own progress.
    emit_progress = ctx["emit_progress"]
    task_id = params.get("task_id", f"train_all_{int(time.time()*1000)}")

    stages = [
        ("preprocess", rpc_preprocess),
        ("extract_f0", rpc_extract_f0),
        ("train", rpc_train),
        ("train_index", rpc_train_index),
    ]
    results = {}
    for name, fn in stages:
        stage_params = dict(params)
        stage_params["task_id"] = f"{task_id}_{name}"
        emit_progress(task_id, (len(results) / len(stages)) * 100.0,
                      f"stage: {name}", "pipeline")
        r = fn(stage_params, ctx)
        results[name] = r
        # Propagate cancellation distinctly — the UI distinguishes cancelled
        # from error and should not show "train_all failed" when the user hit
        # the cancel button.
        if r.get("status") == "cancelled":
            return {
                "status": "cancelled",
                "cancelled_stage": name,
                "results": results,
            }
        if r.get("status") != "success":
            return {"status": "error", "failed_stage": name, "results": results}
    emit_progress(task_id, 100.0, "all stages done", "pipeline")
    return {"status": "success", "results": results}


# ---------------------------------------------------------------------------
# Realtime VC methods.
# ---------------------------------------------------------------------------


# Module-global realtime state because sounddevice streams must survive across
# RPC calls (start -> update_params -> stop).
#
# Guarded by _realtime_lock: start, update_params and stop each perform a
# check-then-act on this dict and could race when the UI triggers them in
# quick succession. The audio callback intentionally does NOT take the lock
# (callbacks run on the sounddevice thread at real-time priority); it only
# reads stop_event, which is thread-safe on its own.
_realtime_lock = threading.Lock()
_realtime_state = {
    "rvc": None,
    "stream": None,
    "stop_event": None,
    "thread": None,
    "params": None,
    "sample_rate": None,
    "device_input": None,
    "device_output": None,
}


def _rms(x) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def rpc_list_audio_devices(params: dict, ctx):
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
                "index": i, "name": d["name"],
                "hostapi": hostapis[d["hostapi"]]["name"],
                "max_channels": d["max_input_channels"],
                "default_sr": d.get("default_samplerate"),
            }
            for i, d in enumerate(devs) if d["max_input_channels"] > 0
        ],
        "output": [
            {
                "index": i, "name": d["name"],
                "hostapi": hostapis[d["hostapi"]]["name"],
                "max_channels": d["max_output_channels"],
                "default_sr": d.get("default_samplerate"),
            }
            for i, d in enumerate(devs) if d["max_output_channels"] > 0
        ],
    }


def rpc_realtime_start(params: dict, ctx):
    """Start a sounddevice stream doing live RVC inference.

    This is a minimal viable implementation — for production, further tuning
    of block size, crossfading, and noise reduction is needed.
    """
    # Reservation pattern: briefly take the lock to check-and-claim the
    # "stream" slot, then do the heavy RVC/sounddevice init outside the
    # lock so that concurrent update_params/stop don't stall for seconds.
    with _realtime_lock:
        if _realtime_state["stream"] is not None:
            return {"status": "error", "error": "already running"}
        _realtime_state["stream"] = "starting"  # reservation sentinel

    import numpy as np
    import sounddevice as sd
    import torch

    from infer.lib.rtrvc import RVC

    config = ctx["config"]
    status = ctx["status"]

    pth_path = params["pth_path"]
    index_path = params.get("index_path", "")
    key = float(params.get("pitch", 0))
    formant = float(params.get("formant", 0.0))
    index_rate = float(params.get("index_rate", 0.0))
    f0_method = params.get("f0_method", "fcpe")
    block_time = float(params.get("block_time", 0.25))
    sample_rate = int(params.get("sample_rate", 48000))
    input_device = params.get("input_device")  # index or name
    output_device = params.get("output_device")
    threshold_db = float(params.get("threshold", -60.0))
    protect = float(params.get("protect", 0.33))

    # Apply device selection to sounddevice defaults.
    if input_device is not None or output_device is not None:
        sd.default.device = (input_device, output_device)

    rvc = RVC(
        key=key,
        formant=formant,
        pth_path=pth_path,
        index_path=index_path,
        index_rate=index_rate,
        n_cpu=min(config.n_cpu, 4),
        device=str(config.device),
        use_jit=False,
        is_half=config.is_half,
        is_dml=config.dml,
    )

    # Resample to 16k for the model, resample back to sample_rate on output.
    sr_model = rvc.tgt_sr
    block_frame_16k = int(block_time * 16000)
    extra_frame_16k = int(2.5 * 16000)

    input_buffer = np.zeros(block_frame_16k + extra_frame_16k, dtype=np.float32)

    stop_event = threading.Event()

    from torchaudio.transforms import Resample

    res_in = Resample(orig_freq=sample_rate, new_freq=16000).to(str(config.device))
    res_out = Resample(orig_freq=sr_model, new_freq=sample_rate).to(str(config.device))

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time_info, sdstatus):
        if stop_event.is_set():
            raise sd.CallbackStop

        # Downmix to mono float32.
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata
        # Threshold (noise gate).
        if _rms(mono) < 10 ** (threshold_db / 20.0):
            outdata.fill(0)
            return

        try:
            with torch.no_grad():
                x = torch.from_numpy(mono).to(str(config.device)).float()
                x16 = res_in(x.unsqueeze(0)).squeeze(0)
                # Slide buffer.
                shift = x16.shape[0]
                input_buffer[:-shift] = input_buffer[shift:]
                input_buffer[-shift:] = x16.detach().cpu().numpy()
                xin = torch.from_numpy(input_buffer).to(str(config.device)).float()

                audio = rvc.infer(
                    xin,
                    block_frame_16k=block_frame_16k,
                    skip_head=extra_frame_16k,
                    return_length=block_frame_16k,
                    f0method=f0_method,
                    protect=protect,
                )
                audio = res_out(audio.unsqueeze(0)).squeeze(0)
                y = audio.detach().cpu().numpy().astype(np.float32)
            # Match output channels.
            if outdata.ndim > 1:
                outdata[:, 0] = y[: outdata.shape[0]]
                if outdata.shape[1] > 1:
                    outdata[:, 1] = outdata[:, 0]
            else:
                outdata[:] = y[: outdata.shape[0]]
        except Exception as e:
            logger.exception("realtime callback error: %s", e)
            outdata.fill(0)

    stream = sd.Stream(
        samplerate=sample_rate,
        blocksize=int(block_time * sample_rate),
        channels=(1, 2),
        dtype="float32",
        callback=callback,
    )
    try:
        stream.start()
    except Exception as e:
        # Release the reservation so the next start() call can proceed.
        with _realtime_lock:
            _realtime_state["stream"] = None
        return {"status": "error", "error": f"stream.start failed: {e}"}

    with _realtime_lock:
        _realtime_state["rvc"] = rvc
        _realtime_state["stream"] = stream  # replace "starting" sentinel
        _realtime_state["stop_event"] = stop_event
        _realtime_state["params"] = dict(params)
        _realtime_state["sample_rate"] = sample_rate

    status("realtime")
    return {"status": "success", "sample_rate": sample_rate, "model_sr": int(sr_model)}


def rpc_realtime_update_params(params: dict, ctx):
    with _realtime_lock:
        rvc = _realtime_state.get("rvc")
        if rvc is None:
            return {"status": "error", "error": "not running"}
        # rvc.set_* are cheap and only touch the RVC instance, so it is safe
        # to call them while holding the state lock. This guarantees we never
        # dereference a partially-initialized or torn-down rvc handle.
        if "pitch" in params:
            rvc.set_key(float(params["pitch"]))
        if "formant" in params:
            rvc.set_formant(float(params["formant"]))
        if "index_rate" in params:
            rvc.set_index_rate(float(params["index_rate"]))
    return {"status": "success"}


def rpc_realtime_stop(params: dict, ctx):
    status = ctx["status"]
    # Snapshot + clear under the lock so the stream handle cannot be replaced
    # out from under us by a concurrent start(); release the lock before the
    # potentially-blocking stream.stop()/close() calls.
    with _realtime_lock:
        stream = _realtime_state.get("stream")
        stop_event = _realtime_state.get("stop_event")
        # Treat the "starting" reservation as "nothing to stop yet".
        if not isinstance(stream, sounddevice_stream_type()):  # type: ignore[name-defined]
            stream = None
        _realtime_state.update({
            "rvc": None, "stream": None, "stop_event": None,
            "thread": None, "params": None,
        })
    if stop_event is not None:
        stop_event.set()
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            logger.warning("stream.stop error: %s", e)
    status("idle")
    return {"status": "success"}


def sounddevice_stream_type():
    """Return sounddevice.Stream, imported lazily to avoid a hard dependency.

    Used by rpc_realtime_stop to distinguish the "starting" sentinel string
    from a real Stream handle.
    """
    try:
        import sounddevice as _sd
        return _sd.Stream
    except Exception:
        return type(None)


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
        "list_audio_devices": lambda p: rpc_list_audio_devices(p, ctx),
        "realtime_start": lambda p: rpc_realtime_start(p, ctx),
        "realtime_update_params": lambda p: rpc_realtime_update_params(p, ctx),
        "realtime_stop": lambda p: rpc_realtime_stop(p, ctx),
    }

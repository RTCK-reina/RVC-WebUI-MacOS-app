"""Runtime configuration for the macOS-focused RVC build.

This file is deliberately slim: only MPS (Apple Silicon) and CPU backends are
supported. All CUDA, XPU (Intel Arc), and DirectML code paths from the upstream
project have been removed because they cannot run on macOS.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from multiprocessing import cpu_count

import torch

logger = logging.getLogger(__name__)


version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if wrapper.instance is None:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        # macOS does not support CUDA; MPS if available, CPU otherwise.
        self.device = "cpu"
        self.is_half = False  # MPS forces fp32 on Apple Silicon
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.global_link,
            self.noparallel,
            self.noautoopen,
            self.nocheck,
            self.update,
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument(
            "--global_link", action="store_true", help="Generate a global proxy link"
        )
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--nocheck", action="store_true", help="Run without checking assets"
        )
        parser.add_argument(
            "--update", action="store_true", help="Update to latest assets"
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.global_link,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.nocheck,
            cmd_opts.update,
        )

    # has_mps is only available in nightly pytorch (for now) and macOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
            inuse_path = f"configs/inuse/{config_file}"
            with open(inuse_path, "r") as f:
                data = json.load(f)
            if data.get("train", {}).get("fp16_run") is not False:
                data.setdefault("train", {})["fp16_run"] = False
                with open(inuse_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info("overwrite " + config_file)
        self.preprocess_per = 3.0
        logger.info("overwrite preprocess_per to %.1f" % (self.preprocess_per))

    def device_config(self):
        if self.has_mps():
            logger.info("Using Apple Silicon MPS backend (fp32 forced)")
            self.device = self.instead = "mps"
        else:
            logger.info("No GPU acceleration available; falling back to CPU")
            self.device = self.instead = "cpu"
        self.is_half = False
        self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        # fp32 path → upstream "5G" profile
        x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        logger.info(
            "Half-precision floating-point: %s, device: %s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max


@singleton_variable
class CPUConfig:
    def __init__(self):
        self.device = "cpu"
        self.is_half = False
        self.use_jit = False
        self.n_cpu = 1
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.instead = "cpu"
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            with open(f"configs/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
        self.preprocess_per = 3.0

    def device_config(self):
        self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        x_pad, x_query, x_center, x_max = 1, 6, 38, 41
        return x_pad, x_query, x_center, x_max

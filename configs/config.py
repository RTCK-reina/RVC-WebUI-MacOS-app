import argparse
import os
import sys
import json
import shutil
from multiprocessing import cpu_count
from pathlib import Path

import torch

# TODO: move device selection into rvc
import logging

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


def _default_base_dir() -> Path:
    """Resolve the bundle base dir.

    Priority:
      1. RVC_BASE_DIR env var (set by the .app launcher to .../Contents/Resources)
      2. Current working directory (development / legacy behavior)
    """
    env = os.environ.get("RVC_BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(os.getcwd()).resolve()


DEFAULT_USER_DIR_NAME = "RVC-WebUI"


def _default_user_dir() -> Path:
    """Resolve the user data dir under ~/Documents/RVC-WebUI by default.

    Priority:
      1. RVC_USER_DIR env var
      2. ~/Documents/RVC-WebUI (the .app convention)
      3. Current working directory (legacy)
    """
    env = os.environ.get("RVC_USER_DIR")
    if env:
        return Path(env).expanduser().resolve()
    home = Path.home()
    documents = home / "Documents"
    if documents.is_dir():
        return (documents / DEFAULT_USER_DIR_NAME).resolve()
    return Path(os.getcwd()).resolve()


def ensure_user_layout(user_dir: Path) -> None:
    """Create the user data directory tree (idempotent)."""
    subdirs = [
        "input/audio",
        "input/training",
        "output/inference",
        "output/batch",
        "output/separation/vocals",
        "output/separation/accompaniment",
        "output/onnx",
        "models",
        "indices",
        "logs",
        "configs/inuse/v1",
        "configs/inuse/v2",
        "temp",
    ]
    for sd in subdirs:
        (user_dir / sd).mkdir(parents=True, exist_ok=True)


def _populate_env_paths(base_dir: Path, user_dir: Path) -> None:
    """Populate os.environ with the canonical paths used across the codebase.

    This replaces the .env-driven configuration so the RPC server and every
    subprocess it spawns (training/feature extraction) agree on where to read
    and write files, without any relative-cwd dependency.

    IMPORTANT: this must OVERRIDE whatever a prior load_dotenv() may have
    populated. The upstream ``.env`` bundled with this repo still carries
    cwd-relative defaults (``weight_root = assets/weights`` etc.) that make
    no sense for the .app: the bundle is read-only, cwd is user_dir, and user
    weights must live at ``~/Documents/RVC-WebUI/models/``. If we use
    ``setdefault`` here, ``load_dotenv`` wins and small-model saves end up in
    ``~/Documents/RVC-WebUI/assets/weights/`` — an unexpected location the
    rest of the app does not index. Force the assignment.
    """
    os.environ["weight_root"] = str(user_dir / "models")
    os.environ["weight_uvr5_root"] = str(base_dir / "assets" / "uvr5_weights")
    os.environ["index_root"] = str(user_dir / "logs")
    os.environ["outside_index_root"] = str(user_dir / "indices")
    os.environ["rmvpe_root"] = str(base_dir / "assets" / "rmvpe")
    # Newly exposed knobs — consumed by rpc_server.py output logic.
    os.environ["output_root"] = str(user_dir / "output")
    os.environ["input_root"] = str(user_dir / "input")
    os.environ["TEMP"] = str(user_dir / "temp")


@singleton_variable
class Config:
    def __init__(self, base_dir=None, user_dir=None):
        # Resolve dirs first so that load_config_json / path helpers can use them.
        self.base_dir: Path = Path(base_dir).expanduser().resolve() if base_dir else _default_base_dir()
        self.user_dir: Path = Path(user_dir).expanduser().resolve() if user_dir else _default_user_dir()
        ensure_user_layout(self.user_dir)
        _populate_env_paths(self.base_dir, self.user_dir)

        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.use_onnx = os.environ.get("RVC_USE_ONNX", "").lower() in ("1", "true", "yes")
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
            self.dml,
            self.nocheck,
            self.update,
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def asset_path(self, *parts) -> Path:
        return self.base_dir.joinpath(*parts)

    def user_path(self, *parts) -> Path:
        return self.user_dir.joinpath(*parts)

    def load_config_json(self) -> dict:
        """Load the per-SR training JSON configs.

        Templates live inside the bundle (base_dir/configs/*.json). The editable
        copies live under user_dir/configs/inuse/*.json so the app can freely
        write to them.
        """
        d = {}
        inuse_root = self.user_dir / "configs" / "inuse"
        src_root = self.base_dir / "configs"
        inuse_root.mkdir(parents=True, exist_ok=True)
        for config_file in version_config_list:
            dst = inuse_root / config_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                src = src_root / config_file
                if src.exists():
                    shutil.copy(src, dst)
            with open(dst, "r") as f:
                d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
        """Parse command line args.

        Compatible with the legacy web.py / gui.py entrypoints but now used by
        rpc_server.py as well. Web-specific flags are preserved only so that
        launching via the old scripts still works; the .app launcher never sets
        them.
        """
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port (legacy web.py)")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument(
            "--global_link", action="store_true", help="Generate a global proxy link (legacy web.py)"
        )
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically (legacy web.py)",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        parser.add_argument(
            "--nocheck", action="store_true", help="Run without checking assets"
        )
        parser.add_argument(
            "--update", action="store_true", help="Update to latest assets"
        )
        # Known but ignored here (rpc_server.py parses these before Config()):
        parser.add_argument("--base-dir", type=str, default=None, help=argparse.SUPPRESS)
        parser.add_argument("--user-dir", type=str, default=None, help=argparse.SUPPRESS)
        cmd_opts, _unknown = parser.parse_known_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.global_link,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
            cmd_opts.nocheck,
            cmd_opts.update,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
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

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        inuse_root = self.user_dir / "configs" / "inuse"
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
            target = inuse_root / config_file
            if not target.exists():
                continue
            with open(target, "r") as f:
                strr = f.read().replace("true", "false")
            with open(target, "w") as f:
                f.write(strr)
            logger.info("overwrite " + config_file)
        self.preprocess_per = 3.0
        logger.info("overwrite preprocess_per to %.1f" % (self.preprocess_per))

    def device_config(self):
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.dml:
            logger.info("Use DirectML instead")
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"Use {self.instead} instead")
        logger.info(
            "Half-precision floating-point: %s, device: %s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max


@singleton_variable
class CPUConfig:
    def __init__(self, base_dir=None, user_dir=None):
        self.base_dir: Path = Path(base_dir).expanduser().resolve() if base_dir else _default_base_dir()
        self.user_dir: Path = Path(user_dir).expanduser().resolve() if user_dir else _default_user_dir()
        ensure_user_layout(self.user_dir)
        _populate_env_paths(self.base_dir, self.user_dir)

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

    def load_config_json(self) -> dict:
        d = {}
        for config_file in version_config_list:
            path = self.base_dir / "configs" / config_file
            with open(path, "r") as f:
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

        # 5G显存配置
        x_pad = 1
        x_query = 6
        x_center = 38
        x_max = 41

        return x_pad, x_query, x_center, x_max

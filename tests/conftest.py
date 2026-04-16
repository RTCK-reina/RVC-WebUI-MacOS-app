"""pytest conftest — stub out heavy third-party modules so the project's
Python sources can be imported without the full RVC conda environment.

デザイン方針:
- infer/, configs/ などの「プロジェクトパッケージディレクトリ」は実ファイルから
  インポートできるよう sys.modules には登録しない。
- 実際にはインストールされていない重量サードパーティ (torch, numpy 等) と、
  テストで使わない重量プロジェクトリーフモジュールのみをスタブ化する。
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# torch スタブ
# ---------------------------------------------------------------------------
def _make_torch_stub() -> MagicMock:
    t = MagicMock(name="torch")
    t.__version__ = "2.0.0+mock"
    t.__file__ = "<stub>"

    mps = MagicMock(name="torch.backends.mps")
    mps.is_available = MagicMock(return_value=False)
    mps.is_built = MagicMock(return_value=False)

    cuda = MagicMock(name="torch.cuda")
    cuda.is_available = MagicMock(return_value=False)
    cuda.empty_cache = MagicMock()

    backends = MagicMock(name="torch.backends")
    backends.mps = mps
    backends.cuda = cuda

    mps_mod = MagicMock(name="torch.mps")
    mps_mod.empty_cache = MagicMock()

    t.backends = backends
    t.cuda = cuda
    t.mps = mps_mod

    # torch_compat.py の二重パッチ防止チェック用マーカーを明示的に False に設定。
    # MagicMock はアクセスされると新しい MagicMock（truthy）を自動生成するため、
    # setattr で False を置いておかないと「既にパッチ済み」と誤判定される。
    t.load._rvc_torch_compat_wrapper = False

    return t


def _make_numba_stub() -> types.ModuleType:
    """numba スタブ — @jit(nopython=True) が no-op デコレータとして動作する。"""
    mod = types.ModuleType("numba")

    def jit(*args, **kwargs):
        # @jit(nopython=True) 形式のファクトリ呼び出し
        if len(args) == 1 and callable(args[0]):
            return args[0]  # @jit 直接適用
        return lambda f: f  # @jit(...) ファクトリ

    mod.jit = jit  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# sys.modules へスタブを注入 (未インストールのもののみ)
# ---------------------------------------------------------------------------
def _stub(name: str, obj=None) -> None:
    if name not in sys.modules:
        sys.modules[name] = obj if obj is not None else MagicMock(name=name)


# --- torch ---
if "torch" not in sys.modules:
    _torch = _make_torch_stub()
    sys.modules["torch"] = _torch
    sys.modules["torch.backends"] = _torch.backends
    sys.modules["torch.backends.mps"] = _torch.backends.mps
    sys.modules["torch.backends.cuda"] = _torch.backends.cuda
    sys.modules["torch.cuda"] = _torch.cuda
    sys.modules["torch.mps"] = _torch.mps
    for _sub in ("torch.nn", "torch.nn.functional", "torch.optim",
                 "torch.utils", "torch.utils.data"):
        _stub(_sub)

# --- numba (特殊: @jit デコレータが機能する必要がある) ---
if "numba" not in sys.modules:
    sys.modules["numba"] = _make_numba_stub()

# --- 純粋なサードパーティスタブ (MagicMock で十分) ---
for _m in [
    "numpy", "scipy", "scipy.io", "scipy.io.wavfile", "scipy.signal",
    "av", "av.audio", "av.audio.resampler", "av.audio.frame",
    "librosa", "faiss", "psutil", "sounddevice",
    "tqdm", "tqdm.auto",
    "dotenv",
    "yaml",
    "colorama",
    "einops",
    "praat_parselmouth",
    "pyworld",
    "resampy",
    "sklearn", "sklearn.cluster",
]:
    _stub(_m)

# --- configs (プロジェクト内だが rpc_server が require する) ---
# 実 configs/ ディレクトリは sys.path にあるので、configs.config だけスタブ化する。
# これで `from configs.config import Config` が MagicMock.Config を返す。
if "configs.config" not in sys.modules:
    _cfg_mod = MagicMock(name="configs.config")
    _cfg_cls = MagicMock(name="Config")
    _cfg_cls.return_value = MagicMock(
        device="cpu",
        is_half=False,
        gpu_name="Mock GPU",
        gpu_mem=0,
        n_cpu=2,
        nocheck=True,
        base_dir="/mock/base",
        user_dir="/mock/user",
        dml=False,
    )
    _cfg_mod.Config = _cfg_cls
    sys.modules["configs.config"] = _cfg_mod

# --- infer 内の重量リーフモジュール (テスト対象外のもののみ) ---
# NOTE: infer.lib.torch_compat / infer.lib.device は テスト対象 → スタブしない。
# NOTE: infer.lib.audio も test_audio.py が直接インポートするため、ここではスタブしない。
_INFER_LEAF_STUBS = [
    "infer.lib.rtrvc",
    "infer.lib.slicer2",
    "infer.lib.rvcmd",
    "infer.lib.train.data_utils",
    "infer.lib.train.losses",
    "infer.lib.train.mel_processing",
    "infer.lib.train.process_ckpt",
    "infer.lib.train.utils",
    "infer.modules.vc.pipeline",
    "infer.modules.vc.utils",
    "infer.modules.vc.info",
    "infer.modules.vc.hash",
    "infer.modules.vc.rmvpe",
    "infer.modules.uvr5",
    "infer.modules.uvr5.modules",
    "infer.modules.uvr5.vr",
    "infer.modules.uvr5.mdxnet",
    "rvc.synthesizer",
    "rvc.f0",
]
for _m in _INFER_LEAF_STUBS:
    _stub(_m)

# infer.modules.vc.modules — VC クラスを持つ特殊スタブ
if "infer.modules.vc.modules" not in sys.modules:
    _vc_mod = MagicMock(name="infer.modules.vc.modules")
    _vc_cls = MagicMock(name="VC")
    _vc_cls.return_value = MagicMock()
    _vc_mod.VC = _vc_cls
    sys.modules["infer.modules.vc.modules"] = _vc_mod

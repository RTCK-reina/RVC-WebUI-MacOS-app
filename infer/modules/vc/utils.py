import os, pathlib

# Scoped compatibility helper: relaxes torch.load's weights_only default only
# around fairseq's HuBERT loader, which still pickles a Dictionary instance.
from infer.lib.torch_compat import legacy_load

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                str(pathlib.Path(root, name))
                for path in [os.getenv("outside_index_root"), os.getenv("index_root")]
                if path and os.path.isdir(path)
                for root, _, files in os.walk(path, topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def _hubert_path() -> str:
    # Prefer an explicit env override (set by rpc_server.py on start), otherwise
    # fall back to the bundle base dir, and last resort the legacy relative path.
    env_path = os.environ.get("hubert_path")
    if env_path and os.path.exists(env_path):
        return env_path
    base = os.environ.get("RVC_BASE_DIR")
    if base:
        candidate = os.path.join(base, "assets", "hubert", "hubert_base.pt")
        if os.path.exists(candidate):
            return candidate
    return "assets/hubert/hubert_base.pt"


def load_hubert(device, is_half):
    # fairseq checkpoint_utils uses torch.load internally and the HuBERT
    # base ckpt pickles a Dictionary instance; relax weights_only just here.
    with legacy_load():
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [_hubert_path()],
            suffix="",
        )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()

import os
import traceback
from collections import OrderedDict
from time import time

import torch

from infer.lib.path_safety import safe_leaf_name
from infer.lib.safe_torch_load import load_weights
from i18n.i18n import I18nAuto
from infer.modules.vc import model_hash_ckpt, hash_id

i18n = I18nAuto()


def _user_weight_path(basename: str) -> str:
    """Return the absolute save path for a user-trained weight file.

    Resolution order (first match wins):
      1. ``weight_root`` env — the canonical knob set by
         configs.config._populate_env_paths on app start; points at
         ``user_dir/models/`` on the macOS .app.
      2. ``RVC_USER_DIR`` env — set very early by rpc_server.py (pre-Config)
         and always inherited by subprocesses. Acts as a safety net when
         the training subprocess never actually imports Config (which is
         what was happening in train.py: it never read weight_root, so
         small-model saves ended up in cwd/assets/weights/ instead of
         user_dir/models/).
      3. ``assets/weights`` — upstream-compatible fallback (relative to
         the current working directory). Matches the original repo's
         behavior when neither of the two env hints is set.

    The parent directory is always created so the very first save on a
    fresh install does not trip
    ``Parent directory assets/weights does not exist``.
    """
    safe_basename = safe_leaf_name(basename, "model output name")
    root = os.environ.get("weight_root")
    if not root:
        user_dir = os.environ.get("RVC_USER_DIR")
        if user_dir:
            root = os.path.join(user_dir, "models")
    if not root:
        root = "assets/weights"
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, safe_basename)


# add author sign
def save_small_model(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["name"] = name
        opt["timestamp"] = int(time())
        if hps.author:
            opt["author"] = hps.author
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        h = model_hash_ckpt(opt)
        opt["hash"] = h
        opt["id"] = hash_id(h)
        torch.save(opt, _user_weight_path("%s.pth" % name))
        return "Success."
    except Exception as e:
        raise RuntimeError(traceback.format_exc()) from e


def _synth_config(sr, version):
    """Return the synthesizer ``config`` list for a sample-rate key + version.

    ``sr`` is the STRING key ("40k"/"48k"/"32k"); returns ``None`` for an
    unrecognised key. Shared by extract_small_model() and merge() so a model
    derived from a raw G_*.pth checkpoint (which carries no embedded "config")
    still gets the correct config stamped, and the two code paths cannot drift.
    """
    if sr == "40k":
        return [
            1025,
            32,
            192,
            192,
            768,
            2,
            6,
            3,
            0,
            "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 10, 2, 2],
            512,
            [16, 16, 4, 4],
            109,
            256,
            40000,
        ]
    if sr == "48k":
        if version == "v1":
            return [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 6, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                109,
                256,
                48000,
            ]
        return [
            1025,
            32,
            192,
            192,
            768,
            2,
            6,
            3,
            0,
            "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [12, 10, 2, 2],
            512,
            [24, 20, 4, 4],
            109,
            256,
            48000,
        ]
    if sr == "32k":
        if version == "v1":
            return [
                513,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 4, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                109,
                256,
                32000,
            ]
        return [
            513,
            32,
            192,
            192,
            768,
            2,
            6,
            3,
            0,
            "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 8, 2, 2],
            512,
            [20, 16, 4, 4],
            109,
            256,
            32000,
        ]
    return None


def extract_small_model(path, name, author, sr, if_f0, info, version):
    try:
        ckpt = load_weights(path, map_location="cpu")
        if "model" in ckpt:
            # Raw training checkpoint (G_xxx.pth): unwrap to state_dict
            ckpt = ckpt["model"]
        elif isinstance(ckpt.get("weight"), dict) and all(
            hasattr(v, "half") for v in ckpt["weight"].values()
        ):
            # Already-extracted inference checkpoint — treat ckpt["weight"] as the state_dict
            ckpt = ckpt["weight"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        cfg = _synth_config(sr, version)
        if cfg is not None:
            opt["config"] = cfg
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["name"] = name
        opt["timestamp"] = int(time())
        if author:
            opt["author"] = author
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        h = model_hash_ckpt(opt)
        opt["hash"] = h
        opt["id"] = hash_id(h)
        torch.save(opt, _user_weight_path("%s.pth" % name))
        return "Success."
    except Exception as e:
        raise RuntimeError(traceback.format_exc()) from e


def change_info(path, info, name):
    try:
        ckpt = load_weights(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, _user_weight_path(name))
        return "Success."
    except Exception as e:
        raise RuntimeError(traceback.format_exc()) from e


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:

        def extract(ckpt):
            a = ckpt["model"]
            out = OrderedDict()
            for key in a.keys():
                if "enc_q" in key:
                    continue
                out[key] = a[key]
            return out

        def authors(c1, c2):
            a1, a2 = c1.get("author", ""), c2.get("author", "")
            if a1 == a2:
                return a1
            if not a1:
                a1 = "Unknown"
            if not a2:
                a2 = "Unknown"
            return f"{a1} & {a2}"

        ckpt1 = load_weights(path1, map_location="cpu")
        ckpt2 = load_weights(path2, map_location="cpu")
        # A raw training checkpoint (G_*.pth) carries no "config" key — indexing
        # it directly used to raise KeyError and the bare except swallowed it
        # into a returned traceback, so merging a raw checkpoint as path1
        # silently failed. Derive config from the user-selected sr/version
        # (the same table extract_small_model uses) when it is absent.
        cfg = ckpt1.get("config")
        if cfg is None:
            cfg = _synth_config(sr, version)
            if cfg is None:
                return (
                    "Fail to merge: the first model has no embedded config and "
                    "sample rate '%s' is unrecognised (expected 40k/48k/32k)." % sr
                )
        author = authors(ckpt1, ckpt2)
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        opt["config"] = cfg
        """
        if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
        elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
        elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
        """
        opt["name"] = name
        opt["timestamp"] = int(time())
        if author:
            opt["author"] = author
        opt["sr"] = sr
        opt["f0"] = int(f0)
        opt["version"] = version
        opt["info"] = info
        h = model_hash_ckpt(opt)
        opt["hash"] = h
        opt["id"] = hash_id(h)
        torch.save(opt, _user_weight_path("%s.pth" % name))
        return "Success."
    except Exception as e:
        raise RuntimeError(traceback.format_exc()) from e

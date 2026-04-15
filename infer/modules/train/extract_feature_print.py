"""Feature extraction preprocess — macOS single-device edition.

Runs HuBERT-based feature extraction over the 16 kHz wav slices produced by
the preprocessing step. Originally the script supported CUDA / DirectML and
branched on a user-supplied device string; on macOS we only ever want MPS or
CPU, so the argv shape is simplified accordingly.
"""

import os
import sys
import traceback

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# argv layout (macOS build):
#   sys.argv[1]   requested device hint ("mps" | "cpu" | anything else ignored)
#   sys.argv[2]   n_part
#   sys.argv[3]   i_part
#   sys.argv[4]   exp_dir
#   sys.argv[5]   version ("v1" | "v2")
#   sys.argv[6]   is_half ("true" | "false")  — MPS ignores this (forced fp32)
# The upstream 8-argument form (including i_gpu / CUDA_VISIBLE_DEVICES) has
# been dropped; callers that still pass the legacy extra element are tolerated
# by reading positionally.
device_hint = sys.argv[1].lower() if len(sys.argv) > 1 else ""
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    # Legacy 8-arg form from upstream: slot 4 was i_gpu. Ignore it on macOS.
    exp_dir = sys.argv[5]
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"

import numpy as np
import torch
import torch.nn.functional as F
import fairseq

# Pick device: explicit hint wins if valid, otherwise auto-detect MPS / CPU.
if device_hint == "mps" and torch.backends.mps.is_available():
    device = "mps"
elif device_hint == "cpu":
    device = "cpu"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# MPS does not support fp16 reliably for HuBERT. Force fp32 on macOS.
if device != "cpu" and is_half:
    is_half = False

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

printt("exp_dir: " + exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = load_audio(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt("move model to %s" % device)
# fp16 is off for MPS / CPU (forced above). Left for completeness in case a
# future backend on macOS supports it.
if is_half and device not in ("mps", "cpu"):
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                source = (
                    feats.half().to(device)
                    if is_half and device not in ("mps", "cpu")
                    else feats.to(device)
                )
                inputs = {
                    "source": source,
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except Exception:
            printt(traceback.format_exc())
    printt("all-feature-done")

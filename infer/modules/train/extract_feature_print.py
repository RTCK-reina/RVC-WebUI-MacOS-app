import os
import sys
import traceback
import atexit

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio

# Scoped compatibility helper: relaxes torch.load's weights_only default only
# around fairseq's HuBERT loader. See infer/lib/torch_compat.py for rationale.
from infer.lib.torch_compat import legacy_load

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Bounded allocator (see configs/config.py): 0.0 disabled the MPS limit and
# let runaway allocations push macOS into swap instead of raising OOM.
os.environ.setdefault(
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    os.environ.get("RVC_MPS_WATERMARK_RATIO", "1.7"),
)

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
import fairseq
import numpy as np
import torch
import torch.nn.functional as F

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")
atexit.register(f.close)


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
    # Exit non-zero so the parent (rpc_extract_f0) detects the failure instead
    # of advancing to train.py with an empty feature directory.
    exit(1)
with legacy_load():
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
model = models[0]
model = model.to(device)
printt("move model to %s" % device)
if is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条

# Batched extraction. Preprocess emits mostly identically-sized slices
# (per * sr samples), so files are grouped by EXACT sample count and each
# group is forwarded through HuBERT as one batch. Same-length batching means
# no padding and therefore bit-identical outputs vs. the old one-file-at-a-
# time loop — while cutting the number of model invocations by ~batch size.
BATCH = max(1, int(os.environ.get("RVC_EXTRACT_BATCH", "8")))
_done_count = 0
failures = []


def _flush_bucket(bucket):
    """Run one same-length batch through the model and save each output."""
    global _done_count
    files = [fname for fname, _ in bucket]
    try:
        batch = torch.cat([t for _, t in bucket], dim=0)
        batch = (
            batch.half().to(device)
            if is_half and device not in ["mps", "cpu"]
            else batch.to(device)
        )
        with torch.no_grad():
            logits = model.extract_features(
                source=batch,
                padding_mask=None,  # equivalent to the old all-False mask
                output_layer=9 if version == "v1" else 12,
            )
            out = model.final_proj(logits[0]) if version == "v1" else logits[0]
        out = out.float().cpu().numpy()
        for i, fname in enumerate(files):
            feats_i = out[i]
            out_path = "%s/%s" % (outPath, fname.replace("wav", "npy"))
            if np.isnan(feats_i).sum() == 0:
                np.save(out_path, feats_i, allow_pickle=False)
            else:
                printt("%s-contains nan" % fname)
            _done_count += 1
            if _done_count % n == 0:
                printt(
                    "now-%s,all-%s,%s,%s"
                    % (len(todo), _done_count, fname, feats_i.shape)
                )
    except Exception:
        failures.extend(files)
        printt("%s-feature-fail-%s" % (files, traceback.format_exc()))


if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s (batch=%s)" % (len(todo), BATCH))
    buckets = {}  # sample_count -> list[(file, tensor[1, L])]
    for file in todo:
        try:
            if not file.endswith(".wav"):
                continue
            wav_path = "%s/%s" % (wavPath, file)
            out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
            if os.path.exists(out_path):
                continue
            feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
            key = feats.shape[-1]
            buckets.setdefault(key, []).append((file, feats))
            if len(buckets[key]) >= BATCH:
                _flush_bucket(buckets.pop(key))
        except Exception:
            failures.append(file)
            printt("%s-feature-fail-%s" % (file, traceback.format_exc()))
    for bucket in buckets.values():
        _flush_bucket(bucket)
    if failures:
        printt(
            "feature extraction failed for %s file(s): %s" % (len(failures), failures)
        )
        raise SystemExit(1)
    printt("all-feature-done")

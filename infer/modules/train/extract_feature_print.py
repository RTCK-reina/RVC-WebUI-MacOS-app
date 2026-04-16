import os
import sys
import traceback

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio
# Scoped compatibility helper: relaxes torch.load's weights_only default only
# around fairseq's HuBERT loader. See infer/lib/torch_compat.py for rationale.
from infer.lib.torch_compat import legacy_load

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))

    # D-1: the original loop read each .wav synchronously right before
    # running HuBERT on it, so the GPU (or CPU on MPS-fallback paths) sat
    # idle during the I/O read. The HuBERT forward pass itself is kept
    # strictly sequential (batching across heterogeneous lengths adds
    # padding complexity and does not always speed MPS up), but we
    # overlap the disk read of wav N+1 with the inference of wav N using
    # a tiny 2-worker prefetch pool. For a dataset of many short clips
    # this is the single most effective change; for a few long clips
    # the overhead is negligible.
    from concurrent.futures import ThreadPoolExecutor
    prefetch_pool = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="feat-prefetch"
    )

    def _readwave_safe(path: str):
        try:
            return readwave(path, normalize=saved_cfg.task.normalize)
        except Exception:
            # Keep parity with the original serial loop which printed the
            # full traceback via `traceback.format_exc()`; without this,
            # read failures surfaced as a bare "*-read-failed" line with
            # no diagnostic context.
            printt("read-failed: %s\n%s" % (path, traceback.format_exc()))
            return None

    # Pre-compute the list of (file, wav_path, out_path) triples that
    # actually need work (skip already-written outputs), so the prefetch
    # queue never fills with already-done entries.
    work: list = []
    for file in todo:
        if not file.endswith(".wav"):
            continue
        wav_path = "%s/%s" % (wavPath, file)
        out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
        if os.path.exists(out_path):
            continue
        work.append((file, wav_path, out_path))

    # Kick off the first two reads, then pipeline: while we compute item k,
    # worker threads are already reading k+1 / k+2.
    PREFETCH_AHEAD = 2
    in_flight: list = []  # list[Future[tensor]]
    for j in range(min(PREFETCH_AHEAD, len(work))):
        _, wp, _ = work[j]
        in_flight.append(prefetch_pool.submit(_readwave_safe, wp))

    try:
        for idx, (file, wav_path, out_path) in enumerate(work):
            try:
                feats = in_flight.pop(0).result()
                # Queue up the next prefetch so we always keep PREFETCH_AHEAD
                # reads in the pipeline.
                next_j = idx + PREFETCH_AHEAD
                if next_j < len(work):
                    _, next_wp, _ = work[next_j]
                    in_flight.append(prefetch_pool.submit(_readwave_safe, next_wp))

                if feats is None:
                    printt("%s-read-failed" % file)
                    continue

                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
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
                    printt("now-%s,all-%s,%s,%s" % (len(work), idx, file, feats.shape))
            except Exception:
                printt(traceback.format_exc())
    finally:
        prefetch_pool.shutdown(wait=False)
    printt("all-feature-done")

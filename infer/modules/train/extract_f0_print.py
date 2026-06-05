import os
import sys
import traceback
import atexit
from pathlib import Path

from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np

from infer.lib.audio import load_audio

from rvc.f0 import Generator

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process

exp_dir = sys.argv[1]
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")
atexit.register(f.close)


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


n_p = int(sys.argv[2])
f0method = sys.argv[3]
device = sys.argv[4]
is_half = sys.argv[5] == "True"


class FeatureInput(object):
    # Batch size for RMVPE batched inference. Larger values amortize GPU
    # kernel dispatch overhead but use more VRAM due to mel padding.
    RMVPE_BATCH_SIZE = int(os.environ.get("RMVPE_BATCH_SIZE", "8"))

    def __init__(self, is_half: bool, device="cpu", samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.f0_gen = Generator(
            Path(os.environ["rmvpe_root"]),
            is_half,
            0,
            device,
            hop_size,
            samplerate,
        )

    def _post_process_f0(self, f0):
        """Apply the same post-processing as Generator.calculate (f0_up_key=0, x_pad=0)."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0

    def go(self, paths, f0_method):
        if len(paths) == 0:
            printt("no-f0-todo")
            return

        printt("todo-f0-%s" % len(paths))
        n = max(len(paths) // 5, 1)

        # Use batched RMVPE inference when the method is rmvpe.
        if f0_method == "rmvpe" and self.RMVPE_BATCH_SIZE > 1:
            self._go_batched(paths, n)
        else:
            self._go_sequential(paths, f0_method, n)

    def _go_sequential(self, paths, f0_method, n):
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                if (
                    os.path.exists(opt_path1 + ".npy") == True
                    and os.path.exists(opt_path2 + ".npy") == True
                ):
                    continue
                x = load_audio(inp_path, self.fs)
                coarse_pit, feature_pit = self.f0_gen.calculate(
                    x, x.shape[0] // self.hop, 0, f0_method, None
                )
                np.save(opt_path2, feature_pit, allow_pickle=False)
                np.save(opt_path1, coarse_pit, allow_pickle=False)
            except:
                printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))

    def _go_batched(self, paths, n):
        """Batch RMVPE inference: accumulate files, run model once per batch."""
        # Lazy-init the RMVPE model via Generator's lazy loading.
        rmvpe = self.f0_gen.rmvpe if hasattr(self.f0_gen, "rmvpe") else None
        if rmvpe is None:
            from rvc.f0.rmvpe import RMVPE

            rmvpe = RMVPE(
                str(self.f0_gen.rmvpe_root / "rmvpe.pt"),
                is_half=self.f0_gen.is_half,
                device=self.f0_gen.device,
            )
            self.f0_gen.rmvpe = rmvpe

        batch_size = self.RMVPE_BATCH_SIZE
        # Filter to only files that need processing.
        todo = []
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            if os.path.exists(opt_path1 + ".npy") and os.path.exists(
                opt_path2 + ".npy"
            ):
                continue
            todo.append((idx, inp_path, opt_path1, opt_path2))

        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start : batch_start + batch_size]
            wavs = []
            p_lens = []
            valid = []
            for idx, inp_path, opt_path1, opt_path2 in batch:
                try:
                    x = load_audio(inp_path, self.fs)
                    wavs.append(x)
                    p_lens.append(x.shape[0] // self.hop)
                    valid.append((idx, inp_path, opt_path1, opt_path2))
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))

            if not wavs:
                continue

            try:
                f0_list = rmvpe.compute_f0_batch(
                    wavs, p_lens=p_lens, filter_radius=0.03
                )
            except:
                printt("f0batch-fail-%s" % traceback.format_exc())
                # Fallback to sequential on batch failure.
                for idx, inp_path, opt_path1, opt_path2 in valid:
                    try:
                        x = load_audio(inp_path, self.fs)
                        coarse_pit, feature_pit = self.f0_gen.calculate(
                            x, x.shape[0] // self.hop, 0, "rmvpe", None
                        )
                        np.save(opt_path2, feature_pit, allow_pickle=False)
                        np.save(opt_path1, coarse_pit, allow_pickle=False)
                    except:
                        printt(
                            "f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc())
                        )
                continue

            for i, (idx, inp_path, opt_path1, opt_path2) in enumerate(valid):
                try:
                    coarse_pit, feature_pit = self._post_process_f0(f0_list[i])
                    np.save(opt_path2, feature_pit, allow_pickle=False)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
                    if idx % n == 0:
                        printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                except:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
    # exp_dir=r"E:\codes\py39\dataset\mi-test"
    # n_p=16
    # f = open("%s/log_extract_f0.log"%exp_dir, "w")
    printt(" ".join(sys.argv))
    featureInput = FeatureInput(is_half, device)
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])

    ps = []
    for i in range(n_p):
        p = Process(
            target=featureInput.go,
            args=(
                paths[i::n_p],
                f0method,
            ),
        )
        ps.append(p)
        p.start()
    for i in range(n_p):
        ps[i].join()

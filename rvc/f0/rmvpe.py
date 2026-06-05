from io import BytesIO
import os
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort

from rvc.jit import load_inputs, get_jit_model, export_jit_model, save_pickle

from .e2e import E2E
from .mel import MelSpectrogram
from .f0 import F0Predictor
from .models import get_rmvpe


def rmvpe_jit_export(
    model_path: str,
    mode: str = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")

    model = get_rmvpe(model_path, device, is_half)
    inputs = None
    if mode == "trace":
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export_jit_model(model, mode, inputs, device, is_half)
    ckpt["device"] = str(device)
    save_pickle(ckpt, save_path)
    return ckpt


class RMVPE(F0Predictor):
    """RMVPE pitch extractor.

    Wraps the :class:`E2E` model + :class:`MelSpectrogram` front-end + 360-class
    pitch decode grid into the :class:`F0Predictor` contract used by
    :class:`rvc.f0.gen.Generator`.

    Parameters come from the RMVPE reference implementation:
      * 16 kHz mono audio, 10 ms hop (160 samples)
      * 128-band log-mel with 1024-pt window, fmin=30, fmax=8000
      * 360 cents-per-bin pitch grid @ 20 cents resolution, zero-padded ±4

    The earlier macOS port built ``self.model`` as a Conv2d×3 placeholder that
    never matched ``rmvpe.pt``; it also inherited from ``nn.Module`` instead of
    :class:`F0Predictor`, leaving ``_interpolate_f0`` / ``_resize_f0`` (used by
    ``compute_f0``) undefined. Both are fixed here.
    """

    def __init__(self, model_path, is_half=False, device="cpu"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RMVPE model file not found: {model_path}")

        # RMVPE operates on 16 kHz mono audio with a 10 ms frame hop.
        super().__init__(
            hop_length=160,
            f0_min=50,
            f0_max=1100,
            sampling_rate=16000,
            device=device,
        )
        self.is_half = is_half
        self.resample_kernel = {}

        # Build the real 741-parameter RMVPE architecture and load its weights.
        model = E2E(n_blocks=4, n_gru=1, kernel_size=(2, 2))
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        if is_half:
            model = model.half()
        self.model = model.to(device)

        # Log-mel front-end: 128-band, 1024-pt window, 10 ms hop, 30–8000 Hz.
        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=1024,
            mel_fmin=30,
            mel_fmax=8000,
            device=device,
        ).to(device)

        # 360-class pitch grid, 20 cents per bin, zero-padded ±4 so the
        # argmax-centered averaging window always has a valid lookup range
        # (_to_local_average_cents pads salience the same way).
        self.cents_mapping = np.pad(
            20 * np.arange(360) + 1997.3794084376191, (4, 4)
        )

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            if self.is_half:
                x = x.half()
            return self.model(x)

    def calculate(self, x):
        return self.forward(x)

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: Optional[int] = None,
        filter_radius: Optional[Union[int, float]] = None,
    ):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        if not torch.is_tensor(wav):
            wav = torch.from_numpy(wav)
        mel = self.mel_extractor(wav.float().to(self.device).unsqueeze(0), center=True)
        hidden = self._mel2hidden(mel)
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        if self.is_half == True:
            hidden = hidden.astype("float32")

        f0 = self._decode(hidden, thred=filter_radius)

        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]

    def compute_f0_batch(
        self,
        wavs: list,
        p_lens: Optional[list] = None,
        filter_radius: Optional[Union[int, float]] = None,
    ):
        """Batch F0 extraction: run the E2E model once on multiple audios.

        Extracts mel spectrograms individually (FFT is sequential), pads them
        to the same frame count, runs the model in a single batched forward
        pass, then decodes each output separately.
        """
        n = len(wavs)
        if n == 0:
            return []
        if p_lens is None:
            p_lens = [w.shape[0] // self.hop_length for w in wavs]

        # Extract mels individually (FFT can't be trivially batched for
        # variable-length inputs).
        mels = []
        frame_counts = []
        for wav in wavs:
            if not torch.is_tensor(wav):
                wav = torch.from_numpy(wav)
            mel = self.mel_extractor(
                wav.float().to(self.device).unsqueeze(0), center=True
            )
            mels.append(mel.squeeze(0))  # [128, T_i]
            frame_counts.append(mel.shape[-1])

        # Pad all mels to the max frame count (aligned to 32).
        max_frames = max(frame_counts)
        max_padded = 32 * ((max_frames - 1) // 32 + 1)
        padded = []
        for mel in mels:
            pad_size = max_padded - mel.shape[-1]
            if pad_size > 0:
                mel = F.pad(mel, (0, pad_size), mode="constant")
            padded.append(mel)
        batch_mel = torch.stack(padded, dim=0)  # [B, 128, T_max]

        # Single batched forward pass.
        with torch.no_grad():
            batch_mel = batch_mel.half() if self.is_half else batch_mel.float()
            batch_hidden = self.model(batch_mel)  # [B, T_max, 360]

        # Decode each output individually.
        results = []
        for i in range(n):
            t = frame_counts[i]
            hidden = batch_hidden[i, :t].cpu().numpy()
            if self.is_half:
                hidden = hidden.astype("float32")
            f0 = self._decode(hidden, thred=filter_radius)
            results.append(
                self._interpolate_f0(self._resize_f0(f0, p_lens[i]))[0]
            )
        return results

    def _to_local_average_cents(self, salience, threshold=0.05):
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        center += 4
        starts = center - 4
        # Vectorised gather (equivalent to the per-frame Python loop): each
        # frame takes the 9 contiguous salience/cents bins centred on argmax.
        # The post-pad index range [starts, starts+9) is always in-bounds
        # because salience and cents_mapping are both ±4 padded.
        idx9 = starts[:, None] + np.arange(9)  # 帧长，9
        todo_salience = salience[np.arange(salience.shape[0])[:, None], idx9]  # 帧长，9
        todo_cents_mapping = self.cents_mapping[idx9]  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= threshold] = 0
        return devided

    def _mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if "privateuseone" in str(self.device):
                onnx_input_name = self.model.get_inputs()[0].name
                onnx_outputs_names = self.model.get_outputs()[0].name
                hidden = self.model.run(
                    [onnx_outputs_names],
                    input_feed={onnx_input_name: mel.cpu().numpy()},
                )[0]
            else:
                mel = mel.half() if self.is_half else mel.float()
                hidden = self.model(mel)
            return hidden[:, :n_frames]

    def _decode(self, hidden, thred=0.03):
        if thred is None:
            thred = 0.03
        cents_pred = self._to_local_average_cents(hidden, threshold=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0

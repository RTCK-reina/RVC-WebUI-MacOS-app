from typing import Optional, Union

import torch
import numpy as np
from numba import jit


@jit(nopython=True)
def _interpolate_f0_inplace(data: np.ndarray) -> np.ndarray:
    """In-place gap interpolation over unvoiced (<=0) F0 frames.

    Numba-compiled version of the previous pure-Python nested loop (which was
    O(n^2)-ish on unvoiced-heavy audio and ran per chunk / realtime block).
    Semantics are identical, including the intentional aliasing where filled
    frames become visible to later iterations.
    """
    frame_number = data.shape[0]
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number - 1:
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    data[k] = last_value
        else:
            last_value = data[i]
    return data


class F0Predictor(object):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sampling_rate=44100,
        device: Optional[str] = None,
    ):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        if device is None:
            # macOS build: MPS if available, CPU otherwise.
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: Optional[int] = None,
        filter_radius: Optional[Union[int, float]] = None,
    ): ...

    def _interpolate_f0(self, f0: np.ndarray):
        """
        对F0进行插值处理
        """
        data = np.ascontiguousarray(f0, dtype=np.float64).reshape(-1).copy()
        vuv_vector = (data > 0.0).astype(np.float32)
        ip_data = _interpolate_f0_inplace(data)
        return ip_data, vuv_vector

    def _resize_f0(self, x: np.ndarray, target_len: int):
        source = np.array(x)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res = np.nan_to_num(target)
        return res

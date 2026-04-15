import os
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from time import time

import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from infer.lib.device import empty_device_cache
from rvc.f0 import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device

        self.f0_gen = Generator(
            Path(os.environ["rmvpe_root"]),
            self.is_half,
            self.x_pad,
            self.device,
            self.window,
            self.sr,
        )

        # A-2: FAISS index cache. For the duration of a Pipeline instance
        # (= one VC model load, shared across every vc_single / vc_multi
        # call), keep the last-loaded index and its reconstructed vector
        # matrix so repeated inferences against the same .index file skip
        # the faiss.read_index + reconstruct_n cost (hundreds of ms + a
        # full-ntotal allocation per call).
        #
        # Invalidation: stored with the file path AND mtime; if the index
        # file is rewritten (e.g. user re-trained the index), the cache
        # drops transparently on the next call.
        self._index_cache_path: str = ""
        self._index_cache_mtime: float = 0.0
        self._index_cache: tuple = (None, None)  # (faiss.Index, big_npy)

    def _load_or_reuse_index(self, file_index: str):
        """Return (index, big_npy) for `file_index`, reusing the in-memory
        cache when possible. Returns (None, None) on load failure so the
        caller can proceed without index-based refinement.
        """
        if not file_index or not os.path.exists(file_index):
            return None, None
        try:
            mtime = os.path.getmtime(file_index)
        except OSError:
            return None, None
        if (
            file_index == self._index_cache_path
            and mtime == self._index_cache_mtime
            and self._index_cache[0] is not None
        ):
            return self._index_cache
        try:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception:
            logger.exception("failed to load faiss index: %s", file_index)
            self._index_cache = (None, None)
            self._index_cache_path = ""
            self._index_cache_mtime = 0.0
            return None, None
        self._index_cache = (index, big_npy)
        self._index_cache_path = file_index
        self._index_cache_mtime = mtime
        return index, big_npy

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = time()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if (
            not isinstance(index, type(None))
            and not isinstance(big_npy, type(None))
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            try:
                score, ix = index.search(npy, k=8)
            except Exception as e:
                raise RuntimeError("index mismatch") from e
            # FAISS returns L2 distance squared; a perfect match yields
            # score = 0, which would make 1/score → inf and propagate NaN
            # through the softmax-style normalization below. Clamp to a tiny
            # epsilon so an exact match collapses cleanly to weight ≈ 1.
            score = np.maximum(score, 1e-10)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = time()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            audio1 = (
                (
                    net_g.infer(
                        feats,
                        p_len,
                        sid,
                        pitch=pitch,
                        pitchf=pitchf,
                    )[0, 0]
                )
                .data.cpu()
                .float()
                .numpy()
            )
        del feats, p_len, padding_mask
        t2 = time()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        f0_file=None,
    ):
        # A-2: the index is cached on the Pipeline instance and only
        # re-read from disk when the path or mtime changes. Previously this
        # block executed `faiss.read_index + reconstruct_n` on every call,
        # even for back-to-back inferences against the same index, which
        # was the dominant latency for short files.
        if index_rate != 0:
            index, big_npy = self._load_or_reuse_index(file_index)
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = time()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    raw_lines = f.read()
                    if len(raw_lines) > 0:
                        lines = raw_lines.strip("\n").split("\n")
                        inp_f0 = []
                        for line in lines:
                            inp_f0.append([float(i) for i in line.split(",")])
                        inp_f0 = np.array(inp_f0, dtype="float32")
            except Exception:
                logger.exception("failed to parse f0 file: %s", f0_file.name)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0:
            if if_f0 == 1:
                pitch, pitchf = self.f0_gen.calculate(
                    audio_pad,
                    p_len,
                    f0_up_key,
                    f0_method,
                    filter_radius,
                    inp_f0,
                )
            elif if_f0 == 2:
                pitch, pitchf = f0_method
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            # MPS does not support float64; normalize to float32 on every device
            # to avoid "Cannot convert a MPS Tensor to float64 dtype" at .to(mps).
            pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = time()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr and resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        np.multiply(audio_opt, max_int16, audio_opt)
        del pitch, pitchf, sid
        # A-3: the previous implementation called empty_device_cache() at
        # the tail of every pipeline() invocation, which forces MPS/CUDA
        # to free and then re-allocate working memory on the next call.
        # In the continuous-inference case (batch / realtime) this is pure
        # overhead because the allocator would have reused those arenas.
        # Cache eviction is now the responsibility of VC.load_vc's unload
        # path, which runs at most once per model switch.
        return audio_opt

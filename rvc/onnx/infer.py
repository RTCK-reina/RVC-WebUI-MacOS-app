import typing
import os

import numpy as np

from rvc.f0 import Generator

# onnxruntime / librosa は省略可能。未インストールでも import だけは通るようにする。
# 実際の推論セッション生成時（Model.__init__）に ImportError が発生するため、
# _try_load_onnx() の try/except が正しくキャッチできる。
try:
    import onnxruntime as _ort
except ImportError:  # pragma: no cover
    _ort = None  # type: ignore[assignment]

try:
    import librosa as _librosa_check  # noqa: F401 – 推論ユーティリティ用
except ImportError:
    pass


def _make_providers(
    device: typing.Literal["cpu", "cuda", "dml", "coreml"],
) -> typing.List[str]:
    """デバイス名から ONNX Runtime の ExecutionProvider リストを返す。

    "coreml" を指定すると CoreML ExecutionProvider が優先され、
    Apple Neural Engine (ANE) へのオフロードが有効になる。
    fp32 export であれば品質劣化なしで動作する。
    """
    if device == "cpu":
        return ["CPUExecutionProvider"]
    elif device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "dml":
        return ["DmlExecutionProvider"]
    elif device == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        raise RuntimeError(f"Unsupported device: {device!r}")


class Model:
    def __init__(
        self,
        path: typing.Union[str, bytes, os.PathLike],
        device: typing.Literal["cpu", "cuda", "dml", "coreml"] = "cpu",
    ):
        if _ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX inference but is not installed. "
                "Install it with: pip install onnxruntime"
            )
        self.model = _ort.InferenceSession(path, providers=_make_providers(device))


class ContentVec(Model):
    def __init__(
        self,
        vec_path: typing.Union[str, bytes, os.PathLike],
        device: typing.Literal["cpu", "cuda", "dml", "coreml"] = "cpu",
    ):
        super().__init__(vec_path, device)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        return self.forward(wav)

    def forward(self, wav: np.ndarray) -> np.ndarray:
        if wav.ndim == 2:  # double channels
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        wav = np.expand_dims(np.expand_dims(wav, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: wav}
        logits = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)


class OnnxSynthesizer:
    """PyTorch の net_g.infer() と同じインターフェースを持つ ONNX ラッパー。

    pipeline.py の Pipeline.vc() がそのまま使えるよう、infer() メソッドが
    torch.Tensor (float32) を [1, 1, samples] 形状で返す。

    CoreML ExecutionProvider を指定することで Apple Neural Engine (ANE) に
    オフロードできる（fp32 export であれば品質劣化なし）。
    """

    def __init__(
        self,
        path: typing.Union[str, bytes, os.PathLike],
        device: typing.Literal["cpu", "cuda", "dml", "coreml"] = "cpu",
    ):
        if _ort is None:
            raise ImportError(
                "onnxruntime is required for ONNX inference but is not installed. "
                "Install it with: pip install onnxruntime"
            )
        self._session = _ort.InferenceSession(path, providers=_make_providers(device))
        self._input_names = [inp.name for inp in self._session.get_inputs()]

    def infer(
        self,
        phone: "torch.Tensor",  # [1, T, C]  – HuBERT features
        phone_lengths: "torch.Tensor",  # [1]
        sid: "torch.Tensor",  # [1]
        pitch: typing.Optional["torch.Tensor"] = None,  # [1, T]
        pitchf: typing.Optional["torch.Tensor"] = None,  # [1, T]
        **_kwargs,
    ) -> "torch.Tensor":
        """net_g.infer() と同じシグネチャ。戻り値は [1, 1, samples] の Tensor。

        ONNX export 時の input_names:
            ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
        """
        import torch

        feats_np = phone.cpu().numpy().astype(np.float32)
        p_len_np = phone_lengths.cpu().numpy().astype(np.int64)
        sid_np = sid.cpu().numpy().astype(np.int64)
        T = feats_np.shape[1]

        pitch_np = (
            pitch.cpu().numpy().astype(np.int64)
            if pitch is not None
            else np.zeros((1, T), dtype=np.int64)
        )
        pitchf_np = (
            pitchf.cpu().numpy().astype(np.float32)
            if pitchf is not None
            else np.zeros((1, T), dtype=np.float32)
        )

        # ランダムノイズ（合成の多様性に影響するが品質には影響しない）
        rnd = np.random.randn(1, 192, T).astype(np.float32)

        inputs = dict(
            zip(
                self._input_names,
                [feats_np, p_len_np, pitch_np, pitchf_np, sid_np, rnd],
            )
        )

        audio_np = self._session.run(None, inputs)[0]  # [1, 1, samples]
        return torch.from_numpy(audio_np)


class RVC(Model):
    """スタンドアロン ONNX 推論クラス（HuBERT + RVC 一体型）。

    既存の用途向けに残す。新しい統合推論パスには OnnxSynthesizer を使う。
    """

    def __init__(
        self,
        model_path: typing.Union[str, bytes, os.PathLike],
        hop_len=512,
        model_sr=40000,
        vec_path: typing.Union[str, bytes, os.PathLike] = "vec-768-layer-12.onnx",
        device: typing.Literal["cpu", "cuda", "dml", "coreml"] = "cpu",
    ):
        super().__init__(model_path, device)
        self.vec_model = ContentVec(vec_path, device)
        self.hop_len = hop_len
        self.f0_gen = Generator(None, False, 0, window=hop_len, sr=model_sr)

    def infer(
        self,
        wav: np.ndarray,
        wav_sr: int,
        sid: int = 0,
        f0_method="dio",
        f0_up_key=0,
    ) -> np.ndarray:
        import librosa

        org_length = len(wav)
        if org_length / wav_sr > 50.0:
            raise RuntimeError("wav max length exceeded")

        hubert = self.vec_model(librosa.resample(wav, orig_sr=wav_sr, target_sr=16000))
        hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]

        pitch, pitchf = self.f0_gen.calculate(
            wav, hubert_length, f0_up_key, f0_method, None
        )
        pitch = pitch.astype(np.int64)

        pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch = pitch.reshape(1, len(pitch))
        ds = np.array([sid]).astype(np.int64)

        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length = np.array([hubert_length]).astype(np.int64)

        out_wav = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()

        out_wav = np.pad(out_wav, (0, 2 * self.hop_len), "constant")

        return out_wav[0:org_length]

    def forward(
        self,
        hubert: np.ndarray,
        hubert_length: int,
        pitch: np.ndarray,
        pitchf: np.ndarray,
        ds: np.ndarray,
        rnd: np.ndarray,
    ) -> np.ndarray:
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        return (self.model.run(None, onnx_input)[0] * 32767).astype(np.int16)

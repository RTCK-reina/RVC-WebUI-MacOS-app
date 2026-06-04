"""tests/test_audio.py

infer/lib/audio.py のユニットテスト。

numpy / av が未インストールの場合は全テストをスキップ。
GPU 不要・CPU のみで動作。

テスト対象:
- float_to_int16: 変換精度・境界値・ゼロ入力
- float_np_array_to_wav_buf: mono/stereo、f32/int16 モード
- save_audio: ファイル書き込みと WAV ヘッダー検証
- load_audio: WAV ファイルの読み込みと形状検証
- get_audio_properties: チャンネル数・サンプルレート取得
"""

from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path
from typing import Tuple

import pytest

# ---------------------------------------------------------------------------
# numpy / av が実際にインストールされているかチェック
# ---------------------------------------------------------------------------
np = pytest.importorskip("numpy", reason="numpy が必要")
_av = pytest.importorskip("av", reason="PyAV (av) が必要")

# numpy が使えるなら audio モジュールも試みる
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# numba の @jit を無効化してインポートできるようにする
try:
    import numba  # noqa: F401
except ImportError:
    # numba がなければ @jit を no-op デコレータに差し替える
    import types

    numba_stub = types.ModuleType("numba")
    numba_stub.jit = lambda *args, **kwargs: (lambda f: f)
    sys.modules.setdefault("numba", numba_stub)

try:
    from infer.lib.audio import (
        float_to_int16,
        float_np_array_to_wav_buf,
        save_audio,
        load_audio,
        get_audio_properties,
    )
except ImportError as e:
    pytest.skip(f"audio モジュールのインポート失敗: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _sine_wave(freq: float, sr: int, duration: float) -> np.ndarray:
    """テスト用サイン波（mono float32, -1〜1）。"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _read_wav_header(buf: io.BytesIO) -> Tuple[int, int, int]:
    """BytesIO から (チャンネル数, サンプルレート, サンプル幅_bytes) を返す。"""
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        return wf.getnchannels(), wf.getframerate(), wf.getsampwidth()


# ---------------------------------------------------------------------------
# float_to_int16
# ---------------------------------------------------------------------------


class TestFloatToInt16:
    def test_output_dtype_is_int16(self):
        audio = np.array([0.5, -0.5, 0.1], dtype=np.float32)
        result = float_to_int16(audio)
        assert result.dtype == np.int16

    def test_positive_values_positive(self):
        audio = np.array([0.5], dtype=np.float32)
        result = float_to_int16(audio)
        assert result[0] > 0

    def test_negative_values_negative(self):
        audio = np.array([-0.5], dtype=np.float32)
        result = float_to_int16(audio)
        assert result[0] < 0

    def test_zero_input(self):
        audio = np.zeros(10, dtype=np.float32)
        result = float_to_int16(audio)
        assert np.all(result == 0)

    def test_preserves_length(self):
        audio = np.random.rand(1000).astype(np.float32) * 2 - 1
        result = float_to_int16(audio)
        assert len(result) == len(audio)

    def test_max_amplitude_clamps_to_int16_range(self):
        audio = np.array([1.0, -1.0], dtype=np.float32)
        result = float_to_int16(audio)
        assert np.all(np.abs(result) <= 32767)

    def test_sign_preservation(self):
        """符号が保持される。"""
        audio = np.array([0.3, -0.7, 0.1, -0.2], dtype=np.float32)
        result = float_to_int16(audio)
        assert result[0] > 0
        assert result[1] < 0
        assert result[2] > 0
        assert result[3] < 0


# ---------------------------------------------------------------------------
# float_np_array_to_wav_buf
# ---------------------------------------------------------------------------


class TestFloatNpArrayToWavBuf:
    def test_returns_bytesio(self):
        audio = _sine_wave(440, 16000, 0.1)
        result = float_np_array_to_wav_buf(audio, 16000)
        assert hasattr(result, "read")

    def test_mono_int16_wav(self):
        audio = _sine_wave(440, 16000, 0.1)
        buf = float_np_array_to_wav_buf(audio, 16000, f32=False)
        ch, sr, sw = _read_wav_header(buf)
        assert ch == 1
        assert sr == 16000
        assert sw == 2  # int16 = 2 bytes

    def test_mono_float32_wav(self):
        audio = _sine_wave(440, 16000, 0.1)
        buf = float_np_array_to_wav_buf(audio, 16000, f32=True)
        buf.seek(0)
        import scipy.io.wavfile as wavfile

        sr, data = wavfile.read(buf)
        assert sr == 16000
        assert data.dtype == np.float32

    def test_stereo_int16_wav(self):
        mono = _sine_wave(440, 16000, 0.1)
        stereo = np.stack([mono, mono], axis=0)  # (2, N)
        buf = float_np_array_to_wav_buf(stereo, 16000, f32=False)
        ch, sr, sw = _read_wav_header(buf)
        assert ch == 2
        assert sr == 16000

    def test_different_sample_rates(self):
        for sr in [8000, 22050, 44100, 48000]:
            audio = _sine_wave(440, sr, 0.05)
            buf = float_np_array_to_wav_buf(audio, sr, f32=False)
            _, actual_sr, _ = _read_wav_header(buf)
            assert actual_sr == sr

    def test_output_not_empty(self):
        audio = _sine_wave(440, 16000, 0.1)
        buf = float_np_array_to_wav_buf(audio, 16000)
        buf.seek(0, 2)  # seek to end
        size = buf.tell()
        assert size > 44  # WAV ヘッダは最低 44 bytes


# ---------------------------------------------------------------------------
# save_audio
# ---------------------------------------------------------------------------


class TestSaveAudio:
    def test_saves_wav_file(self, tmp_path):
        audio = _sine_wave(440, 16000, 0.1)
        path = str(tmp_path / "test.wav")
        save_audio(path, audio, 16000)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_wav_content_readable(self, tmp_path):
        audio = _sine_wave(440, 16000, 0.5)
        path = str(tmp_path / "out.wav")
        save_audio(path, audio, 16000)
        with wave.open(path, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1

    def test_f32_format(self, tmp_path):
        audio = _sine_wave(440, 16000, 0.1)
        path = str(tmp_path / "f32.wav")
        save_audio(path, audio, 16000, f32=True)
        import scipy.io.wavfile as wavfile

        sr, data = wavfile.read(path)
        assert sr == 16000
        assert data.dtype == np.float32

    def test_different_sample_rates(self, tmp_path):
        for sr in [8000, 44100]:
            audio = _sine_wave(440, sr, 0.05)
            path = str(tmp_path / f"audio_{sr}.wav")
            save_audio(path, audio, sr)
            with wave.open(path) as wf:
                assert wf.getframerate() == sr


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------


class TestLoadAudio:
    def _make_wav_bytes(self, sr: int = 16000, duration: float = 0.1) -> bytes:
        """インメモリ WAV バイト列を生成。"""
        audio = _sine_wave(440, sr, duration)
        buf = float_np_array_to_wav_buf(audio, sr, f32=False)
        buf.seek(0)
        return buf.read()

    def test_load_returns_ndarray(self, tmp_path):
        data = self._make_wav_bytes()
        p = tmp_path / "test.wav"
        p.write_bytes(data)
        result = load_audio(str(p), sr=16000)
        assert isinstance(result, np.ndarray)

    def test_load_with_sr_returns_resampled(self, tmp_path):
        data = self._make_wav_bytes(sr=16000, duration=0.5)
        p = tmp_path / "t.wav"
        p.write_bytes(data)
        result = load_audio(str(p), sr=16000)
        assert result.ndim == 1  # mono
        assert len(result) > 0

    def test_load_without_sr_returns_tuple(self, tmp_path):
        data = self._make_wav_bytes(sr=22050, duration=0.1)
        p = tmp_path / "t.wav"
        p.write_bytes(data)
        result = load_audio(str(p), sr=None)
        # sr=None のとき (ndarray, rate) タプル
        assert isinstance(result, tuple)
        arr, rate = result
        assert isinstance(arr, np.ndarray)
        assert isinstance(rate, int)
        assert rate == 22050

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/no/such/file.wav", sr=16000)

    def test_mono_output_is_1d(self, tmp_path):
        data = self._make_wav_bytes(sr=16000, duration=0.1)
        p = tmp_path / "m.wav"
        p.write_bytes(data)
        result = load_audio(str(p), sr=16000, mono=True)
        assert result.ndim == 1

    def test_float32_output(self, tmp_path):
        data = self._make_wav_bytes(sr=16000, duration=0.1)
        p = tmp_path / "f.wav"
        p.write_bytes(data)
        result = load_audio(str(p), sr=16000)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# get_audio_properties
# ---------------------------------------------------------------------------


class TestGetAudioProperties:
    def _make_wav_file(self, tmp_path: Path, sr: int, channels: int) -> str:
        if channels == 1:
            audio = _sine_wave(440, sr, 0.1)
        else:
            mono = _sine_wave(440, sr, 0.1)
            audio = np.stack([mono] * channels, axis=0)
        buf = float_np_array_to_wav_buf(audio, sr, f32=False)
        path = tmp_path / f"audio_{sr}_{channels}ch.wav"
        path.write_bytes(buf.getvalue())
        return str(path)

    def test_mono_properties(self, tmp_path):
        path = self._make_wav_file(tmp_path, 16000, 1)
        ch, sr = get_audio_properties(path)
        assert ch == 1
        assert sr == 16000

    def test_stereo_properties(self, tmp_path):
        path = self._make_wav_file(tmp_path, 44100, 2)
        ch, sr = get_audio_properties(path)
        assert ch == 2
        assert sr == 44100

    def test_various_sample_rates(self, tmp_path):
        for expected_sr in [8000, 22050, 48000]:
            path = self._make_wav_file(tmp_path, expected_sr, 1)
            _, sr = get_audio_properties(path)
            assert sr == expected_sr

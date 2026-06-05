from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import os
import math
import wave

import numpy as np
from numba import jit
import av
from av.audio.resampler import AudioResampler
from av.audio.frame import AudioFrame
import scipy.io.wavfile as wavfile

video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}


@jit(nopython=True)
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    peak = float(np.abs(audio).max())
    if peak <= 0.0:
        return np.zeros_like(audio, dtype=np.int16)
    am = int(math.ceil(peak * 32768))
    if am <= 0:
        return np.zeros_like(audio, dtype=np.int16)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def _to_int16_pcm(wav: np.ndarray) -> np.ndarray:
    """Return int16 PCM samples for WAV writing, dtype-aware.

    Float arrays are assumed to be in roughly [-1, 1] and are peak-normalised
    by ``float_to_int16``. Integer arrays are ALREADY PCM — e.g. the inference
    pipeline (``infer/modules/vc/pipeline.py``) deliberately preserves output
    loudness and the rms-mix envelope, then casts to int16. Running such an
    array back through ``float_to_int16`` would peak-normalise it to full scale
    (a quiet passage gets amplified Nx), destroying that loudness and making
    batch files inconsistent. For integer input we only clip+cast, never
    re-normalise.
    """
    if np.issubdtype(wav.dtype, np.integer):
        return np.clip(wav, -32768, 32767).astype(np.int16)
    return float_to_int16(wav)


def float_np_array_to_wav_buf(wav: np.ndarray, sr: int, f32=False) -> BytesIO:
    buf = BytesIO()
    if f32:
        wavfile.write(buf, sr, wav.astype(np.float32))
    else:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2 if len(wav.shape) > 1 else 1)
            wf.setsampwidth(2)  # Sample width in bytes
            wf.setframerate(sr)  # Sample rate in Hz
            pcm = _to_int16_pcm(wav.T if len(wav.shape) > 1 else wav)
            wf.writeframes(np.ascontiguousarray(pcm).tobytes())
    buf.seek(0, 0)
    return buf


def save_audio(path: str, audio: np.ndarray, sr: int, f32=False, format="wav"):
    buf = float_np_array_to_wav_buf(audio, sr, f32)
    if format != "wav":
        transbuf = BytesIO()
        wav2(buf, transbuf, format)
        buf = transbuf
    with open(path, "wb") as f:
        f.write(buf.getbuffer())


def wav2(i: BytesIO, o: BufferedWriter, format: str):
    inp = av.open(i, "r")
    format = video_format_dict.get(format, format)
    out = av.open(o, "w", format=format)
    format = audio_format_dict.get(format, format)

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def _audio_stream_channels(audio_stream) -> int:
    """Return the stream channel count across PyAV versions."""
    for attr in ("channels",):
        val = getattr(audio_stream, attr, None)
        if val is not None:
            return max(1, int(val))

    cc = getattr(audio_stream, "codec_context", None)
    if cc is not None:
        val = getattr(cc, "channels", None)
        if val is not None:
            return max(1, int(val))

    layout = getattr(audio_stream, "layout", None)
    if layout is not None:
        layout_channels = getattr(layout, "channels", None)
        if layout_channels is not None:
            try:
                return max(1, len(layout_channels))
            except TypeError:
                try:
                    return max(1, int(layout_channels))
                except (TypeError, ValueError):
                    pass
        layout_name = str(getattr(layout, "name", layout)).lower()
        if layout_name == "mono":
            return 1
        if layout_name == "stereo":
            return 2

    return 1


def _ensure_audio_capacity(audio: np.ndarray, end_index: int) -> np.ndarray:
    """Grow a channel-first audio buffer without np.resize data repetition."""
    if end_index <= audio.shape[1]:
        return audio
    new_len = max(end_index, audio.shape[1] * 2)
    grown = np.zeros((audio.shape[0], new_len), dtype=audio.dtype)
    grown[:, : audio.shape[1]] = audio
    return grown


def load_audio(
    file: Union[str, BytesIO, Path],
    sr: Optional[int] = None,
    format: Optional[str] = None,
    mono=True,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    if (isinstance(file, str) and not Path(file).exists()) or (
        isinstance(file, Path) and not file.exists()
    ):
        raise FileNotFoundError(f"File not found: {file}")

    container = av.open(file, format=format)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
        channels = _audio_stream_channels(audio_stream)
        container.seek(0)
        resampler = (
            AudioResampler(format="fltp", layout=audio_stream.layout, rate=sr)
            if sr is not None
            else None
        )

        # AV stores duration in microseconds. Some streams do not expose it,
        # so fall back to one second and grow the buffer as frames arrive.
        target_rate = sr if sr is not None else _audio_stream_sample_rate(audio_stream)
        estimated_total_samples = (
            int(container.duration * target_rate // 1_000_000)
            if container.duration is not None
            else target_rate
        )
        decoded_audio = np.zeros(
            (channels, estimated_total_samples + 1), dtype=np.float32
        )
        offset = 0
        rate = 0

        def process_packet(packet: List[AudioFrame]):
            frames_data = []
            packet_rate = 0
            for frame in packet:
                # frame.pts = None  # 清除时间戳，避免重新采样问题
                resampled_frames = (
                    resampler.resample(frame) if resampler is not None else [frame]
                )
                for resampled_frame in resampled_frames:
                    frame_data = resampled_frame.to_ndarray()
                    if frame_data.ndim == 1:
                        frame_data = frame_data[np.newaxis, :]
                    packet_rate = resampled_frame.rate
                    frames_data.append(frame_data)
            return packet_rate, frames_data

        def frame_iter():
            for packet in container.demux(container.streams.audio[0]):
                yield packet.decode()

        for packet_rate, frames_data in map(process_packet, frame_iter()):
            if not rate:
                rate = packet_rate
            for frame_data in frames_data:
                end_index = offset + len(frame_data[0])
                frame_channels = frame_data.shape[0]
                if frame_channels != decoded_audio.shape[0]:
                    if offset == 0:
                        decoded_audio = np.zeros(
                            (frame_channels, decoded_audio.shape[1]),
                            dtype=decoded_audio.dtype,
                        )
                    elif frame_channels == 1:
                        frame_data = np.repeat(
                            frame_data, decoded_audio.shape[0], axis=0
                        )
                    elif decoded_audio.shape[0] == 1:
                        decoded_audio = np.repeat(
                            decoded_audio, frame_channels, axis=0
                        )
                    else:
                        raise ValueError(
                            "Audio channel count changed from "
                            f"{decoded_audio.shape[0]} to {frame_channels}"
                        )

                decoded_audio = _ensure_audio_capacity(decoded_audio, end_index)
                np.copyto(decoded_audio[..., offset:end_index], frame_data)
                offset += len(frame_data[0])
    finally:
        container.close()

    decoded_audio = decoded_audio[..., :offset]

    if mono and decoded_audio.shape[0] > 1:
        decoded_audio = decoded_audio.mean(0)
    elif mono and decoded_audio.shape[0] == 1:
        decoded_audio = decoded_audio[0]

    if sr is not None:
        return decoded_audio
    return decoded_audio, rate


def resample_audio(
    input_path: str, output_path: str, codec: str, format: str, sr: int, layout: str
) -> None:
    if not os.path.exists(input_path):
        return

    input_container = av.open(input_path)
    output_container = av.open(output_path, "w")

    # Create a stream in the output container
    input_stream = input_container.streams.audio[0]
    output_stream = output_container.add_stream(codec, rate=sr, layout=layout)

    resampler = AudioResampler(format, layout, sr)

    # Copy packets from the input file to the output file
    for packet in input_container.demux(input_stream):
        for frame in packet.decode():
            # frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            out_frames = resampler.resample(frame)
            for out_frame in out_frames:
                for out_packet in output_stream.encode(out_frame):
                    output_container.mux(out_packet)

    for packet in output_stream.encode():
        output_container.mux(packet)

    # Close the containers
    input_container.close()
    output_container.close()


def _audio_stream_sample_rate(audio_stream) -> int:
    """PyAV renamed `base_rate` across versions — use whichever exists."""
    for attr in ("sample_rate", "rate", "base_rate"):
        val = getattr(audio_stream, attr, None)
        if val is not None:
            return int(val)
    cc = getattr(audio_stream, "codec_context", None)
    if cc is not None:
        for attr in ("sample_rate", "rate"):
            val = getattr(cc, attr, None)
            if val is not None:
                return int(val)
    raise AttributeError("Could not determine sample rate from audio stream")


def get_audio_properties(input_path: str) -> Tuple[int, int]:
    container = av.open(input_path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
        channels = _audio_stream_channels(audio_stream)
        rate = _audio_stream_sample_rate(audio_stream)
        return channels, rate
    finally:
        container.close()

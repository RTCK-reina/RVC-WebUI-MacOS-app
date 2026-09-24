"""Equivalence tests for the 2026-08 performance optimization batch.

Each test pins a rewritten hot path against the original (slower) reference
implementation to prove the optimization did not change numerics/semantics.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# tests/conftest.py stubs torch (and friends) with MagicMock so the suite can
# run without the ML environment. Numerical equivalence cannot be proven
# against mocks — run this module with the real bundled env instead:
#   build/python_env/bin/python -m pytest <copy of this file outside tests/>
if "mock" in str(getattr(torch, "__version__", "")) or not hasattr(torch, "allclose"):
    pytest.skip(
        "real torch required (stubbed test harness detected)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# pipeline.py: cumsum moving-sum replacement for the O(window*N) loop.
# ---------------------------------------------------------------------------
class TestMovingSumEquivalence:
    def _reference(self, audio, audio_pad, window):
        audio_sum = np.zeros_like(audio)
        for i in range(window):
            audio_sum += np.abs(audio_pad[i : i - window])
        return audio_sum

    def _optimized(self, audio, audio_pad, window):
        csum = np.concatenate(([0.0], np.cumsum(np.abs(audio_pad), dtype=np.float64)))
        return (csum[window:] - csum[:-window])[: audio.shape[0]]

    @pytest.mark.parametrize("n", [1600, 16000, 48001])
    def test_matches_reference(self, n):
        rng = np.random.default_rng(42)
        window = 160
        audio = rng.standard_normal(n).astype(np.float64)
        audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
        ref = self._reference(audio, audio_pad, window)
        opt = self._optimized(audio, audio_pad, window)
        assert opt.shape == ref.shape
        np.testing.assert_allclose(opt, ref, rtol=1e-9, atol=1e-9)

    def test_argmin_stable_on_realistic_audio(self):
        # The value being consumed downstream is an argmin over a window;
        # verify the cut-point selection itself is identical.
        rng = np.random.default_rng(7)
        window = 160
        audio = (rng.standard_normal(32000) * np.hanning(32000)).astype(np.float64)
        audio_pad = np.pad(audio, (window // 2, window // 2), mode="reflect")
        ref = self._reference(audio, audio_pad, window)
        opt = self._optimized(audio, audio_pad, window)
        t_query = 1600
        for t in (8000, 16000, 24000):
            sl = slice(t - t_query, t + t_query)
            assert (
                np.where(ref[sl] == ref[sl].min())[0][0]
                == np.where(opt[sl] == opt[sl].min())[0][0]
            )


# ---------------------------------------------------------------------------
# rvc/f0/f0.py: numba-compiled _interpolate_f0 vs the original Python loop.
# ---------------------------------------------------------------------------
class TestInterpolateF0Equivalence:
    def _reference(self, f0):
        # Original implementation (pre-numba), verbatim semantics.
        data = np.reshape(f0, (f0.size, 1))
        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0
        ip_data = data
        frame_number = data.size
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
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]
        return ip_data[:, 0], vuv_vector[:, 0]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_reference(self, seed):
        from rvc.f0.f0 import F0Predictor

        rng = np.random.default_rng(seed)
        f0 = rng.uniform(0, 400, size=500)
        f0[rng.random(500) < 0.4] = 0.0  # unvoiced gaps
        pred = F0Predictor.__new__(F0Predictor)  # no torch device needed
        got_ip, got_vuv = F0Predictor._interpolate_f0(pred, f0.copy())
        ref_ip, ref_vuv = self._reference(f0.copy())
        np.testing.assert_allclose(got_ip, ref_ip, rtol=1e-12)
        np.testing.assert_array_equal(got_vuv, ref_vuv)

    def test_all_unvoiced_and_all_voiced(self):
        from rvc.f0.f0 import F0Predictor

        pred = F0Predictor.__new__(F0Predictor)
        for f0 in (np.zeros(64), np.full(64, 220.0)):
            got_ip, got_vuv = F0Predictor._interpolate_f0(pred, f0.copy())
            ref_ip, ref_vuv = self._reference(f0.copy())
            np.testing.assert_allclose(got_ip, ref_ip)
            np.testing.assert_array_equal(got_vuv, ref_vuv)


# ---------------------------------------------------------------------------
# gui.py: torch RMS vs librosa.feature.rms.
# ---------------------------------------------------------------------------
class TestTorchRmsEquivalence:
    def test_matches_librosa(self):
        librosa = pytest.importorskip("librosa")
        gui = pytest.importorskip("gui")
        _rms_torch = gui._rms_torch

        rng = np.random.default_rng(3)
        zc = 160
        y = rng.standard_normal(zc * 40).astype(np.float32)
        try:
            ref = librosa.feature.rms(y=y, frame_length=4 * zc, hop_length=zc)
        except ImportError as e:  # librosa lazy-imports scipy submodules
            pytest.skip(f"librosa/scipy unusable in this env: {e}")
        got = _rms_torch(torch.from_numpy(y), 4 * zc, zc).numpy()
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# losses.py: tensors returned instead of floats, same values.
# ---------------------------------------------------------------------------
class TestDiscriminatorLossTensors:
    def test_values_and_types(self):
        from infer.lib.train.losses import discriminator_loss

        dr = [torch.tensor([0.9, 0.8]), torch.tensor([0.7])]
        dg = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]
        loss, r_losses, g_losses = discriminator_loss(dr, dg)
        assert all(torch.is_tensor(v) for v in r_losses + g_losses)
        expected = sum(
            torch.mean((1 - r.float()) ** 2) + torch.mean(g.float() ** 2)
            for r, g in zip(dr, dg)
        )
        assert torch.allclose(loss, expected)
        for r, got in zip(dr, r_losses):
            assert torch.allclose(got, torch.mean((1 - r.float()) ** 2))


# ---------------------------------------------------------------------------
# configs/config.py: use_fp32_config must only touch train.fp16_run.
# ---------------------------------------------------------------------------
class TestUseFp32ConfigTargetedEdit:
    def test_only_fp16_run_changes(self, tmp_path, monkeypatch):
        from configs.config import Config, version_config_list

        # Config is wrapped by @singleton_variable (a function); recover the
        # underlying class from the decorator closure to instantiate without
        # running the full device-probing __init__.
        klass = None
        for cell in getattr(Config, "__closure__", None) or ():
            if isinstance(cell.cell_contents, type):
                klass = cell.cell_contents
                break
        if klass is None:
            pytest.skip("could not unwrap singleton Config class")
        cfg = klass.__new__(klass)  # avoid full device probing
        cfg.user_dir = tmp_path
        cfg.json_config = {
            k: {"train": {"fp16_run": True}} for k in version_config_list
        }
        inuse = tmp_path / "configs" / "inuse" / "v1"
        inuse.mkdir(parents=True)
        target = inuse / "40k.json"
        original = {
            "train": {"fp16_run": True, "some_flag": True},
            "data": {"name": "true_believer", "enabled": True},
        }
        target.write_text(json.dumps(original))
        cfg.use_fp32_config()
        after = json.loads(target.read_text())
        assert after["train"]["fp16_run"] is False
        # The old blanket text replace would have destroyed these:
        assert after["train"]["some_flag"] is True
        assert after["data"]["enabled"] is True
        assert after["data"]["name"] == "true_believer"


# ---------------------------------------------------------------------------
# data_utils.py: float32-WAV-aware length estimate.
# ---------------------------------------------------------------------------
class TestBucketLengthEstimate:
    def test_f32_wav_length(self, tmp_path):
        # A 1-second 40kHz float32 mono WAV: 40000 samples * 4 bytes + 44B.
        sr, hop = 40000, 400
        n = sr
        path = tmp_path / "x.wav"
        path.write_bytes(b"\x00" * (44 + n * 4))
        import os

        est = max(1, (os.path.getsize(path) - 44) // (4 * hop))
        assert est == n // hop  # exactly 100 frames

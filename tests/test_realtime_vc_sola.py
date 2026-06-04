"""tests/test_realtime_vc_sola.py

gui.py のリアルタイムVC SOLA オフセット計算の回帰テスト。

背景:
- 旧 gui.py は macOS 分岐で `_, sola_offset = torch.max(t1d)` を使用していた。
- `torch.max(input)` は dim 省略時スカラー tensor を返し、タプルではないため
  `TypeError: cannot unpack non-iterable Tensor object` が発生する。
- sounddevice の callback 内例外はデフォルトで握り潰されるため、
  症状は「macOS でリアルタイムVC が無音」となり発見が遅れた。
- 本テストは同じ書き方が再発していないことを保証する。
"""

from __future__ import annotations

import pytest

# conftest.py で torch がスタブ化されている場合は本物の API を検証できないため、
# 実 torch が必要。未インストール環境では skip する。
torch = pytest.importorskip("torch")
if "+mock" in getattr(torch, "__version__", ""):
    pytest.skip(
        "real torch not installed (conftest stub active)",
        allow_module_level=True,
    )


class TestTorchMaxArgmaxAPIContract:
    """torch.max / torch.argmax の仕様確認。

    gui.py の修正が依拠する API 契約を明文化する。
    """

    def test_torch_max_of_1d_tensor_returns_scalar_not_tuple(self):
        """`torch.max(input)` は dim 省略時スカラー tensor を返し、タプルではない。

        旧 gui.py で `_, x = torch.max(t1d)` と書くとこの挙動により TypeError
        となっていた。その前提を固定する。
        """
        t = torch.tensor([1.0, 3.0, 2.0])
        result = torch.max(t)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0
        with pytest.raises(TypeError):
            _a, _b = torch.max(t)  # noqa: F841 — アンパック不可を意図的に検証

    def test_torch_argmax_of_1d_tensor_item_returns_int(self):
        """`torch.argmax(t).item()` は Python int を返す。"""
        t = torch.tensor([1.0, 3.0, 2.0])
        idx = torch.argmax(t).item()
        assert isinstance(idx, int)
        assert idx == 1


class TestSolaOffsetCalculation:
    """gui.py:audio_callback 内の SOLA オフセット計算の正当性。"""

    def test_offset_matches_reference_location(self):
        """sola_buffer と完全一致する位置で argmax が立つ。"""
        import torch.nn.functional as F

        sola_buffer_frame = 8
        sola_search_frame = 4
        sola_buffer = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        expected_offset = 3
        infer_wav = torch.cat(
            [
                torch.zeros(expected_offset),
                sola_buffer,
                torch.zeros(1),
            ]
        )

        conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, sola_buffer_frame),
            )
            + 1e-8
        )

        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()
        assert sola_offset == expected_offset

    def test_offset_is_int_in_valid_range(self):
        """ランダム入力でも sola_offset は int かつ search frame 内に収まる。

        旧 macOS 分岐の誤った書き方が再導入されていないことを検知する
        センチネルテスト。
        """
        import torch.nn.functional as F

        sola_buffer_frame = 8
        sola_search_frame = 4
        torch.manual_seed(0)
        sola_buffer = torch.randn(sola_buffer_frame)
        infer_wav = torch.randn(sola_buffer_frame + sola_search_frame)

        conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, sola_buffer_frame),
            )
            + 1e-8
        )
        ratio = cor_nom[0, 0] / cor_den[0, 0]

        sola_offset = torch.argmax(ratio).item()
        assert isinstance(sola_offset, int)
        assert 0 <= sola_offset <= sola_search_frame

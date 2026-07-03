from __future__ import annotations

from types import SimpleNamespace

import pytest

from infer.lib.path_safety import (
    PathValidationError,
    optional_file_under,
    require_env_root,
    resolve_under,
    safe_format,
    safe_leaf_name,
)
from rpc_server import _require_existing_file
from rpc_training import _exp_dir, _safe_f0_method, _safe_sr_name, _safe_version


def test_safe_leaf_name_rejects_traversal():
    with pytest.raises(PathValidationError):
        safe_leaf_name("../escape", "exp_name")


def test_safe_leaf_name_allows_unicode_display_name():
    assert safe_leaf_name("戌亥とこ_01", "exp_name") == "戌亥とこ_01"


def test_resolve_under_rejects_absolute_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(PathValidationError):
        resolve_under(root, outside, "output_path")


def test_optional_file_under_allows_default_inside_root(tmp_path):
    root = tmp_path / "root"
    target = optional_file_under(
        root, "", "output_path", default=root / "out" / "sample.flac"
    )
    assert target == (root / "out" / "sample.flac").resolve()
    assert target.parent.is_dir()


def test_safe_format_rejects_unknown_format():
    with pytest.raises(PathValidationError):
        safe_format("aiff", ("flac", "wav"))


def test_training_exp_dir_cannot_escape_user_logs(tmp_path):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    config = SimpleNamespace(user_dir=user_dir)

    with pytest.raises(PathValidationError):
        _exp_dir(config, "../../escape")


def test_training_exp_dir_uses_user_logs_for_valid_name(tmp_path):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    config = SimpleNamespace(user_dir=user_dir)

    exp_dir = _exp_dir(config, "voice_run")
    assert exp_dir == user_dir / "logs" / "voice_run"
    assert exp_dir.is_dir()


def test_rpc_existing_file_validation_rejects_missing(tmp_path):
    with pytest.raises(PathValidationError):
        _require_existing_file(tmp_path / "missing.wav", "input_audio_path")


def test_rpc_existing_file_validation_returns_resolved_file(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF")

    assert _require_existing_file(audio, "input_audio_path") == str(audio.resolve())


def test_training_sr_rejects_unknown_value():
    with pytest.raises(PathValidationError):
        _safe_sr_name("44k")


def test_training_version_rejects_path_like_value():
    with pytest.raises(PathValidationError):
        _safe_version("../v2")


def test_training_f0_method_rejects_unknown_value():
    with pytest.raises(PathValidationError):
        _safe_f0_method("shell")


def test_require_env_root_fails_closed_when_unset(monkeypatch):
    monkeypatch.delenv("weight_root", raising=False)
    with pytest.raises(PathValidationError):
        require_env_root("weight_root")


def test_require_env_root_fails_closed_when_blank(monkeypatch):
    monkeypatch.setenv("weight_root", "   ")
    with pytest.raises(PathValidationError):
        require_env_root("weight_root")


def test_require_env_root_returns_resolved_path(monkeypatch, tmp_path):
    monkeypatch.setenv("weight_root", str(tmp_path))
    assert require_env_root("weight_root") == tmp_path.resolve()

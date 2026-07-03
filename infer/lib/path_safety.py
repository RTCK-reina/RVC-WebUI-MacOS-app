"""Path and file-name validation helpers for RPC-facing code.

The native app accepts several paths from Swift/AppStorage and from the
line-delimited RPC protocol. Treat every such value as untrusted at the Python
boundary: input files may live anywhere the user can read, but app-managed
outputs and generated names must stay inside their configured roots.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


class PathValidationError(ValueError):
    """Raised when a user/RPC supplied path violates an app path policy."""


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def safe_leaf_name(value: object, field: str, *, max_len: int = 120) -> str:
    """Return *value* as a single safe path component.

    Unicode display names are allowed, but directory separators, control
    characters, empty names, and dot-only traversal names are rejected.
    """
    name = str(value or "").strip()
    if not name:
        raise PathValidationError(f"{field} is required")
    if len(name) > max_len:
        raise PathValidationError(f"{field} is too long (max {max_len} characters)")
    if name in {".", ".."}:
        raise PathValidationError(f"{field} must not be {name!r}")
    if "/" in name or "\\" in name or os.sep in name:
        raise PathValidationError(f"{field} must be a file name, not a path")
    if os.altsep and os.altsep in name:
        raise PathValidationError(f"{field} must be a file name, not a path")
    if _CONTROL_CHARS.search(name):
        raise PathValidationError(f"{field} contains control characters")
    return name


def safe_stem(value: object, field: str, *, fallback: str = "item") -> str:
    """Return a filesystem-safe-ish stem derived from an arbitrary label."""
    stem = Path(str(value or "")).stem.strip()
    if not stem:
        stem = fallback
    stem = _CONTROL_CHARS.sub("_", stem)
    stem = stem.replace("/", "_").replace("\\", "_")
    if stem in {".", ".."}:
        stem = fallback
    return stem[:120]


def safe_format(value: object, allowed: Iterable[str], field: str = "format") -> str:
    allowed_set = {str(v).lower() for v in allowed}
    fmt = str(value or "").strip().lower()
    if fmt not in allowed_set:
        allowed_list = ", ".join(sorted(allowed_set))
        raise PathValidationError(f"{field} must be one of: {allowed_list}")
    return fmt


def require_env_root(name: str) -> Path:
    """Return the root directory configured in env var *name*, fail-closed.

    A missing/empty root must never silently degrade to cwd (``Path("")``
    resolves to the current directory, which would defeat containment).
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        raise PathValidationError(f"{name} is not configured")
    return Path(raw).expanduser().resolve()


def resolve_under(root: str | Path, value: str | Path, field: str) -> Path:
    """Resolve *value* and require that it stays under *root*.

    Absolute values are accepted only if they are already inside *root*.
    Relative values are interpreted relative to *root*. Existing symlinked
    parents are resolved before the containment check.
    """
    root_path = Path(root).expanduser().resolve()
    raw = Path(str(value)).expanduser()
    candidate = raw if raw.is_absolute() else root_path / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise PathValidationError(f"{field} must stay under {root_path}") from exc
    return resolved


def optional_dir_under(
    root: str | Path,
    value: object,
    field: str,
    *,
    default: str | Path,
) -> Path:
    if value is None or str(value).strip() == "":
        resolved = Path(default).expanduser().resolve()
    else:
        resolved = resolve_under(root, str(value), field)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def optional_file_under(
    root: str | Path,
    value: object,
    field: str,
    *,
    default: str | Path,
) -> Path:
    if value is None or str(value).strip() == "":
        resolved = Path(default).expanduser().resolve()
        root_path = Path(root).expanduser().resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError as exc:
            raise PathValidationError(f"default {field} escapes {root_path}") from exc
    else:
        resolved = resolve_under(root, str(value), field)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

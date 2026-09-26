#!/usr/bin/env python3
"""Repair Mach-O zero-fill sections that newer macOS dyld refuses to load.

Symptom (seen on macOS 26/27 with gfortran-built extensions, e.g. scipy's
PROPACK/ARPACK modules — even from official PyPI wheels):

    ImportError: dlopen(..._spropack.cpython-310-darwin.so, 0x0002):
      section '__DATA/__thread_bss' has a zero-fill section type,
      but offset field is not zero

Zero-fill sections (S_ZEROFILL / S_THREAD_LOCAL_ZEROFILL / S_GB_ZEROFILL)
occupy no space in the file, so their `offset` field is meaningless; older
toolchains (gfortran/ld64 combinations) left a nonzero value there, which
older dyld ignored and newer dyld hard-rejects. The fix is to zero the
offset field in place — a byte-exact no-op for how the file is actually
loaded — and re-sign ad hoc.

Usage:
    python3 tools/fix_macho_zerofill.py <dir-or-file> [...]

Scans *.so / *.dylib recursively, patches in place, ad-hoc re-signs patched
files (codesign -f -s -), and prints a summary. Idempotent.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
SECTION_TYPE_MASK = 0x000000FF
ZEROFILL_TYPES = {0x1, 0xC, 0x12}  # S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL


def patch_macho(path: Path) -> int:
    """Zero the offset field of zero-fill sections. Returns #sections fixed."""
    try:
        data = bytearray(path.read_bytes())
    except OSError:
        return 0
    if len(data) < 32 or struct.unpack_from("<I", data, 0)[0] != MH_MAGIC_64:
        return 0  # not a thin 64-bit little-endian Mach-O (fat/32-bit: skip)
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    fixed = 0
    for _ in range(ncmds):
        if off + 8 > len(data):
            break
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmdsize < 8 or off + cmdsize > len(data):
            break
        if cmd == LC_SEGMENT_64 and cmdsize >= 72:
            nsects = struct.unpack_from("<I", data, off + 64)[0]
            sect_off = off + 72
            for _ in range(nsects):
                if sect_off + 80 > off + cmdsize:
                    break
                flags = struct.unpack_from("<I", data, sect_off + 64)[0]
                file_offset = struct.unpack_from("<I", data, sect_off + 48)[0]
                if (flags & SECTION_TYPE_MASK) in ZEROFILL_TYPES and file_offset != 0:
                    struct.pack_into("<I", data, sect_off + 48, 0)
                    fixed += 1
                sect_off += 80
        off += cmdsize
    if fixed:
        path.write_bytes(data)
    return fixed


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    targets: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            targets.extend(p.rglob("*.so"))
            targets.extend(p.rglob("*.dylib"))
        elif p.is_file():
            targets.append(p)
    patched_files = []
    for f in targets:
        n = patch_macho(f)
        if n:
            patched_files.append((f, n))
            # In-place mutation invalidates the signature; ad-hoc re-sign.
            subprocess.run(
                ["codesign", "-f", "-s", "-", str(f)],
                check=False,
                capture_output=True,
            )
    for f, n in patched_files:
        print(f"patched {n} zero-fill section(s): {f}")
    print(f"scanned {len(targets)} file(s), patched {len(patched_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

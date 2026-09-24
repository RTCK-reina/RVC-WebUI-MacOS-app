#!/usr/bin/env bash
# Run tests/test_optimization_equivalence.py against the REAL bundled torch.
#
# tests/conftest.py unconditionally stubs torch (and friends) so the normal
# suite can run without the ML environment — numerical equivalence cannot be
# proven against mocks, so the file self-skips under tests/. This script
# copies it to a temp dir outside the stubbed conftest and runs it with the
# bundled env (build/python_env).
#
# Usage:
#   tools/run_equivalence_tests.sh [extra pytest args...]
#   RVC_PYENV=/path/to/env/bin/python tools/run_equivalence_tests.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${RVC_PYENV:-${ROOT}/build/python_env/bin/python}"

if [[ ! -x "${PY}" ]]; then
    echo "error: real python env not found: ${PY}" >&2
    echo "run build_app.sh first, or pass RVC_PYENV=/path/to/python" >&2
    exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cp "${ROOT}/tests/test_optimization_equivalence.py" "${tmp}/"
exec "${PY}" -m pytest -q "${tmp}/test_optimization_equivalence.py" "$@"

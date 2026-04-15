#!/bin/bash

set -fa

# Check if Python is installed
if ! command -v python; then
  echo "Python not found. Please install Python using your package manager or via PyEnv."
  exit 1
fi

requirements_file="requirements/main.txt"
venv_path=".venv"
stamp_file="${venv_path}/.requirements.sha256"

# Compute a portable sha256 of the requirements file (shasum on macOS, sha256sum on Linux).
if command -v shasum >/dev/null 2>&1; then
  current_hash="$(shasum -a 256 "${requirements_file}" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  current_hash="$(sha256sum "${requirements_file}" | awk '{print $1}')"
else
  current_hash=""
fi

needs_install=0
if [[ ! -d "${venv_path}" ]]; then
  echo "Creating venv..."
  python -m venv "${venv_path}"
  needs_install=1
elif [[ -n "${current_hash}" ]] && [[ ! -f "${stamp_file}" || "$(cat "${stamp_file}" 2>/dev/null)" != "${current_hash}" ]]; then
  echo "requirements/main.txt changed; reinstalling dependencies..."
  needs_install=1
fi

# shellcheck disable=SC1091
source "${venv_path}/bin/activate"

if [[ "${needs_install}" -eq 1 ]]; then
  pip install --upgrade -r "${requirements_file}"
  if [[ -n "${current_hash}" ]]; then
    echo "${current_hash}" > "${stamp_file}"
  fi
fi

echo "Activating venv..."

# Run the main script
python web.py --pycmd python

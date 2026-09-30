#!/usr/bin/env bash
# Run from any directory; use only this checkout's environment.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

if [[ "$(uname -s)" != Linux ]]; then
    echo "This helper targets Linux. See README.md for manual setup." >&2
    exit 1
fi
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Install complete CPython 3.11 with venv support first; see README.md." >&2
    exit 1
fi
python3.11 -c 'import sys, lzma, ssl, sqlite3, bz2, ctypes, venv; sys.exit(0 if sys.version_info[:2] == (3, 11) else "python3.11 must resolve to CPython 3.11")'

if [[ ! -e .venv && ! -L .venv ]]; then
    echo "Creating $project_root/.venv with python3.11"
    python3.11 -m venv .venv
fi
if [[ ! -x .venv/bin/python || ! -f .venv/pyvenv.cfg ]]; then
    echo "Existing .venv is incomplete. Move it aside and rerun; it was not deleted." >&2
    exit 1
fi
.venv/bin/python -c '
import pathlib, sys
expected = pathlib.Path.cwd() / ".venv"
if sys.version_info[:2] != (3, 11):
    sys.exit("Existing .venv must use Python 3.11; it was not modified.")
if sys.prefix == sys.base_prefix or pathlib.Path(sys.prefix).resolve() != expected.resolve():
    sys.exit("Existing .venv must belong to this checkout; it was not modified.")
'

echo "Installing requirements into $project_root/.venv (failures stop setup)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python verify_environment.py
echo "Setup complete. Select $project_root/.venv/bin/python in PyCharm."
echo "To activate in your shell: source \"$project_root/.venv/bin/activate\""

#!/usr/bin/env bash
# setup_env.sh - create a local Python venv and install requirements
# Usage (from the repo root, in Git Bash):
#   ./setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment in $VENV ..."
    py -3 -m venv "$VENV" 2>/dev/null || python -m venv "$VENV"
else
    echo "Virtual environment $VENV already exists, reusing it."
fi

PYTHON="$VENV/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON="$VENV/bin/python"

echo "Upgrading pip ..."
"$PYTHON" -m pip install --upgrade pip

echo "Installing packages from requirements.txt ..."
"$PYTHON" -m pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:"
echo "    source $VENV/Scripts/activate   # Git Bash on Windows"

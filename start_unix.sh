#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  echo "Create .venv and install Python requirements first. See README.md." >&2
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then npm --prefix frontend install; fi
export PYTHONPATH="$PWD/backend${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000 &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT INT TERM
npm --prefix frontend run dev

#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

echo "Starting Superconductivity ML API on 0.0.0.0:${PORT}"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT}"

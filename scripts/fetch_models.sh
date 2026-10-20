#!/usr/bin/env bash
# Download and verify the current release using model_manifest.json.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python scripts/fetch_models.py "$@"
fi
exec python3 scripts/fetch_models.py "$@"

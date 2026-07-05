#!/bin/bash
# Container entrypoint:
#   1. Bootstrap the model file into /data on first run.
#   2. Hand off to the wyoming server.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"

echo "[entrypoint] Starting wyoming-silero-tts"
echo "[entrypoint] DATA_DIR=$DATA_DIR"

# 1. Make sure the model file exists (SHA-256 verified download)
python3 -m scripts.bootstrap --data-dir "$DATA_DIR"

# 2. Run the wyoming server (foreground, replace shell so signals propagate)
exec python3 -m wyoming_silero_tts "$@"

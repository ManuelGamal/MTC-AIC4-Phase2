#!/usr/bin/env bash
set -euo pipefail

if [ "$(uname -m)" != "aarch64" ]; then
    echo "This direct setup is intended for Jetson/aarch64." >&2
    exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg

python3 - <<'PY'
try:
    import torch
    print("Found torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as exc:
    raise SystemExit(
        "PyTorch is not installed in this Python. Install the NVIDIA JetPack-compatible "
        "PyTorch package or use the Docker path. Do not install generic pip torch."
    ) from exc
PY

python3 -m pip install --upgrade pip
python3 -m pip install --only-binary=:all: -r requirements.txt
python3 tools/verify_jetson_requirements.py

echo "Direct Jetson Python environment is ready."

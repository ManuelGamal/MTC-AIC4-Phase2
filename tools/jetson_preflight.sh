#!/usr/bin/env bash
set -euo pipefail

echo "== Jetson preflight =="
echo "arch: $(uname -m)"
if [ -f /etc/nv_tegra_release ]; then
    echo "l4t: $(head -n 1 /etc/nv_tegra_release)"
else
    echo "l4t: /etc/nv_tegra_release not found"
fi

echo "disk:"
df -h . || true

if command -v docker >/dev/null 2>&1; then
    echo "docker: $(docker --version)"
    if docker info 2>/dev/null | grep -qi nvidia; then
        echo "nvidia docker runtime: detected"
    else
        echo "nvidia docker runtime: not detected in docker info"
    fi
else
    echo "docker: missing"
fi

if command -v python3 >/dev/null 2>&1; then
    echo "python3: $(python3 --version)"
fi

python3 - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch: import failed:", exc)
PY

if [ -f checkpoints/model_final.pth ] || [ -f checkpoints/model.pth ]; then
    echo "checkpoint: found"
else
    echo "checkpoint: missing; run python3 download.py"
fi

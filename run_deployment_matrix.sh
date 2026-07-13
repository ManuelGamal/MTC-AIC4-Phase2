#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON="${INPUT_JSON:-test.json}"
SPLIT="${SPLIT:-public_lb}"
OUTPUT_CSV="${OUTPUT_CSV:-predictions.csv}"

export INPUT_JSON SPLIT OUTPUT_CSV

bash tools/jetson_preflight.sh || true

echo "== Path 1: Docker image build/run =="
if command -v docker >/dev/null 2>&1; then
    if bash run_jetson.sh; then
        echo "Docker path passed."
        exit 0
    fi
    echo "Docker path failed; trying next fallback." >&2
else
    echo "Docker not found; trying next fallback." >&2
fi

echo "== Path 2: jetson-containers =="
if command -v jetson-containers >/dev/null 2>&1; then
    if bash run_jetson_containers.sh; then
        echo "jetson-containers path passed."
        exit 0
    fi
    echo "jetson-containers path failed; trying direct Python." >&2
else
    echo "jetson-containers not found; trying direct Python." >&2
fi

echo "== Path 3: Direct Jetson Python =="
if bash run_direct_jetson.sh; then
    echo "Direct Python path passed."
    exit 0
fi

echo "All deployment paths failed." >&2
exit 1

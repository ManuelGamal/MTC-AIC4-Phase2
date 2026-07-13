#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON="${INPUT_JSON:-test.json}"
SPLIT="${SPLIT:-public_lb}"
OUTPUT_CSV="${OUTPUT_CSV:-predictions.csv}"

if ! command -v jetson-containers >/dev/null 2>&1; then
    echo "jetson-containers is not installed." >&2
    echo "Install it with:" >&2
    echo "  git clone https://github.com/dusty-nv/jetson-containers" >&2
    echo "  bash jetson-containers/install.sh" >&2
    exit 1
fi

if [ ! -f "checkpoints/model_final.pth" ] && [ ! -f "checkpoints/model.pth" ]; then
    echo "Checkpoint missing. Run: python download.py" >&2
    exit 1
fi

if IMAGE_TAG="$(autotag pytorch 2>/dev/null)"; then
    :
elif IMAGE_TAG="$(autotag l4t-pytorch 2>/dev/null)"; then
    :
else
    IMAGE_TAG="nvcr.io/nvidia/pytorch:25.06-py3-igpu"
fi

jetson-containers run "${IMAGE_TAG}" \
    --volume "$(pwd):/opt/newbiesquad" \
    --workdir /opt/newbiesquad \
    --env ORIN_FP16="${ORIN_FP16:-1}" \
    --env ORIN_TF32="${ORIN_TF32:-1}" \
    --env ORIN_CUDA_AUTOTUNE="${ORIN_CUDA_AUTOTUNE:-1}" \
    --env ORIN_CHANNELS_LAST="${ORIN_CHANNELS_LAST:-1}" \
    --env ORIN_CUDA_WARMUP="${ORIN_CUDA_WARMUP:-1}" \
    --env ORIN_COMPILE="${ORIN_COMPILE:-0}" \
    --env ORIN_TENSORRT="${ORIN_TENSORRT:-0}" \
    --env ORIN_TENSORRT_AUTOBUILD="${ORIN_TENSORRT_AUTOBUILD:-0}" \
    --env ORIN_TENSORRT_ENGINE="${ORIN_TENSORRT_ENGINE:-/opt/newbiesquad/checkpoints/uetrack_fp16.engine}" \
    --env ORIN_TENSORRT_REQUIRED="${ORIN_TENSORRT_REQUIRED:-0}" \
    bash -lc "python -m pip install --only-binary=:all: -r requirements.txt && python tools/verify_jetson_requirements.py && python inference.py '${INPUT_JSON}' '${SPLIT}' '${OUTPUT_CSV}' && python tools/validate_predictions_against_input.py '${INPUT_JSON}' '${SPLIT}' '${OUTPUT_CSV}' --allow-zero-boxes"

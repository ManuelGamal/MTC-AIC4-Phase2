#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-newbiesquad_orin:jp72}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.06-py3-igpu}"
INPUT_JSON="${INPUT_JSON:-test.json}"
SPLIT="${SPLIT:-public_lb}"
OUTPUT_CSV="${OUTPUT_CSV:-predictions.csv}"

if [ ! -f "inference.py" ] || [ ! -f "predictor.py" ]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

if [ ! -f "checkpoints/model_final.pth" ] && [ ! -f "checkpoints/model.pth" ]; then
    echo "Checkpoint missing. Run: python download.py" >&2
    exit 1
fi

docker build \
    -f Dockerfile.jetson \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE_NAME}" .

docker run --rm --runtime nvidia --network none \
    -e ORIN_FP16="${ORIN_FP16:-1}" \
    -e ORIN_TF32="${ORIN_TF32:-1}" \
    -e ORIN_CUDA_AUTOTUNE="${ORIN_CUDA_AUTOTUNE:-1}" \
    -e ORIN_CHANNELS_LAST="${ORIN_CHANNELS_LAST:-1}" \
    -e ORIN_CUDA_WARMUP="${ORIN_CUDA_WARMUP:-1}" \
    -e ORIN_COMPILE="${ORIN_COMPILE:-0}" \
    -e ORIN_TENSORRT="${ORIN_TENSORRT:-0}" \
    -e ORIN_TENSORRT_AUTOBUILD="${ORIN_TENSORRT_AUTOBUILD:-0}" \
    -e ORIN_TENSORRT_ENGINE="${ORIN_TENSORRT_ENGINE:-/opt/newbiesquad/checkpoints/uetrack_fp16.engine}" \
    -e ORIN_TENSORRT_REQUIRED="${ORIN_TENSORRT_REQUIRED:-0}" \
    -v "$(pwd):/opt/newbiesquad" \
    -w /opt/newbiesquad \
    "${IMAGE_NAME}" \
    python inference.py "${INPUT_JSON}" "${SPLIT}" "${OUTPUT_CSV}"

docker run --rm --network none \
    -v "$(pwd):/opt/newbiesquad" \
    -w /opt/newbiesquad \
    "${IMAGE_NAME}" \
    python tools/validate_predictions_against_input.py "${INPUT_JSON}" "${SPLIT}" "${OUTPUT_CSV}" --allow-zero-boxes

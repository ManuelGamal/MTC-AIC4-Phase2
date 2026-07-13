#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-newbiesquad_orin:jp72}"
IMAGE_TAR="${IMAGE_TAR:-newbiesquad_orin_jp72.tar}"
INPUT_JSON="${INPUT_JSON:-test.json}"
SPLIT="${SPLIT:-public_lb}"
OUTPUT_CSV="${OUTPUT_CSV:-predictions.csv}"
SAMPLE_CSV="${SAMPLE_CSV:-sample_submission.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    if [ ! -f "${IMAGE_TAR}" ]; then
        echo "Missing ${IMAGE_TAR}; cannot load ${IMAGE_NAME}." >&2
        exit 1
    fi
    docker load -i "${IMAGE_TAR}"
fi

mkdir -p output

docker run --rm --runtime nvidia --network none \
    -e ORIN_FP16="${ORIN_FP16:-1}" \
    -e ORIN_TF32="${ORIN_TF32:-1}" \
    -e ORIN_CUDA_AUTOTUNE="${ORIN_CUDA_AUTOTUNE:-1}" \
    -e ORIN_CHANNELS_LAST="${ORIN_CHANNELS_LAST:-1}" \
    -e ORIN_CUDA_WARMUP="${ORIN_CUDA_WARMUP:-1}" \
    -e ORIN_COMPILE="${ORIN_COMPILE:-0}" \
    -v "${SCRIPT_DIR}:/offline" \
    -w /offline \
    "${IMAGE_NAME}" \
    python /opt/newbiesquad/inference.py "${INPUT_JSON}" "${SPLIT}" "output/${OUTPUT_CSV}"

docker run --rm --network none \
    -v "${SCRIPT_DIR}:/offline" \
    -w /offline \
    "${IMAGE_NAME}" \
    python /opt/newbiesquad/tools/validate_predictions_against_input.py "${INPUT_JSON}" "${SPLIT}" "output/${OUTPUT_CSV}" --allow-zero-boxes

if [ -f "${SAMPLE_CSV}" ]; then
    docker run --rm --network none \
        -v "${SCRIPT_DIR}:/offline" \
        -w /offline \
        "${IMAGE_NAME}" \
        python /opt/newbiesquad/check_submission.py "${SAMPLE_CSV}" "output/${OUTPUT_CSV}"
fi

echo "Saved output/${OUTPUT_CSV}"

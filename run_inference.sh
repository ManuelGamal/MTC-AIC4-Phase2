#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON="${1:-test.json}"
SPLIT="${2:-public_lb}"
OUTPUT_CSV="${3:-predictions.csv}"

ORIN_TENSORRT="${ORIN_TENSORRT:-1}"
ORIN_TENSORRT_AUTOBUILD="${ORIN_TENSORRT_AUTOBUILD:-1}"
ORIN_TENSORRT_REQUIRED="${ORIN_TENSORRT_REQUIRED:-0}"
ORIN_TENSORRT_ENGINE="${ORIN_TENSORRT_ENGINE:-/workspace/checkpoints/uetrack_fp16.engine}"
ORIN_TENSORRT_ONNX="${ORIN_TENSORRT_ONNX:-/workspace/checkpoints/uetrack_trt.onnx}"
ORIN_TENSORRT_WORKSPACE_MB="${ORIN_TENSORRT_WORKSPACE_MB:-2048}"

cuda_available() {
    python3 - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

if [ "${ORIN_TENSORRT}" = "1" ] && [ "${ORIN_TENSORRT_AUTOBUILD}" = "1" ] && [ ! -f "${ORIN_TENSORRT_ENGINE}" ]; then
    if cuda_available; then
        echo "TensorRT engine not found. Attempting FP16 engine build: ${ORIN_TENSORRT_ENGINE}"
        if ORIN_TENSORRT=0 ORIN_COMPILE=0 python3 /workspace/tools/export_to_tensorrt.py \
            --mode fp16 \
            --onnx "${ORIN_TENSORRT_ONNX}" \
            --fp16-engine "${ORIN_TENSORRT_ENGINE}" \
            --workspace-mb "${ORIN_TENSORRT_WORKSPACE_MB}"; then
            echo "TensorRT FP16 engine ready: ${ORIN_TENSORRT_ENGINE}"
        else
            if [ "${ORIN_TENSORRT_REQUIRED}" = "1" ]; then
                echo "TensorRT engine build failed and ORIN_TENSORRT_REQUIRED=1." >&2
                exit 1
            fi
            echo "TensorRT engine build failed. Falling back to PyTorch FP16." >&2
        fi
    else
        if [ "${ORIN_TENSORRT_REQUIRED}" = "1" ]; then
            echo "CUDA is unavailable and ORIN_TENSORRT_REQUIRED=1." >&2
            exit 1
        fi
        echo "CUDA is unavailable. Falling back to PyTorch." >&2
    fi
fi

exec python3 /workspace/inference.py "${INPUT_JSON}" "${SPLIT}" "${OUTPUT_CSV}"

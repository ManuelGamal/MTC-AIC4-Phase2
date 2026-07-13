#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON="${INPUT_JSON:-test.json}"
SPLIT="${SPLIT:-public_lb}"
OUTPUT_CSV="${OUTPUT_CSV:-predictions.csv}"

if [ ! -f "inference.py" ] || [ ! -f "predictor.py" ]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

if [ ! -f "checkpoints/model_final.pth" ] && [ ! -f "checkpoints/model.pth" ]; then
    echo "Checkpoint missing. Run: python3 download.py" >&2
    exit 1
fi

python3 tools/verify_jetson_requirements.py

python3 - <<'PY'
import sys
import torch

print("direct_python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in direct Python mode.")
PY

ORIN_FP16="${ORIN_FP16:-1}" \
ORIN_TF32="${ORIN_TF32:-1}" \
ORIN_CUDA_AUTOTUNE="${ORIN_CUDA_AUTOTUNE:-1}" \
ORIN_CHANNELS_LAST="${ORIN_CHANNELS_LAST:-1}" \
ORIN_CUDA_WARMUP="${ORIN_CUDA_WARMUP:-1}" \
ORIN_COMPILE="${ORIN_COMPILE:-0}" \
ORIN_TENSORRT="${ORIN_TENSORRT:-0}" \
ORIN_TENSORRT_ENGINE="${ORIN_TENSORRT_ENGINE:-checkpoints/uetrack_fp16.engine}" \
ORIN_TENSORRT_REQUIRED="${ORIN_TENSORRT_REQUIRED:-0}" \
python3 inference.py "${INPUT_JSON}" "${SPLIT}" "${OUTPUT_CSV}"

python3 tools/validate_predictions_against_input.py "${INPUT_JSON}" "${SPLIT}" "${OUTPUT_CSV}" --allow-zero-boxes

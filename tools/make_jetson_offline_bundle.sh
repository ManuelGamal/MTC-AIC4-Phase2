#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-newbiesquad_orin:jp72}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.06-py3-igpu}"
BUNDLE_DIR="${BUNDLE_DIR:-jetson_offline_bundle}"
IMAGE_TAR="${IMAGE_TAR:-newbiesquad_orin_jp72.tar}"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "This must run on a Jetson/aarch64 host so the saved image matches the target board." >&2
    exit 1
fi

if [ ! -f "Dockerfile.jetson" ] || [ ! -f "inference.py" ] || [ ! -f "predictor.py" ]; then
    echo "Run this script from the repository root." >&2
    exit 1
fi

if [ ! -f "checkpoints/model_final.pth" ] && [ ! -f "checkpoints/model.pth" ]; then
    echo "Checkpoint missing. Run: python download.py" >&2
    exit 1
fi

mkdir -p "${BUNDLE_DIR}"

docker build \
    -f Dockerfile.jetson \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE_NAME}" .

docker save "${IMAGE_NAME}" -o "${BUNDLE_DIR}/${IMAGE_TAR}"

cp run_offline.sh "${BUNDLE_DIR}/run_offline.sh"
chmod +x "${BUNDLE_DIR}/run_offline.sh"

cp test.json "${BUNDLE_DIR}/test.json" 2>/dev/null || true
cp sample_submission.csv "${BUNDLE_DIR}/sample_submission.csv" 2>/dev/null || true

cat > "${BUNDLE_DIR}/README_OFFLINE.txt" <<EOF
Jetson offline bundle
=====================

Copy this whole folder to the USB drive.

On the target Jetson:
  cd /path/to/${BUNDLE_DIR}
  bash run_offline.sh

For competition input:
  1. Put the test JSON and video folders in this bundle folder.
  2. Run:
     INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_offline.sh

The Docker image already contains the repository, checkpoint, PyTorch/CUDA stack,
and Python dependencies installed during docker build.
EOF

echo "Created ${BUNDLE_DIR}/"
echo "Copy ${BUNDLE_DIR}/ to the flash drive."

# Phase 3 Docker Submission

This repository contains the complete Docker build context for the Phase 3
submission. The submission Dockerfile is located at the repository root:

```text
Dockerfile
```

The Dockerfile copies the full repository into the image, installs the required
CUDA 13 / PyTorch / TensorRT stack, downloads the model checkpoint during build
if it is not already present, and runs inference through the standard
`inference.py` entry point.

## Target Platform

```text
Platform: linux/amd64
Base image: nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04
PyTorch: 2.9.1 CUDA 13.0 wheel
TensorRT ARG: TRT_VER=10.16.1.11-1+cuda13.2
ONNX: 1.19.1
```

`Dockerfile.jetson` is retained only as an ARM64 Jetson fallback artifact and is
not the Phase 3 submission Dockerfile.

## Build Command

```bash
docker buildx build --platform linux/amd64 -f Dockerfile -t newbiesquad_phase3:latest --load .
```

The Docker build runs:

```bash
python3 download.py
python3 tools/verify_phase3_requirements.py
```

This verifies the dependency imports and UETrack import path before the image is
accepted.

## Default Runtime Behavior

The default container command is:

```dockerfile
CMD ["bash", "run_inference.sh", "test.json", "public_lb", "predictions.csv"]
```

`run_inference.sh` attempts to use TensorRT FP16 automatically:

```text
ORIN_TENSORRT=1
ORIN_TENSORRT_AUTOBUILD=1
ORIN_TENSORRT_REQUIRED=0
ORIN_TENSORRT_ENGINE=/workspace/checkpoints/uetrack_fp16.engine
ORIN_TENSORRT_ONNX=/workspace/checkpoints/uetrack_trt.onnx
```

Runtime order:

```text
1. If the TensorRT FP16 engine exists, load and use it.
2. If the engine is missing and CUDA is available, build it with tools/export_to_tensorrt.py --mode fp16.
3. If TensorRT build or load fails, continue with the PyTorch FP16 path.
4. If ORIN_TENSORRT_REQUIRED=1, TensorRT failure is treated as fatal.
```

This keeps TensorRT as the preferred performance path while preserving a
reliable PyTorch fallback.

## Inference Command

The evaluator may run the default container command, or explicitly run:

```bash
docker run --rm --gpus all newbiesquad_phase3:latest \
  bash run_inference.sh test.json public_lb predictions.csv
```

For evaluator-provided manifests, the same argument contract is used:

```bash
bash run_inference.sh <input_json> <split_name> <output_csv>
```

The CSV output format is:

```text
id,x,y,w,h
```

## Verification Commands

Import verification:

```bash
docker run --rm --gpus all newbiesquad_phase3:latest \
  python3 -c "import torch, torchvision, torchaudio, tensorrt, onnx, cv2; print(torch.__version__); print(torch.cuda.is_available()); print(tensorrt.__version__); print(onnx.__version__); print(cv2.__version__)"
```

Unit tests:

```bash
docker run --rm newbiesquad_phase3:latest \
  python3 -m unittest discover -s tests -p "test*.py" -v
```

Generated-video smoke test:

```bash
docker run --rm \
  -v "${PWD}/tools/phase3_smoke_inference.py:/tmp/phase3_smoke_inference.py:ro" \
  newbiesquad_phase3:latest \
  python3 /tmp/phase3_smoke_inference.py
```

Expected smoke-test result:

```text
Sequences: 1
Predictions: 4
Saved: /tmp/phase3_smoke/predictions.csv
```

## Full Build Context Required

The Dockerfile uses:

```dockerfile
COPY . .
```

Therefore the complete repository must be supplied as the Docker build context,
not only the Dockerfile text. Required build-context files include:

```text
Dockerfile
requirements.phase3.txt
inference.py
predictor.py
download.py
check_submission.py
run_inference.sh
UETrack/
tools/
tests/
```

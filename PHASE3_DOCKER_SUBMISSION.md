# Phase 3 Docker Submission Notes

The competition email changes the required target from the Jetson ARM64 offline
image to an amd64 CUDA 13 submission image.

Use this file for the submission form:

```text
Dockerfile
```

Do not submit `Dockerfile.jetson` for Phase 3. That file is only for the older Jetson offline bundle path.

## Required Base

```text
nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04
```

## Required PyTorch

```bash
python3 -m pip install --break-system-packages \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

## Required TensorRT Arg

```dockerfile
ARG TRT_VER=10.16.1.11-1+cuda13.2
```

## Build Command

On an amd64 PC:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile -t newbiesquad_phase3:latest --load .
```

On an arm64 machine, use QEMU/buildx for amd64:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile -t newbiesquad_phase3:latest --load .
```

## Smoke Test

```bash
docker run --rm --gpus all newbiesquad_phase3:latest python3 -c "import torch, tensorrt, onnx; print(torch.__version__); print(torch.cuda.is_available()); print(tensorrt.__version__); print(onnx.__version__)"
```

Expected:

```text
2.9.1+cu130
True
10.16...
1.19.1
```

On a machine without an NVIDIA driver, run the same command without `--gpus all`.
`torch.cuda.is_available()` will print `False`, but the imports should still pass.

Then run the end-to-end generated-video smoke test:

```bash
docker run --rm -v "${PWD}/tools/phase3_smoke_inference.py:/tmp/phase3_smoke_inference.py:ro" newbiesquad_phase3:latest python3 /tmp/phase3_smoke_inference.py
```

On Windows PowerShell:

```powershell
docker run --rm -v "${PWD}\tools\phase3_smoke_inference.py:/tmp/phase3_smoke_inference.py:ro" newbiesquad_phase3:latest python3 /tmp/phase3_smoke_inference.py
```

Expected:

```text
Sequences: 1
Predictions: 4
Saved: /tmp/phase3_smoke/predictions.csv
```

## Real Contest Release Test

The local contest release used for testing was:

```text
C:\Users\manue\OneDrive\Documents\AIC\contest_release
```

Run one official `public_lb` sequence first:

```powershell
mkdir predictions -Force

docker run --rm -w /workspace/data `
  -v "C:\Users\manue\OneDrive\Documents\AIC\contest_release:/workspace/data:ro" `
  -v "${PWD}\tools\public_lb_car_video.json:/tmp/public_lb_car_video.json:ro" `
  -v "${PWD}\predictions:/outputs" `
  newbiesquad_phase3:latest `
  python3 /workspace/inference.py /tmp/public_lb_car_video.json public_lb /outputs/car_video_predictions.csv
```

Validate that one-sequence CSV:

```powershell
docker run --rm -w /workspace/data `
  -v "C:\Users\manue\OneDrive\Documents\AIC\contest_release:/workspace/data:ro" `
  -v "${PWD}\tools\public_lb_car_video.json:/tmp/public_lb_car_video.json:ro" `
  -v "${PWD}\predictions:/outputs:ro" `
  newbiesquad_phase3:latest `
  python3 /workspace/tools/validate_predictions_against_input.py /tmp/public_lb_car_video.json public_lb /outputs/car_video_predictions.csv
```

Expected:

```text
PREDICTIONS MATCH INPUT JSON
Split: public_lb
Rows: 585
```

Run the full official `public_lb` split:

```powershell
docker run --rm --gpus all -w /workspace/data `
  -v "C:\Users\manue\OneDrive\Documents\AIC\contest_release:/workspace/data:ro" `
  -v "${PWD}\predictions:/outputs" `
  newbiesquad_phase3:latest `
  python3 /workspace/inference.py metadata/contestant_manifest.json public_lb /outputs/predictions.csv
```

Validate the full submission CSV against the official sample submission:

```powershell
docker run --rm `
  -v "C:\Users\manue\OneDrive\Documents\AIC\contest_release:/workspace/data:ro" `
  -v "${PWD}\predictions:/outputs:ro" `
  newbiesquad_phase3:latest `
  python3 /workspace/check_submission.py /workspace/data/metadata/sample_submission.csv /outputs/predictions.csv
```

Expected full `public_lb` size:

```text
74293 prediction rows
```

## Default TensorRT FP16 Fast Path

The default container command now tries TensorRT first. If CUDA, ONNX export,
or engine build/load fails, it falls back to PyTorch FP16 unless strict mode is
enabled:

```text
ORIN_TENSORRT=1
ORIN_TENSORRT_AUTOBUILD=1
ORIN_TENSORRT_ENGINE=/workspace/checkpoints/uetrack_fp16.engine
ORIN_TENSORRT_REQUIRED=0
```

Default `docker run newbiesquad_phase3:latest` behavior:

```text
1. If /workspace/checkpoints/uetrack_fp16.engine exists, load it.
2. If it is missing and CUDA is available, try to build it with tools/export_to_tensorrt.py --mode fp16.
3. If TensorRT cannot be used, continue with the PyTorch FP16 tracker.
```

Build the FP16 TensorRT engine on the same GPU class that will run inference:

```powershell
docker run --rm --gpus all -w /workspace/data `
  -v "C:\Users\manue\OneDrive\Documents\AIC\contest_release:/workspace/data:ro" `
  -v "${PWD}\checkpoints:/workspace/checkpoints" `
  newbiesquad_phase3:latest `
  python3 /workspace/tools/export_to_tensorrt.py `
    --mode fp16 `
    --onnx /workspace/checkpoints/uetrack_trt.onnx `
    --fp16-engine /workspace/checkpoints/uetrack_fp16.engine `
    --workspace-mb 2048 `
    --check-onnx
```

Run inference with TensorRT enabled:

```powershell
docker run --rm --gpus all -w /workspace/data `
  -v "C:\Users\manue\OneDrive\Documents\AIC\contest_release:/workspace/data:ro" `
  -v "${PWD}\checkpoints:/workspace/checkpoints:ro" `
  -v "${PWD}\predictions:/outputs" `
  -e ORIN_TENSORRT=1 `
  -e ORIN_TENSORRT_ENGINE=/workspace/checkpoints/uetrack_fp16.engine `
  newbiesquad_phase3:latest `
  python3 /workspace/inference.py metadata/contestant_manifest.json public_lb /outputs/predictions_trt_fp16.csv
```

For benchmarking, force failure if TensorRT is not actually used:

```powershell
-e ORIN_TENSORRT_REQUIRED=1
```

If `ORIN_TENSORRT_REQUIRED=0`, any missing or invalid engine falls back to the
PyTorch path automatically.

## What Changed Versus Jetson Offline Docker

- Target platform is `linux/amd64`, not `linux/arm64`.
- Base image is CUDA runtime, not NVIDIA PyTorch iGPU.
- PyTorch is installed with the CUDA 13.0 wheel index.
- TensorRT packages are pinned through `TRT_VER`.
- The full repository code is copied into the image.
- `download.py` runs during the Docker build so a clean GitHub checkout can fetch `checkpoints/model_final.pth`.

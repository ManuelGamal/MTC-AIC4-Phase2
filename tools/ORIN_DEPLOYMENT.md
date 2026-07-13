# Jetson Orin Nano 8 GB Deployment Guide

Target board: Jetson Orin Nano 8 GB running JetPack 7.2 / L4T 39.2.

This guide has two supported paths:

1. Build this repository's `Dockerfile.jetson` on the Jetson.
2. Run from a `jetson-containers` PyTorch image without building a custom image.

The Docker path is the reproducible submission path. The `jetson-containers` path is a fast bring-up/debug path.

## Important Corrections to the Original Plan

- Do not depend on `dustynv/pytorch:2.4-r39.2.0` unless you have verified that exact tag exists on the target board. `Dockerfile.jetson` uses NVIDIA NGC `nvcr.io/nvidia/pytorch:25.06-py3-igpu` and lets you override it with `BASE_IMAGE=...`.
- Do not install PyTorch from `requirements.txt`. PyTorch, CUDA, cuDNN, and TensorRT must come from the Jetson-compatible base image.
- `ORIN_COMPILE=1` is experimental. `torch.compile(mode="max-autotune")` can consume too much memory and time on an 8 GB Jetson, and this tracker calls `forward_encoder` / `forward_decoder` directly, so compiling the module wrapper is not a guaranteed speedup.
- `ORIN_FP16=1` is the default acceleration path. `ORIN_FP16=0` now disables autocast so TF32/FP32 tests are real.
- TensorRT export is a second-stage optimization. It must be validated against accuracy and ONNX export support before it is treated as the production path.

## One-Command Docker Deployment

Run on the Jetson from the repository root:

```bash
python download.py
bash run_jetson.sh
```

This builds `newbiesquad_orin:jp72`, runs inference with GPU access, writes `predictions.csv`, then validates it with `check_submission.py`.

Override inputs when needed:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_jetson.sh
```

Override the base image only if NVIDIA publishes a newer JetPack 7.2-compatible PyTorch iGPU tag:

```bash
BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3-igpu bash run_jetson.sh
```

## Alternative: jetson-containers

Install once:

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

Then run:

```bash
python download.py
bash run_jetson_containers.sh
```

The script tries `autotag pytorch`, then `autotag l4t-pytorch`, then falls back to the same NGC PyTorch iGPU image used by the Dockerfile.

## Full Fallback Matrix

Use this when you want the repo to try every supported execution path:

```bash
python download.py
bash run_deployment_matrix.sh
```

It runs:

1. `tools/jetson_preflight.sh`
2. Docker image build/run
3. `jetson-containers`
4. Direct Jetson Python

For the direct Python fallback, prepare the host environment with:

```bash
bash tools/setup_direct_jetson.sh
bash run_direct_jetson.sh
```

This path requires JetPack-compatible PyTorch to already be installed. It intentionally does not install generic PyPI `torch`.

## Flash Drive / Offline Bundle

Use this when the target Jetson has no internet. Build the bundle on an internet-connected Jetson Orin Nano with the same JetPack/L4T:

```bash
python download.py
bash tools/make_jetson_offline_bundle.sh
```

The generated folder is the transfer artifact:

```text
jetson_offline_bundle/
  newbiesquad_orin_jp72.tar
  run_offline.sh
  test.json
  sample_submission.csv
  README_OFFLINE.txt
```

Copy that folder to the flash drive. On the target Jetson:

```bash
cd /path/to/jetson_offline_bundle
bash run_offline.sh
```

For competition input, put the provided JSON and video folders beside `run_offline.sh`, then run:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_offline.sh
```

## Board Setup Checklist

Before benchmarking:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
docker info | grep -i runtime
python - <<'PY'
import platform
print(platform.machine())
PY
```

Expected architecture is `aarch64`. If Docker cannot see the NVIDIA runtime, reinstall or repair NVIDIA Container Toolkit before debugging the model.

## Runtime Flags

Default production flags:

```bash
ORIN_FP16=1
ORIN_TF32=1
ORIN_CUDA_AUTOTUNE=1
ORIN_CHANNELS_LAST=1
ORIN_CUDA_WARMUP=1
ORIN_COMPILE=0
```

Debug flags:

```bash
ORIN_FP16=0 ORIN_TF32=1 bash run_jetson.sh
ORIN_COMPILE=1 bash run_jetson.sh
```

Only keep `ORIN_COMPILE=1` if it improves measured median latency and does not cause out-of-memory failures.

## Verification Gates

Run these on the Jetson/container:

```bash
python -m unittest tests.test_jetson_optimizations tests.test_env_simulator tests.test_phase2_predictor
python inference.py test.json public_lb predictions.csv
python tools/validate_predictions_against_input.py test.json public_lb predictions.csv --allow-zero-boxes
python tools/latency_benchmark.py test.json public_lb --require-gpu
```

Passing unit tests on a desktop is useful, but it does not prove CUDA, FP16, video decoding, or memory behavior on Orin. The final gate is the board-side latency benchmark plus a valid submission CSV.

## TensorRT Follow-Up

Try TensorRT only after the PyTorch FP16 Docker path works. Start with the export-readiness probe:

```bash
python tools/export_to_tensorrt.py --output /tmp/uetrack.onnx --device cuda
```

That script does not currently write an ONNX file. A production TensorRT path needs a tensor-only wrapper around the UETrack encoder/decoder, then accuracy and latency checks against PyTorch FP16 output.

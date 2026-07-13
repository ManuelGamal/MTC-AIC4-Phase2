# Reproducible Jetson Orin Nano Deployment

Target: Jetson Orin Nano 8 GB, JetPack 7.2, L4T 39.2.

NVIDIA lists JetPack 7.2 as L4T 39.2 with Jetson Orin Family support. The deployment below is built around that pairing and avoids pip-installing PyTorch on Jetson.

## Critique of the Original Plan

- Good: adding a deployment guide, unit tests, and a CPU pipeline simulation is the right shape.
- Risky: `dustynv/pytorch:2.4-r39.2.0` was assumed, not verified. The Jetson-specific `Dockerfile.jetson` now defaults to NVIDIA NGC `nvcr.io/nvidia/pytorch:25.06-py3-igpu` and supports `BASE_IMAGE=...` override.
- Risky: `ORIN_COMPILE=1` as a default is not safe on an 8 GB Orin. It can spend minutes compiling, increase memory pressure, and may not speed this tracker because the hot path calls `forward_encoder` and `forward_decoder` directly.
- Fixed: `ORIN_FP16=0` previously still used CUDA autocast. FP16 autocast is now gated by `ORIN_FP16`.
- Missing: the previous TensorRT INT8 instructions referenced a calibration generator that was not in the repo. TensorRT is now documented as a follow-up, not the default production path.
- Overclaimed: desktop tests cannot prove Orin latency. The final reproducibility gate must run on the Jetson with CUDA visible.

## Primary Path: Jetson Dockerfile

Run this on the Jetson from the repo root:

```bash
python download.py
bash run_jetson.sh
```

The script builds:

```bash
newbiesquad_orin:jp72
```

Then it runs:

```bash
python inference.py test.json public_lb predictions.csv
python check_submission.py sample_submission.csv predictions.csv
```

For a competition package that provides `data/test.json` and a `hidden` split, run:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_jetson.sh
```

Default runtime flags:

```bash
ORIN_FP16=1
ORIN_TF32=1
ORIN_CUDA_AUTOTUNE=1
ORIN_CHANNELS_LAST=1
ORIN_CUDA_WARMUP=1
ORIN_COMPILE=0
```

## Full Fallback Matrix

When you want the most robust “try everything” command, run:

```bash
python download.py
bash run_deployment_matrix.sh
```

It tries these paths in order:

1. Docker image build/run with `Dockerfile`.
2. `jetson-containers` with an auto-tagged PyTorch image.
3. Direct Jetson Python, if PyTorch/CUDA is already installed on the host.

Before running, you can check the board:

```bash
bash tools/jetson_preflight.sh
```

For the no-Docker direct Python fallback:

```bash
bash tools/setup_direct_jetson.sh
bash run_direct_jetson.sh
```

The direct setup intentionally refuses to install generic `pip torch`; PyTorch must come from NVIDIA/JetPack or the Jetson Docker image.

## Alternative Path: jetson-containers

Run this if you want faster iteration without building the repo image:

```bash
python download.py
bash run_jetson_containers.sh
```

The script tries `autotag pytorch`, falls back to `autotag l4t-pytorch`, then falls back to `nvcr.io/nvidia/pytorch:25.06-py3-igpu`.

## Flash Drive / Offline Path

Use this when the target Jetson will not have internet. The final artifact is a saved Docker image containing the CUDA/PyTorch base image, apt packages, Python packages, repository code, and checkpoint.

On an internet-connected Jetson Orin Nano running the same JetPack/L4T:

```bash
python download.py
bash tools/make_jetson_offline_bundle.sh
```

Copy the generated folder to the flash drive:

```text
jetson_offline_bundle/
  newbiesquad_orin_jp72.tar
  run_offline.sh
  test.json
  sample_submission.csv
  README_OFFLINE.txt
```

On the offline target Jetson:

```bash
cd /path/to/jetson_offline_bundle
bash run_offline.sh
```

For competition input, put the provided JSON and video folders beside `run_offline.sh`, then run:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_offline.sh
```

## Required Board Validation

Run these inside the Jetson environment:

```bash
python -m unittest tests.test_jetson_optimizations tests.test_env_simulator tests.test_phase2_predictor
python tools/latency_benchmark.py test.json public_lb --require-gpu --profile
python tools/validate_predictions_against_input.py test.json public_lb predictions.csv --allow-zero-boxes
```

Ship only after:

- CUDA is available in the container.
- The unit tests pass in the Jetson container.
- `predictions.csv` passes `check_submission.py`.
- Median tracked-frame latency is acceptable on the real board.

## Optional TensorRT Follow-Up

Only after PyTorch FP16 is working, run the probe:

```bash
python tools/export_to_tensorrt.py --output /tmp/uetrack.onnx --device cuda
```

That script does not currently write an ONNX file. A production TensorRT path needs a tensor-only wrapper around the UETrack encoder/decoder, then accuracy and latency checks against PyTorch FP16 output.

# Orin Nano 8 GB Deployment Guide

This guide walks through optimizing the UETrack model for <30 ms latency on Jetson Orin Nano 8 GB.

## Overview

The model is trained on specific crop sizes (112×112 templates, 224×224 search) and must stay on that distribution. Optimization paths:

1. **GPU FP16 (recommended first step)** – Fast, good accuracy
2. **TensorRT FP16** – Engine-level optimization
3. **TensorRT INT8** – Maximum compression, requires calibration
4. **Dynamic quantization** – CPU fallback (opt-in via code)

## Quick Start on Orin Nano

### Prerequisites

```bash
# Install required packages
pip install tensorrt onnx onnx-graphsurgeon
```

### Step 1: Run with GPU FP16 (Default)

The model runs on GPU by default. On Orin Nano with FP16:

```bash
python inference.py data/test.json hidden predictions.csv
```

Expected per-frame latency: 15–25 ms (batch size 1, two templates).

### Step 2: Benchmark Current Performance

Use the benchmark script to measure actual latency on your target device:

```bash
python tools/benchmark_inference.py \
    /path/to/video.mp4 \
    /path/to/annotation.txt \
    --device cuda \
    --fp16
```

This prints median, mean, p90, p99 latencies in milliseconds.

### Step 3: If Still Above 30 ms, Try TensorRT FP16

On Orin Nano, export the model and build a TensorRT engine:

```bash
# Step 3a: Export to ONNX (run once on Orin)
cd /path/to/NewbieSquad-MTC-AIC4-main
python tools/export_to_tensorrt.py --output /tmp/uetrack.onnx --device cuda

# Step 3b: Build FP16 engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=/tmp/uetrack.onnx \
    --saveEngine=/tmp/uetrack_fp16.engine \
    --fp16 \
    --workspace=2048

# Step 3c: Benchmark the engine
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=/tmp/uetrack_fp16.engine \
    --iterations=1000 \
    --avgRuns=100
```

TensorRT engines typically achieve 1.5–3x speedup over PyTorch on Orin.

### Step 4: If Still Above 30 ms, Try TensorRT INT8

INT8 provides aggressive quantization but requires a small calibration dataset. 

**Create a calibration dataset:**

```bash
# Generate 200 representative frames from your contest videos
python tools/generate_calibration_data.py \
    --input-dir /path/to/contest_release \
    --output-dir /tmp/calib_data \
    --num-frames 200
```

**Build INT8 engine with calibration:**

```bash
# Manual calibration (uses entropy calibration)
/usr/src/tensorrt/bin/trtexec \
    --onnx=/tmp/uetrack.onnx \
    --int8 \
    --saveEngine=/tmp/uetrack_int8.engine \
    --workspace=2048 \
    --calib=/tmp/calib_data
```

INT8 engines are typically 3–5x smaller than FP32 with ~1–5% accuracy loss.

## Environment Variables

### ORIN_QUANTIZE_DYNAMIC

Enable CPU-side dynamic quantization of Linear layers (PyTorch only, no TensorRT):

```bash
export ORIN_QUANTIZE_DYNAMIC=1
python inference.py data/test.json hidden predictions.csv
```

This is a fallback; TensorRT FP16 is preferred.

## Benchmarking Best Practices

1. **Warm-up first:** Run 20 iterations before measuring to avoid JIT/cuDNN setup overhead.
2. **Use avgRuns and iterations:** `trtexec --iterations=1000 --avgRuns=100` gives stable numbers.
3. **Disable frequency scaling:** 
   ```bash
   sudo jetson_clocks
   ```
4. **Monitor memory:** 
   ```bash
   tegrastats
   ```

## If Latency is Still Too High

1. **Check batch size:** Ensure batch size is 1.
2. **Profile the hot path:** Use `nsys` to find bottlenecks.
3. **Consider reducing precision further:**
   - TensorRT INT8 with larger workspace
   - Quantize-aware training (out of scope here)
4. **Check GPU clock:** Orin Nano runs at 1020 MHz; the RTX 4060 is 10–40x faster.

## Expected Performance on Orin Nano 8 GB

| Configuration | Latency (ms) | Notes |
|---|---|---|
| PyTorch FP32 | 80–150 | CPU fallback, very slow |
| PyTorch FP16 | 15–30 | GPU FP16, typical winner |
| TensorRT FP16 | 10–20 | Engine optimization |
| TensorRT INT8 | 5–15 | Aggressive quantization |

Your target is <30 ms, so **PyTorch FP16 (default) should work**.

## Files in this Directory

- `benchmark_inference.py` – Per-frame latency measurement
- `export_to_tensorrt.py` – ONNX export helper
- `torchscript_convert.py` – TorchScript export (reference, limited use)
- `generate_calibration_data.py` – Calibration dataset builder (to be added)

## Troubleshooting

**"trtexec not found"**
- trtexec comes with TensorRT on Jetson. Path is typically `/usr/src/tensorrt/bin/trtexec`.
- Alternatively, use Python TensorRT API if preferred.

**"Out of memory during engine build"**
- Reduce workspace: `--workspace=1024` (default 2048)
- Split the model into smaller subgraphs if needed.

**"ONNX export failed with custom ops"**
- The UETrack encoder/decoder may use custom PyTorch operations not supported by ONNX.
- Fallback: Use Torch-TensorRT or run full PyTorch on Orin GPU.

## Contact & Next Steps

1. Run `tools/benchmark_inference.py` on Orin to measure current latency.
2. Report the median latency to the team.
3. If <30 ms, you're done.
4. If >30 ms, try TensorRT FP16 next.
5. If TensorRT FP16 is still >30 ms, open a calibration issue or try INT8.

Good luck!

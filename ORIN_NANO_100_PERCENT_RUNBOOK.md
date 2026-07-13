# Jetson Orin Nano Exact Runbook

Target board: Jetson Orin Nano 8 GB, JetPack 7.2 / L4T 39.2.

This is the literal transfer-and-run checklist. Follow it in order. The safest path is **Path A** if the Jetson has internet. Use **Path B** if the final Jetson is offline and you need to move everything with a flash drive.

## Success Criteria

The run is successful only if the Jetson prints:

```text
PREDICTIONS MATCH INPUT JSON
```

and one of:

```text
Docker path passed.
jetson-containers path passed.
Direct Python path passed.
```

For the evaluator-style dataset, the final command is:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_deployment_matrix.sh
```

The output file will be:

```text
predictions.csv
```

## Verified Status From This PC

These checks were run before this runbook was finalized:

```text
Docker ARM64 image build: PASS
Image tag: newbiesquad_orin:jp72
Image architecture: arm64/linux
Docker image size shown by `docker images`: 18.8 GB
Base image: nvcr.io/nvidia/pytorch:25.06-py3-igpu
Dependency verifier inside ARM64 image: PASS
Full ARM64 unit suite inside Docker: 38 tests OK, 1 expected skip
Local Windows unit suite: 35 tests OK, 10 expected skips
Offline runner bash syntax inside ARM64 image: PASS
Windows offline bundle script PowerShell syntax: PASS
```

Important limit: this PC does not have the Jetson GPU/runtime attached, so CUDA latency and TensorRT engine execution still must be checked on the Orin. The image itself and Python dependencies were verified as ARM64.

## USB Format

Use one of these formats for the flash drive:

```text
exFAT
NTFS
ext4
```

Do **not** use FAT32. Docker image files can be larger than 4 GB.

## Path A: Jetson Has Internet

Use this path if the Jetson can download Docker images, Python wheels, and the checkpoint.

### A1. Copy the Repo to the Jetson

Option 1: clone directly on the Jetson:

```bash
cd ~
git clone <your-repo-url>
cd NewbieSquad-MTC-AIC4-main
```

Option 2: copy from Windows to USB:

1. On Windows, open:

```text
C:\Users\manue\Downloads\NewbieSquad-MTC-AIC4-main (1)\
```

2. Copy the whole folder:

```text
NewbieSquad-MTC-AIC4-main
```

3. Paste it onto the flash drive root. The USB should look like:

```text
USB_DRIVE/
  NewbieSquad-MTC-AIC4-main/
    Dockerfile
    Dockerfile.jetson
    inference.py
    predictor.py
    requirements.txt
    tools/
    UETrack/
```

4. Eject the flash drive safely from Windows.

5. Plug the flash drive into the Jetson.

6. On the Jetson, find the USB:

```bash
lsblk
ls /media/$USER
```

7. Copy the repo from USB to the Jetson home folder:

```bash
cp -r /media/$USER/<USB_NAME>/NewbieSquad-MTC-AIC4-main ~/
cd ~/NewbieSquad-MTC-AIC4-main
```

Replace `<USB_NAME>` with the actual USB name shown by `ls /media/$USER`.

### A2. Prepare Jetson Basics

Run on the Jetson:

```bash
sudo apt update
sudo apt install -y git docker.io python3-pip
sudo usermod -aG docker $USER
```

Log out and log back in, then return to the repo:

```bash
cd ~/NewbieSquad-MTC-AIC4-main
```

Check Jetson/Docker:

```bash
uname -m
cat /etc/nv_tegra_release
docker --version
docker info
```

Expected architecture:

```text
aarch64
```

### A3. Run Preflight

```bash
bash tools/jetson_preflight.sh
```

If it says the checkpoint is missing, that is okay. The next step downloads it.

### A4. Download Model Checkpoint

```bash
python download.py
ls -lh checkpoints/
```

Expected one of:

```text
model_final.pth
model.pth
```

### A5. Public Repo Test

Run:

```bash
bash run_deployment_matrix.sh
```

This automatically tries:

1. Docker build/run.
2. `jetson-containers`.
3. Direct Jetson Python.

The test passes if you see:

```text
PREDICTIONS MATCH INPUT JSON
```

Important: Path A builds and validates the safe **PyTorch FP16** runtime. It does **not** build TensorRT automatically. TensorRT is a separate step after the PyTorch path passes.

### A6. Hidden/Test Dataset Run

Your evaluator data should be copied into the repo like this:

```text
NewbieSquad-MTC-AIC4-main/
  data/
    test.json
    <video folders...>
```

If the data is on USB, copy it onto the Jetson:

```bash
cd ~/NewbieSquad-MTC-AIC4-main
mkdir -p data
cp -r /media/$USER/<USB_NAME>/data/* data/
```

Then run:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_deployment_matrix.sh
```

If the JSON split is not called `hidden`, inspect it:

```bash
python - <<'PY'
import json
print(json.load(open("data/test.json")).keys())
PY
```

Then replace `hidden`:

```bash
INPUT_JSON=data/test.json SPLIT=<split_name> OUTPUT_CSV=predictions.csv bash run_deployment_matrix.sh
```

### A6.1. Single-Sequence Cows Smoke Test

Use this when you want to test one known sequence before running the full split.

Copy cows data into:

```text
NewbieSquad-MTC-AIC4-main/
  data/
    dataset1/
      cows/
        cows.mp4
        annotation.txt
```

Create the smoke JSON from the bundled template:

```bash
cp tools/cows_single_sequence_jetson.json data/cows_single_sequence.json
```

Run only the cows sequence:

```bash
INPUT_JSON=data/cows_single_sequence.json SPLIT=test OUTPUT_CSV=predictions_cows.csv bash run_deployment_matrix.sh
```

Expected validation output:

```text
PREDICTIONS MATCH INPUT JSON
Split: test
Rows: 489
```

### A6.2. Build TensorRT After Path A Passes

Do this only after `run_deployment_matrix.sh` passes with PyTorch FP16. TensorRT is optional acceleration; PyTorch FP16 remains the fallback.

Build TensorRT FP16:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode fp16 \
  --input-json data/test.json \
  --split hidden \
  --onnx checkpoints/uetrack_trt.onnx \
  --fp16-engine checkpoints/uetrack_fp16.engine \
  --workspace-mb 2048
```

Then build TensorRT INT8:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode int8 \
  --input-json data/test.json \
  --split hidden \
  --onnx checkpoints/uetrack_trt.onnx \
  --int8-engine checkpoints/uetrack_int8.engine \
  --calib-npz checkpoints/uetrack_int8_calib.npz \
  --calib-cache checkpoints/uetrack_int8_calib.cache \
  --calib-samples 128 \
  --workspace-mb 2048
```

Confirm files exist:

```bash
ls -lh checkpoints/uetrack_trt.onnx
ls -lh checkpoints/uetrack_fp16.engine
ls -lh checkpoints/uetrack_int8.engine
```

### A7. Copy Output Back to USB

After success:

```bash
mkdir -p /media/$USER/<USB_NAME>/orin_results
cp predictions.csv /media/$USER/<USB_NAME>/orin_results/
sync
```

Then eject/unmount from the Jetson file manager or run:

```bash
sync
```

Now the USB contains:

```text
USB_DRIVE/
  orin_results/
    predictions.csv
```

## Path B: Final Jetson Is Offline

Use this path when the final Jetson cannot download anything. You can build the offline Docker image in either of these places:

```text
Option B1: an internet-connected Jetson with the same JetPack.
Option B2: this Windows PC with Docker Desktop cross-building linux/arm64.
```

The saved Docker image must be ARM64. The verified image tag from this PC is:

```text
newbiesquad_orin:jp72
```

Final inspected image metadata from this PC:

```text
architecture: arm64
os: linux
docker image size shown by `docker images`: 18.8 GB
base image: nvcr.io/nvidia/pytorch:25.06-py3-igpu
```

### B1. Option 1: Build Bundle on Internet-Connected Jetson

```bash
cd ~
git clone <your-repo-url>
cd NewbieSquad-MTC-AIC4-main
```

or copy the repo from USB as described in Path A.

Download the checkpoint:

```bash
python download.py
ls -lh checkpoints/
```

Build the offline bundle:

```bash
bash tools/make_jetson_offline_bundle.sh
```

Expected output folder:

```text
jetson_offline_bundle/
  newbiesquad_orin_jp72.tar
  run_offline.sh
  test.json
  sample_submission.csv
  README_OFFLINE.txt
```

### B2. Option 2: Build ARM64 Image on This Windows PC

Use this if no Jetson has internet but this Windows PC does.

Open PowerShell in:

```text
C:\Users\manue\Downloads\NewbieSquad-MTC-AIC4-main (1)\NewbieSquad-MTC-AIC4-main
```

Build the ARM64 image:

```powershell
docker buildx build --platform linux/arm64 -f Dockerfile.jetson --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3-igpu --build-arg SKIP_BUILD_VERIFY=1 -t newbiesquad_orin:jp72 --load .
```

Verify the image metadata:

```powershell
docker image inspect newbiesquad_orin:jp72 --format "{{.Id}} {{.Architecture}} {{.Os}} {{.Size}}"
```

Expected shape:

```text
sha256:<image-id> arm64 linux <byte-count>
```

For storage planning, use `docker images newbiesquad_orin`; this build showed `18.8GB`.

Recommended: create the offline bundle with the committed PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File tools\create_windows_offline_bundle.ps1
```

Manual equivalent: save the image tar to a flash drive or local folder with at least 32 GB free. Use a 64 GB USB drive if possible:

```powershell
mkdir jetson_offline_bundle
docker save newbiesquad_orin:jp72 -o jetson_offline_bundle\newbiesquad_orin_jp72.tar
```

Copy these into `jetson_offline_bundle`:

```text
run_offline.sh
data/test.json
sample_submission.csv
checkpoints/model_final.pth
```

`run_offline.sh` is now committed in the repository root.

### B3. Copy Offline Bundle to Flash Drive

If you built the bundle on Windows, copy this folder with File Explorer:

```text
C:\Users\manue\Downloads\NewbieSquad-MTC-AIC4-main (1)\NewbieSquad-MTC-AIC4-main\jetson_offline_bundle
```

Paste it onto the USB root. The USB should look like:

```text
USB_DRIVE/
  jetson_offline_bundle/
    newbiesquad_orin_jp72.tar
    run_offline.sh
    README_OFFLINE.txt
    data/
    checkpoints/
```

If you built the bundle on a Jetson, plug in the flash drive and find it:

```bash
ls /media/$USER
```

Copy the whole bundle folder:

```bash
cp -r jetson_offline_bundle /media/$USER/<USB_NAME>/
sync
```

The USB should now look like:

```text
USB_DRIVE/
  jetson_offline_bundle/
    newbiesquad_orin_jp72.tar
    run_offline.sh
    test.json
    sample_submission.csv
    README_OFFLINE.txt
```

If you already have hidden/test data, also copy it into the bundle on the USB:

```bash
mkdir -p /media/$USER/<USB_NAME>/jetson_offline_bundle/data
cp -r data/* /media/$USER/<USB_NAME>/jetson_offline_bundle/data/
sync
```

Final USB layout for hidden data:

```text
USB_DRIVE/
  jetson_offline_bundle/
    newbiesquad_orin_jp72.tar
    run_offline.sh
    data/
      test.json
      <video folders...>
```

### B5. On Offline Jetson: Copy Bundle from USB

Plug the USB into the offline Jetson.

Find the USB:

```bash
lsblk
ls /media/$USER
```

Copy the bundle to the Jetson home folder:

```bash
cp -r /media/$USER/<USB_NAME>/jetson_offline_bundle ~/
cd ~/jetson_offline_bundle
```

You can also run directly from the USB, but copying to the Jetson is safer and faster.

### B6. Load and Run Offline Docker Image

Public/default test:

```bash
bash run_offline.sh
```

Hidden/evaluator-style data:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_offline.sh
```

Success output must include:

```text
PREDICTIONS MATCH INPUT JSON
Saved output/predictions.csv
```

### B7. Copy Offline Output Back to USB

```bash
mkdir -p /media/$USER/<USB_NAME>/orin_results
cp output/predictions.csv /media/$USER/<USB_NAME>/orin_results/
sync
```

USB result:

```text
USB_DRIVE/
  orin_results/
    predictions.csv
```

## Path C: Direct Python Fallback

Use this only if Docker is broken but JetPack-compatible PyTorch is already installed on the Jetson.

From the repo folder:

```bash
bash tools/setup_direct_jetson.sh
python download.py
bash run_direct_jetson.sh
```

Hidden data:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_direct_jetson.sh
```

Do **not** run:

```bash
pip install torch
```

Generic PyPI PyTorch is usually wrong for Jetson. Use Docker or NVIDIA JetPack-compatible PyTorch.

## Manual Validation Commands

Validate against any input JSON:

```bash
python tools/validate_predictions_against_input.py data/test.json hidden predictions.csv --allow-zero-boxes
```

If you have a matching sample submission CSV:

```bash
python check_submission.py sample_submission.csv predictions.csv
```

Latency check:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
python tools/latency_benchmark.py data/test.json hidden --require-gpu --profile
```

## Optional TensorRT Build

The normal deployment path is PyTorch FP16. TensorRT is an optional acceleration build that must be created on the Jetson. Keep PyTorch FP16 as the fallback even if TensorRT builds successfully.

TensorRT artifacts are written here:

```text
checkpoints/uetrack_trt.onnx
checkpoints/uetrack_fp16.engine
checkpoints/uetrack_int8.engine
checkpoints/uetrack_int8_calib.npz
checkpoints/uetrack_int8_calib.cache
```

### TensorRT Prerequisites

Run from the repo folder on the Jetson:

```bash
cd ~/NewbieSquad-MTC-AIC4-main
bash tools/jetson_preflight.sh
python download.py
```

Confirm CUDA is available:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Expected:

```text
True
```

### Build ONNX Only

Use this first. It proves the model wrapper can export:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode export \
  --input-json data/test.json \
  --split hidden \
  --onnx checkpoints/uetrack_trt.onnx
```

If using the public repo JSON:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode export \
  --input-json test.json \
  --split public_lb \
  --onnx checkpoints/uetrack_trt.onnx
```

Expected file:

```bash
ls -lh checkpoints/uetrack_trt.onnx
```

### Build TensorRT FP16 Engine

Build FP16 first. It is safer than INT8:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode fp16 \
  --input-json data/test.json \
  --split hidden \
  --onnx checkpoints/uetrack_trt.onnx \
  --fp16-engine checkpoints/uetrack_fp16.engine \
  --workspace-mb 2048
```

Expected file:

```bash
ls -lh checkpoints/uetrack_fp16.engine
```

Benchmark the engine with TensorRT:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=checkpoints/uetrack_fp16.engine \
  --iterations=1000 \
  --avgRuns=100
```

Run the tracker through the TensorRT FP16 fast path:

```bash
ORIN_TENSORRT=1 \
ORIN_TENSORRT_ENGINE=checkpoints/uetrack_fp16.engine \
INPUT_JSON=data/test.json \
SPLIT=hidden \
OUTPUT_CSV=predictions_trt_fp16.csv \
bash run_deployment_matrix.sh
```

For proof that TensorRT is actually being used, add:

```bash
ORIN_TENSORRT_REQUIRED=1
```

If `ORIN_TENSORRT_REQUIRED=0` or unset, the tracker falls back to PyTorch FP16
when the engine is missing or invalid.

### Build TensorRT INT8 Engine

Only do this after FP16 builds. INT8 needs calibration frames from sequences like the test dataset:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode int8 \
  --input-json data/test.json \
  --split hidden \
  --onnx checkpoints/uetrack_trt.onnx \
  --int8-engine checkpoints/uetrack_int8.engine \
  --calib-npz checkpoints/uetrack_int8_calib.npz \
  --calib-cache checkpoints/uetrack_int8_calib.cache \
  --calib-samples 128 \
  --workspace-mb 2048
```

Expected files:

```bash
ls -lh checkpoints/uetrack_int8.engine
ls -lh checkpoints/uetrack_int8_calib.npz
ls -lh checkpoints/uetrack_int8_calib.cache
```

Benchmark INT8:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=checkpoints/uetrack_int8.engine \
  --iterations=1000 \
  --avgRuns=100
```

### Build Everything in One Command

This exports ONNX, builds FP16, then builds INT8:

```bash
ORIN_COMPILE=0 python tools/export_to_tensorrt.py \
  --mode all \
  --input-json data/test.json \
  --split hidden \
  --workspace-mb 2048 \
  --calib-samples 128
```

### TensorRT Failure Rule

If ONNX export, FP16 engine build, or INT8 calibration fails, do not block the deployment. Use the validated PyTorch FP16 path:

```bash
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_deployment_matrix.sh
```

TensorRT should only be used after comparing predictions and latency against the PyTorch FP16 output.

## Troubleshooting

### USB Name Unknown

```bash
lsblk
ls /media/$USER
```

### Permission Denied Running Script from USB

Use:

```bash
bash run_offline.sh
```

instead of:

```bash
./run_offline.sh
```

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
```

Then log out and log back in.

### Checkpoint Missing

```bash
python download.py
ls -lh checkpoints/
```

### CUDA Missing

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

If CUDA is false in direct Python, use Docker.

### Input JSON Paths Broken

Run from the repo or offline bundle folder:

```bash
python - <<'PY'
import json
from pathlib import Path

input_json = Path("data/test.json")
split = "hidden"
data = json.loads(input_json.read_text())
for name, info in data[split].items():
    p = Path(info["video_path"])
    if not p.is_absolute():
        p = input_json.parent / p
    print(name, p, p.exists())
PY
```

Any `False` path must be fixed before inference.

## Final Checklist

For online Jetson:

```bash
cd ~/NewbieSquad-MTC-AIC4-main
bash tools/jetson_preflight.sh
python download.py
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_deployment_matrix.sh
python tools/validate_predictions_against_input.py data/test.json hidden predictions.csv --allow-zero-boxes
```

For offline Jetson:

```bash
cd ~/jetson_offline_bundle
INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_offline.sh
```

Copy final output to USB:

```bash
mkdir -p /media/$USER/<USB_NAME>/orin_results
cp predictions.csv /media/$USER/<USB_NAME>/orin_results/ 2>/dev/null || cp output/predictions.csv /media/$USER/<USB_NAME>/orin_results/
sync
```

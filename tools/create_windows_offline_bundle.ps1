param(
    [string]$BundleDir = "jetson_offline_bundle",
    [string]$ImageName = "newbiesquad_orin:jp72",
    [string]$BaseImage = "nvcr.io/nvidia/pytorch:25.06-py3-igpu",
    [string]$Dockerfile = "Dockerfile.jetson",
    [string]$ImageTar = "newbiesquad_orin_jp72.tar",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dockerfile) -or -not (Test-Path "inference.py") -or -not (Test-Path "predictor.py")) {
    throw "Run this script from the repository root."
}

if (-not (Test-Path "checkpoints/model_final.pth") -and -not (Test-Path "checkpoints/model.pth")) {
    throw "Checkpoint missing. Run python download.py or copy model_final.pth into checkpoints/ first."
}

if (-not $SkipBuild) {
    docker buildx build `
        --platform linux/arm64 `
        -f $Dockerfile `
        --build-arg "BASE_IMAGE=$BaseImage" `
        --build-arg "SKIP_BUILD_VERIFY=1" `
        -t $ImageName `
        --load .
}

$inspect = docker image inspect $ImageName --format "{{.Architecture}} {{.Os}} {{.Size}}"
if ($inspect -notmatch "^arm64 linux ") {
    throw "Image $ImageName is not linux/arm64. Got: $inspect"
}

New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null
docker save $ImageName -o (Join-Path $BundleDir $ImageTar)

Copy-Item "run_offline.sh" (Join-Path $BundleDir "run_offline.sh") -Force

if (Test-Path "test.json") {
    Copy-Item "test.json" (Join-Path $BundleDir "test.json") -Force
}

if (Test-Path "sample_submission.csv") {
    Copy-Item "sample_submission.csv" (Join-Path $BundleDir "sample_submission.csv") -Force
}

if (Test-Path "data") {
    Copy-Item "data" (Join-Path $BundleDir "data") -Recurse -Force
}

New-Item -ItemType Directory -Force -Path (Join-Path $BundleDir "checkpoints") | Out-Null
if (Test-Path "checkpoints/model_final.pth") {
    Copy-Item "checkpoints/model_final.pth" (Join-Path $BundleDir "checkpoints/model_final.pth") -Force
} elseif (Test-Path "checkpoints/model.pth") {
    Copy-Item "checkpoints/model.pth" (Join-Path $BundleDir "checkpoints/model.pth") -Force
}

@"
Jetson offline bundle
=====================

Copy this whole folder to the USB drive.

On the target Jetson:
  cd /path/to/$BundleDir
  chmod +x run_offline.sh
  bash run_offline.sh

For competition input:
  INPUT_JSON=data/test.json SPLIT=hidden OUTPUT_CSV=predictions.csv bash run_offline.sh

Image metadata:
  $inspect
"@ | Set-Content -Encoding UTF8 (Join-Path $BundleDir "README_OFFLINE.txt")

Write-Host "Created $BundleDir"
Write-Host "Image metadata: $inspect"

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VER=10.16.1.11-1+cuda13.2
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY \
    OMP_NUM_THREADS=4 \
    ORIN_FP16=1 \
    ORIN_TF32=1 \
    ORIN_CUDA_AUTOTUNE=1 \
    ORIN_CHANNELS_LAST=1 \
    ORIN_CUDA_WARMUP=1 \
    ORIN_COMPILE=0 \
    ORIN_TENSORRT=1 \
    ORIN_TENSORRT_AUTOBUILD=1 \
    ORIN_TENSORRT_REQUIRED=0 \
    ORIN_TENSORRT_ENGINE=/workspace/checkpoints/uetrack_fp16.engine \
    ORIN_TENSORRT_ONNX=/workspace/checkpoints/uetrack_trt.onnx \
    ORIN_TENSORRT_WORKSPACE_MB=2048

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
        libnvinfer10=${TRT_VER} \
        libnvinfer-plugin10=${TRT_VER} \
        libnvonnxparsers10=${TRT_VER} \
        libnvinfer-bin=${TRT_VER} \
        python3-libnvinfer=${TRT_VER} \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchaudio==2.9.1 \
        --index-url ${PYTORCH_INDEX_URL}

WORKDIR /workspace

COPY requirements.phase3.txt .
RUN python3 -m pip install --break-system-packages --only-binary=:all: -r requirements.phase3.txt

COPY . .
RUN python3 download.py
RUN python3 tools/verify_phase3_requirements.py

CMD ["bash", "run_inference.sh", "test.json", "public_lb", "predictions.csv"]

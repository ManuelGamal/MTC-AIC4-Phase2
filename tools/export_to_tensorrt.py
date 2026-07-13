"""
Export UETrack to ONNX and build TensorRT engines on Jetson.

This script must be run on the Jetson, inside the Docker image or another
JetPack-compatible Python environment with CUDA, PyTorch, ONNX, and TensorRT.

It exports the hot model path as a tensor-only wrapper:
    template0, template1, search, anno0, anno1

and returns:
    score_map, size_map, offset_map

The Python tracker still performs crop generation, bbox mapping, template update,
and absence logic. TensorRT accelerates the encoder/decoder forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import predictor


class UETrackTensorRTWrapper(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(
        self,
        template0: torch.Tensor,
        template1: torch.Tensor,
        search: torch.Tensor,
        anno0: torch.Tensor,
        anno1: torch.Tensor,
    ):
        raw = self.network.forward_encoder(
            [template0, template1],
            [search],
            [anno0, anno1],
            text_src=None,
            task_index=0,
        )
        features = raw[0]
        out = self.network.forward_decoder(features)
        return out["score_map"], out["size_map"], out["offset_map"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/build UETrack TensorRT artifacts.")
    parser.add_argument("--input-json", default="test.json")
    parser.add_argument("--split", default="public_lb")
    parser.add_argument("--onnx", default="checkpoints/uetrack_trt.onnx")
    parser.add_argument("--fp16-engine", default="checkpoints/uetrack_fp16.engine")
    parser.add_argument("--int8-engine", default="checkpoints/uetrack_int8.engine")
    parser.add_argument("--calib-npz", default="checkpoints/uetrack_int8_calib.npz")
    parser.add_argument("--calib-cache", default="checkpoints/uetrack_int8_calib.cache")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["export", "fp16", "int8", "all"], default="all")
    parser.add_argument("--check-onnx", action="store_true", help="Run onnx.checker after export.")
    return parser.parse_args()


def tensor_dtype(use_fp16: bool) -> torch.dtype:
    return torch.float16 if use_fp16 else torch.float32


def load_wrapper(device: str, use_fp16: bool = True) -> tuple[UETrackTensorRTWrapper, predictor.RuntimeConfig]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT export/build on Jetson.")
    model = predictor.load_model(device=device)
    network = model.network.eval()
    if use_fp16 and model.device.type == "cuda":
        network = network.half()
    wrapper = UETrackTensorRTWrapper(network).to(model.device).eval()
    return wrapper, model.cfg


def dummy_inputs(cfg: predictor.RuntimeConfig, device: torch.device, dtype: torch.dtype):
    template_shape = (1, 3, cfg.template_size, cfg.template_size)
    search_shape = (1, 3, cfg.search_size, cfg.search_size)
    template0 = torch.randn(template_shape, device=device, dtype=dtype)
    template1 = torch.randn(template_shape, device=device, dtype=dtype)
    search = torch.randn(search_shape, device=device, dtype=dtype)
    anno0 = torch.tensor([[0.25, 0.25, 0.5, 0.5]], device=device, dtype=dtype)
    anno1 = torch.tensor([[0.25, 0.25, 0.5, 0.5]], device=device, dtype=dtype)
    return template0, template1, search, anno0, anno1


def export_onnx(output_path: Path, opset: int, check_onnx: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper, cfg = load_wrapper("cuda", use_fp16=True)
    device = next(wrapper.parameters()).device
    inputs = dummy_inputs(cfg, device, torch.float16)

    with torch.inference_mode():
        outputs = wrapper(*inputs)
    print("PyTorch wrapper outputs:", [tuple(out.shape) for out in outputs])

    torch.onnx.export(
        wrapper,
        inputs,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["template0", "template1", "search", "anno0", "anno1"],
        output_names=["score_map", "size_map", "offset_map"],
        dynamic_axes=None,
    )
    print(f"Wrote ONNX: {output_path}")

    if check_onnx:
        import onnx

        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        print("ONNX checker: ok")


def run_trtexec(args: list[str]) -> None:
    trtexec = Path("/usr/src/tensorrt/bin/trtexec")
    exe = str(trtexec) if trtexec.exists() else "trtexec"
    print("Running:", " ".join([exe, *args]))
    subprocess.run([exe, *args], check=True)


def build_fp16_engine(onnx_path: Path, engine_path: Path, workspace_mb: int) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    run_trtexec(
        [
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            f"--memPoolSize=workspace:{workspace_mb}",
            "--useSpinWait",
            "--separateProfileRun",
        ]
    )


def resolve_path(input_json: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return input_json.resolve().parent / path


def read_sequences(input_json: Path, split: str) -> Iterable[tuple[str, dict]]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if split not in data:
        raise ValueError(f"Split '{split}' not found in {input_json}.")
    return data[split].items()


def make_sample_from_sequence(
    input_json: Path,
    name: str,
    info: dict,
    cfg: predictor.RuntimeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    del name
    video_path = resolve_path(input_json, info["video_path"])
    anno_path = resolve_path(input_json, info["annotation_path"])
    init_box = predictor.read_init_box(anno_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        ok, frame_bgr = capture.read()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    template_patch, resize_factor = predictor.sample_target(
        frame_rgb,
        init_box,
        cfg.template_factor,
        output_sz=cfg.template_size,
    )
    search_patch, _search_resize = predictor.sample_target(
        frame_rgb,
        init_box,
        cfg.search_factor,
        output_sz=cfg.search_size,
    )
    anno = predictor.transform_image_to_crop(
        torch.tensor(init_box),
        torch.tensor(init_box),
        resize_factor,
        torch.tensor([cfg.template_size, cfg.template_size]),
        normalize=True,
    ).numpy()

    template = predictor.preprocess_image(template_patch).unsqueeze(0).numpy()
    search = predictor.preprocess_image(search_patch).unsqueeze(0).numpy()
    anno = anno.reshape(1, 4).astype(np.float32)
    return template, template.copy(), search, anno, anno.copy()


def write_calibration_npz(input_json: Path, split: str, output_path: Path, max_samples: int) -> None:
    _wrapper, cfg = load_wrapper("cuda", use_fp16=False)
    samples = []
    for name, info in read_sequences(input_json, split):
        sample = make_sample_from_sequence(input_json, name, info, cfg)
        if sample is not None:
            samples.append(sample)
        if len(samples) >= max_samples:
            break
    if not samples:
        raise RuntimeError("No calibration samples could be read from the input JSON.")

    arrays = {}
    input_names = ["template0", "template1", "search", "anno0", "anno1"]
    for input_idx, input_name in enumerate(input_names):
        arrays[input_name] = np.concatenate([sample[input_idx] for sample in samples], axis=0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    print(f"Wrote calibration tensors: {output_path}")
    for key, value in arrays.items():
        print(f"  {key}: {value.shape} {value.dtype}")


def make_entropy_calibrator(npz_path: Path, cache_path: Path, batch_size: int = 1):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    class UETrackEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.cache_path = cache_path
            self.batch_size = batch_size
            data = np.load(npz_path)
            self.names = ["template0", "template1", "search", "anno0", "anno1"]
            self.arrays = [np.ascontiguousarray(data[name].astype(np.float32)) for name in self.names]
            self.num_samples = min(array.shape[0] for array in self.arrays)
            self.index = 0
            self.device_allocs = [cuda.mem_alloc(array[0:batch_size].nbytes) for array in self.arrays]

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            del names
            if self.index + self.batch_size > self.num_samples:
                return None
            for array, alloc in zip(self.arrays, self.device_allocs):
                batch = np.ascontiguousarray(array[self.index : self.index + self.batch_size])
                cuda.memcpy_htod(alloc, batch)
            self.index += self.batch_size
            return [int(alloc) for alloc in self.device_allocs]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return UETrackEntropyCalibrator()


def build_int8_engine_python(onnx_path: Path, engine_path: Path, calib_npz: Path, calib_cache: Path, workspace_mb: int) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    else:
        config.max_workspace_size = workspace_mb * 1024 * 1024

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = make_entropy_calibrator(calib_npz, calib_cache)

    print("Building INT8 TensorRT engine. This can take several minutes on Orin.")
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized = engine.serialize() if engine is not None else None
    if serialized is None:
        raise RuntimeError("TensorRT INT8 engine build failed.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Wrote INT8 engine: {engine_path}")


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx)
    fp16_engine = Path(args.fp16_engine)
    int8_engine = Path(args.int8_engine)
    calib_npz = Path(args.calib_npz)
    calib_cache = Path(args.calib_cache)
    input_json = Path(args.input_json)

    if args.mode in {"export", "fp16", "int8", "all"}:
        export_onnx(onnx_path, args.opset, args.check_onnx)
    if args.mode in {"fp16", "all"}:
        build_fp16_engine(onnx_path, fp16_engine, args.workspace_mb)
    if args.mode in {"int8", "all"}:
        write_calibration_npz(input_json, args.split, calib_npz, args.calib_samples)
        build_int8_engine_python(onnx_path, int8_engine, calib_npz, calib_cache, args.workspace_mb)

    print("TensorRT artifact build complete.")
    return 0


if __name__ == "__main__":
    # Keep TensorRT export conservative by disabling torch.compile at import/load time.
    os.environ.setdefault("ORIN_COMPILE", "0")
    raise SystemExit(main())

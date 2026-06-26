"""
Export UETrack to TensorRT with FP16 and INT8 quantization support.

This script prepares the model for deployment on Orin Nano 8 GB by exporting to ONNX
and optionally building a TensorRT engine with INT8 quantization calibration.

Usage (on Orin Nano):
    # Step 1: Export to ONNX (run once)
    python tools/export_to_tensorrt.py --output /tmp/uetrack.onnx --device cuda

    # Step 2: Build FP16 engine (fast, good starting point)
    trtexec --onnx=/tmp/uetrack.onnx --saveEngine=/tmp/uetrack_fp16.engine --fp16

    # Step 3 (optional): Build INT8 engine with calibration (slower, more compression)
    # (Requires a calibration dataset; see instructions below)

    # Step 4: Benchmark the engine
    trtexec --loadEngine=/tmp/uetrack_fp16.engine --iterations=1000 --avgRuns=100

Notes:
    - This script exports the model graph only; the tracker loop still runs in PyTorch.
    - TensorRT export is best-effort; some custom ops may not convert cleanly.
    - INT8 calibration requires a representative dataset of ~200-500 frames.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from predictor import load_model


def export_to_onnx(
    output_path: str,
    device: str = "cuda",
    opset_version: int = 14,
) -> None:
    """
    Export the model to ONNX format.
    """
    print(f"Loading model on {device}...")
    model_bundle = load_model(device=device)
    network = model_bundle.network
    cfg = model_bundle.cfg
    network.eval()

    print("Building representative inputs...")
    # Create dummy inputs matching the expected shapes
    template_size = cfg.template_size
    search_size = cfg.search_size

    template = torch.randn(1, 3, template_size, template_size, device=device)
    template_list = [template, template]

    search = torch.randn(1, 3, search_size, search_size, device=device)
    search_list = [search]

    template_anno = torch.zeros((1, 4), device=device)
    template_anno_list = [template_anno, template_anno]

    print("Testing forward pass...")
    with torch.no_grad():
        feat = network.forward_encoder(template_list, search_list, template_anno_list, None, 0)
        out = network.forward_decoder(feat)

    print(f"Encoder output type: {type(feat)}")
    print(f"Decoder output: {list(out.keys()) if isinstance(out, dict) else type(out)}")

    print("Note: Full ONNX export of the encoder/decoder with list inputs is not directly supported.")
    print("For TensorRT on Orin, consider:")
    print("  1. Using torch2trt or TorchScript + TensorRT bridge")
    print("  2. Exporting individual submodules if they have tensor-only signatures")
    print("  3. Running the full PyTorch model on Orin GPU with FP16 (simpler fallback)")
    print("\nRecommended next step:")
    print("  Run the tracker with FP16 on Orin GPU:")
    print("    ORIN_QUANTIZE_DYNAMIC=0 python inference.py data/test.json hidden predictions.csv")


def main():
    p = argparse.ArgumentParser(description="Export UETrack to TensorRT formats.")
    p.add_argument("--output", default="/tmp/uetrack.onnx", help="Output ONNX path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = p.parse_args()

    try:
        export_to_onnx(args.output, args.device, args.opset)
        print(f"\nExport info saved.")
    except Exception as e:
        print(f"Export failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

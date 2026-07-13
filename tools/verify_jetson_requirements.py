"""
Fail-fast dependency check for the Jetson Docker image.

This is intentionally import-focused: it catches missing binary wheels, broken
OpenCV/numpy ABI combinations, missing PyTorch, and UETrack import regressions
during docker build instead of at runtime on the board.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UETRACK_ROOT = REPO_ROOT / "UETrack"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UETRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(UETRACK_ROOT))


REQUIRED_IMPORTS = [
    "cv2",
    "numpy",
    "onnx",
    "pandas",
    "yaml",
    "timm",
    "einops",
    "easydict",
    "gdown",
    "torch",
]


def main() -> int:
    loaded = {}
    for module_name in REQUIRED_IMPORTS:
        module = importlib.import_module(module_name)
        loaded[module_name] = module
        version = getattr(module, "__version__", "unknown")
        print(f"{module_name}: {version}")

    import predictor

    predictor._ensure_runtime_ready()
    importlib.import_module("lib.config.uetrack.config")
    importlib.import_module("lib.models.uetrack")
    print("UETrack imports: ok")
    print(f"python: {sys.version.split()[0]}")
    print(f"cuda_available_at_build: {loaded['torch'].cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

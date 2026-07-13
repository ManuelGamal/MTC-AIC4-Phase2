import ast
import importlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = REPO_ROOT / "predictor.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestRuntimeOptimizationSource(unittest.TestCase):
    def test_compile_is_opt_in(self):
        source = PREDICTOR_PATH.read_text(encoding="utf-8")
        self.assertIn('USE_COMPILE = os.getenv("ORIN_COMPILE", "0") == "1"', source)

    def test_autocast_is_gated_by_fp16_flag(self):
        source = PREDICTOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        funcs = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("_cuda_autocast_context", funcs)
        helper_source = ast.get_source_segment(source, funcs["_cuda_autocast_context"])
        self.assertIn("USE_FP16", helper_source)
        self.assertIn("nullcontext", helper_source)

    def test_tensorrt_runtime_is_optional_and_guarded(self):
        source = PREDICTOR_PATH.read_text(encoding="utf-8")
        self.assertIn('USE_TENSORRT = os.getenv("ORIN_TENSORRT", "1") == "1"', source)
        self.assertIn('TENSORRT_REQUIRED = os.getenv("ORIN_TENSORRT_REQUIRED", "0") == "1"', source)
        self.assertIn("class TensorRTRunner", source)
        self.assertIn("execute_async_v3", source)
        self.assertIn("Falling back to PyTorch", source)


@unittest.skipUnless(HAS_TORCH, "torch is required for runtime optimization tests")
class TestRuntimeOptimizationRuntime(unittest.TestCase):
    def setUp(self):
        self.original_environ = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_environ)

    def _reload_predictor(self):
        import predictor

        return importlib.reload(predictor)

    def test_fp16_conversion_in_image_tensor(self):
        import numpy as np
        import torch

        os.environ["ORIN_FP16"] = "1"
        os.environ["ORIN_CHANNELS_LAST"] = "0"
        predictor = self._reload_predictor()

        tensor = predictor.image_to_device_tensor(np.zeros((16, 16, 3), dtype=np.uint8), torch.device("cpu"))
        self.assertEqual("float32", str(tensor.dtype).replace("torch.", ""))

    def test_tf32_enabled_by_default(self):
        os.environ.pop("ORIN_TF32", None)
        predictor = self._reload_predictor()
        self.assertTrue(predictor.USE_TF32)

    def test_compile_disabled_by_default(self):
        os.environ.pop("ORIN_COMPILE", None)
        predictor = self._reload_predictor()
        self.assertFalse(predictor.USE_COMPILE)

    def test_tensorrt_forward_is_used_when_runner_is_present(self):
        import torch

        predictor = self._reload_predictor()

        class FakeRunner:
            def __init__(self):
                self.called = False

            def forward(self, templates, searches, annos):
                self.called = True
                return {"score_map": torch.ones(1), "size_map": torch.ones(1), "offset_map": torch.ones(1)}

        tracker = object.__new__(predictor.UETrackOnline)
        tracker.trt_runner = FakeRunner()
        tracker.network = None
        tracker.device = torch.device("cpu")
        tracker.template_list = [torch.zeros(1), torch.zeros(1)]
        tracker.template_anno_list = [torch.zeros(1), torch.zeros(1)]

        out = tracker._run_network_forward(torch.zeros(1))
        self.assertTrue(tracker.trt_runner.called)
        self.assertEqual({"score_map", "size_map", "offset_map"}, set(out))

    def test_tensorrt_failure_falls_back_to_pytorch(self):
        import torch

        os.environ["ORIN_TENSORRT_REQUIRED"] = "0"
        predictor = self._reload_predictor()

        class FailingRunner:
            def forward(self, *_args):
                raise RuntimeError("engine failed")

        class FakeNetwork:
            def forward_encoder(self, *_args, **_kwargs):
                return ["features"]

            def forward_decoder(self, features):
                return {"score_map": features}

        tracker = object.__new__(predictor.UETrackOnline)
        tracker.trt_runner = FailingRunner()
        tracker.network = FakeNetwork()
        tracker.device = torch.device("cpu")
        tracker.template_list = [torch.zeros(1), torch.zeros(1)]
        tracker.template_anno_list = [torch.zeros(1), torch.zeros(1)]

        out = tracker._run_network_forward(torch.zeros(1))
        self.assertIsNone(tracker.trt_runner)
        self.assertEqual({"score_map": "features"}, out)


if __name__ == "__main__":
    unittest.main()

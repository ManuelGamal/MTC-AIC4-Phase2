import importlib.util
import random
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_PATH = REPO_ROOT / "tools" / "orin_live_simulator.py"
HAS_RUNTIME_DEPS = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None
)

if HAS_RUNTIME_DEPS:
    import numpy as np
    spec = importlib.util.spec_from_file_location("orin_live_simulator", SIMULATOR_PATH)
    assert spec is not None and spec.loader is not None
    orin_live_simulator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orin_live_simulator)


@unittest.skipUnless(HAS_RUNTIME_DEPS, "OpenCV and NumPy are required for live simulator tests")
class TestOrinLiveSimulator(unittest.TestCase):
    def test_resolve_execution_plan_uses_contest_preset(self):
        class Args:
            preset = "contest"
            input_json = None
            split = None
            sequence = None

        input_json, split, sequence = orin_live_simulator.resolve_execution_plan(Args())
        self.assertEqual(REPO_ROOT / "data" / "contest_input.json", input_json)
        self.assertEqual("test", split)
        self.assertIsNone(sequence)

    def test_apply_camera_artifacts_preserves_frame_shape(self):
        frame = np.full((48, 64, 3), 120, dtype=np.uint8)

        class Args:
            simulate_camera = True
            motion_jitter = 1.0
            blur_kernel = 3
            brightness_jitter = 0.05
            jpeg_quality = 70

        transformed = orin_live_simulator.apply_camera_artifacts(frame, random.Random(7), Args())
        self.assertEqual(frame.shape, transformed.shape)
        self.assertEqual(frame.dtype, transformed.dtype)

    def test_summarize_latencies_reports_expected_quantiles(self):
        stats = orin_live_simulator.summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(5.0, stats["count"])
        self.assertAlmostEqual(30.0, stats["median"], places=6)
        self.assertAlmostEqual(50.0, stats["p99"], places=6)


if __name__ == "__main__":
    unittest.main()

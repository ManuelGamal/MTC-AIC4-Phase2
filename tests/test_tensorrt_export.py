import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRT_SCRIPT = REPO_ROOT / "tools" / "export_to_tensorrt.py"
PHASE3_DOC = REPO_ROOT / "PHASE3_DOCKER_SUBMISSION.md"
REQUIREMENTS = REPO_ROOT / "requirements.txt"


class TestTensorRTExportScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRT_SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_script_defines_tensor_only_wrapper(self):
        self.assertIn("class UETrackTensorRTWrapper", self.source)
        self.assertIn("template0", self.source)
        self.assertIn("template1", self.source)
        self.assertIn("search", self.source)
        self.assertIn("anno0", self.source)
        self.assertIn("anno1", self.source)
        self.assertIn("[template0, template1]", self.source)
        self.assertIn("[search]", self.source)
        self.assertIn("[anno0, anno1]", self.source)

    def test_onnx_export_has_fixed_io_contract(self):
        self.assertIn('input_names=["template0", "template1", "search", "anno0", "anno1"]', self.source)
        self.assertIn('output_names=["score_map", "size_map", "offset_map"]', self.source)
        self.assertIn("dynamic_axes=None", self.source)
        self.assertIn("opset_version=opset", self.source)

    def test_cli_modes_cover_fp16_and_int8(self):
        self.assertIn('choices=["export", "fp16", "int8", "all"]', self.source)
        self.assertIn('--fp16-engine", default="checkpoints/uetrack_fp16.engine"', self.source)
        self.assertIn('--int8-engine", default="checkpoints/uetrack_int8.engine"', self.source)
        self.assertIn('--calib-samples", type=int, default=128', self.source)

    def test_fp16_engine_uses_trtexec_fp16(self):
        self.assertIn("def build_fp16_engine", self.source)
        self.assertIn('"--fp16"', self.source)
        self.assertIn('"--separateProfileRun"', self.source)
        self.assertIn("run_trtexec", self.source)

    def test_int8_engine_uses_entropy_calibrator(self):
        self.assertIn("def make_entropy_calibrator", self.source)
        self.assertIn("trt.IInt8EntropyCalibrator2", self.source)
        self.assertIn("config.set_flag(trt.BuilderFlag.INT8)", self.source)
        self.assertIn("write_calibration_npz", self.source)

    def test_import_time_disables_torch_compile_for_export(self):
        self.assertIn('os.environ.setdefault("ORIN_COMPILE", "0")', self.source)


class TestTensorRTDocs(unittest.TestCase):
    def test_phase3_doc_contains_tensorrt_runtime_contract(self):
        text = PHASE3_DOC.read_text(encoding="utf-8")
        self.assertIn("TensorRT", text)
        self.assertIn("tools/export_to_tensorrt.py --mode fp16", text)
        self.assertIn("--mode fp16", text)
        self.assertIn("checkpoints/uetrack_fp16.engine", text)
        self.assertIn("PyTorch fallback", text)

    def test_requirements_include_onnx_for_export(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("onnx==", requirements)


if __name__ == "__main__":
    unittest.main()

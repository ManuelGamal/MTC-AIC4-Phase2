import ast
import csv
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_ZIP = REPO_ROOT / "mtc-aic4-phase2-main.zip"


class TestPhase2Scaffold(unittest.TestCase):
    def test_inference_matches_phase2_scaffold(self):
        if not PHASE2_ZIP.exists():
            self.skipTest("phase-2 scaffold zip is not present")
        with zipfile.ZipFile(PHASE2_ZIP) as archive:
            expected = archive.read("mtc-aic4-phase2-main/inference.py").decode("utf-8")
        actual = (REPO_ROOT / "inference.py").read_text(encoding="utf-8")
        self.assertEqual(expected.replace("\r\n", "\n"), actual.replace("\r\n", "\n"))

    def test_no_generated_uetrack_local_files_with_machine_paths(self):
        forbidden = [
            REPO_ROOT / "UETrack/lib/test/evaluation/local.py",
            REPO_ROOT / "UETrack/lib/train/admin/local.py",
        ]
        self.assertEqual([], [str(path.relative_to(REPO_ROOT)) for path in forbidden if path.exists()])

    def test_no_hardcoded_local_paths_in_submission_surface(self):
        scan_files = [
            REPO_ROOT / "predictor.py",
            REPO_ROOT / "download.py",
            REPO_ROOT / "requirements.txt",
        ]
        forbidden = ["C:/", "C:\\", "/workspace/", "git clone", "pip install git+"]
        for path in scan_files:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} found in {path.relative_to(REPO_ROOT)}")

    def test_local_or_heavy_artifacts_are_ignored(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        required_patterns = {"*.pth", "*.pt", "/data/", "predictions*.csv", "__pycache__/"}
        self.assertTrue(required_patterns <= set(gitignore), required_patterns - set(gitignore))


class TestPredictorContracts(unittest.TestCase):
    def test_predictor_public_api_signatures_exist(self):
        tree = ast.parse((REPO_ROOT / "predictor.py").read_text(encoding="utf-8"))
        funcs = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("load_model", funcs)
        self.assertIn("run_tracker", funcs)
        self.assertEqual(["device"], [arg.arg for arg in funcs["load_model"].args.args])
        self.assertEqual(
            ["model", "video_path", "init_box_path"],
            [arg.arg for arg in funcs["run_tracker"].args.args],
        )

    def test_download_uses_pinned_gdown_and_checkpoint_name(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        download = (REPO_ROOT / "download.py").read_text(encoding="utf-8")
        self.assertIn("gdown==5.2.0", requirements)
        self.assertIn("model_final.pth", download)
        self.assertIn("download_folder", download)

    def test_latency_benchmark_exists_and_requires_gpu_option(self):
        bench = REPO_ROOT / "tools/latency_benchmark.py"
        self.assertTrue(bench.exists())
        source = bench.read_text(encoding="utf-8")
        self.assertIn("--require-gpu", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn("predictor.run_tracker", source)


class TestCsvAndBoxValidation(unittest.TestCase):
    def test_csv_rows_match_expected_ids_and_bbox_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred_path = tmp_path / "predictions.csv"
            rows = [
                {"id": "seq_0", "x": "0", "y": "0", "w": "10", "h": "10"},
                {"id": "seq_1", "x": "5", "y": "6", "w": "8", "h": "9"},
            ]
            with pred_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "x", "y", "w", "h"])
                writer.writeheader()
                writer.writerows(rows)

            with pred_path.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(["id", "x", "y", "w", "h"], list(loaded[0].keys()))
            self.assertEqual(["seq_0", "seq_1"], [row["id"] for row in loaded])
            for row in loaded:
                x, y, w, h = [float(row[key]) for key in ["x", "y", "w", "h"]]
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertGreaterEqual(w, 0)
                self.assertGreaterEqual(h, 0)
                self.assertLessEqual(x + w, 20)
                self.assertLessEqual(y + h, 20)

    def test_inference_writes_csv_with_mocked_tracker(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is needed to import inference.py")
        import inference

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_json = tmp_path / "input.json"
            output_csv = tmp_path / "predictions.csv"
            input_json.write_text(
                json.dumps(
                    {
                        "split": {
                            "seq": {
                                "video_path": "unused.mp4",
                                "annotation_path": "unused.txt",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            fake_predictions = [
                {"frame_idx": 0, "x": 1, "y": 2, "w": 3, "h": 4},
                {"frame_idx": 1, "x": 5, "y": 6, "w": 7, "h": 8},
            ]
            with mock.patch.object(inference, "load_model", return_value=object()), mock.patch.object(
                inference, "run_tracker", return_value=fake_predictions
            ), mock.patch("sys.argv", ["inference.py", str(input_json), "split", str(output_csv)]):
                inference.infer_dataset()

            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(["seq_0", "seq_1"], [row["id"] for row in rows])
            self.assertEqual(["id", "x", "y", "w", "h"], list(rows[0].keys()))


if __name__ == "__main__":
    unittest.main()

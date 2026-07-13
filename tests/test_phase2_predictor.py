import ast
import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = REPO_ROOT / "predictor.py"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
SUBMISSION_FILES = [
    REPO_ROOT / "predictor.py",
    REPO_ROOT / "inference.py",
    REPO_ROOT / "download.py",
    REPO_ROOT / "check_submission.py",
]
HAS_RUNTIME_DEPS = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
)


class TestSubmissionConstraints(unittest.TestCase):
    def test_no_jupyter_notebooks_are_submitted(self):
        notebooks = list(REPO_ROOT.rglob("*.ipynb"))
        self.assertEqual([], [str(path.relative_to(REPO_ROOT)) for path in notebooks])

    def test_phase2_entrypoint_files_exist(self):
        for path in SUBMISSION_FILES:
            self.assertTrue(path.exists(), str(path.relative_to(REPO_ROOT)))

    def test_requirements_are_version_pinned(self):
        lines = [
            line.strip()
            for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertGreater(len(lines), 0)
        for line in lines:
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[^=<>!~]+$")

    def test_submission_files_have_no_hardcoded_paths_or_runtime_setup(self):
        forbidden_patterns = [
            r"/workspace/",
            r"C:\\",
            r"pip\s+install",
            r"git\s+clone",
            r"input\(",
        ]
        for path in SUBMISSION_FILES:
            source = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                self.assertIsNone(
                    re.search(pattern, source),
                    f"{path.relative_to(REPO_ROOT)} matched {pattern}",
                )

    def test_submission_imports_are_available_in_requirements_or_stdlib(self):
        stdlib_or_local = {
            "__future__",
            "csv",
            "contextlib",
            "dataclasses",
            "importlib",
            "json",
            "lib",
            "math",
            "os",
            "pathlib",
            "re",
            "sys",
            "time",
            "types",
            "typing",
            "predictor",
            "torch",
        }
        package_to_imports = {
            "opencv-python-headless": {"cv2"},
            "numpy": {"numpy"},
            "torch": {"torch"},
        }
        allowed_imports = set(stdlib_or_local)
        for package_line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
            package = package_line.strip().split("==", 1)[0]
            allowed_imports.add(package.replace("-", "_"))
            allowed_imports.update(package_to_imports.get(package, set()))

        for path in SUBMISSION_FILES:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            self.assertTrue(
                imports <= allowed_imports,
                f"{path.relative_to(REPO_ROOT)}: {imports - allowed_imports}",
            )


class TestPredictorPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_RUNTIME_DEPS:
            raise unittest.SkipTest("OpenCV and NumPy are required for predictor pipeline tests")
        global cv2
        global np
        global predictor
        import cv2
        import numpy as np
        import predictor

    def test_read_init_box_accepts_commas_and_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            comma_path = Path(tmp) / "comma.txt"
            space_path = Path(tmp) / "space.txt"
            comma_path.write_text("10,20,30,40\n", encoding="utf-8")
            space_path.write_text("10 20 30 40\n", encoding="utf-8")

            self.assertEqual([10.0, 20.0, 30.0, 40.0], predictor.read_init_box(comma_path))
            self.assertEqual([10.0, 20.0, 30.0, 40.0], predictor.read_init_box(space_path))

    def test_postprocessing_clips_boxes_to_frame_bounds(self):
        self.assertEqual(
            [0.0, 0.0, 20.0, 20.0],
            predictor.clip_bbox([-10, -5, 20, 20], width=100, height=80),
        )
        self.assertEqual(
            [90.0, 70.0, 10.0, 10.0],
            predictor.clip_bbox([90, 70, 50, 50], width=100, height=80),
        )

    def test_load_model_returns_framework_compatible_model(self):
        if not any(path.exists() for path in predictor.CHECKPOINT_CANDIDATES):
            with self.assertRaises(FileNotFoundError):
                predictor.load_model(device="cpu")
            return
        model = predictor.load_model(device="cpu")
        self.assertTrue(hasattr(model, "network"))
        self.assertTrue(hasattr(model, "cfg"))
        self.assertEqual("cpu", model.device.type)

    def test_run_tracker_returns_one_prediction_per_frame(self):
        if not any(path.exists() for path in predictor.CHECKPOINT_CANDIDATES):
            self.skipTest("UETrack checkpoint is required for video integration test")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sequence.avi"
            init_path = tmp_path / "annotation_first_box.txt"
            init_box = [12, 16, 18, 14]
            init_path.write_text(",".join(str(v) for v in init_box), encoding="utf-8")
            self._write_synthetic_video(video_path, init_box, n_frames=6)

            model = predictor.load_model(device="cpu")
            predictions = predictor.run_tracker(model, video_path, init_path)

            self.assertEqual(6, len(predictions))
            self.assertEqual(
                {"frame_idx": 0, "x": 12.0, "y": 16.0, "w": 18.0, "h": 14.0},
                predictions[0],
            )
            for idx, row in enumerate(predictions):
                self.assertEqual(idx, row["frame_idx"])
                self.assertEqual({"frame_idx", "x", "y", "w", "h"}, set(row.keys()))
                self.assertGreaterEqual(row["x"], 0.0)
                self.assertGreaterEqual(row["y"], 0.0)
                self.assertGreater(row["w"], 0.0)
                self.assertGreater(row["h"], 0.0)
                self.assertLessEqual(row["x"] + row["w"], 96.0)
                self.assertLessEqual(row["y"] + row["h"], 72.0)

    @staticmethod
    def _write_synthetic_video(video_path, init_box, n_frames):
        width, height = 96, 72
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            8.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create synthetic test video")

        x, y, w, h = init_box
        try:
            for frame_idx in range(n_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                x1 = int(x + frame_idx * 3)
                y1 = int(y + frame_idx * 2)
                x2 = x1 + int(w)
                y2 = y1 + int(h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (x1 + 5, y1 + 5), 3, (255, 0, 0), -1)
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()

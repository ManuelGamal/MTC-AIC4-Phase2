import os
import tempfile
import json
import csv
import unittest
import importlib.util
from unittest import mock
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAS_RUNTIME_DEPS = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
)

if HAS_RUNTIME_DEPS:
    import inference
    import predictor

@unittest.skipUnless(HAS_RUNTIME_DEPS, "OpenCV, NumPy, and torch are required")
class TestEnvironmentSimulator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        
    def tearDown(self):
        self.tmp_dir.cleanup()

    @mock.patch("inference.load_model")
    @mock.patch("inference.run_tracker")
    def test_full_pipeline_simulation(self, mock_run_tracker, mock_load_model):
        """Simulate running inference.py in the environment."""
        
        # 1. Create a fake input.json
        input_json_path = self.tmp_path / "test_input.json"
        video_file = self.tmp_path / "mock_video.mp4"
        annotation_file = self.tmp_path / "mock_anno.txt"
        
        # Write dummy annotation
        annotation_file.write_text("10,20,30,40\n")
        
        # Write dummy json
        input_json_path.write_text(json.dumps({
            "test_split": {
                "seq_1": {
                    "video_path": str(video_file),
                    "annotation_path": str(annotation_file)
                }
            }
        }))
        
        output_csv = self.tmp_path / "predictions.csv"
        
        # 2. Mock predictor behavior
        mock_load_model.return_value = mock.MagicMock()
        mock_run_tracker.return_value = [
            {"frame_idx": 0, "x": 10, "y": 20, "w": 30, "h": 40},
            {"frame_idx": 1, "x": 11, "y": 21, "w": 31, "h": 41}
        ]
        
        # 3. Simulate CLI arguments and run
        with mock.patch("sys.argv", ["inference.py", str(input_json_path), "test_split", str(output_csv)]):
            inference.infer_dataset()
            
        # 4. Verify outputs are saved successfully
        self.assertTrue(output_csv.exists(), "Prediction CSV should be generated")
        
        with open(output_csv, 'r') as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[0]["id"], "seq_1_0")
            self.assertEqual(reader[0]["x"], "10")
            
    def test_video_capture_and_formatting(self):
        """Test formatting of predictions matches the expected standard output."""
        pred = predictor.format_prediction(5, [12.1234, 15.5678, 100.1, 200.2])
        self.assertEqual(pred["frame_idx"], 5)
        self.assertEqual(pred["x"], 12.123)
        self.assertEqual(pred["y"], 15.568)
        self.assertEqual(pred["w"], 100.1)
        self.assertEqual(pred["h"], 200.2)

if __name__ == "__main__":
    unittest.main()

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


root = Path("/tmp/phase3_smoke")
sequence_dir = root / "dataset1" / "synthetic"
sequence_dir.mkdir(parents=True, exist_ok=True)

video_path = sequence_dir / "synthetic.mp4"
writer = cv2.VideoWriter(
    str(video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    5,
    (128, 128),
)
for idx in range(4):
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30 + idx * 2, 40), (60 + idx * 2, 70), (0, 255, 0), -1)
    writer.write(frame)
writer.release()

annotation_path = sequence_dir / "annotation_first_box.txt"
annotation_path.write_text("30,40,30,30\n", encoding="utf-8")

input_path = root / "input.json"
output_path = root / "predictions.csv"
input_path.write_text(
    json.dumps(
        {
            "public_lb": {
                "dataset1_synthetic": {
                    "video_path": str(video_path),
                    "annotation_path": str(annotation_path),
                    "n_frames": 4,
                }
            }
        }
    ),
    encoding="utf-8",
)

subprocess.run(
    [
        "python3",
        "/workspace/inference.py",
        str(input_path),
        "public_lb",
        str(output_path),
    ],
    check=True,
)

print(output_path.read_text(encoding="utf-8"))

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REQUIRED_COLUMNS = ["id", "x", "y", "w", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prediction CSV against an input JSON split.")
    parser.add_argument("input_json")
    parser.add_argument("split")
    parser.add_argument("predictions_csv")
    parser.add_argument("--allow-zero-boxes", action="store_true", help="Allow 0,0,0,0 absent-target rows.")
    return parser.parse_args()


def resolve_path(input_json: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return input_json.resolve().parent / path


def count_video_frames(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video to count frames: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            return frame_count

        count = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
        return count
    finally:
        capture.release()


def expected_ids(input_json: Path, split: str) -> list[str]:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if split not in data:
        raise ValueError(f"Split '{split}' not found in {input_json}")

    ids: list[str] = []
    for sequence_name, info in data[split].items():
        if "n_frames" in info:
            frame_count = int(info["n_frames"])
        else:
            frame_count = count_video_frames(resolve_path(input_json, info["video_path"]))
        if frame_count <= 0:
            raise ValueError(f"Sequence {sequence_name} has no frames.")
        ids.extend(f"{sequence_name}_{frame_idx}" for frame_idx in range(frame_count))
    return ids


def load_rows(predictions_csv: Path) -> list[dict[str, str]]:
    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"Invalid columns in {predictions_csv}: {reader.fieldnames}; expected {REQUIRED_COLUMNS}")
        return list(reader)


def validate_values(rows: list[dict[str, str]], allow_zero_boxes: bool) -> None:
    for idx, row in enumerate(rows):
        values = []
        for key in ["x", "y", "w", "h"]:
            try:
                value = float(row[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {idx} has non-numeric {key}: {row[key]!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Row {idx} has non-finite {key}: {value}")
            values.append(value)

        x, y, w, h = values
        if x < 0 or y < 0 or w < 0 or h < 0:
            raise ValueError(f"Row {idx} has negative bbox value: {row}")
        if not allow_zero_boxes and (w == 0 or h == 0):
            raise ValueError(f"Row {idx} has zero-size bbox: {row}")


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    predictions_csv = Path(args.predictions_csv)

    expected = expected_ids(input_json, args.split)
    rows = load_rows(predictions_csv)
    validate_values(rows, args.allow_zero_boxes)

    found = [row["id"] for row in rows]
    if len(found) != len(expected):
        raise ValueError(f"Row count mismatch: expected {len(expected)}, found {len(found)}")

    expected_set = set(expected)
    found_set = set(found)
    missing = expected_set - found_set
    extra = found_set - expected_set
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise ValueError(f"Missing {len(missing)} prediction IDs. First missing: {preview}")
    if extra:
        preview = ", ".join(sorted(extra)[:10])
        raise ValueError(f"Found {len(extra)} extra prediction IDs. First extra: {preview}")
    if len(found) != len(found_set):
        raise ValueError("Prediction CSV contains duplicate IDs.")

    print("PREDICTIONS MATCH INPUT JSON")
    print(f"Split: {args.split}")
    print(f"Rows: {len(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

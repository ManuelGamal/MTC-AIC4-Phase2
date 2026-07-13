"""
Build an inference manifest from an existing submission CSV and dataset folder.

This is useful for comparing the refactored predictor against an older
submission on exactly the same sequence IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv")
    parser.add_argument("data_root")
    parser.add_argument("output_json")
    parser.add_argument("--split", default="public_lb")
    parser.add_argument("--prefer-suffix", default="_96")
    return parser.parse_args()


def sequence_ids(submission_csv: Path) -> list[str]:
    ids = set()
    with submission_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq_id, _frame_idx = row["id"].rsplit("_", 1)
            ids.add(seq_id)
    return sorted(ids)


def choose_video(seq_dir: Path, prefer_suffix: str) -> Path | None:
    videos = sorted(seq_dir.glob("*.mp4"))
    preferred = [path for path in videos if path.stem.endswith(prefer_suffix)]
    for path in preferred + videos:
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        if ok:
            read_ok, _ = cap.read()
            ok = bool(read_ok)
        cap.release()
        if ok:
            return path
    return None


def frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def main() -> int:
    args = parse_args()
    submission_csv = Path(args.submission_csv)
    data_root = Path(args.data_root)
    output_json = Path(args.output_json)

    manifest = {args.split: {}}
    missing = []

    for seq_id in sequence_ids(submission_csv):
        seq_dir = data_root / seq_id
        annotation = seq_dir / "annotation.txt"
        video = choose_video(seq_dir, args.prefer_suffix)

        if not seq_dir.exists() or video is None or not annotation.exists():
            missing.append(
                {
                    "seq_id": seq_id,
                    "seq_dir_exists": seq_dir.exists(),
                    "annotation_exists": annotation.exists(),
                    "videos": [path.name for path in seq_dir.glob("*.mp4")] if seq_dir.exists() else [],
                }
            )
            continue

        manifest[args.split][seq_id] = {
            "video_path": str(video.relative_to(data_root)),
            "annotation_path": str(annotation.relative_to(data_root)),
            "n_frames": frame_count(video),
        }

    output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote: {output_json}")
    print(f"sequences: {len(manifest[args.split])}")
    print(f"missing: {len(missing)}")
    if missing:
        print("first missing:")
        for item in missing[:20]:
            print(item)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

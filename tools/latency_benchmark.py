"""
Latency benchmark for the phase-2 UETrack submission.

Measures the same public API used by the organizer:
    predictor.load_model()
    predictor.run_tracker()

Example:
    python tools/latency_benchmark.py data/contest_input.json test --max-frames 120 --require-gpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark end-to-end tracker latency.")
    parser.add_argument("input_json", help="Competition JSON file.")
    parser.add_argument("split", help="Split name inside JSON.")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--sequence", default=None, help="Optional sequence name to benchmark.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument("--include-load", action="store_true", help="Include model load time in reported total.")
    parser.add_argument("--decode-only", action="store_true", help="Measure video read + BGR->RGB conversion only.")
    parser.add_argument("--profile", action="store_true", help="Report decode/init/track timing buckets.")
    return parser.parse_args()


def resolve_path(input_json: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return input_json.resolve().parent / path


def benchmark_sequence(
    model,
    input_json: Path,
    name: str,
    info: dict,
    max_frames: int,
    decode_only: bool,
    profile: bool,
) -> dict:
    video_path = resolve_path(input_json, info["video_path"])
    annotation_path = resolve_path(input_json, info["annotation_path"])
    init_box = None if decode_only else predictor.read_init_box(annotation_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        tracker = None if decode_only else predictor.UETrackOnline(model.network, model.cfg, model.device)
        predictions = []
        decode_s = 0.0
        init_s = 0.0
        track_s = 0.0
        post_s = 0.0
        if model is not None and model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        frame_idx = 0
        while frame_idx < max_frames:
            t_decode = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decode_s += time.perf_counter() - t_decode
            if not decode_only:
                if frame_idx == 0:
                    t_init = time.perf_counter()
                    tracker.initialize(frame_rgb, init_box)
                    if model.device.type == "cuda":
                        torch.cuda.synchronize(model.device)
                    init_s += time.perf_counter() - t_init
                    pred_box = init_box
                else:
                    t_track = time.perf_counter()
                    pred_box = tracker.track(frame_rgb)
                    if model.device.type == "cuda":
                        torch.cuda.synchronize(model.device)
                    track_s += time.perf_counter() - t_track
                t_post = time.perf_counter()
                height, width = frame_rgb.shape[:2]
                pred_box = predictor.clip_bbox(pred_box, width, height)
                predictions.append(predictor.format_prediction(frame_idx, pred_box))
                post_s += time.perf_counter() - t_post
            frame_idx += 1
        if model is not None and model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - start
    finally:
        cap.release()

    measured_frames = frame_idx if decode_only else max(0, len(predictions) - 1)
    ms_per_frame = (elapsed / measured_frames * 1000.0) if measured_frames else 0.0
    return {
        "sequence": name,
        "clip_frames": frame_idx,
        "predictions": frame_idx if decode_only else len(predictions),
        "tracked_frames": measured_frames,
        "elapsed_s": elapsed,
        "ms_per_tracked_frame": ms_per_frame,
        "fps": (frame_idx / elapsed) if elapsed else 0.0,
        "decode_s": decode_s,
        "init_s": init_s,
        "track_s": track_s,
        "post_s": post_s,
        "profile": profile,
    }


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if args.split not in data:
        raise ValueError(f"Split '{args.split}' not found in {input_json}.")

    if args.require_gpu and not torch.cuda.is_available():
        print("CUDA is not available; cannot run required GPU latency benchmark.", file=sys.stderr)
        return 2

    model = None
    load_elapsed = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.decode_only:
        load_start = time.perf_counter()
        model = predictor.load_model(device=device)
        load_elapsed = time.perf_counter() - load_start

    sequences = data[args.split]
    if args.sequence is not None:
        sequences = {args.sequence: sequences[args.sequence]}

    results = [
        benchmark_sequence(model, input_json, name, info, args.max_frames, args.decode_only, args.profile)
        for name, info in sequences.items()
    ]
    ms_values = [row["ms_per_tracked_frame"] for row in results if row["tracked_frames"] > 0]

    print(f"mode: {'decode_only' if args.decode_only else 'decode_plus_track'}")
    print(f"device: {model.device if model is not None else 'none'}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"model_load_s: {load_elapsed:.3f}")
    for row in results:
        print(
            f"{row['sequence']}: frames={row['predictions']} "
            f"elapsed_s={row['elapsed_s']:.3f} "
            f"fps={row['fps']:.2f} "
            f"tracked_ms={row['ms_per_tracked_frame']:.3f}"
        )
        if args.profile:
            tracked = max(row["tracked_frames"], 1)
            frames = max(row["clip_frames"], 1)
            print(
                f"  decode_ms/frame={row['decode_s'] / frames * 1000.0:.3f} "
                f"init_ms={row['init_s'] * 1000.0:.3f} "
                f"track_ms/tracked={row['track_s'] / tracked * 1000.0:.3f} "
                f"post_ms/frame={row['post_s'] / frames * 1000.0:.3f}"
            )

    if ms_values:
        total_elapsed = sum(row["elapsed_s"] for row in results)
        if args.include_load:
            total_elapsed += load_elapsed
        print(f"mean_tracked_ms: {statistics.mean(ms_values):.3f}")
        print(f"median_tracked_ms: {statistics.median(ms_values):.3f}")
        print(f"total_elapsed_s: {total_elapsed:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

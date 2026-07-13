"""Simulate the Jetson Orin Nano live-tracking demo.

This utility mirrors the competition demo loop as closely as possible on a
desktop machine:

- load the real UETrack checkpoint through ``predictor.load_model()``
- initialize the tracker from a provided bounding-box file
- stream frames from either a USB camera index or a video file
- optionally degrade frames to resemble a screen being filmed by a webcam
- report per-frame latency and fail if it exceeds a configured threshold

Typical usage:
    python tools/orin_live_simulator.py data/contest_input.json test \
        --sequence seq_0 --source-video /path/to/video.mp4 \
        --init-box /path/to/init_box.txt --simulate-camera --threshold-ms 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import threading
import queue
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the Orin Nano live demo loop.")
    parser.add_argument("input_json", nargs="?", default=None, help="Competition JSON file.")
    parser.add_argument("split", nargs="?", default=None, help="Split name inside the competition JSON.")
    parser.add_argument("--preset", choices=["contest"], default=None, help="Shortcut for the bundled contest JSON and demo settings.")
    parser.add_argument("--sequence", default=None, help="Optional sequence name to run.")
    parser.add_argument("--source-video", default=None, help="Optional video file to replay.")
    parser.add_argument("--camera-index", type=int, default=None, help="USB camera index to open.")
    parser.add_argument("--init-box", default=None, help="Optional init-box text file.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum frames to consume.")
    parser.add_argument("--threshold-ms", type=float, default=30.0, help="Latency threshold per tracked frame.")
    parser.add_argument("--target-fps", type=float, default=0.0, help="Optional replay pacing FPS; 0 disables sleeping.")
    parser.add_argument("--loop-video", action="store_true", help="Loop the video source when it ends.")
    parser.add_argument("--simulate-camera", action="store_true", help="Apply camera/screen artifacts to frames.")
    parser.add_argument("--jpeg-quality", type=int, default=72, help="JPEG quality when simulating camera artifacts.")
    parser.add_argument("--blur-kernel", type=int, default=3, help="Blur kernel size for simulated camera artifacts.")
    parser.add_argument("--motion-jitter", type=float, default=0.8, help="Random crop jitter in pixels for simulated camera artifacts.")
    parser.add_argument("--brightness-jitter", type=float, default=0.05, help="Brightness jitter ratio for simulated camera artifacts.")
    parser.add_argument("--drop-prob", type=float, default=0.0, help="Probability of dropping a replayed frame.")
    parser.add_argument("--display", action="store_true", help="Show the simulated feed in a window.")
    parser.add_argument("--fp16", action="store_true", help="Convert the network to FP16 on CUDA.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is unavailable.")
    parser.add_argument("--include-load", action="store_true", help="Include model load time in the summary.")
    return parser.parse_args()


def resolve_execution_plan(args: argparse.Namespace) -> tuple[Path, str, str | None]:
    if args.preset == "contest":
        input_json = REPO_ROOT / "data" / "contest_input.json"
        split = "test"
        sequence = args.sequence
        return input_json, split, sequence

    if args.input_json is None or args.split is None:
        raise ValueError("input_json and split are required unless --preset contest is used")

    return Path(args.input_json), args.split, args.sequence


def resolve_path(input_json: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return input_json.resolve().parent / path


def open_capture(camera_index: int | None, source_video: Path | None) -> cv2.VideoCapture:
    if camera_index is not None:
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
    if source_video is None:
        raise ValueError("Either --camera-index or --source-video must be provided")
    return cv2.VideoCapture(str(source_video))


class PrefetchCapture:
    def __init__(self, capture: cv2.VideoCapture, enabled: bool = True):
        self.capture = capture
        self.enabled = enabled
        self.queue: queue.Queue[tuple[bool, np.ndarray | None] | None] = queue.Queue(maxsize=1)
        self.thread: threading.Thread | None = None
        self.stopped = False

    def start(self) -> "PrefetchCapture":
        if not self.enabled:
            return self
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return self

    def _worker(self) -> None:
        try:
            while not self.stopped:
                ok, frame = self.capture.read()
                self.queue.put((ok, frame), timeout=0.1)
                if not ok:
                    break
        except Exception:
            self.queue.put(None)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.enabled:
            return self.capture.read()
        item = self.queue.get()
        if item is None:
            return False, None
        return item

    def release(self) -> None:
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_camera_artifacts(frame_bgr: np.ndarray, rng: random.Random, args: argparse.Namespace) -> np.ndarray:
    if not args.simulate_camera:
        return frame_bgr

    height, width = frame_bgr.shape[:2]
    working = frame_bgr.copy()

    jitter = max(0.0, float(args.motion_jitter))
    if jitter > 0:
        shift_x = int(round(rng.uniform(-jitter, jitter)))
        shift_y = int(round(rng.uniform(-jitter, jitter)))
        matrix = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
        working = cv2.warpAffine(working, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    blur_kernel = max(1, int(args.blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel > 1:
        working = cv2.GaussianBlur(working, (blur_kernel, blur_kernel), 0)

    brightness = 1.0 + rng.uniform(-float(args.brightness_jitter), float(args.brightness_jitter))
    working = np.clip(working.astype(np.float32) * brightness, 0.0, 255.0).astype(np.uint8)

    quality = int(np.clip(args.jpeg_quality, 30, 100))
    encode_ok, encoded = cv2.imencode(".jpg", working, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if encode_ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            working = decoded

    return working


def summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    count = len(ordered)
    if count == 0:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "count": float(count),
        "mean": float(statistics.mean(ordered)),
        "median": float(statistics.median(ordered)),
        "p90": float(ordered[min(count - 1, max(0, int(round(count * 0.90)) - 1))]),
        "p99": float(ordered[min(count - 1, max(0, int(round(count * 0.99)) - 1))]),
    }


def load_init_box(input_json: Path, info: dict, explicit_init_box: Path | None) -> list[float]:
    if explicit_init_box is not None:
        return predictor.read_init_box(explicit_init_box)
    return predictor.read_init_box(resolve_path(input_json, info["annotation_path"]))


def run_sequence(model: predictor.UETrackModelBundle, input_json: Path, name: str, info: dict, args: argparse.Namespace) -> dict:
    source_video = resolve_path(input_json, args.source_video or info["video_path"])
    init_box_path = resolve_path(input_json, args.init_box)
    init_box = load_init_box(input_json, info, init_box_path)
    capture = open_capture(args.camera_index, source_video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open capture source: {source_video if source_video is not None else args.camera_index}")
    prefetch = PrefetchCapture(capture, enabled=True).start()

    tracker = predictor.UETrackOnline(model.network, model.cfg, model.device)
    rng = random.Random(42)
    tracked_frame_latencies_ms: list[float] = []
    track_only_latencies_ms: list[float] = []
    decode_seconds = 0.0
    init_seconds = 0.0
    track_seconds = 0.0
    frame_idx = 0
    target_interval = 1.0 / args.target_fps if args.target_fps > 0 else 0.0

    try:
        while frame_idx < args.max_frames:
            frame_start = time.perf_counter()
            if args.drop_prob > 0.0 and frame_idx > 0 and rng.random() < args.drop_prob:
                ok, _ = prefetch.read()
                if not ok:
                    if args.loop_video and source_video is not None:
                        prefetch.release()
                        capture = open_capture(args.camera_index, source_video)
                        prefetch = PrefetchCapture(capture, enabled=True).start()
                        continue
                    break
                frame_idx += 1
                continue

            decode_start = time.perf_counter()
            ok, frame_bgr = prefetch.read()
            if not ok:
                if args.loop_video and source_video is not None:
                    prefetch.release()
                    capture = open_capture(args.camera_index, source_video)
                    prefetch = PrefetchCapture(capture, enabled=True).start()
                    continue
                break
            frame_bgr = apply_camera_artifacts(frame_bgr, rng, args)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decode_seconds += time.perf_counter() - decode_start

            if frame_idx == 0:
                init_start = time.perf_counter()
                _sync_if_cuda(model.device)
                tracker.initialize(frame_rgb, init_box)
                _sync_if_cuda(model.device)
                init_seconds += time.perf_counter() - init_start
                pred_box = init_box
            else:
                track_start = time.perf_counter()
                _sync_if_cuda(model.device)
                pred_box = tracker.track(frame_rgb)
                _sync_if_cuda(model.device)
                track_seconds += time.perf_counter() - track_start
                track_only_latencies_ms.append((time.perf_counter() - track_start) * 1000.0)

            if args.display:
                preview = frame_bgr.copy()
                if frame_idx > 0:
                    x, y, w, h = [int(round(v)) for v in predictor.clip_bbox(pred_box, preview.shape[1], preview.shape[0])]
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("orin-live-simulator", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx > 0:
                tracked_frame_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)

            frame_idx += 1
    finally:
        prefetch.release()
        if args.display:
            cv2.destroyAllWindows()

    stats_total = summarize_latencies(tracked_frame_latencies_ms)
    stats_track = summarize_latencies(track_only_latencies_ms)
    threshold_pass = stats_total["median"] <= args.threshold_ms
    return {
        "sequence": name,
        "frames": frame_idx,
        "tracked_frames": len(tracked_frame_latencies_ms),
        "decode_s": decode_seconds,
        "init_s": init_seconds,
        "track_s": track_seconds,
        "stats_total": stats_total,
        "stats_track": stats_track,
        "pass": threshold_pass,
        "source_video": str(source_video) if source_video is not None else None,
    }


def main() -> int:
    args = parse_args()
    input_json, split_name, requested_sequence = resolve_execution_plan(args)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if split_name not in data:
        raise ValueError(f"Split '{split_name}' not found in {input_json}.")

    if args.require_gpu and not torch.cuda.is_available():
        print("CUDA is not available; the Orin simulator was asked to require GPU.", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_start = time.perf_counter()
    model = predictor.load_model(device=device)
    load_seconds = time.perf_counter() - load_start
    if args.fp16 and model.device.type == "cuda":
        try:
            model.network.half()
            print("Converted model to FP16")
        except Exception as exc:  # pragma: no cover - best-effort optimization path
            print(f"FP16 conversion failed: {exc}", file=sys.stderr)

    sequences = data[split_name]
    if requested_sequence is not None:
        sequences = {requested_sequence: sequences[requested_sequence]}
    elif args.preset == "contest":
        first_name = next(iter(sequences))
        sequences = {first_name: sequences[first_name]}

    results = [run_sequence(model, input_json, name, info, args) for name, info in sequences.items()]
    tracked_means = [row["stats_track"]["mean"] for row in results if row["tracked_frames"] > 0]

    print(f"device: {model.device}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"model_load_s: {load_seconds:.3f}")
    for row in results:
        stats_total = row["stats_total"]
        stats_track = row["stats_track"]
        print(
            f"{row['sequence']}: frames={row['frames']} tracked_frames={row['tracked_frames']} "
            f"median_total_ms={stats_total['median']:.3f} mean_total_ms={stats_total['mean']:.3f} "
            f"median_track_ms={stats_track['median']:.3f} mean_track_ms={stats_track['mean']:.3f} "
            f"p90_total_ms={stats_total['p90']:.3f} p99_total_ms={stats_total['p99']:.3f} pass={row['pass']}"
        )
        if args.include_load:
            total_seconds = row["decode_s"] + row["init_s"] + row["track_s"] + load_seconds
        else:
            total_seconds = row["decode_s"] + row["init_s"] + row["track_s"]
        print(f"  total_seconds={total_seconds:.3f}")

    if tracked_means:
        print(f"overall_mean_track_ms: {statistics.mean(tracked_means):.3f}")
        print(f"overall_median_track_ms: {statistics.median(tracked_means):.3f}")

    if any(not row["pass"] for row in results):
        print(f"Latency threshold exceeded: median must be <= {args.threshold_ms:.2f} ms", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Benchmark UETrack inference latency per-frame.

Usage:
  python tools/benchmark_inference.py <video_path> <annotation_path> [--device cpu|cuda] [--fp16]

Outputs per-frame timings and summary (median, mean, p90, p99).

Note: Run this on the target device (Orin Nano) for realistic numbers.
"""

import sys
import time
import argparse
import statistics
from pathlib import Path

import torch

from predictor import load_model, UETrackOnline, read_init_box, clip_bbox


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('annotation')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fp16', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    model_bundle = load_model(device=args.device)
    device = model_bundle.device
    network = model_bundle.network

    if args.fp16 and device.type == 'cuda':
        try:
            network.half()
            print('Converted network to FP16')
        except Exception as e:
            print('FP16 conversion failed:', e)

    tracker = UETrackOnline(network, model_bundle.cfg, device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError('Could not open video: ' + str(args.video))

    init_box = read_init_box(args.annotation)

    frame_idx = 0
    timings = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_idx == 0:
            tracker.initialize(frame_rgb, init_box)
        else:
            t0 = time.perf_counter()
            tracker.track(frame_rgb)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
        frame_idx += 1

    cap.release()

    if not timings:
        print('No frames measured')
        return

    timings_sorted = sorted(timings)
    mean = statistics.mean(timings_sorted)
    median = statistics.median(timings_sorted)
    p90 = timings_sorted[int(len(timings_sorted)*0.9)-1]
    p99 = timings_sorted[int(len(timings_sorted)*0.99)-1]

    print(f'Frames measured: {len(timings_sorted)}')
    print(f'Mean latency (ms): {mean:.2f}')
    print(f'Median latency (ms): {median:.2f}')
    print(f'P90 latency (ms): {p90:.2f}')
    print(f'P99 latency (ms): {p99:.2f}')


if __name__ == '__main__':
    import cv2
    main()

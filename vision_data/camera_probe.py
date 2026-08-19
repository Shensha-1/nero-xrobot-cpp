#!/usr/bin/env python3
"""Read a UVC camera and verify timestamped RGB frames without robot I/O."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        parser.error("width, height, and frames must be positive")

    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open UVC camera {args.device}")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        timestamps_ns: list[int] = []
        shapes: set[tuple[int, int, int]] = set()
        for _ in range(args.frames):
            ok, frame = camera.read()
            timestamp_ns = time.monotonic_ns()
            if not ok or frame is None:
                raise RuntimeError("UVC camera returned an empty frame")
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"expected BGR frame HxWx3, got {frame.shape}")
            timestamps_ns.append(timestamp_ns)
            shapes.add(tuple(int(value) for value in frame.shape))
        intervals_ms = [
            (current - previous) / 1e6
            for previous, current in zip(timestamps_ns, timestamps_ns[1:])
        ]
        report = {
            "event": "camera_probe",
            "device": args.device,
            "frames": len(timestamps_ns),
            "frame_shapes_bgr": sorted(shapes),
            "first_monotonic_ns": timestamps_ns[0],
            "last_monotonic_ns": timestamps_ns[-1],
            "mean_frame_interval_ms": sum(intervals_ms) / len(intervals_ms) if intervals_ms else 0.0,
            "max_frame_interval_ms": max(intervals_ms, default=0.0),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

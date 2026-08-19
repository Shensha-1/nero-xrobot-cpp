#!/usr/bin/env python3
"""Record timestamped right_wrist UVC video without opening robot interfaces."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means until Ctrl+C")
    parser.add_argument("--preview", action="store_true", help="show a live local preview; press q to stop")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0.0 or args.seconds < 0.0:
        parser.error("width, height, fps must be positive and seconds must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open UVC camera {args.device}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    video_path = args.output_dir / "right_wrist.avi"
    timestamps_path = args.output_dir / "right_wrist_frames.jsonl"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video writer {video_path}")

    start_ns = time.monotonic_ns()
    frame_count = 0
    try:
        with timestamps_path.open("x", encoding="utf-8") as timestamps:
            while True:
                ok, frame = camera.read()
                monotonic_ns = time.monotonic_ns()
                if not ok or frame is None:
                    raise RuntimeError("UVC camera returned an empty frame")
                if frame.shape != (args.height, args.width, 3):
                    raise RuntimeError(
                        f"camera format changed: expected {(args.height, args.width, 3)}, got {frame.shape}"
                    )
                writer.write(frame)
                if args.preview:
                    cv2.imshow("NERO right_wrist", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                timestamps.write(json.dumps({
                    "event": "right_wrist_frame",
                    "sequence": frame_count,
                    "monotonic_ns": monotonic_ns,
                    "wall_time_ns": time.time_ns(),
                }) + "\n")
                frame_count += 1
                if args.seconds > 0.0 and (monotonic_ns - start_ns) / 1e9 >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        writer.release()
        camera.release()
        if args.preview:
            cv2.destroyAllWindows()

    elapsed_s = (time.monotonic_ns() - start_ns) / 1e9
    report = {
        "event": "right_wrist_recording_complete",
        "device": args.device,
        "video": str(video_path),
        "timestamps": str(timestamps_path),
        "frames": frame_count,
        "elapsed_s": elapsed_s,
        "mean_fps": frame_count / elapsed_s if elapsed_s > 0.0 else 0.0,
    }
    (args.output_dir / "recording_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

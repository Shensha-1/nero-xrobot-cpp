#!/usr/bin/env python3
"""Record two timestamped UVC streams without opening robot interfaces."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


class QtPreview:
    """GUI preview independent of OpenCV HighGUI support."""

    def __init__(self) -> None:
        from PyQt5 import QtCore, QtWidgets

        self._core = QtCore
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self._label = QtWidgets.QLabel()
        self._label.setWindowTitle("NERO dual-camera recording: wrist | external")
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setMinimumSize(640, 480)
        self._label.show()

    def show(self, frame: object) -> bool:
        from PyQt5 import QtGui

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QtGui.QImage(rgb.data, width, height, width * 3, QtGui.QImage.Format_RGB888).copy()
        self._label.setPixmap(QtGui.QPixmap.fromImage(image))
        self._label.resize(width, height)
        self._app.processEvents()
        return self._label.isVisible()



@dataclass
class CameraStream:
    name: str
    device: str
    capture: cv2.VideoCapture
    writer: cv2.VideoWriter
    timestamps: object
    sequence: int = 0


def open_stream(name: str, device: str, output_dir: Path, width: int, height: int, fps: float) -> CameraStream:
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {name} UVC camera: {device}")
    writer = cv2.VideoWriter(str(output_dir / f"{name}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create {name} video writer")
    return CameraStream(name, device, capture, writer, (output_dir / f"{name}_frames.jsonl").open("x", encoding="utf-8"))


def record_one(stream: CameraStream, width: int, height: int) -> object:
    ok, frame = stream.capture.read()
    captured_ns = time.monotonic_ns()
    if not ok or frame is None:
        raise RuntimeError(f"{stream.name} camera returned an empty frame")
    if frame.shape != (height, width, 3):
        raise RuntimeError(f"{stream.name} format changed: expected {(height, width, 3)}, got {frame.shape}")
    stream.writer.write(frame)
    stream.timestamps.write(json.dumps({"event": "camera_frame", "stream": stream.name, "sequence": stream.sequence, "monotonic_ns": captured_ns, "wall_time_ns": time.time_ns()}) + "\n")
    stream.sequence += 1
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-wrist-device", default="/dev/video0")
    parser.add_argument("--external-device", default="/dev/video2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means until Ctrl+C")
    parser.add_argument("--preview", action="store_true", help="show both streams; close its window or press Ctrl+C to stop recording")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0.0 or args.seconds < 0.0:
        parser.error("width, height, fps must be positive and seconds must be non-negative")
    if args.right_wrist_device == args.external_device:
        parser.error("the two camera devices must differ")

    preview = QtPreview() if args.preview else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    wrist = external = None
    start_ns = time.monotonic_ns()
    try:
        wrist = open_stream("right_wrist", args.right_wrist_device, args.output_dir, args.width, args.height, args.fps)
        external = open_stream("external", args.external_device, args.output_dir, args.width, args.height, args.fps)
        while True:
            wrist_frame = record_one(wrist, args.width, args.height)
            external_frame = record_one(external, args.width, args.height)
            if preview is not None and not preview.show(cv2.hconcat([wrist_frame, external_frame])):
                break
            if args.seconds > 0.0 and (time.monotonic_ns() - start_ns) / 1e9 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for stream in (wrist, external):
            if stream is not None:
                stream.timestamps.close()
                stream.writer.release()
                stream.capture.release()


    elapsed_s = (time.monotonic_ns() - start_ns) / 1e9
    report = {"event": "dual_camera_recording_complete", "right_wrist": {"device": args.right_wrist_device, "frames": wrist.sequence if wrist else 0}, "external": {"device": args.external_device, "frames": external.sequence if external else 0}, "elapsed_s": elapsed_s, "mean_fps": {"right_wrist": wrist.sequence / elapsed_s if wrist and elapsed_s > 0 else 0.0, "external": external.sequence / elapsed_s if external and elapsed_s > 0 else 0.0}}
    (args.output_dir / "recording_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

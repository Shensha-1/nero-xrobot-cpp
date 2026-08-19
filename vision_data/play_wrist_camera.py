#!/usr/bin/env python3
"""Display a recorded wrist-camera AVI using OpenCV."""

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--window", default="NERO wrist recording")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imshow(args.window, frame)
        if cv2.waitKey(16) & 0xFF in (ord("q"), 27):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

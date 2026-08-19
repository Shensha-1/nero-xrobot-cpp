#!/usr/bin/env python3
"""Open a live UVC camera preview without recording or robot control."""

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--window", default="NERO camera preview")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera: {args.device}")

    while True:
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"camera read failed: {args.device}")
        cv2.imshow(args.window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

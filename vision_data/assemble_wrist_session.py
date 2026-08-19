#!/usr/bin/env python3
"""Align a right_wrist camera recording with xrobot_cpp real-XR telemetry."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any


def json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSON in {path}:{line_number}") from error
    return records


def nearest_frame(frame_times: list[int], timestamp_ns: int) -> tuple[int, int]:
    index = bisect.bisect_left(frame_times, timestamp_ns)
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(frame_times)]
    if not candidates:
        raise RuntimeError("camera recording contains no frames")
    selected = min(candidates, key=lambda candidate: abs(frame_times[candidate] - timestamp_ns))
    return selected, abs(frame_times[selected] - timestamp_ns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--telemetry-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-image-age-ms", type=float, default=50.0)
    args = parser.parse_args()
    if args.max_image_age_ms <= 0.0:
        parser.error("--max-image-age-ms must be positive")
    if args.output_dir.exists():
        raise RuntimeError(f"output directory already exists: {args.output_dir}")

    camera_dir = args.camera_dir
    frame_records = json_lines(camera_dir / "right_wrist_frames.jsonl")
    if not frame_records or any(record.get("event") != "right_wrist_frame" for record in frame_records):
        raise RuntimeError("camera timestamp file does not contain right_wrist_frame records")
    frame_times = [int(record["monotonic_ns"]) for record in frame_records]
    if frame_times != sorted(frame_times):
        raise RuntimeError("camera timestamps are not monotonic")

    telemetry_records = json_lines(args.telemetry_log)
    mit_cycles = [record for record in telemetry_records if record.get("event") == "mit_cycle" and record.get("mode") == "real_xr_teleop"]
    if not mit_cycles:
        raise RuntimeError("telemetry log contains no real_xr_teleop mit_cycle records")
    gripper_events = [record for record in telemetry_records if record.get("event") == "xr_gripper_command"]
    gripper_times = [int(record["monotonic_ns"]) for record in gripper_events]
    maximum_age_ns = int(args.max_image_age_ms * 1e6)

    args.output_dir.mkdir(parents=True)
    accepted = 0
    rejected = 0
    with (args.output_dir / "samples.jsonl").open("x", encoding="utf-8") as output:
        for cycle in mit_cycles:
            timestamp_ns = int(cycle["monotonic_ns"])
            frame_index, image_age_ns = nearest_frame(frame_times, timestamp_ns)
            if image_age_ns > maximum_age_ns:
                rejected += 1
                continue
            gripper_target_m = None
            gripper_index = bisect.bisect_right(gripper_times, timestamp_ns) - 1
            if gripper_index >= 0:
                gripper_target_m = gripper_events[gripper_index].get("target_width_m")
            sample = {
                "event": "aligned_right_wrist_sample",
                "monotonic_ns": timestamp_ns,
                "task": args.task,
                "observation": {
                    "image": {
                        "stream": "right_wrist",
                        "video": str(camera_dir / "right_wrist.avi"),
                        "frame_index": frame_index,
                        "monotonic_ns": frame_times[frame_index],
                        "age_ms": image_age_ns / 1e6,
                    },
                    "state": cycle["q_rad"],
                    "gripper_width_m": cycle.get("gripper_width_m"),
                },
                "action": {
                    "joint_target_rad": cycle["q_des_rad"],
                    "gripper_target_m": gripper_target_m,
                },
                "source": {
                    "telemetry_log": str(args.telemetry_log),
                    "joint_state_sequence": cycle.get("joint_state_sequence"),
                },
            }
            output.write(json.dumps(sample) + "\n")
            accepted += 1

    summary = {
        "event": "right_wrist_session_assembled",
        "task": args.task,
        "camera_frames": len(frame_records),
        "telemetry_cycles": len(mit_cycles),
        "accepted_samples": accepted,
        "rejected_for_image_age": rejected,
        "max_image_age_ms": args.max_image_age_ms,
        "image_stream": "right_wrist",
        "action_semantics": "C++ q_des_rad trajectory target with latest prior gripper target",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

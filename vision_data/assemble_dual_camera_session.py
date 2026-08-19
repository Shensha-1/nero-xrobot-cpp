#!/usr/bin/env python3
"""Align dual-camera recordings to xrobot_cpp real-XR MIT telemetry."""

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
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid JSON in {path}:{line_number}") from error
    return records


def frame_records(camera_dir: Path, stream: str) -> list[dict[str, Any]]:
    records = json_lines(camera_dir / f"{stream}_frames.jsonl")
    if not records or any(record.get("event") != "camera_frame" or record.get("stream") != stream for record in records):
        raise RuntimeError(f"invalid or missing {stream} timestamp stream")
    times = [int(record["monotonic_ns"]) for record in records]
    if times != sorted(times):
        raise RuntimeError(f"{stream} timestamps are not monotonic")
    return records


def nearest(records: list[dict[str, Any]], timestamp_ns: int) -> tuple[dict[str, Any], int]:
    times = [int(record["monotonic_ns"]) for record in records]
    insertion = bisect.bisect_left(times, timestamp_ns)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(records)]
    selected = min(candidates, key=lambda index: abs(times[index] - timestamp_ns))
    return records[selected], abs(times[selected] - timestamp_ns)


def image_sample(camera_dir: Path, stream: str, record: dict[str, Any], age_ns: int) -> dict[str, Any]:
    return {"stream": stream, "video": str(camera_dir / f"{stream}.avi"), "frame_index": record["sequence"], "monotonic_ns": record["monotonic_ns"], "age_ms": age_ns / 1e6}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--telemetry-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-camera-age-ms", type=float, default=40.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=15.0)
    args = parser.parse_args()
    if args.max_camera_age_ms <= 0.0 or args.max_camera_skew_ms <= 0.0:
        parser.error("camera age and skew limits must be positive")
    if args.output_dir.exists():
        raise RuntimeError(f"output directory already exists: {args.output_dir}")

    wrist_frames = frame_records(args.camera_dir, "right_wrist")
    external_frames = frame_records(args.camera_dir, "external")
    telemetry = json_lines(args.telemetry_log)
    cycles = [record for record in telemetry if record.get("event") == "mit_cycle" and record.get("mode") == "real_xr_teleop"]
    if not cycles:
        raise RuntimeError("telemetry log contains no real_xr_teleop MIT cycles")
    gripper_events = [record for record in telemetry if record.get("event") == "xr_gripper_command"]
    gripper_times = [int(record["monotonic_ns"]) for record in gripper_events]

    max_age_ns = int(args.max_camera_age_ms * 1e6)
    max_skew_ns = int(args.max_camera_skew_ms * 1e6)
    accepted = rejected_age = rejected_skew = 0
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "samples.jsonl").open("x", encoding="utf-8") as output:
        for cycle in cycles:
            timestamp_ns = int(cycle["monotonic_ns"])
            wrist, wrist_age_ns = nearest(wrist_frames, timestamp_ns)
            external, external_age_ns = nearest(external_frames, timestamp_ns)
            skew_ns = abs(int(wrist["monotonic_ns"]) - int(external["monotonic_ns"]))
            if wrist_age_ns > max_age_ns or external_age_ns > max_age_ns:
                rejected_age += 1
                continue
            if skew_ns > max_skew_ns:
                rejected_skew += 1
                continue
            gripper_index = bisect.bisect_right(gripper_times, timestamp_ns) - 1
            gripper_target_m = gripper_events[gripper_index].get("target_width_m") if gripper_index >= 0 else None
            sample = {"event": "aligned_dual_camera_sample", "monotonic_ns": timestamp_ns, "task": args.task, "observation": {"images": {"right_wrist": image_sample(args.camera_dir, "right_wrist", wrist, wrist_age_ns), "external": image_sample(args.camera_dir, "external", external, external_age_ns)}, "state": cycle["q_rad"], "gripper_width_m": cycle.get("gripper_width_m")}, "action": {"joint_target_rad": cycle["q_des_rad"], "gripper_target_m": gripper_target_m}, "alignment": {"camera_skew_ms": skew_ns / 1e6, "telemetry_mode": cycle["mode"]}, "source": {"telemetry_log": str(args.telemetry_log), "joint_state_sequence": cycle.get("joint_state_sequence")}}
            output.write(json.dumps(sample) + "\n")
            accepted += 1

    summary = {"event": "dual_camera_session_assembled", "task": args.task, "right_wrist_frames": len(wrist_frames), "external_frames": len(external_frames), "telemetry_cycles": len(cycles), "accepted_samples": accepted, "rejected_for_camera_age": rejected_age, "rejected_for_camera_skew": rejected_skew, "max_camera_age_ms": args.max_camera_age_ms, "max_camera_skew_ms": args.max_camera_skew_ms}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

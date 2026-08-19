# NERO xrobot_cpp Project Guide

`xrobot_cpp` is the single control project for NERO. The compiled C++ binary
is the only component that opens SocketCAN or commands the arm. Python tools
only record cameras, assemble datasets, and prepare learning inputs.

## Layout

| Path | Purpose | Retention |
| --- | --- | --- |
| `src/`, `include/`, `tests/` | C++ controller, IK, CAN, dynamics, tests | Source of truth |
| `assets/urdf/` | Control URDFs | Source of truth |
| `assets/ros_packages/` | Vendored NERO URDF and visual mesh package | Source of truth |
| `third_party/placo/` | Local Placo QP IK dependency | Source of truth |
| `build-xr-placo/` | Physical-arm build | Keep |
| `build-mujoco-xr-placo/` | MuJoCo verification build | Keep |
| `scripts/` | Short operational entry points | Keep |
| `vision_data/` | Camera capture and offline alignment | Keep |
| `learning/datasets/raw/` | Immutable camera and telemetry recordings | Keep |
| `learning/datasets/lerobot/` | Generated aligned episode samples | Regenerate from raw if needed |
| `learning/openpi/`, `learning/cache/` | OpenPI environment and download/cache data | Regenerable, do not delete during active setup |
| `results/` | Controller and simulation experiment logs | Archive selectively |

## Bash Environment

For direct execution of a binary, load the project environment first:

```bash
cd ~/nero/xrobot_cpp
source scripts/xrobot_cpp_env.sh
```

It loads system ROS and exposes the vendored NERO model package. No old Python
control project is used by the teleoperation launcher.

## Daily Collection

Use a fresh three-digit episode number. Start the camera recorder first:

```bash
cd ~/nero/xrobot_cpp/vision_data
./record_episode.sh NNN
```

Then start physical teleoperation in another terminal:

```bash
cd ~/nero/xrobot_cpp
./scripts/run_teleop_episode.sh NNN
```

The teleoperation launcher refuses to start until the camera directory exists,
so failed teleoperation startup cannot create an empty episode. Stop
teleoperation first, then stop the recorder. Assemble only after both stop:

```bash
cd ~/nero/xrobot_cpp
PYTHONPATH= PYTHONHOME= /usr/bin/python3 vision_data/assemble_dual_camera_session.py \
  --camera-dir learning/datasets/raw/pick_place_NNN \
  --telemetry-log learning/datasets/raw/pick_place_NNN/real_xr_teleop.jsonl \
  --output-dir learning/datasets/lerobot/pick_place_NNN \
  --task "pick up the object and place it at the target"
```

## Dataset Admission

Accept an episode only when:

- both AVI files and both timestamp streams are present;
- the assembler reports accepted samples and no camera-age rejections;
- telemetry contains normal `mit_cycle`, `xr_ik_target`, and optional
  `xr_gripper_command` events without explicit safety, timeout, stale-feedback,
  watchdog, or recovery events;
- a visual review confirms object visibility, successful grasp, placement, and
  no unsafe or failed behavior.

Episodes 001 through 010 passed the automatic structural and synchronization
checks on 2026-08-19. They contain 2,264 aligned samples; semantic task success
still requires visual review before training.

## Regenerable Storage

Do not delete raw episodes, assembled episodes, or the two maintained build
directories during normal work. The main reclaimable storage is
`learning/cache/` (currently the largest directory), followed by obsolete
experiment logs in `results/`. Deleting either should be a deliberate action,
not part of routine cleanup.

# NERO Vision Data

`vision_data/` is the camera and offline-alignment layer for `xrobot_cpp`. It
never opens SocketCAN or commands the arm.

## Devices

- `right_wrist`: `/dev/video0`
- `external`: `/dev/video2`

Verify a device without opening CAN:

```bash
cd ~/nero/xrobot_cpp
PYTHONPATH= PYTHONHOME= /usr/bin/python3 vision_data/camera_probe.py --device /dev/video0
```

## Training Episode Capture

Use the short record and teleoperation entry points described in
[`docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md). They avoid long terminal
commands being split across lines:

```bash
cd ~/nero/xrobot_cpp/vision_data
./record_episode.sh NNN
```

The recorder writes both AVI streams, per-frame monotonic timestamps, and a
recording summary to `learning/datasets/raw/pick_place_NNN/`. Start the C++
teleoperation launcher in another terminal only after the preview opens.

## Offline Alignment

`assemble_dual_camera_session.py` matches each `real_xr_teleop` MIT telemetry
cycle to the nearest wrist and external frames. It rejects a sample if either
camera is more than 40 ms away from the control timestamp or if the two camera
timestamps differ by more than 15 ms. This is software timestamp alignment, not
hardware-trigger synchronization.

The assembled output contains `samples.jsonl` and `summary.json` under
`learning/datasets/lerobot/pick_place_NNN/`. See `learning/README.md` for the
actual state/action schema.

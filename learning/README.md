# NERO Learning Workspace

This is the learning layer of `xrobot_cpp`. The C++ executable remains the
only process allowed to open SocketCAN, select motor modes, or command MIT
control. Python here records cameras, aligns data, and prepares offline policy
inputs only.

See [`docs/PROJECT_GUIDE.md`](../docs/PROJECT_GUIDE.md) for the full project
layout and collection workflow.

## Layout

- `datasets/raw/`: immutable dual-camera recordings and matching C++ telemetry.
- `datasets/lerobot/`: generated aligned samples; regenerate from raw data.
- `configs/`: task, feature, training, and deployment configuration.
- `tools/`: environment setup and verification tools.
- `openpi/`: official OpenPI source and its project-local environment.
- `checkpoints/`, `runs/`: future model weights and evaluation outputs.
- `cache/`: regenerable package and model-download cache.

## Current Dual-Camera Schema

The current assembler writes one JSON object per accepted real-XR MIT cycle:

```text
observation.images.right_wrist  AVI path, frame index, monotonic timestamp
observation.images.external     AVI path, frame index, monotonic timestamp
observation.state               [q1, q2, q3, q4, q5, q6, q7] in radians
observation.gripper_width_m     measured gripper opening in metres
action.joint_target_rad         next C++ joint target [q1..q7] in radians
action.gripper_target_m         latest gripper target in metres, or null
```

`joint_target_rad` is an absolute C++ target, not a torque command and not a
joint increment. Any pi0.5 adapter must explicitly transform this schema into
the policy representation, and the C++ safety layer must continue to clamp all
limits before actuation.

## Environment

```bash
cd ~/nero/xrobot_cpp
bash learning/tools/setup_pi05_env.sh
source learning/openpi/.venv/bin/activate
python learning/tools/verify_pi05_environment.py
```

The RTX 3050 has 8 GB VRAM. It is suitable for collection, conversion, and
software integration, but not approved for local pi0.5 inference or LoRA/full
fine-tuning. Keep the model-serving and training plan separate from the
physical-arm control process.

## Dataset Admission

Keep an episode only when both camera streams are readable, timestamp alignment
passes, telemetry is continuous with no explicit safety/timeout/recovery event,
and a visual review confirms a successful task execution. Automatic validation
has passed for episodes 001 through 010; semantic success still needs human
review before training.

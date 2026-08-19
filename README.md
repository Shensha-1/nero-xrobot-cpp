# NERO C++ XR Control

This repository implements the C++ control path for the AgileX NERO arm.

```text
Pico XR frame -> TCP mapping -> Placo QP inverse kinematics
              -> S-curve joint trajectory -> G(q) + MIT impedance
              -> SocketCAN MIT commands -> NERO arm
```

## Supported Builds

Only these two build directories are maintained. All other `build*` directories are regenerable CMake artifacts.

| Directory | Purpose | Opens CAN |
| --- | --- | --- |
| `build-xr-placo` | Physical NERO XR teleoperation with Placo QP IK | Only for real-control commands |
| `build-mujoco-xr-placo` | MuJoCo, XR, and the same Placo QP IK backend | No |

## Build

```bash
cd ~/nero/xrobot_cpp

# Physical-arm build: XR + Placo, no MuJoCo.
cmake -S . -B build-xr-placo \
  -DXROBOT_CPP_WITH_XR=ON \
  -DXROBOT_CPP_WITH_MUJOCO=OFF \
  -DXROBOT_CPP_WITH_PLACO=ON
cmake --build build-xr-placo -j1

# Simulation build: MuJoCo + XR + the same Placo backend.
cmake -S . -B build-mujoco-xr-placo \
  -DXROBOT_CPP_WITH_XR=ON \
  -DXROBOT_CPP_WITH_MUJOCO=ON \
  -DXROBOT_CPP_WITH_PLACO=ON
cmake --build build-mujoco-xr-placo -j1
```

Placo runs locally inside the C++ process. It solves constrained QP inverse kinematics from the NERO URDF, current joint state, TCP tasks, joint limits, and velocity limits. It does not call a network or cloud API.

## MuJoCo Verification

The commands below are simulation-only and never open `can0`.

```bash
cd ~/nero/xrobot_cpp
ctest --test-dir build-mujoco-xr-placo --output-on-failure
./build-mujoco-xr-placo/xrobot_cpp --mujoco-model-parity
./build-mujoco-xr-placo/xrobot_cpp \
  --mujoco-full-chain-test \
  --output results/mujoco_full_chain.json
```

For interactive XR simulation, hold Pico A to rebase and track; release A to hold the last target. Trigger closes the simulated gripper and Grip opens it.

```bash
./build-mujoco-xr-placo/xrobot_cpp \
  --mujoco-xr-teleop \
  --orientation-mode absolute \
  --translation-scale 0.65 \
  --gripper-width-m 0.060 \
  --payload-mass-kg 0.50 \
  --telemetry-log results/mujoco_xr_placo_trial.csv
```

## Real Arm Preflight

The physical controller requires an enabled arm and fresh feedback on `can0`. This preflight is receive-only except for `--bootstrap-feedback`, which sends the non-motion mode and feedback-request frames used to start feedback. It sends no enable, MIT, joint, or gripper movement frame.

```bash
cd ~/nero/xrobot_cpp
./build-xr-placo/xrobot_cpp \
  --real-control-preflight \
  --seconds 5 \
  --can can0 \
  --bootstrap-feedback
```

Proceed only when preflight reports continuously ready feedback.

## Real XR Teleoperation

Validate parsing without opening Pico, CAN, motor modes, MIT, joint, or gripper commands:

```bash
./build-xr-placo/xrobot_cpp \
  --real-xr-teleop \
  --arm-command-unlock \
  --supported-arm \
  --orientation-mode absolute \
  --enable-xr-gripper \
  --translation-scale 0.65 \
  --session-radius-m 0 \
  --seconds 0 \
  --dry-run
```

The following command controls the physical arm. It requires an enabled arm, working CAN feedback, the Pico PC service, a clear workspace, and physical support acknowledgement.

```bash
./build-xr-placo/xrobot_cpp \
  --real-xr-teleop \
  --arm-command-unlock \
  --supported-arm \
  --orientation-mode absolute \
  --enable-xr-gripper \
  --translation-scale 0.65 \
  --session-radius-m 0 \
  --joint-v-max 0.40 \
  --joint-a-max 0.80 \
  --joint-j-max 8.0 \
  --seconds 0 \
  --telemetry-log results/real_xr_placo_absolute_gripper.jsonl
```

Controls:

- Hold Pico A: rebase at the measured TCP and enter MIT tracking.
- Release A: restore MOVE J and command the measured joint position as hold.
- Controller translation: TCP translation in the calibrated base frame.
- `--orientation-mode absolute`: controller attitude maps to absolute TCP attitude through `configs/nero_xr_calibration.json`.
- Trigger closes the gripper; Grip opens it. The XR gripper target changes by at most 1 mm per input update, which is 50% of the previous rate. Both are ignored unless `--enable-xr-gripper` is present.
- `Ctrl+C`, XR timeout, genuinely stale CAN feedback, or a safety rejection restore MOVE J measured-position hold. Feedback age is evaluated after reading the latest receiver snapshot, so a frame arriving during a control cycle is not misclassified as stale. Recovery sends no zero-torque MIT frame.

`MOVE J` is the NERO firmware position-control mode. It is used for the non-motion hold after leaving MIT tracking; MIT mode is used while A is held to send impedance and gravity control commands.

## Telemetry

`--telemetry-log PATH` writes JSONL or CSV depending on the command. Real XR logs record measured and desired joints, velocity, gravity and impedance torque at 10 Hz, plus Placo IK targets with solve time and residual errors, and safety events. With `--enable-xr-gripper`, each transmitted gripper command adds an `xr_gripper_command` event containing XR sequence, Trigger, Grip, target width, and force.

`XROBOT_CPP_TX_AUDIT_PATH=PATH` records transmitted CAN frames for short protocol inspection. Do not set it during real-time teleoperation: it adds high-rate disk I/O to every transmitted frame.

## Model and Control Notes

- The NERO URDF supplies joint origins, limits, link masses, centers of mass, and inertias for FK and nominal gravity compensation `G(q)`.
- `--mujoco-model-parity` compares MuJoCo and URDF/KDL FK and gravity terms.
- The real controller computes nominal `G(q)` each cycle and combines it with MIT position and velocity impedance.
- CAN feedback runs in a dedicated receiver and MIT transmission is scheduled at 100 Hz. The safety gate evaluates feedback age after reading that receiver snapshot.
- Placo performs IK only. This project implements SocketCAN framing, trajectory generation, gravity feedforward, impedance torque construction, watchdogs, and gripper commands.

## Final Verification

The maintained builds are validated locally with:

```bash
ctest --test-dir build-xr-placo --output-on-failure
ctest --test-dir build-mujoco-xr-placo --output-on-failure
./build-mujoco-xr-placo/xrobot_cpp --mujoco-model-parity
```

These checks verify protocol parsing, dynamics, gravity/FK parity, QP IK integration, and the feedback-safety timebase. They do not replace enabled-arm preflight before every physical session.

## Cleaning Generated Files

Keep `build-xr-placo` and `build-mujoco-xr-placo`. Other build directories can be removed and rebuilt with CMake. `results/` contains experiment logs only; retain validated trials and CAN audit files needed for later analysis.


## Project Operations

For the current project layout, Bash environment, short data-collection
commands, dataset admission rules, and storage policy, see
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md). Direct Bash invocations of
`xrobot_cpp` should load:

```bash
source scripts/xrobot_cpp_env.sh
```

The NERO URDF and mesh package used by the C++ controller is vendored in
`assets/ros_packages/`; the physical teleoperation launcher does not depend on
the old Python control project path.

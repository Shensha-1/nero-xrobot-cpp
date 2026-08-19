#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]{3}$ ]]; then
  echo "Usage: $0 NNN" >&2
  exit 2
fi

root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
episode_dir="$root_dir/learning/datasets/raw/pick_place_$1"
if [[ ! -d "$episode_dir" ]]; then
  echo "Camera directory is missing: $episode_dir. Start record_episode.sh $1 first." >&2
  exit 1
fi

source "$root_dir/scripts/xrobot_cpp_env.sh"
set -u

exec "$root_dir/build-xr-placo/xrobot_cpp" \
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
  --telemetry-log "$episode_dir/real_xr_teleop.jsonl"

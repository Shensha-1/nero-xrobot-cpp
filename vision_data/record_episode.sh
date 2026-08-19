#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]{3}$ ]]; then
  echo "Usage: $0 NNN" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd -- "$script_dir/.." && pwd)

exec env -u PYTHONPATH -u PYTHONHOME /usr/bin/python3 \
  "$script_dir/record_dual_camera.py" \
  --output-dir "$root_dir/learning/datasets/raw/pick_place_$1" \
  --preview

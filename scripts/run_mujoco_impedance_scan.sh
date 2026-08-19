#!/usr/bin/env bash
set -euo pipefail

# Simulation-only K/D/gravity sweep. It never opens SocketCAN.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-$root_dir/build-mujoco/xrobot_cpp}"
output_dir="${2:-$root_dir/results/impedance_scan}"
mkdir -p "$output_dir"
csv="$output_dir/summary.csv"
printf 'stiffness_nm_rad,damping_nm_s_rad,gravity_scale,exit_code,result_path\n' > "$csv"
trajectories="${XROBOT_CPP_SCAN_TRAJECTORIES:-3}"
for stiffness in 4 6 8; do
  for damping in 0.6 1.0 1.4; do
    for gravity_scale in 0.90 1.00 1.10; do
      tag="k${stiffness}_d${damping}_g${gravity_scale}"
      result="$output_dir/$tag.jsonl"
      set +e
      "$binary" --mujoco-impedance-convergence --trajectories "$trajectories" --seconds 3 --payload-mass-kg 0.20 --seed 20260810 --spring-stiffness "$stiffness" --damping "$damping" --gravity-scale "$gravity_scale" --output "$result"
      code=$?
      set -e
      printf '%s,%s,%s,%s,%s\n' "$stiffness" "$damping" "$gravity_scale" "$code" "$result" >> "$csv"
    done
  done
done
echo "MuJoCo impedance scan completed: $csv"

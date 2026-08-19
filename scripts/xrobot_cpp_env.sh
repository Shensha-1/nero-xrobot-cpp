#!/usr/bin/env bash
# Source this file before directly running an xrobot_cpp binary from Bash.

_xrobot_cpp_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/jazzy/setup.bash
export AMENT_PREFIX_PATH="${_xrobot_cpp_root}/assets/ros_packages:${AMENT_PREFIX_PATH}"
unset _xrobot_cpp_root

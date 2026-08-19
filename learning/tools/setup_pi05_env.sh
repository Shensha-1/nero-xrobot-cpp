#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
openpi_dir="${workspace_dir}/learning/openpi"
uv_bin="${HOME}/.local/bin/uv"
cache_dir="${workspace_dir}/learning/cache/uv"

# Use a domestic PyPI mirror for registry packages. The two OpenPI Git
# dependencies remain pinned to upstream commits and are intentionally not
# substituted with an unverified code mirror.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_dir}}"
# OpenPI pins its two Git dependencies to exact commits. Route only this setup
# process through a GitHub transport proxy; do not modify the user's global Git
# configuration.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.https://ghfast.top/https://github.com/.insteadOf"
export GIT_CONFIG_VALUE_0="https://github.com/"

if [[ ! -x "${uv_bin}" ]]; then
  echo "ERROR: uv is not installed at ${uv_bin}." >&2
  exit 1
fi

if [[ ! -f "${openpi_dir}/pyproject.toml" ]]; then
  echo "Fetching official Physical Intelligence OpenPI source..."
  git clone --depth 1 --no-recurse-submodules \
    https://github.com/Physical-Intelligence/openpi.git "${openpi_dir}"
fi

cd "${openpi_dir}"
GIT_LFS_SKIP_SMUDGE=1 "${uv_bin}" lock --upgrade
GIT_LFS_SKIP_SMUDGE=1 "${uv_bin}" sync --frozen
"${openpi_dir}/.venv/bin/python" "${workspace_dir}/learning/tools/verify_pi05_environment.py"

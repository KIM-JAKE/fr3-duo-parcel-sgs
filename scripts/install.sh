#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
mkdir -p .local_isaaclab3
if [[ ! -d .local_isaaclab3/IsaacLab/.git ]]; then
 git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git .local_isaaclab3/IsaacLab
fi
git -C .local_isaaclab3/IsaacLab fetch origin e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
git -C .local_isaaclab3/IsaacLab checkout --detach e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
uv venv --python 3.12 --seed .local_isaaclab3/venv
source .local_isaaclab3/venv/bin/activate
(cd .local_isaaclab3/IsaacLab && ./isaaclab.sh -i 'newton,rl[rsl-rl]')
uv pip install --python "$VIRTUAL_ENV/bin/python" --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-lock.txt
python -c 'import isaaclab,isaaclab_newton,newton,torch; print(torch.__version__, newton.__version__)'

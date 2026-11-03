#!/usr/bin/env bash
# Usage: tools/check.sh {validator|bag|web} [arguments...]
set -eo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:---help}" in
  validator) script=validator.py ;;
  bag) script=bag_pipeline.py ;;
  web) script=web.py ;;
  -h|--help)
    echo 'Usage: tools/check.sh {validator|bag|web} [arguments...]'
    echo 'validator: synthetic NPZ + real validator + atomic commit (ROS, NumPy)'
    echo 'bag: rosbag -> running Capture/Reconstruct Actions (ROS; see bag --help)'
    echo 'web: local dashboard browser check (Chromium, Node; see web --help)'
    exit 0 ;;
  *) echo "Unknown check: $1" >&2; exit 2 ;;
esac
shift
check_args=("$@")
set --
if [[ "$script" != web.py ]]; then
  if [[ ! -f "$workspace/devel/setup.bash" ]]; then
    echo 'Build the ROS workspace with catkin_make first.' >&2
    exit 1
  fi
  source "$workspace/devel/setup.bash"
fi
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" "$workspace/tools/checks/$script" "${check_args[@]}"

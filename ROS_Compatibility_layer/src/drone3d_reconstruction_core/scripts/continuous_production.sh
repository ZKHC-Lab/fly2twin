#!/usr/bin/env bash
# Unified production start for the single-NX continuous reconstruction line.
#
# This one command performs the frozen preflight (existing ROS master +
# worker CUDA/checkpoint) and then launches the capture Actions, the
# Reconstruct Action, the continuous session scheduler, and (with --web)
# the production Web dashboard. It never starts roscore and never starts
# hardware-group components.
#
# The Web node reuses the SAME capture profiles and continuous profile that
# the scheduler already loads; no new profile file is required. Set
# web_host / web_port / web_preview_fps inside continuous_profile YAML
# (default web_host is 127.0.0.1; use 0.0.0.0 only on a trusted intranet).
#
# Usage:
#   continuous_production.sh \
#     --front-profile   /abs/front/capture_profile.yaml \
#     --down-profile    /abs/down/capture_profile.yaml \
#     --reconstruct-profile /abs/reconstruct_profile.yaml \
#     --continuous-profile  /abs/continuous_profile.yaml [--web]
set -euo pipefail

usage() {
  echo "usage: $0 --front-profile F --down-profile D --reconstruct-profile R --continuous-profile C [--web]" >&2
  exit 2
}

FRONT=""
DOWN=""
RECONSTRUCT=""
CONTINUOUS=""
WEB=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --front-profile) FRONT="${2:-}"; shift 2 ;;
    --down-profile) DOWN="${2:-}"; shift 2 ;;
    --reconstruct-profile) RECONSTRUCT="${2:-}"; shift 2 ;;
    --continuous-profile) CONTINUOUS="${2:-}"; shift 2 ;;
    --web) WEB=1; shift ;;
    *) usage ;;
  esac
done
[[ -n "$FRONT" && -n "$DOWN" && -n "$RECONSTRUCT" && -n "$CONTINUOUS" ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_PYTHON_BIN="${ROS_PYTHON:-python3}"

echo "production preflight: checking ROS master, CUDA and checkpoint"
"$ROS_PYTHON_BIN" "$SCRIPT_DIR/continuous_production_preflight.py" \
  --reconstruct-profile "$RECONSTRUCT" --continuous-profile "$CONTINUOUS"

WEB_ARG="false"
if [[ "$WEB" -eq 1 ]]; then
  WEB_ARG="true"
fi

exec roslaunch drone3d_reconstruction_core continuous_production.launch \
  front_capture_profile:="$FRONT" \
  down_capture_profile:="$DOWN" \
  reconstruct_profile:="$RECONSTRUCT" \
  continuous_profile:="$CONTINUOUS" \
  web_enabled:="$WEB_ARG"

#!/usr/bin/env bash
# One command for a recording flight: sensors + TF (no production stack),
# camera check, then bag recording (JPEG color, no depth) until Ctrl-C.
#
#   bash nx_capture.sh NAME [EXPOSURE_MS GAIN]      (default 10 ms, gain 128)
#
# The camera often comes up without images on the first start; bringup is
# retried once, which recovers it (see NX_BRINGUP.md).
set -eo pipefail

NAME="${1:?usage: $0 NAME [EXPOSURE_MS GAIN]}"
EXPOSURE_MS="${2:-10}"
GAIN="${3:-128}"
S="$(cd "$(dirname "$0")" && pwd)"

bringup() {
  bash "$S/nx_step2_bringup.sh" --no-production --color 1280x720@30 \
    --rgb-exposure-ms "$EXPOSURE_MS" --rgb-gain "$GAIN"
}
bringup || { echo "== bringup failed once, retrying"; bringup; }

source /opt/ros/noetic/setup.bash
python3 "$S/camera_check.py" --seconds 4
read -r -p "Brightness OK? Enter starts recording ${NAME} (Ctrl-C stops it), Ctrl-C now aborts: "
exec bash "$S/nx_record.sh" --name "$NAME" --no-depth   # 3DGS uses Omega depth, not the D435i

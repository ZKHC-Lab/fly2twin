#!/usr/bin/env bash
# Record a data bag for offline reconstruction (3DGS / replay).
# Run after `nx_step2_bringup.sh --no-production ...`; stop with Ctrl-C.
#
#   bash nx_record.sh [--out DIR] [--name NAME] [--no-depth] [--no-lidar] [--raw]
#
# Default: color as JPEG quality 95 via image_transport (/compressed), depth as
# lossless PNG level 1 (/compressedDepth). JPEG 95 differs from the raw frame by
# ~42 dB PSNR, below the sensor noise, at ~1/9 of the size. --raw records
# uncompressed images (~5.8 GB/min). No bag-level compression: an LZ4 chunk in an
# earlier outdoor bag could not be read back. 3 GB input buffer: the default
# 256 MB holds under 3 s at ~100 MB/s, and office_gs_01 lost 1.7 s of color once.
set -eo pipefail

OUT="${HOME}/bags"
NAME="flight_$(date +%Y%m%d_%H%M%S)"
DEPTH=1
LIDAR=1
RAW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --no-depth) DEPTH=0; shift ;;
    --no-lidar) LIDAR=0; shift ;;
    --raw) RAW=1; shift ;;
    *) echo "usage: $0 [--out DIR] [--name NAME] [--no-depth] [--no-lidar] [--raw]" >&2; exit 2 ;;
  esac
done

source /opt/ros/noetic/setup.bash
mkdir -p "$OUT"

color=/camera/color/image_raw
depth=/camera/aligned_depth_to_color/image_raw
if [[ "$RAW" -eq 0 ]]; then
  rosrun dynamic_reconfigure dynparam set "$color/compressed" "{format: jpeg, jpeg_quality: 95}"
  [[ "$DEPTH" -eq 1 ]] && rosrun dynamic_reconfigure dynparam set "$depth/compressedDepth" png_level 1
  color="$color/compressed"
  depth="$depth/compressedDepth"
fi
topics=(
  "$color" /camera/color/camera_info /camera/color/metadata
  /Odometry /tf /tf_static
)
[[ "$DEPTH" -eq 1 ]] && topics+=("$depth" /camera/aligned_depth_to_color/camera_info)
[[ "$LIDAR" -eq 1 ]] && topics+=(/livox/lidar /livox/imu)

size="$(timeout 3 rostopic echo -n 1 /camera/color/camera_info 2>/dev/null \
  | awk '/^height:/{h=$2} /^width:/{w=$2} END{if (w) print w "x" h}')"
echo "color: ${size:-NOT PUBLISHING}"
echo "free space in $OUT: $(df -h --output=avail "$OUT" | tail -1 | tr -d ' ')"
echo "NX measured: raw color + aligned depth 5.8 GB/min; JPEG color + lidar (no depth) 0.86 GB/min"
echo "recording $OUT/$NAME.bag (Ctrl-C to stop)"
exec rosbag record --buffsize=3072 -O "$OUT/$NAME.bag" "${topics[@]}"

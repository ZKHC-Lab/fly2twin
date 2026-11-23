#!/usr/bin/env bash
# Single-NX Step 2 one-click bringup (bench only; never starts propellers).
#
# Verified start order on the Orin NX 16GB bench host (Ubuntu 20.04 / ROS Noetic):
#   roscore -> livox MID360 -> FAST-LIO -> Realsense -> body->camera TF
#   -> production stack (capture + reconstruct + continuous session + web)
# The production stack runs the mainline frame_pose placement from the
# tools/nx_bench profiles; that profile has not been run on the NX yet.
#
# With --external-odom TOPIC the FAST-LIO step is replaced by an externally
# published nav_msgs/Odometry topic: our FAST-LIO is never started and the
# front capture profile is pointed at TOPIC (a derived profile file is used,
# the original profile is never modified).
#
# Every step is idempotent: components already publishing their expected ROS
# topic are skipped, and a second production launch is never created.
# See docs/NX_BRINGUP.md in this package for details.
# (No `set -u`: /opt/ros/noetic/setup.bash and its profile snippets are not
#  nounset-clean, and this script sources them.)
set -eo pipefail

# Defaults follow the deployment this script lives in: <ws>/src/drone3d_reconstruction_core/scripts.
WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
STEP2_ROOT="${STEP2_ROOT:-$WORKSPACE/run/nx_step2}"
WS_LIVOX_SETUP="${WS_LIVOX_SETUP:-$HOME/ws_livox/devel/setup.bash}"
WS_SLAM_SETUP="${WS_SLAM_SETUP:-$HOME/slam_ws/devel/setup.bash}"
ROS_SETUP="/opt/ros/noetic/setup.bash"
WEB=1
EXTERNAL_ODOM_TOPIC=""
NO_LIVOX=0
PRODUCTION=1
COLOR="640x480@15"
RGB_EXPOSURE_MS=""
RGB_GAIN=""

usage() {
  cat >&2 <<EOF
usage: $0 [--step2-root DIR] [--workspace DIR] [--no-web]
          [--external-odom TOPIC] [--no-livox] [--no-production]
          [--color WxH@FPS] [--rgb-exposure-ms MS --rgb-gain G]

  --step2-root DIR      NX bench root containing the four profiles and fastlio/
                        (default: <workspace>/run/nx_step2)
  --workspace DIR       catkin workspace root (default: the workspace containing this script)
  --no-web              start production without the Web dashboard
  --external-odom TOPIC do not start our FAST-LIO; require TOPIC to publish
                        nav_msgs/Odometry and point front capture at it
                        (example: --external-odom /Odometry)
  --no-livox            do not start the livox MID360 driver either
                        (for fully external odometry setups)
  --no-production       stop after sensors + TF (data recording flights)
  --color WxH@FPS       RealSense color stream (default 640x480@15).
                        The production profiles assume 640x480; use another
                        size only with --no-production
  --rgb-exposure-ms MS  manual RGB exposure (disables auto exposure);
  --rgb-gain G          requires --rgb-gain (0-128)
EOF
  exit 2
}

say() { printf '[nx-bringup %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[nx-bringup] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step2-root) STEP2_ROOT="${2:-}"; shift 2 ;;
    --workspace) WORKSPACE="${2:-}"; shift 2 ;;
    --no-web) WEB=0; shift ;;
    --external-odom) EXTERNAL_ODOM_TOPIC="${2:-}"; shift 2 ;;
    --no-livox) NO_LIVOX=1; shift ;;
    --no-production) PRODUCTION=0; shift ;;
    --color) COLOR="${2:-}"; shift 2 ;;
    --rgb-exposure-ms) RGB_EXPOSURE_MS="${2:-}"; shift 2 ;;
    --rgb-gain) RGB_GAIN="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

if [[ "$NO_LIVOX" -eq 1 && -z "$EXTERNAL_ODOM_TOPIC" ]]; then
  die "--no-livox requires --external-odom TOPIC"
fi
[[ "$COLOR" =~ ^([0-9]+)x([0-9]+)@([0-9]+)$ ]] || die "--color must look like 1280x720@30"
COLOR_W="${BASH_REMATCH[1]}"; COLOR_H="${BASH_REMATCH[2]}"; COLOR_FPS="${BASH_REMATCH[3]}"
if [[ "$PRODUCTION" -eq 1 && "$COLOR" != "640x480@15" ]]; then
  die "--color $COLOR needs --no-production (production profiles assume 640x480)"
fi
if [[ -n "$RGB_EXPOSURE_MS$RGB_GAIN" && ( -z "$RGB_EXPOSURE_MS" || -z "$RGB_GAIN" ) ]]; then
  die "--rgb-exposure-ms and --rgb-gain go together"
fi

FRONT_PROFILE="$STEP2_ROOT/capture_profile.front.yaml"
FRONT_EXTERNAL_PROFILE="$STEP2_ROOT/capture_profile.front.external_odom.yaml"
DOWN_PROFILE="$STEP2_ROOT/capture_profile.down.yaml"
RECONSTRUCT_PROFILE="$STEP2_ROOT/reconstruct_profile.yaml"
CONTINUOUS_PROFILE="$STEP2_ROOT/continuous_profile.yaml"
FASTLIO_LAUNCH="$STEP2_ROOT/fastlio/mapping_mid360_step2.launch"
PROD_SCRIPT="$WORKSPACE/src/drone3d_reconstruction_core/scripts/continuous_production.sh"
PREFLIGHT="$WORKSPACE/src/drone3d_reconstruction_core/scripts/continuous_production_preflight.py"

FRONT_PROFILE_RUNTIME="$FRONT_PROFILE"
if [[ -n "$EXTERNAL_ODOM_TOPIC" ]]; then
  if [[ "$EXTERNAL_ODOM_TOPIC" != /* || "$EXTERNAL_ODOM_TOPIC" == *" "* ]]; then
    die "external odometry topic must be an absolute ROS topic name without spaces"
  fi
  CURRENT_ODOM_TOPIC="$(sed -n \
    's/^odometry_topic:[[:space:]]*"\(.*\)"[[:space:]]*$/\1/p' \
    "$FRONT_PROFILE" | head -1)"
  [[ -n "$CURRENT_ODOM_TOPIC" ]] || die "could not read odometry_topic from $FRONT_PROFILE"
  if [[ "$CURRENT_ODOM_TOPIC" != "$EXTERNAL_ODOM_TOPIC" ]]; then
    cp "$FRONT_PROFILE" "$FRONT_EXTERNAL_PROFILE"
    sed -i \
      "s|^odometry_topic: .*|odometry_topic: \"$EXTERNAL_ODOM_TOPIC\"|" \
      "$FRONT_EXTERNAL_PROFILE"
    FRONT_PROFILE_RUNTIME="$FRONT_EXTERNAL_PROFILE"
    printf '[nx-bringup %s] %s\n' "$(date +%H:%M:%S)" \
      "external odometry: derived front profile $FRONT_EXTERNAL_PROFILE"
  fi
fi

for profile in \
  "$FRONT_PROFILE" "$DOWN_PROFILE" "$RECONSTRUCT_PROFILE" "$CONTINUOUS_PROFILE"
do
  [[ -f "$profile" ]] || die "missing profile: $profile"
done
for path in "$WS_LIVOX_SETUP" "$WS_SLAM_SETUP" "$ROS_SETUP" "$FASTLIO_LAUNCH" "$PROD_SCRIPT" "$PREFLIGHT"; do
  [[ -f "$path" ]] || die "missing required file: $path"
done

source "$ROS_SETUP"
source "$WS_LIVOX_SETUP"

master_ok() { timeout 3 rostopic list >/dev/null 2>&1; }
topic_ok() { timeout 3 rostopic echo -n 2 "$1" >/dev/null 2>&1; }
process_ok() { pgrep -f "$1" >/dev/null 2>&1; }

wait_topic() {
  local topic="$1"
  local attempts="${2:-20}"
  for _ in $(seq 1 "$attempts"); do
    topic_ok "$topic" && return 0
    sleep 2
  done
  return 1
}

start_roscore() {
  local log="$STEP2_ROOT/roscore.log"
  if master_ok; then
    say "roscore already up"
    return 0
  fi
  say "starting roscore -> $log"
  nohup setsid bash -c 'source "$0"; exec roscore' "$ROS_SETUP" \
    >"$log" 2>&1 </dev/null &
  disown
  for _ in $(seq 1 15); do
    master_ok && { say "roscore ready"; return 0; }
    sleep 2
  done
  die "roscore did not come up (log: $log)"
}

start_livox() {
  local log="$STEP2_ROOT/livox_step2.log"
  if [[ "$NO_LIVOX" -eq 1 ]]; then
    say "external odometry mode: NOT starting livox MID360"
    return 0
  fi
  if topic_ok /livox/lidar; then
    say "livox already publishing /livox/lidar"
    return 0
  fi
  if process_ok '[r]oslaunch livox_ros_driver2 msg_MID360s.launch'; then
    say "livox process exists without point data; cycling driver"
    pkill -INT -f '[r]oslaunch livox_ros_driver2 msg_MID360s.launch' || true
    pkill -TERM -f '[l]ivox_ros_driver2_node' || true
    sleep 3
  fi
  if ! process_ok '[r]oslaunch livox_ros_driver2 msg_MID360s.launch'; then
    say "starting livox MID360 -> $log"
    nohup setsid bash -c \
      'source /opt/ros/noetic/setup.bash; source "$0"; exec roslaunch livox_ros_driver2 msg_MID360s.launch' \
      "$WS_LIVOX_SETUP" >"$log" 2>&1 </dev/null &
    disown
  fi
  wait_topic /livox/lidar 25 || die "livox not publishing; check MID360 cable and $log"
  say "livox ready"
}

start_fastlio() {
  local log="$STEP2_ROOT/fastlio_step2.log"
  if [[ -n "$EXTERNAL_ODOM_TOPIC" ]]; then
    say "external odometry mode: NOT starting our FAST-LIO"
    if topic_ok "$EXTERNAL_ODOM_TOPIC"; then
      say "external odometry already publishing $EXTERNAL_ODOM_TOPIC"
      return 0
    fi
    say "waiting for external odometry topic $EXTERNAL_ODOM_TOPIC"
    wait_topic "$EXTERNAL_ODOM_TOPIC" 25 \
      || die "external odometry topic $EXTERNAL_ODOM_TOPIC is not publishing; start the external source first"
    say "external odometry ready"
    return 0
  fi
  if topic_ok /Odometry; then
    say "FAST-LIO already publishing /Odometry"
    return 0
  fi
  if process_ok '[r]oslaunch .*mapping_mid360_step2.launch'; then
    say "FAST-LIO process exists without odometry data; cycling mapper"
    pkill -INT -f '[r]oslaunch .*mapping_mid360_step2.launch' || true
    pkill -TERM -f '[f]astlio_mapping' || true
    sleep 3
  fi
  if ! process_ok '[r]oslaunch .*mapping_mid360_step2.launch'; then
    say "starting FAST-LIO -> $log"
    nohup setsid bash -c \
      'source /opt/ros/noetic/setup.bash; source "$0"; exec roslaunch "$1"' \
      "$WS_SLAM_SETUP" "$FASTLIO_LAUNCH" >"$log" 2>&1 </dev/null &
    disown
  fi
  wait_topic /Odometry 25 || die "FAST-LIO not publishing /Odometry; check $log"
  say "FAST-LIO ready"
}

color_size() {
  timeout 3 rostopic echo -n 1 /camera/color/camera_info 2>/dev/null \
    | awk '/^height:/{h=$2} /^width:/{w=$2} END{if (w) print w "x" h}'
}

start_camera() {
  local log="$STEP2_ROOT/realsense_pausetest.log"
  if topic_ok /camera/color/image_raw; then
    local size
    size="$(color_size)"
    [[ "$size" == "${COLOR_W}x${COLOR_H}" ]] \
      || die "realsense already running at ${size:-unknown size}, wanted ${COLOR_W}x${COLOR_H}; stop it first"
    say "realsense already publishing /camera/color/image_raw ($size)"
    return 0
  fi
  if process_ok '[r]oslaunch realsense2_camera rs_camera.launch'; then
    # Known post-boot failure mode: the process is alive but no image topic.
    # The verified recovery is a full camera-stack cycle, then one retry.
    say "camera process exists without image topic; cycling camera stack"
    pkill -INT -f '[r]oslaunch realsense2_camera rs_camera.launch' || true
    pkill -TERM -f '[r]ealsense2_camera_manager' || true
    sleep 6
  fi
  say "starting realsense (color ${COLOR}, depth 640x480@15, IMU off) -> $log"
  nohup setsid bash -c \
    'source /opt/ros/noetic/setup.bash; exec roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_gyro:=false enable_accel:=false color_width:=$0 color_height:=$1 color_fps:=$2 depth_width:=640 depth_height:=480 depth_fps:=15' \
    "$COLOR_W" "$COLOR_H" "$COLOR_FPS" >"$log" 2>&1 </dev/null &
  disown
  wait_topic /camera/color/image_raw 25 \
    || die "realsense has no image topic; unplug/replug USB and rerun (log: $log)"
  say "realsense ready"
}

set_rgb_exposure() {
  [[ -n "$RGB_EXPOSURE_MS" ]] || return 0
  # D400 RGB exposure is in 100 us units: 5 ms -> 50.
  local units
  units="$(awk -v ms="$RGB_EXPOSURE_MS" 'BEGIN{printf "%d", ms*10+0.5}')"
  say "RGB manual exposure ${RGB_EXPOSURE_MS} ms (${units} x 100us), gain ${RGB_GAIN}"
  rosrun dynamic_reconfigure dynparam set /camera/rgb_camera \
    "{'enable_auto_exposure': False, 'exposure': ${units}, 'gain': ${RGB_GAIN}}" \
    || die "could not set RGB exposure via dynamic_reconfigure /camera/rgb_camera"
  rosrun dynamic_reconfigure dynparam get /camera/rgb_camera \
    | tr ',' '\n' | grep -E "'(enable_auto_exposure|exposure|gain)'" | sort -u || true
}

tf_ok() {
  # tf_echo is an infinite stream: capture 3 s of it and pattern-match the
  # text. SIGINT (not timeout's default SIGTERM, and no `| head`) lets the
  # node unregister; otherwise the master keeps a dead /tf_echo_* entry and
  # every later `rosbag record` logs "XmlRpcClient ... Connection refused".
  local output
  output="$(timeout -s INT 3 rosrun tf tf_echo body camera_color_optical_frame 2>&1 || true)"
  case "$output" in
    *"[0.057, -0.007, -0.078]"*"[-0.494, 0.506, -0.499, 0.501]"*) return 0 ;;
    *"[0.057, -0.007, -0.078]"*"[0.494, -0.506, 0.499, -0.501]"*) return 0 ;;
    *) return 1 ;;
  esac
}

start_body_camera_tf() {
  local log="$STEP2_ROOT/body_camera_tf_$(date +%Y%m%d).log"
  if tf_ok; then
    say "body->camera TF already connected"
    return 0
  fi
  if ! process_ok '[s]tatic_transform_publisher .* body camera_color_optical_frame'; then
    # tf2_ros publishes a latched /tf_static. The old tf publisher repeats it on /tf, which
    # snapshots record as dynamic, so the G4 adapter finds no static extrinsic and drops localization.
    # 2026-10-23 recalibration (docs/CAMERA_EXTRINSIC.md): D435i<->LiDAR registration on three
    # recordings. The previous value (0.329365 0.033964 -0.079647, from FAST-LIVO2 Rcl/Pcl with
    # body==lidar) put the camera ~27 cm too far forward along its optical axis.
    say "starting body->camera TF (2026-10-23 calibration) -> $log"
    nohup setsid bash -c \
      'source /opt/ros/noetic/setup.bash; exec rosrun tf2_ros static_transform_publisher 0.056558 -0.006717 -0.077751 -0.494176 0.506142 -0.498990 0.500618 body camera_color_optical_frame' \
      >"$log" 2>&1 </dev/null &
    disown
  else
    die "existing body->camera publisher has stale/wrong calibration; stop it and rerun"
  fi
  for _ in $(seq 1 8); do
    tf_ok && { say "body->camera TF ready"; return 0; }
    sleep 2
  done
  die "body->camera TF not connected; frame_pose cannot place frames without the body->camera extrinsic"
}

production_running() {
  process_ok '[r]oslaunch drone3d_reconstruction_core continuous_production.launch'
}

start_production() {
  local log="$STEP2_ROOT/production.log"
  if production_running; then
    say "production stack already running; not starting a second one"
    return 0
  fi
  source "$WORKSPACE/devel/setup.bash"
  say "production preflight (ROS master + CUDA + checkpoint)"
  python3 "$PREFLIGHT" --reconstruct-profile "$RECONSTRUCT_PROFILE"

  local web_arg=""
  if [[ "$WEB" -eq 1 ]]; then
    web_arg="--web"
  fi
  export STEP2_ROOT WORKSPACE ROS_SETUP PROD_SCRIPT
  export FRONT_PROFILE_RUNTIME DOWN_PROFILE RECONSTRUCT_PROFILE CONTINUOUS_PROFILE WEB_ARG="$web_arg"
  say "starting production stack -> $log"
  nohup setsid bash -c '
    source "$ROS_SETUP"
    source "$WORKSPACE/devel/setup.bash"
    cd "$STEP2_ROOT"
    exec bash "$PROD_SCRIPT" \
      --front-profile "$FRONT_PROFILE_RUNTIME" \
      --down-profile "$DOWN_PROFILE" \
      --reconstruct-profile "$RECONSTRUCT_PROFILE" \
      --continuous-profile "$CONTINUOUS_PROFILE" \
      $WEB_ARG
  ' >"$log" 2>&1 </dev/null &
  disown
  wait_topic /reconstruct/reconstruct_segment/status 20 \
    || die "production stack did not come up; see $log"
  say "production stack ready"
}

start_roscore
start_livox
start_fastlio
start_camera
set_rgb_exposure
start_body_camera_tf
if [[ "$PRODUCTION" -eq 0 ]]; then
  say "sensors + TF up; production stack not started (--no-production)"
  exit 0
fi
start_production

say "bringup complete"
say "Web UI (on NX): http://127.0.0.1:8080"
say "Web UI (from dev): ssh -L 8080:127.0.0.1:8080 <nx-user>@<nx-host>"
say "logs: $STEP2_ROOT/{roscore,livox_step2,fastlio_step2,realsense_pausetest,body_camera_tf_*,production}.log"

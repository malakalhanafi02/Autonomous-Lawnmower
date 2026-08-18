#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Record a 3-pane timelapse (Gazebo + RViz + camera image) from X11 windows.

Usage:
  record_sim_timelapse.sh [options]

Options:
  --output <file>            Output video path (.mp4)
  --display <display>        X11 display (default: $DISPLAY)
  --gazebo-pattern <regex>   Window title regex for Gazebo
  --rviz-pattern <regex>     Window title regex for RViz
  --camera-pattern <regex>   Window title regex for camera viewer
  --capture-fps <n>          Input capture FPS before timelapse speed-up
  --output-fps <n>           Output video FPS
  --speed <n>                Timelapse speed factor (12 means 12x faster)
  --duration <s>             Optional recording duration in seconds (0 = until Ctrl+C)
  --help                     Show this help

Example:
  rosrun rqt_image_view rqt_image_view /mower_camera/camera/image_raw
  ./record_sim_timelapse.sh --speed 16 --duration 300
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

find_window_id() {
  local pattern="$1"
  xdotool search --name "$pattern" 2>/dev/null | head -n 1 || true
}

window_geometry() {
  local win_id="$1"
  xwininfo -id "$win_id" | awk '
    /Absolute upper-left X:/ {x=$4}
    /Absolute upper-left Y:/ {y=$4}
    /^  Width:/ {w=$2}
    /^  Height:/ {h=$2}
    END { printf "%d %d %d %d\n", x, y, w, h }
  '
}

timestamp="$(date +%Y%m%d_%H%M%S)"
display="${DISPLAY:-:0.0}"
output="${HOME}/Videos/mower_timelapse_${timestamp}.mp4"
gazebo_pattern='Gazebo|gzclient'
rviz_pattern='rviz|RViz'
camera_pattern='rqt_image_view|Image View|camera/image_raw'
capture_fps=15
output_fps=30
speed=12
duration=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --display) display="$2"; shift 2 ;;
    --gazebo-pattern) gazebo_pattern="$2"; shift 2 ;;
    --rviz-pattern) rviz_pattern="$2"; shift 2 ;;
    --camera-pattern) camera_pattern="$2"; shift 2 ;;
    --capture-fps) capture_fps="$2"; shift 2 ;;
    --output-fps) output_fps="$2"; shift 2 ;;
    --speed) speed="$2"; shift 2 ;;
    --duration) duration="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_cmd ffmpeg
require_cmd xdotool
require_cmd xwininfo

gazebo_id="$(find_window_id "$gazebo_pattern")"
rviz_id="$(find_window_id "$rviz_pattern")"
camera_id="$(find_window_id "$camera_pattern")"

if [[ -z "$gazebo_id" || -z "$rviz_id" || -z "$camera_id" ]]; then
  echo "Could not find all required windows." >&2
  echo "Gazebo pattern: ${gazebo_pattern} -> ${gazebo_id:-NOT FOUND}" >&2
  echo "RViz pattern:   ${rviz_pattern} -> ${rviz_id:-NOT FOUND}" >&2
  echo "Camera pattern: ${camera_pattern} -> ${camera_id:-NOT FOUND}" >&2
  echo "Tip: run camera viewer with:" >&2
  echo "  rosrun rqt_image_view rqt_image_view /mower_camera/camera/image_raw" >&2
  exit 1
fi

read -r gx gy gw gh <<<"$(window_geometry "$gazebo_id")"
read -r rx ry rw rh <<<"$(window_geometry "$rviz_id")"
read -r cx cy cw ch <<<"$(window_geometry "$camera_id")"

mkdir -p "$(dirname "$output")"

duration_args=()
if [[ "$duration" != "0" ]]; then
  duration_args=(-t "$duration")
fi

echo "Recording timelapse to: $output"
echo "Gazebo window: id=$gazebo_id geom=${gw}x${gh}+${gx},${gy}"
echo "RViz window:   id=$rviz_id geom=${rw}x${rh}+${rx},${ry}"
echo "Camera window: id=$camera_id geom=${cw}x${ch}+${cx},${cy}"
echo "Press Ctrl+C to stop (unless --duration was set)."

ffmpeg -y \
  -f x11grab -framerate "$capture_fps" -video_size "${gw}x${gh}" -i "${display}+${gx},${gy}" \
  -f x11grab -framerate "$capture_fps" -video_size "${rw}x${rh}" -i "${display}+${rx},${ry}" \
  -f x11grab -framerate "$capture_fps" -video_size "${cw}x${ch}" -i "${display}+${cx},${cy}" \
  -filter_complex "\
[0:v]setpts=PTS/${speed},fps=${output_fps},scale=960:540[gz];\
[1:v]setpts=PTS/${speed},fps=${output_fps},scale=960:540[rv];\
[2:v]setpts=PTS/${speed},fps=${output_fps},scale=1920:540[cam];\
[gz][rv][cam]xstack=inputs=3:layout=0_0|w0_0|0_h0[v]" \
  -map "[v]" \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -crf 22 \
  -pix_fmt yuv420p \
  "${duration_args[@]}" \
  "$output"

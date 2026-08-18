# Jetson Nano Bringup

This package launches a working baseline for:
- Manual control via `/cmd_vel`
- Safety watchdog stop at 500 ms command loss
- LiDAR SLAM (`gmapping` or `hector`)
- AI-assisted pre-planning of cut regions

## 1) Dependencies on Jetson (ROS1 Noetic)

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-gmapping \
  ros-noetic-hector-mapping \
  ros-noetic-rplidar-ros \
  ros-noetic-teleop-twist-keyboard \
  ros-noetic-cv-bridge \
  ros-noetic-tf2-ros \
  ros-noetic-tf2-geometry-msgs \
  python3-serial

pip3 install ultralytics
```

## 2) Build

```bash
cd ~/autonomous_mower/mower_ws
catkin_make
source devel/setup.bash
```

## 3) Bringup

```bash
roslaunch mower_bringup jetson_nano_bringup.launch serial_port:=/dev/ttyACM0 serial_baud:=115200

# Typical RPLIDAR C1M1-R2 setup:
roslaunch mower_bringup jetson_nano_bringup.launch \
  serial_port:=/dev/ttyACM0 serial_baud:=115200 \
  lidar_serial_port:=/dev/ttyUSB0 lidar_serial_baudrate:=460800
```

Optional backend switch:

```bash
roslaunch mower_bringup jetson_nano_bringup.launch slam_backend:=hector
```

If you want to run without the physical LiDAR node:

```bash
roslaunch mower_bringup jetson_nano_bringup.launch use_rplidar:=false
```

Full cohesive autonomy stack (SLAM + EKF + cut planner + segmentation adapter + obstacle safety + cmd arbitration):

```bash
roslaunch mower_bringup full_autonomy.launch \
  use_mega:=true serial_port:=/dev/ttyACM0 serial_baud:=115200 \
  use_rplidar:=true lidar_serial_port:=/dev/ttyUSB0 lidar_serial_baudrate:=460800 \
  use_rtk:=true use_segmentation_adapter:=true \
  mode:=manual autonomy_enabled:=false start_corner:=lower_left
```

Legacy teleop compatibility (optional):

```bash
# Only needed for tools that publish to /cmd_vel instead of /cmd_vel_manual.
# Uses a separate legacy topic to avoid /cmd_vel feedback loops.
roslaunch mower_bringup full_autonomy.launch \
  use_legacy_cmd_vel_relay:=true \
  legacy_cmd_vel_input_topic:=/cmd_vel
```

Run from a previously saved map (skip SLAM):

```bash
roslaunch mower_bringup full_autonomy.launch \
  map_source:=saved \
  saved_map_yaml:=$(rospack find mower_bringup)/maps/latest_scan.yaml
```

Simulation with the same autonomy stack:

```bash
roslaunch mower_bringup sim_full_stack.launch gui:=true
```

Two-phase simulation workflow (recommended for repeatable trials):

```bash
# Phase 1: scan + build map with SLAM (manual driving)
roslaunch mower_bringup sim_phase1_scan.launch gui:=true
# Also loads the current saved map on /map_saved for RViz overlay comparison.
```

In a second terminal (after driving the exploration route):

```bash
# Save/overwrite reusable scan map at mower_bringup/maps/latest_scan.{yaml,pgm}
rosrun mower_bringup mower_trial_workflow.py save-map

# Optional: merge this run into the existing saved map (iterative map refinement)
rosrun mower_bringup mower_trial_workflow.py save-map --merge-existing
# Conflict mode options: occupied_wins (default), newest, average
# If you intentionally want to replace with a sparse map:
rosrun mower_bringup mower_trial_workflow.py save-map --force-overwrite
```

Then start cut phase from saved static map:

```bash
# Phase 2: load saved map as read-only /map (no SLAM map persistence)
roslaunch mower_bringup sim_phase2_cut.launch gui:=true
# Optional: allow live SLAM updates in phase-2 (off by default to protect baseline view)
roslaunch mower_bringup sim_phase2_cut.launch gui:=true enable_live_map_updates:=true

# Generate path once from saved map
rosrun mower_bringup mower_trial_workflow.py build-plan

# Start auto-cut
rosrun mower_bringup mower_trial_workflow.py start-cut
```

Stop auto-cut:

```bash
rosrun mower_bringup mower_trial_workflow.py stop-cut
```

Map persistence behavior:
- `save-map` now guards against accidental wipe: it refuses overwrite when known area drops too far.
- Use `save-map --merge-existing` to apply only new observations onto the saved baseline.
- Use `save-map --force-overwrite` only when you intentionally want full replacement.
- Phase 2 uses `map_server` with the saved YAML/PGM, so runtime map changes do not write back to disk.
- Re-scan + `save-map` when you want to replace the reusable baseline map.

This simulation includes:
- bounded lawn fences
- a concrete patch (blade auto-off over that region)
- randomized fixed obstacles for avoidance testing

Start corner options for coverage planning:
- `lower_left`
- `lower_right`
- `upper_left`
- `upper_right`

## 4) Manual Drive

In a second terminal:

```bash
cd ~/autonomous_mower/mower_ws
source devel/setup.bash
rosrun mower_bringup manual_drive_teleop.py
```

This manual teleop:
- publishes only to `/cmd_vel_manual`
- continuously republishes the last command (so motion holds)
- supports explicit turn-only keys (`a/d` and `j/l`)
- forces `mode=manual` and `autonomy_enabled=false` while it is running
- auto-clears motion after 1.0s and ignores key auto-repeat while held (prevents latching)
- drains keyboard input buffer each cycle (uses latest key only)

If you want the stock keyboard teleop instead, use:

```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel_manual _repeat_rate:=20 _key_timeout:=0.3
```

Switch to autonomy at runtime:

```bash
rostopic pub -1 /mower/mode std_msgs/String "data: 'auto'"
rostopic pub -1 /mower/autonomy_enabled std_msgs/Bool "data: true"
```

When autonomy is enabled:
- mission manager automatically calls `/cut_region_planner/build_plan`
- follower executes the coverage path
- follower returns to the mission start point when coverage is complete
- blade guard publishes `/mower/cutter_enabled` and controls cutter speed based on camera no-cut polygons

Return to manual:

```bash
rostopic pub -1 /mower/autonomy_enabled std_msgs/Bool "data: false"
rostopic pub -1 /mower/mode std_msgs/String "data: 'manual'"
```

## 5) Arduino Mega Command Contract

The Jetson serial bridge sends ASCII lines:

```text
V:<velocity_mps>,A:<steering_rad>
```

Example:

```text
V:0.250,A:-0.120
```

Stop command (watchdog or shutdown):

```text
V:0.000,A:0.000
```

Firmware-side safety also applies:
- ultrasonic hard stop under stop distance
- ultrasonic slow zone before stop distance
- one-shot timeout stop when commands are stale

Bridge publishes hardware safety status:
- `jetson_mega_bridge/ultrasonic_distance_cm`
- `jetson_mega_bridge/collision_blocked`

## 6) Camera Obstacle Detection

The `camera_detector` node runs YOLOv8 inference on camera images and
publishes obstacle positions for the cut-region planner and a safety
alert flag.

```bash
# Bringup includes the detector by default.  Override model path:
roslaunch mower_bringup jetson_nano_bringup.launch model_path:=/home/jetson/models/best.pt

# Disable detector if running without a camera:
roslaunch mower_bringup jetson_nano_bringup.launch enable_detector:=false
```

Topics published:

| Topic | Type | Purpose |
|---|---|---|
| `/camera/detections_json` | `std_msgs/String` | JSON with `no_cut_zones` and `detections` |
| `/camera/detection_image` | `sensor_msgs/Image` | Annotated image for visualization |
| `/camera/obstacle_alert`  | `std_msgs/Bool` | `true` when Person or Dog within 2 m |

Detection classes: Person, Dog, Tree, Bicycle, Electric pole, Uncovered manhole.

## 7) AI Cut Region Planning

Planner node:
- Subscribes map: `/map`
- Subscribes detection JSON: `/camera/detections_json`
- Publishes coverage path: `/mower_ai/cut_plan`
- Service to generate plan: `/cut_region_planner/build_plan`

Detection payload format (auto-generated by `camera_detector`):

```json
{
  "no_cut_zones": [
    [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 2.0}, {"x": 2.0, "y": 3.0}]
  ],
  "detections": [
    {"class": "Tree", "confidence": 0.85, "position": {"x": 3.1, "y": 1.2}, "distance_m": 3.33}
  ]
}
```

If your segmentation model already outputs polygons as JSON, publish to:

```text
/camera/segmentation_zones_json
```

The segmentation adapter converts that to:

```text
/camera/detections_json
```

for the cut planner.

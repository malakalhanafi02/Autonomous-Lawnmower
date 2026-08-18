# System Architecture — Autonomous Lawn Mower (MSE 4499)

This repo contains the **AI-perception module** of a larger capstone project: a
boundary-free autonomous lawn mower built on **ROS 1 (Noetic)**, validated in
**Gazebo** + **RViz**, targeting a **Jetson Nano** (compute) + **Arduino Mega**
(motor control). The project earned an **Honourable Mention at Western
Engineering Design Day**.

**Goal:** map a yard, plan an efficient mowing path, avoid people/pets/obstacles,
and only cut where it's safe — without buried perimeter wires. Target
requirements: localization error ≤ 10 cm, operating speed < 1 m/s, fail-safe
shutdown, obstacle awareness.

## The full pipeline

```
 CAMERA ──► AI PERCEPTION (YOLOv8) ──► no-cut zones + obstacle alerts
 LiDAR  ──► SLAM (map) ─┐
 IMU/GPS/odom ──► EKF (pose) ─┴─► COVERAGE PLANNER ──► path (waypoints)
                                        │
                        PATH FOLLOWER ──► /cmd_vel ──► SAFETY FILTER ──► MOTOR BRIDGE ──► Arduino/motors
                                                          ▲
                              LiDAR + ultrasonic ─────────┘ (last line of defense)
```

Independent ROS nodes communicate over topics/services so a failure in one
subsystem can be isolated and fixed without touching the rest.

## Where each piece lives in this repo

- **Perception — `src/mower_perception/`:** two YOLOv8 models — object
  **detection** (people/pets/obstacles) and lawn **segmentation** (cuttable
  grass vs. boundary/non-grass). Trained on a custom, hand-labeled dataset.
  In the ROS graph, `mower_ws/src/mower_ai/scripts/camera_detector.py` and
  `segmentation_adapter.py` run these models on the live camera feed and
  publish detections as "no-cut" polygons for the planner — used
  **conservatively** (cautious hints, not exact boundaries), because in a
  safety-critical system it's better to be slightly over-cautious than to
  trust an imperfect mask.
- **Localization & mapping — `mower_navigation/`:** LiDAR SLAM (`slam.launch`,
  gmapping) produces an occupancy grid; `ekf.launch` fuses wheel odometry +
  IMU + GPS (or RTK GPS via `ekf_rtk.launch`) into one pose estimate — this
  fusion is what gets localization under the 10 cm target, since no single
  sensor is accurate enough alone.
- **Coverage planning — `mower_ai/scripts/cut_region_planner.py`:** a
  boustrophedon ("back-and-forth") sweep over the occupancy map, skipping
  occupied cells and AI no-cut zones via a point-in-polygon test, producing a
  waypoint path. `mower_navigation/scripts/coverage_path_follower.py` (and a
  `_v2` revision) executes it, arbitrated by `coverage_mission_manager.py`.
- **Motion control — `mower_hardware_interface/scripts/jetson_mega_bridge.py`:**
  a Jetson↔Arduino serial bridge that converts `/cmd_vel` into motor commands,
  clamped to a max speed, with a **500 ms watchdog** — if commands stop
  arriving, the mower halts instead of running away.
  `mower_navigation/scripts/cmd_vel_arbiter.py` arbitrates between autonomous
  and manual command sources before it reaches the bridge.
- **Safety (defense in depth) — `mower_navigation/scripts/`:** the command
  watchdog, `scan_safety_filter.py` (LiDAR + ultrasonic, with hysteresis to
  avoid chattering between blocked/unblocked), `scan_self_filter.py`
  (distinguishing real obstacles from the robot's own body/self-hits), and
  `surface_blade_guard.py` (cutter gating — only enables the blade when the
  mission state is valid, the mower is outside no-cut zones, *and* the camera
  confirms grass-like pixels underneath).
- **Simulation & validation — `mower_sim/`, `mower_bringup/launch/`:** each
  subsystem was validated independently in Gazebo/RViz
  (`sim_phase1_scan.launch`, `sim_phase2_cut.launch`) before being combined
  into a full-stack autonomy trial (`sim_full_stack.launch`,
  `full_autonomy.launch`), so failures could be isolated rather than debugged
  all at once.

## Results

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Obstacle detection | YOLOv8 | box mAP@0.5 | **0.764** |
| Lawn segmentation | YOLOv8-seg | mask mAP@0.5 | **0.745** |

## Notable engineering problems (and fixes)

- **Waypoint over-advancement during turns** caused uneven coverage rows —
  fixed by refining heading thresholds and waypoint-advancement logic in the
  path follower.
- **Safety-filter chattering** (rapidly toggling blocked/unblocked) — fixed
  with hysteresis so the filter commits to a state.
- **False obstacles from self-hits** — LiDAR occasionally picked up the
  robot's own body or scan artifacts as obstacles; the safety logic had to
  distinguish real obstacles from these.
- **Cutter flicker** from a naive color check — fixed by fusing mission state
  with visual validation instead of trusting a single frame.

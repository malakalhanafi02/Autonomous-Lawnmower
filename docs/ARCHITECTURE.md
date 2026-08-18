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

## Where this repo fits

- **Perception (this repo):** two YOLOv8 models — object **detection**
  (people/pets/obstacles) and lawn **segmentation** (cuttable grass vs.
  boundary/non-grass). Trained on a custom, hand-labeled dataset. Detections
  are published as "no-cut" polygons that the planner consumes, and used
  **conservatively** — treated as cautious hints rather than exact boundaries,
  because in a safety-critical system it's better to be slightly over-cautious
  than to trust an imperfect mask.
- **Localization & mapping:** LiDAR SLAM (gmapping / hector_mapping) produces
  an occupancy grid; an Extended Kalman Filter fuses wheel odometry + IMU +
  GPS into one pose estimate — this fusion is what gets localization under the
  10 cm target, since no single sensor is accurate enough alone.
- **Coverage planning:** a boustrophedon ("back-and-forth") sweep over the
  occupancy map, skipping occupied cells and AI no-cut zones via a
  point-in-polygon test, producing a waypoint path.
- **Motion control:** a Jetson↔Arduino serial bridge converts `/cmd_vel` into
  motor commands, clamped to a max speed, with a **500 ms watchdog** — if
  commands stop arriving, the mower halts instead of running away.
- **Safety (defense in depth):** the command watchdog, a LiDAR + ultrasonic
  safety filter (with hysteresis to avoid chattering between blocked/unblocked),
  and cutter gating that only enables the blade when the mission state is
  valid, the mower is outside no-cut zones, *and* the camera confirms
  grass-like pixels underneath.
- **Simulation & validation:** each subsystem was validated independently in
  Gazebo/RViz before being combined into a full-stack autonomy trial, so
  failures could be isolated rather than debugged all at once.

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

## Full report

The complete capstone report (system design, mechanical/electrical details,
full validation results) isn't included here to keep this repo focused and
free of non-public project material — this repo is the standalone, runnable
extract of the perception module.

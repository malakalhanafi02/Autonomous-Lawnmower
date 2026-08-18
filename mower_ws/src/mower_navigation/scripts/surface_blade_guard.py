#!/usr/bin/env python3
import json
import math
import os

import rospy
try:
    import cv2
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover
    cv2 = None
    CvBridge = None

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, String


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SurfaceBladeGuard:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")
        self.detections_topic = rospy.get_param("~detections_topic", "/camera/detections_json")
        self.enabled_topic = rospy.get_param("~enabled_topic", "/mower/autonomy_enabled")
        self.cut_cycle_topic = rospy.get_param("~cut_cycle_topic", "/mower/cut_cycle_active")
        self.require_cut_cycle = _as_bool(rospy.get_param("~require_cut_cycle", True))
        self.cut_cycle_timeout_s = float(rospy.get_param("~cut_cycle_timeout_s", 1.0))
        self.motion_cmd_topic = rospy.get_param("~motion_cmd_topic", "/cmd_vel")
        self.motion_cmd_timeout_s = float(rospy.get_param("~motion_cmd_timeout_s", 0.6))
        self.min_cmd_linear_for_lookahead_mps = float(
            rospy.get_param("~min_cmd_linear_for_lookahead_mps", 0.02)
        )
        self.camera_roi_grass_gate_enabled = _as_bool(
            rospy.get_param("~camera_roi_grass_gate_enabled", True)
        )
        self.camera_roi_image_topic = rospy.get_param(
            "~camera_roi_image_topic", "/camera/image_raw"
        )
        self.camera_roi_timeout_s = float(rospy.get_param("~camera_roi_timeout_s", 0.8))
        self.camera_roi_min_grass_ratio = float(
            rospy.get_param("~camera_roi_min_grass_ratio", 0.90)
        )
        self.camera_roi_x_min_frac = float(rospy.get_param("~camera_roi_x_min_frac", 0.0))
        self.camera_roi_x_max_frac = float(rospy.get_param("~camera_roi_x_max_frac", 1.0))
        self.camera_roi_y_min_frac = float(rospy.get_param("~camera_roi_y_min_frac", 0.80))
        self.camera_roi_y_max_frac = float(rospy.get_param("~camera_roi_y_max_frac", 1.0))
        self.camera_roi_h_min = int(rospy.get_param("~camera_roi_h_min", 30))
        self.camera_roi_h_max = int(rospy.get_param("~camera_roi_h_max", 95))
        self.camera_roi_s_min = int(rospy.get_param("~camera_roi_s_min", 35))
        self.camera_roi_s_max = int(rospy.get_param("~camera_roi_s_max", 255))
        self.camera_roi_v_min = int(rospy.get_param("~camera_roi_v_min", 30))
        self.camera_roi_v_max = int(rospy.get_param("~camera_roi_v_max", 255))
        self.non_grass_snapshot_enabled = _as_bool(
            rospy.get_param("~non_grass_snapshot_enabled", True)
        )
        self.non_grass_snapshot_interval_s = float(
            rospy.get_param("~non_grass_snapshot_interval_s", 1.0)
        )
        self.non_grass_snapshot_dir = rospy.get_param(
            "~non_grass_snapshot_dir", "~/.ros/non_grass_snapshots"
        )
        self.grass_hold_during_turn = _as_bool(rospy.get_param("~grass_hold_during_turn", True))
        self.grass_hold_turn_min_angular_rps = float(
            rospy.get_param("~grass_hold_turn_min_angular_rps", 0.35)
        )
        self.grass_hold_turn_max_linear_mps = float(
            rospy.get_param("~grass_hold_turn_max_linear_mps", 0.10)
        )
        self.grass_hold_max_s = float(rospy.get_param("~grass_hold_max_s", 6.0))

        self.output_enabled_topic = rospy.get_param("~output_enabled_topic", "/mower/cutter_enabled")
        self.output_reason_topic = rospy.get_param("~output_reason_topic", "/mower/cutter_reason")
        self.command_topic = rospy.get_param("~command_topic", "/cutter_controller/command")

        self.lookahead_m = float(rospy.get_param("~lookahead_m", 0.8))
        self.blade_speed_on = float(rospy.get_param("~blade_speed_on", 45.0))
        self.blade_speed_off = float(rospy.get_param("~blade_speed_off", 0.0))
        self.command_cutter = _as_bool(rospy.get_param("~command_cutter", True))
        self.disable_in_manual = _as_bool(rospy.get_param("~disable_in_manual", False))
        self.fail_closed_if_no_detection = _as_bool(
            rospy.get_param("~fail_closed_if_no_detection", False)
        )
        self.loop_hz = float(rospy.get_param("~loop_hz", 20.0))

        self.autonomy_enabled = False
        self._autonomy_prev = False
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.no_cut_polygons = []
        self.last_detection_stamp = rospy.Time(0)
        self.cut_cycle_active = False
        self.cut_cycle_stamp = rospy.Time(0)
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.cmd_stamp = rospy.Time(0)
        self.last_surface_grass_ok = False
        self.grass_hold_active = False
        self.grass_hold_started = rospy.Time(0)
        self.camera_roi_grass_ok = True
        self.camera_roi_grass_ratio = 1.0
        self.camera_roi_stamp = rospy.Time(0)
        self.camera_bridge = None
        self.latest_camera_image = None
        self.latest_camera_image_stamp = rospy.Time(0)
        self.non_grass_last_snapshot = rospy.Time(0)
        self.non_grass_snapshot_idx = 0

        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.detections_topic, String, self._detections_cb, queue_size=10)
        rospy.Subscriber(self.enabled_topic, Bool, self._enabled_cb, queue_size=5)
        rospy.Subscriber(self.cut_cycle_topic, Bool, self._cut_cycle_cb, queue_size=10)
        rospy.Subscriber(self.motion_cmd_topic, Twist, self._cmd_cb, queue_size=20)
        if self.camera_roi_grass_gate_enabled:
            if cv2 is None or CvBridge is None:
                rospy.logwarn(
                    "surface_blade_guard: OpenCV/cv_bridge unavailable, disabling camera ROI grass gate."
                )
                self.camera_roi_grass_gate_enabled = False
            else:
                self.camera_bridge = CvBridge()
                rospy.Subscriber(
                    self.camera_roi_image_topic, Image, self._camera_image_cb, queue_size=2
                )
        if self.non_grass_snapshot_enabled:
            if cv2 is None:
                rospy.logwarn(
                    "surface_blade_guard: OpenCV unavailable, disabling non-grass snapshots."
                )
                self.non_grass_snapshot_enabled = False
            else:
                self._prepare_non_grass_snapshot_dir(clear=False)
        self.camera_roi_ok_pub = rospy.Publisher("/mower/camera_roi_grass_ok", Bool, queue_size=10)
        self.camera_roi_ratio_pub = rospy.Publisher("/mower/camera_roi_grass_ratio", Float64, queue_size=10)

        self.enabled_pub = rospy.Publisher(self.output_enabled_topic, Bool, queue_size=10)
        self.reason_pub = rospy.Publisher(self.output_reason_topic, String, queue_size=10)
        self.command_pub = rospy.Publisher(self.command_topic, Float64, queue_size=10)

    def _enabled_cb(self, msg):
        now_enabled = bool(msg.data)
        if now_enabled and not self._autonomy_prev:
            # New run start: clear previous run snapshots.
            self._prepare_non_grass_snapshot_dir(clear=True)
            self.non_grass_last_snapshot = rospy.Time(0)
        self.autonomy_enabled = now_enabled
        self._autonomy_prev = now_enabled

    def _cut_cycle_cb(self, msg):
        self.cut_cycle_active = msg.data
        self.cut_cycle_stamp = rospy.Time.now()

    def _cmd_cb(self, msg):
        self.cmd_linear = float(msg.linear.x)
        self.cmd_angular = float(msg.angular.z)
        self.cmd_stamp = rospy.Time.now()

    def _odom_cb(self, msg):
        self.have_odom = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = _yaw_from_quat(msg.pose.pose.orientation)

    def _detections_cb(self, msg):
        self.last_detection_stamp = rospy.Time.now()
        try:
            payload = json.loads(msg.data)
            polygons = payload.get("no_cut_zones", [])
            parsed = []
            for poly in polygons:
                pts = []
                for p in poly:
                    pts.append((float(p["x"]), float(p["y"])))
                if len(pts) >= 3:
                    parsed.append(pts)
            self.no_cut_polygons = parsed
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Surface blade guard: detection parse failed: %s", exc)

    def _camera_image_cb(self, msg):
        if not self.camera_roi_grass_gate_enabled or self.camera_bridge is None:
            return
        try:
            img = self.camera_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_camera_image = img
            self.latest_camera_image_stamp = rospy.Time.now()
            h, w = img.shape[:2]
            if h < 8 or w < 8:
                return

            x0 = max(0, min(w - 1, int(self.camera_roi_x_min_frac * w)))
            x1 = max(x0 + 1, min(w, int(self.camera_roi_x_max_frac * w)))
            y0 = max(0, min(h - 1, int(self.camera_roi_y_min_frac * h)))
            y1 = max(y0 + 1, min(h, int(self.camera_roi_y_max_frac * h)))
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                return

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                (self.camera_roi_h_min, self.camera_roi_s_min, self.camera_roi_v_min),
                (self.camera_roi_h_max, self.camera_roi_s_max, self.camera_roi_v_max),
            )
            grass_ratio = float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])
            self.camera_roi_grass_ratio = grass_ratio
            self.camera_roi_grass_ok = grass_ratio >= self.camera_roi_min_grass_ratio
            self.camera_roi_stamp = rospy.Time.now()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Surface blade guard: camera ROI parse failed: %s", exc)

    def _prepare_non_grass_snapshot_dir(self, clear):
        if not self.non_grass_snapshot_enabled:
            return
        path = os.path.expanduser(self.non_grass_snapshot_dir)
        try:
            os.makedirs(path, exist_ok=True)
            if clear:
                for name in os.listdir(path):
                    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    fp = os.path.join(path, name)
                    if os.path.isfile(fp):
                        os.remove(fp)
                self.non_grass_snapshot_idx = 0
            self.non_grass_snapshot_dir = path
        except Exception as exc:
            rospy.logwarn("Surface blade guard: snapshot dir setup failed: %s", exc)
            self.non_grass_snapshot_enabled = False

    def _maybe_save_non_grass_snapshot(self, reason):
        if not self.non_grass_snapshot_enabled:
            return
        if reason != "camera_roi_not_grass":
            return
        if self.latest_camera_image is None:
            return
        now = rospy.Time.now()
        if self.latest_camera_image_stamp == rospy.Time(0):
            return
        if (now - self.latest_camera_image_stamp).to_sec() > self.camera_roi_timeout_s:
            return
        if self.non_grass_last_snapshot != rospy.Time(0):
            if (now - self.non_grass_last_snapshot).to_sec() < self.non_grass_snapshot_interval_s:
                return
        ts = now.to_sec()
        fname = "non_grass_{:06d}_{:.3f}.png".format(self.non_grass_snapshot_idx, ts)
        self.non_grass_snapshot_idx += 1
        fp = os.path.join(self.non_grass_snapshot_dir, fname)
        try:
            cv2.imwrite(fp, self.latest_camera_image)
            self.non_grass_last_snapshot = now
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Surface blade guard: snapshot write failed: %s", exc)

    @staticmethod
    def _point_in_poly(x, y, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            denom = y2 - y1
            if abs(denom) < 1e-12:
                continue
            intersects = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / denom + x1
            )
            if intersects:
                inside = not inside
        return inside

    def _in_no_cut(self, x, y):
        for poly in self.no_cut_polygons:
            if self._point_in_poly(x, y, poly):
                return True
        return False

    def _compute_enable_and_reason(self):
        now = rospy.Time.now()
        if not self.have_odom:
            return False, "no_odom"

        if self.disable_in_manual and not self.autonomy_enabled:
            return False, "manual_mode"

        if self.require_cut_cycle:
            if self.cut_cycle_stamp == rospy.Time(0):
                return False, "not_on_cut_cycle"
            if (now - self.cut_cycle_stamp).to_sec() > self.cut_cycle_timeout_s:
                return False, "cut_cycle_timeout"
            if not self.cut_cycle_active:
                return False, "not_on_cut_cycle"

        if self.fail_closed_if_no_detection and self.last_detection_stamp == rospy.Time(0):
            return False, "no_surface_data"

        look_heading = self.yaw
        cmd_fresh = (
            self.cmd_stamp != rospy.Time(0)
            and (now - self.cmd_stamp).to_sec() <= self.motion_cmd_timeout_s
        )
        if cmd_fresh and abs(self.cmd_linear) >= self.min_cmd_linear_for_lookahead_mps:
            if self.cmd_linear < 0.0:
                look_heading = self.yaw + math.pi
        look_x = self.x + self.lookahead_m * math.cos(look_heading)
        look_y = self.y + self.lookahead_m * math.sin(look_heading)
        map_surface_grass_ok = not (
            self._in_no_cut(self.x, self.y) or self._in_no_cut(look_x, look_y)
        )
        camera_surface_grass_ok = True
        if self.camera_roi_grass_gate_enabled:
            if self.camera_roi_stamp == rospy.Time(0):
                return False, "camera_roi_unavailable"
            if (now - self.camera_roi_stamp).to_sec() > self.camera_roi_timeout_s:
                return False, "camera_roi_timeout"
            camera_surface_grass_ok = self.camera_roi_grass_ok
        surface_grass_ok = map_surface_grass_ok and camera_surface_grass_ok

        turning_now = (
            cmd_fresh
            and abs(self.cmd_angular) >= self.grass_hold_turn_min_angular_rps
            and abs(self.cmd_linear) <= self.grass_hold_turn_max_linear_mps
        )
        if self.grass_hold_during_turn and turning_now:
            if not self.grass_hold_active and self.last_surface_grass_ok:
                self.grass_hold_active = True
                self.grass_hold_started = now
            if self.grass_hold_active:
                if (now - self.grass_hold_started).to_sec() <= self.grass_hold_max_s:
                    return True, "grass_hold_turn"
                self.grass_hold_active = False
        else:
            self.grass_hold_active = False

        self.last_surface_grass_ok = surface_grass_ok
        if not surface_grass_ok:
            if not map_surface_grass_ok:
                return False, "no_cut_surface"
            return False, "camera_roi_not_grass"
        return True, "grass_ok"

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            enabled, reason = self._compute_enable_and_reason()
            self.camera_roi_ok_pub.publish(Bool(data=self.camera_roi_grass_ok))
            self.camera_roi_ratio_pub.publish(Float64(data=self.camera_roi_grass_ratio))
            self._maybe_save_non_grass_snapshot(reason)
            self.enabled_pub.publish(Bool(data=enabled))
            self.reason_pub.publish(String(data=reason))
            if self.command_cutter:
                self.command_pub.publish(Float64(data=self.blade_speed_on if enabled else self.blade_speed_off))
            rate.sleep()


def main():
    rospy.init_node("surface_blade_guard")
    SurfaceBladeGuard().spin()


if __name__ == "__main__":
    main()

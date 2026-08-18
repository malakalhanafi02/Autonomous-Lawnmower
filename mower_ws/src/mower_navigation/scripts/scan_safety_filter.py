#!/usr/bin/env python3
import math

import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class ScanSafetyFilter:
    def __init__(self):
        self.cmd_in_topic = rospy.get_param("~cmd_in_topic", "/cmd_vel_raw")
        self.cmd_out_topic = rospy.get_param("~cmd_out_topic", "/cmd_vel")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.use_ultrasonic_front = _as_bool(rospy.get_param("~use_ultrasonic_front", True))
        self.ultrasonic_topic = rospy.get_param(
            "~ultrasonic_topic", "/jetson_mega_bridge/ultrasonic_distance_cm"
        )
        self.ultrasonic_scale = float(rospy.get_param("~ultrasonic_scale", 0.01))
        self.ultrasonic_timeout_s = float(rospy.get_param("~ultrasonic_timeout_s", 0.6))
        self.ultrasonic_stop_distance_m = float(
            rospy.get_param("~ultrasonic_stop_distance_m", 0.55)
        )
        self.ultrasonic_clear_distance_m = float(
            rospy.get_param(
                "~ultrasonic_clear_distance_m", self.ultrasonic_stop_distance_m + 0.20
            )
        )
        self.ultrasonic_clear_distance_m = max(
            self.ultrasonic_clear_distance_m, self.ultrasonic_stop_distance_m + 0.01
        )
        self.ultrasonic_reverse_speed_mps = float(
            rospy.get_param("~ultrasonic_reverse_speed_mps", 0.18)
        )
        self.ultrasonic_turn_rate_rps = float(
            rospy.get_param("~ultrasonic_turn_rate_rps", 0.9)
        )

        self.stop_distance_m = float(rospy.get_param("~stop_distance_m", 0.55))
        self.slow_distance_m = float(rospy.get_param("~slow_distance_m", 1.2))
        self.forward_fov_deg = float(rospy.get_param("~forward_fov_deg", 70.0))
        self.avoid_turn_rate = float(rospy.get_param("~avoid_turn_rate", 0.8))
        self.blocked_reverse_speed_mps = float(
            rospy.get_param("~blocked_reverse_speed_mps", 0.18)
        )
        self.scan_timeout_s = float(rospy.get_param("~scan_timeout_s", 0.6))
        self.tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.2))
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.robot_half_length_m = float(rospy.get_param("~robot_half_length_m", 0.40))
        self.robot_half_width_m = float(rospy.get_param("~robot_half_width_m", 0.25))
        self.footprint_padding_m = float(rospy.get_param("~footprint_padding_m", 0.03))
        self.self_filter_distance_m = float(rospy.get_param("~self_filter_distance_m", 0.24))
        self.min_points_to_block = max(1, int(rospy.get_param("~min_points_to_block", 4)))
        self.clear_distance_m = float(
            rospy.get_param("~clear_distance_m", self.stop_distance_m + 0.15)
        )
        self.turn_bias_epsilon_m = float(rospy.get_param("~turn_bias_epsilon_m", 0.05))
        self.idle_linear_threshold = float(rospy.get_param("~idle_linear_threshold", 0.01))
        self.idle_angular_threshold = float(rospy.get_param("~idle_angular_threshold", 0.01))
        self.allow_reverse_when_blocked = _as_bool(
            rospy.get_param("~allow_reverse_when_blocked", True)
        )
        self.fail_open = _as_bool(rospy.get_param("~fail_open", False))
        self.loop_hz = float(rospy.get_param("~loop_hz", 30.0))

        self.latest_cmd = Twist()
        self.latest_scan = None
        self.latest_scan_stamp = rospy.Time(0)
        self.latest_ultrasonic_m = None
        self.latest_ultrasonic_stamp = rospy.Time(0)
        self.ultrasonic_override_latched = False
        self.blocked_latched = False
        self.block_turn_sign = 1.0
        self._warned_tf_missing = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(self.cmd_in_topic, Twist, self._cmd_cb, queue_size=20)
        rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=10)
        if self.use_ultrasonic_front:
            rospy.Subscriber(self.ultrasonic_topic, Float32, self._ultrasonic_cb, queue_size=10)
            rospy.loginfo(
                "scan_safety_filter ultrasonic override enabled: stop=%.2fm clear=%.2fm reverse=%.2fm/s turn=%.2frad/s",
                self.ultrasonic_stop_distance_m,
                self.ultrasonic_clear_distance_m,
                self.ultrasonic_reverse_speed_mps,
                self.ultrasonic_turn_rate_rps,
            )

        self.cmd_pub = rospy.Publisher(self.cmd_out_topic, Twist, queue_size=20)
        self.blocked_pub = rospy.Publisher("~blocked", Bool, queue_size=5)

    def _cmd_cb(self, msg):
        self.latest_cmd = msg

    def _scan_cb(self, msg):
        self.latest_scan = msg
        self.latest_scan_stamp = rospy.Time.now()

    def _ultrasonic_cb(self, msg):
        raw = float(msg.data)
        if not math.isfinite(raw) or raw <= 0.0:
            return
        self.latest_ultrasonic_m = raw * self.ultrasonic_scale
        self.latest_ultrasonic_stamp = rospy.Time.now()

    def _ultrasonic_front_distance(self):
        if not self.use_ultrasonic_front or self.latest_ultrasonic_m is None:
            return float("inf"), False
        age = (rospy.Time.now() - self.latest_ultrasonic_stamp).to_sec()
        if age > self.ultrasonic_timeout_s:
            return float("inf"), False
        return self.latest_ultrasonic_m, True

    def _choose_turn_sign_from_points(self, points, hfov_rad):
        if not points:
            return self.block_turn_sign

        left_points = []
        right_points = []
        for bx, by in points:
            if bx <= 0.0:
                continue
            ang = abs(math.atan2(by, bx))
            if ang > hfov_rad:
                continue
            d = math.hypot(bx, by)
            if d < self.self_filter_distance_m:
                continue
            if by >= 0.0:
                left_points.append(d)
            else:
                right_points.append(d)

        if not left_points and not right_points:
            return self.block_turn_sign

        lmin = min(left_points) if left_points else float("inf")
        rmin = min(right_points) if right_points else float("inf")
        if abs(lmin - rmin) < self.turn_bias_epsilon_m:
            return self.block_turn_sign
        return 1.0 if lmin > rmin else -1.0

    def _ultrasonic_override_cmd(self, cmd, points):
        hfov = math.radians(self.forward_fov_deg) / 2.0
        turn_sign = self._choose_turn_sign_from_points(points, hfov)
        if abs(cmd.angular.z) > self.idle_angular_threshold:
            turn_sign = 1.0 if cmd.angular.z >= 0.0 else -1.0
        self.block_turn_sign = turn_sign

        safe = Twist()
        safe.linear.x = -abs(self.ultrasonic_reverse_speed_mps)
        safe.angular.z = abs(self.ultrasonic_turn_rate_rps) * turn_sign
        return safe

    @staticmethod
    def _quat_to_rot_2d(q):
        # 2D projection of full quaternion rotation (x/y components only).
        xx = q.x * q.x
        yy = q.y * q.y
        zz = q.z * q.z
        xy = q.x * q.y
        xz = q.x * q.z
        yz = q.y * q.z
        wx = q.w * q.x
        wy = q.w * q.y
        wz = q.w * q.z
        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        return r00, r01, r10, r11

    def _scan_points_in_base(self, scan):
        scan_frame = scan.header.frame_id.strip() if scan.header.frame_id else ""
        if not scan_frame:
            return None
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame,
                scan_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_s),
            )
            self._warned_tf_missing = False
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ):
            if not self._warned_tf_missing:
                rospy.logwarn(
                    "scan_safety_filter waiting for TF %s <- %s",
                    self.base_frame,
                    scan_frame,
                )
                self._warned_tf_missing = True
            return None

        tx = tfm.transform.translation.x
        ty = tfm.transform.translation.y
        r00, r01, r10, r11 = self._quat_to_rot_2d(tfm.transform.rotation)

        pts = []
        a = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                sx = r * math.cos(a)
                sy = r * math.sin(a)
                bx = tx + r00 * sx + r01 * sy
                by = ty + r10 * sx + r11 * sy
                pts.append((bx, by))
            a += scan.angle_increment
        return pts

    def _outside_robot_footprint(self, x, y):
        lx = self.robot_half_length_m + self.footprint_padding_m
        ly = self.robot_half_width_m + self.footprint_padding_m
        return not (-lx <= x <= lx and -ly <= y <= ly)

    def _safe_cmd(self):
        cmd = Twist()
        cmd.linear.x = self.latest_cmd.linear.x
        cmd.angular.z = self.latest_cmd.angular.z

        ultra_front_m, ultra_valid = self._ultrasonic_front_distance()
        ultra_blocked = False
        if ultra_valid:
            if self.ultrasonic_override_latched:
                ultra_blocked = ultra_front_m <= self.ultrasonic_clear_distance_m
            else:
                ultra_blocked = ultra_front_m <= self.ultrasonic_stop_distance_m
            self.ultrasonic_override_latched = ultra_blocked
        else:
            self.ultrasonic_override_latched = False

        if self.latest_scan is None:
            if ultra_blocked:
                return self._ultrasonic_override_cmd(cmd, None), True
            if not self.fail_open:
                return Twist(), True
            return cmd, False

        scan_age = (rospy.Time.now() - self.latest_scan_stamp).to_sec()
        if scan_age > self.scan_timeout_s:
            if ultra_blocked:
                return self._ultrasonic_override_cmd(cmd, None), True
            if not self.fail_open:
                return Twist(), True
            return cmd, False

        pts = self._scan_points_in_base(self.latest_scan)
        if pts is None:
            if ultra_blocked:
                return self._ultrasonic_override_cmd(cmd, None), True
            if not self.fail_open:
                return Twist(), True
            return cmd, False

        if ultra_blocked:
            rospy.logwarn_throttle(
                1.0,
                "scan_safety_filter ultrasonic override active (front=%.2fm <= %.2fm): forcing reverse-turn escape.",
                ultra_front_m,
                self.ultrasonic_stop_distance_m,
            )
            self.blocked_latched = False
            return self._ultrasonic_override_cmd(cmd, pts), True

        # This filter reasons only about the forward sector. Let reverse
        # commands pass so higher-level recovery can back away from walls.
        if self.allow_reverse_when_blocked and cmd.linear.x < -self.idle_linear_threshold:
            self.blocked_latched = False
            return cmd, False

        hfov = math.radians(self.forward_fov_deg) / 2.0
        candidate_points = []
        fwd_points = []
        left_points = []
        right_points = []
        for bx, by in pts:
            if bx <= 0.0:
                continue
            ang = abs(math.atan2(by, bx))
            if ang > hfov:
                continue
            d = math.hypot(bx, by)
            if d < self.self_filter_distance_m:
                continue
            candidate_points.append((d, bx, by))
            if not self._outside_robot_footprint(bx, by):
                continue
            fwd_points.append((d, bx, by))
            if by >= 0.0:
                left_points.append(d)
            else:
                right_points.append(d)

        if not candidate_points:
            self.blocked_latched = False
            return cmd, False

        # Prefer points outside the footprint; if footprint filtering removes all points,
        # fall back to candidate points so near-front walls cannot become invisible.
        active_points = fwd_points if fwd_points else candidate_points
        if not left_points:
            left_points = [p[0] for p in active_points if p[2] >= 0.0]
        if not right_points:
            right_points = [p[0] for p in active_points if p[2] < 0.0]

        distances = [p[0] for p in active_points]
        min_fwd = min(distances)
        stop_count = sum(1 for d in distances if d < self.stop_distance_m)
        clear_count = sum(1 for d in distances if d < self.clear_distance_m)
        if self.blocked_latched:
            blocked = clear_count >= self.min_points_to_block
        else:
            blocked = stop_count >= self.min_points_to_block

        if blocked:
            self.blocked_latched = True
            # Never inject autonomous turning when upstream command is idle.
            if (
                abs(cmd.linear.x) < self.idle_linear_threshold
                and abs(cmd.angular.z) < self.idle_angular_threshold
            ):
                return Twist(), True

            # If caller is already doing an in-place turn, do not override turn direction.
            if (
                abs(cmd.linear.x) < self.idle_linear_threshold
                and abs(cmd.angular.z) > self.idle_angular_threshold
            ):
                safe = Twist()
                safe.angular.z = cmd.angular.z
                self.block_turn_sign = 1.0 if safe.angular.z >= 0.0 else -1.0
                return safe, True

            safe = Twist()
            # Otherwise stop forward motion and turn toward the side with more free space.
            lmin = min(left_points) if left_points else min_fwd
            rmin = min(right_points) if right_points else min_fwd
            if abs(lmin - rmin) < self.turn_bias_epsilon_m:
                turn_sign = self.block_turn_sign
            else:
                turn_sign = 1.0 if lmin > rmin else -1.0
            self.block_turn_sign = turn_sign
            if min_fwd <= self.stop_distance_m:
                # Near-contact recovery: back out while turning.
                safe.linear.x = -abs(self.blocked_reverse_speed_mps)
            safe.angular.z = abs(self.avoid_turn_rate) * turn_sign
            return safe, True

        self.blocked_latched = False
        if min_fwd < self.slow_distance_m and cmd.linear.x > 0.0:
            scale = max(0.0, (min_fwd - self.stop_distance_m) / (self.slow_distance_m - self.stop_distance_m))
            cmd.linear.x *= scale

        return cmd, False

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            cmd, blocked = self._safe_cmd()
            self.blocked_pub.publish(Bool(data=blocked))
            self.cmd_pub.publish(cmd)
            rate.sleep()


def main():
    rospy.init_node("scan_safety_filter")
    ScanSafetyFilter().spin()


if __name__ == "__main__":
    main()

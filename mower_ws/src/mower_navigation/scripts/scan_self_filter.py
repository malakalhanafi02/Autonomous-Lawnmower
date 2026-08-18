#!/usr/bin/env python3
import math

import rospy
import tf2_ros
from sensor_msgs.msg import LaserScan


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class ScanSelfFilter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/scan")
        self.output_topic = rospy.get_param("~output_topic", "/scan_self_filtered")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.robot_half_length_m = float(rospy.get_param("~robot_half_length_m", 0.48))
        self.robot_half_width_m = float(rospy.get_param("~robot_half_width_m", 0.30))
        self.footprint_padding_m = float(rospy.get_param("~footprint_padding_m", 0.05))
        self.self_filter_distance_m = float(rospy.get_param("~self_filter_distance_m", 0.0))
        self.max_range_to_inf_margin_m = float(
            rospy.get_param("~max_range_to_inf_margin_m", 0.02)
        )
        self.tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.15))
        self.fail_open = _as_bool(rospy.get_param("~fail_open", True))

        self._warned_tf_missing = False
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.output_topic, LaserScan, queue_size=10)
        rospy.Subscriber(self.input_topic, LaserScan, self._scan_cb, queue_size=10)

    @staticmethod
    def _quat_to_rot_2d(q):
        xx = q.x * q.x
        yy = q.y * q.y
        zz = q.z * q.z
        xy = q.x * q.y
        wz = q.w * q.z
        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        return r00, r01, r10, r11

    def _inside_robot_footprint(self, bx, by):
        lx = self.robot_half_length_m + self.footprint_padding_m
        ly = self.robot_half_width_m + self.footprint_padding_m
        return (-lx <= bx <= lx) and (-ly <= by <= ly)

    def _scan_cb(self, scan):
        scan_frame = scan.header.frame_id.strip() if scan.header.frame_id else ""
        if not scan_frame:
            if self.fail_open:
                self.pub.publish(scan)
            return

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
                    "scan_self_filter waiting for TF %s <- %s",
                    self.base_frame,
                    scan_frame,
                )
                self._warned_tf_missing = True
            if self.fail_open:
                self.pub.publish(scan)
            return

        tx = tfm.transform.translation.x
        ty = tfm.transform.translation.y
        r00, r01, r10, r11 = self._quat_to_rot_2d(tfm.transform.rotation)

        out = LaserScan()
        out.header = scan.header
        out.angle_min = scan.angle_min
        out.angle_max = scan.angle_max
        out.angle_increment = scan.angle_increment
        out.time_increment = scan.time_increment
        out.scan_time = scan.scan_time
        out.range_min = scan.range_min
        out.range_max = scan.range_max

        ranges = list(scan.ranges)
        angle = scan.angle_min
        for i, rng in enumerate(scan.ranges):
            if math.isfinite(rng) and scan.range_min < rng < scan.range_max:
                # Treat near-max returns as no-obstacle to avoid a dense range ring
                # dominating RViz and downstream obstacle logic.
                if rng >= (scan.range_max - self.max_range_to_inf_margin_m):
                    ranges[i] = float("inf")
                    angle += scan.angle_increment
                    continue
                sx = rng * math.cos(angle)
                sy = rng * math.sin(angle)
                bx = tx + r00 * sx + r01 * sy
                by = ty + r10 * sx + r11 * sy
                if self._inside_robot_footprint(bx, by):
                    ranges[i] = float("inf")
                elif (
                    self.self_filter_distance_m > 0.0
                    and math.hypot(bx, by) < self.self_filter_distance_m
                ):
                    ranges[i] = float("inf")
            angle += scan.angle_increment

        out.ranges = ranges
        out.intensities = list(scan.intensities)
        self.pub.publish(out)


def main():
    rospy.init_node("scan_self_filter")
    ScanSelfFilter()
    rospy.spin()


if __name__ == "__main__":
    main()

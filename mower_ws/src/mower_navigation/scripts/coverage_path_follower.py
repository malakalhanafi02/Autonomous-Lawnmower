#!/usr/bin/env python3
import math

import rospy
import tf2_geometry_msgs  # noqa: F401 (needed by tf2 for PoseStamped transforms)
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class CoveragePathFollower:
    def __init__(self):
        self.path_topic = rospy.get_param("~path_topic", "/mower_ai/cut_plan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel_nav")
        self.cmd_raw_topic = rospy.get_param("~cmd_raw_topic", "/cmd_vel_raw")
        self.cmd_out_topic = rospy.get_param("~cmd_out_topic", "/cmd_vel")
        self.joint_states_topic = rospy.get_param("~joint_states_topic", "/joint_states")
        self.rear_left_joint_name = rospy.get_param(
            "~rear_left_joint_name", "rear_left_wheel_joint"
        )
        self.rear_right_joint_name = rospy.get_param(
            "~rear_right_joint_name", "rear_right_wheel_joint"
        )
        self.enabled_topic = rospy.get_param("~enabled_topic", "/mower/autonomy_enabled")

        self.max_linear = float(rospy.get_param("~max_linear_mps", 0.5))
        self.max_angular = float(rospy.get_param("~max_angular_rps", 1.0))
        self.kp_angular = float(rospy.get_param("~kp_angular", 1.6))
        self.rotate_in_place_threshold = float(
            rospy.get_param("~rotate_in_place_threshold_rad", 1.45)
        )
        self.coverage_allow_in_place_turn = _as_bool(
            rospy.get_param("~coverage_allow_in_place_turn", False)
        )
        self.coverage_stop_turn_go = _as_bool(rospy.get_param("~coverage_stop_turn_go", True))
        self.coverage_turn_enter_rad = float(rospy.get_param("~coverage_turn_enter_rad", 0.30))
        self.coverage_turn_exit_rad = float(rospy.get_param("~coverage_turn_exit_rad", 0.08))
        self.coverage_turn_min_angular_rps = float(
            rospy.get_param("~coverage_turn_min_angular_rps", 0.35)
        )
        self.coverage_turn_radius_m = float(rospy.get_param("~coverage_turn_radius_m", 0.75))
        self.coverage_turn_speed_mps = float(rospy.get_param("~coverage_turn_speed_mps", 0.16))
        self.tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.2))
        self.start_nearest_waypoint = _as_bool(rospy.get_param("~start_nearest_waypoint", True))
        self.reseed_on_new_path = _as_bool(rospy.get_param("~reseed_on_new_path", False))
        self.min_linear = float(rospy.get_param("~min_linear_mps", 0.05))
        self.waypoint_tolerance = float(rospy.get_param("~waypoint_tolerance_m", 0.35))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance_m", 0.25))
        self.coverage_max_waypoint_advance = int(
            rospy.get_param("~coverage_max_waypoint_advance", 4)
        )
        self.odom_timeout_s = float(rospy.get_param("~odom_timeout_s", 0.6))
        self.loop_hz = float(rospy.get_param("~loop_hz", 20.0))
        self.return_to_start = _as_bool(rospy.get_param("~return_to_start", True))

        # Phase 1: perimeter bootstrap mapping.
        self.perimeter_bootstrap_enabled = _as_bool(
            rospy.get_param("~perimeter_bootstrap_enabled", True)
        )
        self.perimeter_repeat_each_enable = _as_bool(
            rospy.get_param("~perimeter_repeat_each_enable", True)
        )
        self.perimeter_complete_topic = rospy.get_param(
            "~perimeter_complete_topic", "/mower/perimeter_complete"
        )
        self.perimeter_scan_topic = rospy.get_param("~perimeter_scan_topic", "/scan")
        self.perimeter_base_frame = rospy.get_param("~perimeter_base_frame", "base_footprint")
        self.perimeter_scan_timeout_s = float(rospy.get_param("~perimeter_scan_timeout_s", 0.7))
        self.perimeter_sector_half_width_deg = float(
            rospy.get_param("~perimeter_sector_half_width_deg", 22.0)
        )
        self.perimeter_sector_min_points = max(
            1, int(rospy.get_param("~perimeter_sector_min_points", 5))
        )
        self.perimeter_sector_distance_rank = max(
            1, int(rospy.get_param("~perimeter_sector_distance_rank", 3))
        )
        self.perimeter_self_filter_m = float(rospy.get_param("~perimeter_self_filter_m", 0.2))
        self.perimeter_robot_half_length_m = float(
            rospy.get_param("~perimeter_robot_half_length_m", 0.48)
        )
        self.perimeter_robot_half_width_m = float(
            rospy.get_param("~perimeter_robot_half_width_m", 0.30)
        )
        self.perimeter_footprint_padding_m = float(
            rospy.get_param("~perimeter_footprint_padding_m", 0.03)
        )
        self.perimeter_min_duration_s = float(rospy.get_param("~perimeter_min_duration_s", 35.0))
        self.perimeter_max_duration_s = float(rospy.get_param("~perimeter_max_duration_s", 0.0))
        self.perimeter_timeout_min_progress_ratio = float(
            rospy.get_param("~perimeter_timeout_min_progress_ratio", 0.75)
        )
        self.perimeter_timeout_max_restarts = int(
            rospy.get_param("~perimeter_timeout_max_restarts", 2)
        )
        self.perimeter_min_travel_m = float(rospy.get_param("~perimeter_min_travel_m", 12.0))
        self.perimeter_min_extent_x_m = float(
            rospy.get_param("~perimeter_min_extent_x_m", 4.0)
        )
        self.perimeter_min_extent_y_m = float(
            rospy.get_param("~perimeter_min_extent_y_m", 4.0)
        )
        self.perimeter_far_distance_m = float(rospy.get_param("~perimeter_far_distance_m", 1.5))
        self.perimeter_close_tolerance_m = float(
            rospy.get_param("~perimeter_close_tolerance_m", 1.0)
        )
        self.perimeter_forward_speed_mps = float(
            rospy.get_param("~perimeter_forward_speed_mps", 0.28)
        )
        self.perimeter_turn_rate_rps = float(rospy.get_param("~perimeter_turn_rate_rps", 0.45))
        self.perimeter_right_turn_rate_rps = float(
            rospy.get_param("~perimeter_right_turn_rate_rps", 0.9)
        )
        self.perimeter_front_stop_m = float(rospy.get_param("~perimeter_front_stop_m", 0.75))
        self.perimeter_front_clear_m = float(rospy.get_param("~perimeter_front_clear_m", 1.0))
        self.perimeter_front_predict_time_s = float(
            rospy.get_param("~perimeter_front_predict_time_s", 2.0)
        )
        self.perimeter_emergency_stop_m = float(
            rospy.get_param("~perimeter_emergency_stop_m", 1.20)
        )
        self.perimeter_emergency_stop_votes = max(
            1, int(rospy.get_param("~perimeter_emergency_stop_votes", 2))
        )
        self.perimeter_corner_preturn_m = float(
            rospy.get_param("~perimeter_corner_preturn_m", 1.8)
        )
        self.perimeter_corner_commit_m = float(
            rospy.get_param("~perimeter_corner_commit_m", 1.2)
        )
        self.perimeter_corner_min_linear_mps = float(
            rospy.get_param("~perimeter_corner_min_linear_mps", 0.10)
        )
        requested_corner_turn_angle = float(
            rospy.get_param("~perimeter_corner_turn_angle_rad", math.pi / 2.0)
        )
        self.perimeter_corner_turn_angle_rad = math.pi / 2.0
        if abs(requested_corner_turn_angle - self.perimeter_corner_turn_angle_rad) > 1e-3:
            rospy.logwarn(
                "Forcing perimeter corner turn angle to 90 degrees (%.2f rad). Ignoring configured %.2f rad.",
                self.perimeter_corner_turn_angle_rad,
                requested_corner_turn_angle,
            )
        self.perimeter_corner_turn_exit_rad = float(
            rospy.get_param("~perimeter_corner_turn_exit_rad", 0.08)
        )
        self.perimeter_corner_turn_rate_rps = float(
            rospy.get_param("~perimeter_corner_turn_rate_rps", 1.0)
        )
        self.perimeter_corner_turn_linear_mps = float(
            rospy.get_param("~perimeter_corner_turn_linear_mps", 0.16)
        )
        self.perimeter_corner_turn_max_s = float(
            rospy.get_param("~perimeter_corner_turn_max_s", 4.0)
        )
        self.perimeter_corner_timeout_accept_rad = float(
            rospy.get_param("~perimeter_corner_timeout_accept_rad", 0.28)
        )
        self.perimeter_corner_backoff_s = float(
            rospy.get_param("~perimeter_corner_backoff_s", 1.2)
        )
        self.perimeter_corner_retry_cooldown_s = float(
            rospy.get_param("~perimeter_corner_retry_cooldown_s", 1.8)
        )
        self.perimeter_corner_backoff_speed_mps = float(
            rospy.get_param("~perimeter_corner_backoff_speed_mps", 0.22)
        )
        self.perimeter_corner_backoff_turn_rps = float(
            rospy.get_param("~perimeter_corner_backoff_turn_rps", 0.55)
        )
        self.perimeter_corner_retry_clearance_m = float(
            rospy.get_param("~perimeter_corner_retry_clearance_m", 1.35)
        )
        self.perimeter_corner_trigger_count = max(
            1, int(rospy.get_param("~perimeter_corner_trigger_count", 4))
        )
        self.perimeter_right_target_m = float(rospy.get_param("~perimeter_right_target_m", 0.75))
        self.perimeter_right_open_m = float(rospy.get_param("~perimeter_right_open_m", 1.2))
        self.perimeter_right_close_m = float(rospy.get_param("~perimeter_right_close_m", 0.45))
        self.perimeter_wall_kp = float(rospy.get_param("~perimeter_wall_kp", 1.1))
        self.perimeter_wall_max_turn_rps = float(
            rospy.get_param("~perimeter_wall_max_turn_rps", 0.65)
        )
        self.perimeter_seek_turn_rps = float(rospy.get_param("~perimeter_seek_turn_rps", 0.0))
        self.perimeter_right_acquire_m = float(
            rospy.get_param("~perimeter_right_acquire_m", 1.8)
        )
        self.perimeter_require_wall_contact = _as_bool(
            rospy.get_param("~perimeter_require_wall_contact", True)
        )
        self.perimeter_stall_window_s = float(rospy.get_param("~perimeter_stall_window_s", 2.0))
        self.perimeter_stall_min_progress_m = float(
            rospy.get_param("~perimeter_stall_min_progress_m", 0.08)
        )
        self.perimeter_turn_stall_min_yaw_rad = float(
            rospy.get_param("~perimeter_turn_stall_min_yaw_rad", 0.14)
        )
        self.perimeter_turn_stall_min_cmd_rps = float(
            rospy.get_param("~perimeter_turn_stall_min_cmd_rps", 0.35)
        )
        self.perimeter_front_block_reverse_speed_mps = float(
            rospy.get_param("~perimeter_front_block_reverse_speed_mps", 0.14)
        )
        self.perimeter_front_turn_bias_m = float(
            rospy.get_param("~perimeter_front_turn_bias_m", 0.20)
        )
        self.perimeter_escape_turn_s = float(rospy.get_param("~perimeter_escape_turn_s", 1.5))
        self.perimeter_blind_turn_rps = float(rospy.get_param("~perimeter_blind_turn_rps", 0.7))
        self.perimeter_escape_reverse_speed_mps = float(
            rospy.get_param("~perimeter_escape_reverse_speed_mps", 0.14)
        )
        self.perimeter_escape_reverse_only_s = float(
            rospy.get_param("~perimeter_escape_reverse_only_s", 0.60)
        )
        self.perimeter_escape_pivot_turn_s = float(
            rospy.get_param("~perimeter_escape_pivot_turn_s", 2.0)
        )
        self.perimeter_escape_max_reverse_attempts = int(
            rospy.get_param("~perimeter_escape_max_reverse_attempts", 2)
        )
        self.perimeter_escape_pivot_turn_rps = float(
            rospy.get_param("~perimeter_escape_pivot_turn_rps", 1.2)
        )
        self.perimeter_drive_wheel_separation_m = float(
            rospy.get_param("~perimeter_drive_wheel_separation_m", 0.50)
        )
        self.perimeter_reverse_turn_ratio = _clamp(
            float(rospy.get_param("~perimeter_reverse_turn_ratio", 0.90)),
            0.1,
            1.0,
        )
        # Global stuck recovery: wheels spinning but odom not changing.
        self.motion_stuck_enabled = _as_bool(rospy.get_param("~motion_stuck_enabled", True))
        self.motion_stuck_window_s = float(rospy.get_param("~motion_stuck_window_s", 2.5))
        self.motion_stuck_min_progress_m = float(
            rospy.get_param("~motion_stuck_min_progress_m", 0.03)
        )
        self.motion_stuck_min_yaw_rad = float(
            rospy.get_param("~motion_stuck_min_yaw_rad", 0.08)
        )
        self.motion_stuck_min_cmd_linear_mps = float(
            rospy.get_param("~motion_stuck_min_cmd_linear_mps", 0.10)
        )
        self.motion_stuck_min_cmd_angular_rps = float(
            rospy.get_param("~motion_stuck_min_cmd_angular_rps", 0.30)
        )
        self.motion_stuck_min_wheel_rps = float(
            rospy.get_param("~motion_stuck_min_wheel_rps", 0.45)
        )
        self.motion_stuck_min_effective_ratio = float(
            rospy.get_param("~motion_stuck_min_effective_ratio", 0.35)
        )
        self.motion_stuck_escape_reverse_mps = float(
            rospy.get_param("~motion_stuck_escape_reverse_mps", 0.20)
        )
        self.motion_stuck_escape_turn_rps = float(
            rospy.get_param("~motion_stuck_escape_turn_rps", 1.00)
        )
        self.motion_stuck_escape_duration_s = float(
            rospy.get_param("~motion_stuck_escape_duration_s", 1.8)
        )
        self.motion_stuck_escape_reverse_only_s = float(
            rospy.get_param("~motion_stuck_escape_reverse_only_s", 0.6)
        )
        self.motion_stuck_keep_turn_direction = _as_bool(
            rospy.get_param("~motion_stuck_keep_turn_direction", True)
        )
        self.forward_alignment_guard_enabled = _as_bool(
            rospy.get_param("~forward_alignment_guard_enabled", True)
        )
        self.forward_alignment_min_cmd_mps = float(
            rospy.get_param("~forward_alignment_min_cmd_mps", 0.10)
        )
        self.forward_alignment_reverse_detect_mps = float(
            rospy.get_param("~forward_alignment_reverse_detect_mps", 0.05)
        )
        self.forward_alignment_min_turn_rps = float(
            rospy.get_param("~forward_alignment_min_turn_rps", 0.45)
        )
        self.debug_speed_scale = max(float(rospy.get_param("~debug_speed_scale", 1.0)), 0.1)
        self.debug_turn_scale = max(float(rospy.get_param("~debug_turn_scale", 1.0)), 0.1)

        self.max_linear *= self.debug_speed_scale
        self.min_linear = min(self.max_linear, self.min_linear * self.debug_speed_scale)
        self.perimeter_forward_speed_mps *= self.debug_speed_scale

        self.max_angular *= self.debug_turn_scale
        self.perimeter_turn_rate_rps = min(
            self.perimeter_turn_rate_rps * self.debug_turn_scale, self.max_angular
        )
        self.perimeter_right_turn_rate_rps = min(
            self.perimeter_right_turn_rate_rps * self.debug_turn_scale, self.max_angular
        )
        self.perimeter_corner_turn_rate_rps = min(
            self.perimeter_corner_turn_rate_rps * self.debug_turn_scale, self.max_angular
        )
        self.perimeter_blind_turn_rps = min(
            self.perimeter_blind_turn_rps * self.debug_turn_scale, self.max_angular
        )
        self.perimeter_wall_max_turn_rps = min(
            self.perimeter_wall_max_turn_rps * self.debug_turn_scale, self.max_angular
        )
        self.perimeter_corner_backoff_turn_rps = min(
            self.perimeter_corner_backoff_turn_rps * self.debug_turn_scale, self.max_angular
        )
        self.perimeter_escape_pivot_turn_rps = min(
            self.perimeter_escape_pivot_turn_rps * self.debug_turn_scale, self.max_angular
        )
        self.motion_stuck_escape_turn_rps = min(
            self.motion_stuck_escape_turn_rps * self.debug_turn_scale, self.max_angular
        )
        # Keep recovery speeds bounded even when debug scaling is high.
        self.perimeter_escape_reverse_speed_mps = min(self.perimeter_escape_reverse_speed_mps, 0.24)
        self.perimeter_corner_backoff_speed_mps = min(self.perimeter_corner_backoff_speed_mps, 0.24)
        self.perimeter_front_block_reverse_speed_mps = min(
            self.perimeter_front_block_reverse_speed_mps, 0.24
        )
        # In-place pivots are prone to bind against walls in this model; keep a small arc.
        self.perimeter_corner_turn_linear_mps = max(
            self.perimeter_corner_turn_linear_mps, self.perimeter_corner_min_linear_mps
        )
        self.perimeter_corner_turn_linear_mps = min(self.perimeter_corner_turn_linear_mps, 0.14)
        self.perimeter_corner_turn_rate_rps = min(self.perimeter_corner_turn_rate_rps, 0.85)
        self.perimeter_corner_backoff_turn_rps = min(self.perimeter_corner_backoff_turn_rps, 0.70)
        self.perimeter_escape_pivot_turn_rps = min(self.perimeter_escape_pivot_turn_rps, 0.90)

        if self.debug_speed_scale != 1.0 or self.debug_turn_scale != 1.0:
            rospy.logwarn(
                "Coverage follower debug speed scaling active (linear x%.2f, turn x%.2f).",
                self.debug_speed_scale,
                self.debug_turn_scale,
            )

        # Optional ultrasonic front assist (typically from jetson_mega_bridge).
        self.use_ultrasonic_front = _as_bool(rospy.get_param("~use_ultrasonic_front", True))
        self.ultrasonic_topic = rospy.get_param(
            "~ultrasonic_topic", "/jetson_mega_bridge/ultrasonic_distance_cm"
        )
        self.ultrasonic_scale = float(rospy.get_param("~ultrasonic_scale", 0.01))
        self.ultrasonic_timeout_s = float(rospy.get_param("~ultrasonic_timeout_s", 0.6))

        self.path = None
        self.path_frame = ""
        self.idx = 0
        self.odom = None
        self.odom_stamp = rospy.Time(0)
        self.enabled = False
        self._prev_enabled = False
        self._needs_reseed = False
        self.start_xy = None
        self.returning_home = False
        self.home_reached = False
        self.coverage_turning_in_place = False

        self.latest_scan = None
        self.latest_scan_stamp = rospy.Time(0)
        self.latest_ultrasonic_m = None
        self.latest_ultrasonic_stamp = rospy.Time(0)
        self._dbg_state = "init"
        self._dbg_front = float("inf")
        self._dbg_front_right = float("inf")
        self._dbg_front_left = float("inf")
        self._dbg_right = float("inf")
        self._dbg_left = float("inf")
        self._dbg_dynamic_stop = float("inf")
        self._dbg_actual_linear = 0.0
        self._dbg_actual_angular = 0.0
        self._dbg_turning_in_place = False
        self._dbg_raw_linear = 0.0
        self._dbg_raw_angular = 0.0
        self._dbg_out_linear = 0.0
        self._dbg_out_angular = 0.0
        self._dbg_wheel_left_rps = 0.0
        self._dbg_wheel_right_rps = 0.0

        self.perimeter_done = not self.perimeter_bootstrap_enabled
        self.perimeter_start_xy = None
        self.perimeter_last_xy = None
        self.perimeter_travel_m = 0.0
        self.perimeter_went_far = False
        self.perimeter_started_stamp = rospy.Time(0)
        self.perimeter_min_x = 0.0
        self.perimeter_max_x = 0.0
        self.perimeter_min_y = 0.0
        self.perimeter_max_y = 0.0
        self.perimeter_progress_ref_xy = None
        self.perimeter_progress_ref_stamp = rospy.Time(0)
        self.perimeter_turn_ref_yaw = None
        self.perimeter_turn_ref_stamp = rospy.Time(0)
        self.perimeter_escape_until = rospy.Time(0)
        self.perimeter_escape_reverse_until = rospy.Time(0)
        self.perimeter_escape_turn_sign = -1.0
        self.perimeter_stall_count = 0
        self.perimeter_escape_use_pivot = False
        self.perimeter_wall_seen = False
        self._perimeter_timeout_no_wall_warned = False
        self._warned_perimeter_tf_missing = False
        self.perimeter_corner_turn_active = False
        self.perimeter_corner_turn_target_yaw = 0.0
        self.perimeter_corner_turn_started_stamp = rospy.Time(0)
        self.perimeter_corner_turn_sign = -1.0
        self.perimeter_corner_anchor_yaw = None
        self.perimeter_corner_trigger_hits = 0
        self.perimeter_corner_backoff_until = rospy.Time(0)
        self.perimeter_corner_retry_until = rospy.Time(0)
        self.perimeter_corner_retry_required = False
        self.perimeter_timeout_restart_count = 0
        self.motion_stuck_ref_xy = None
        self.motion_stuck_ref_yaw = None
        self.motion_stuck_ref_stamp = rospy.Time(0)
        self.motion_stuck_escape_until = rospy.Time(0)
        self.motion_stuck_escape_reverse_until = rospy.Time(0)
        self.motion_stuck_escape_turn_sign = 1.0
        self.motion_stuck_count = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(self.path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber(self.cmd_raw_topic, Twist, self._raw_cmd_cb, queue_size=20)
        rospy.Subscriber(self.cmd_out_topic, Twist, self._out_cmd_cb, queue_size=20)
        rospy.Subscriber(self.joint_states_topic, JointState, self._joint_states_cb, queue_size=20)
        rospy.Subscriber(self.enabled_topic, Bool, self._enabled_cb, queue_size=5)
        rospy.Subscriber(self.perimeter_scan_topic, LaserScan, self._scan_cb, queue_size=10)
        if self.use_ultrasonic_front:
            rospy.Subscriber(self.ultrasonic_topic, Float32, self._ultrasonic_cb, queue_size=10)
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=20)
        self.state_pub = rospy.Publisher("~state", String, queue_size=10)
        self.debug_markers_pub = rospy.Publisher("~debug_markers", MarkerArray, queue_size=5)
        self.perimeter_done_pub = rospy.Publisher(
            self.perimeter_complete_topic, Bool, queue_size=5, latch=True
        )
        self.perimeter_done_pub.publish(Bool(data=self.perimeter_done))

    def _path_cb(self, msg):
        old_len = len(self.path.poses) if (self.path is not None and self.path.poses) else 0
        self.path = msg
        self.path_frame = msg.header.frame_id.strip() if msg.header.frame_id else ""

        if self.reseed_on_new_path or old_len == 0:
            self.idx = 0
            self._needs_reseed = True
        else:
            # Preserve progress across replans to avoid repeatedly re-entering the same loop.
            old_idx = max(0, min(self.idx, old_len - 1))
            denom = max(1, old_len - 1)
            progress = float(old_idx) / float(denom)
            new_len = len(msg.poses)
            self.idx = int(round(progress * max(0, new_len - 1)))
            self._needs_reseed = False
            rospy.loginfo(
                "Coverage follower preserved progress on replan: %.1f%% -> idx=%d/%d.",
                100.0 * progress,
                self.idx,
                new_len,
            )

        self.returning_home = False
        self.home_reached = False
        self.coverage_turning_in_place = False
        rospy.loginfo(
            "Coverage follower got new path with %d poses (frame=%s).",
            len(msg.poses),
            self.path_frame or "unset",
        )

    def _odom_cb(self, msg):
        self.odom = msg
        self.odom_stamp = rospy.Time.now()
        self._dbg_actual_linear = msg.twist.twist.linear.x
        self._dbg_actual_angular = msg.twist.twist.angular.z

    def _enabled_cb(self, msg):
        self.enabled = msg.data

    def _raw_cmd_cb(self, msg):
        self._dbg_raw_linear = msg.linear.x
        self._dbg_raw_angular = msg.angular.z

    def _out_cmd_cb(self, msg):
        self._dbg_out_linear = msg.linear.x
        self._dbg_out_angular = msg.angular.z

    def _joint_states_cb(self, msg):
        if not msg.name or not msg.velocity:
            return
        for i, name in enumerate(msg.name):
            if i >= len(msg.velocity):
                break
            if name == self.rear_left_joint_name:
                self._dbg_wheel_left_rps = msg.velocity[i]
            elif name == self.rear_right_joint_name:
                self._dbg_wheel_right_rps = msg.velocity[i]

    def _scan_cb(self, msg):
        self.latest_scan = msg
        self.latest_scan_stamp = rospy.Time.now()

    def _ultrasonic_cb(self, msg):
        raw = float(msg.data)
        if not math.isfinite(raw) or raw <= 0.0:
            return
        self.latest_ultrasonic_m = raw * self.ultrasonic_scale
        self.latest_ultrasonic_stamp = rospy.Time.now()

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

    def _outside_robot_footprint(self, bx, by):
        lx = self.perimeter_robot_half_length_m + self.perimeter_footprint_padding_m
        ly = self.perimeter_robot_half_width_m + self.perimeter_footprint_padding_m
        return not (-lx <= bx <= lx and -ly <= by <= ly)

    def _scan_points_in_base(self):
        if self.latest_scan is None:
            return None
        age = (rospy.Time.now() - self.latest_scan_stamp).to_sec()
        if age > self.perimeter_scan_timeout_s:
            return None

        scan_frame = self.latest_scan.header.frame_id.strip() if self.latest_scan.header.frame_id else ""
        if not scan_frame:
            return None

        try:
            tfm = self.tf_buffer.lookup_transform(
                self.perimeter_base_frame,
                scan_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_s),
            )
            self._warned_perimeter_tf_missing = False
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ):
            if not self._warned_perimeter_tf_missing:
                rospy.logwarn(
                    "coverage_path_follower waiting for TF %s <- %s",
                    self.perimeter_base_frame,
                    scan_frame,
                )
                self._warned_perimeter_tf_missing = True
            return None

        tx = tfm.transform.translation.x
        ty = tfm.transform.translation.y
        r00, r01, r10, r11 = self._quat_to_rot_2d(tfm.transform.rotation)

        pts = []
        a = self.latest_scan.angle_min
        for r in self.latest_scan.ranges:
            if math.isfinite(r) and self.latest_scan.range_min < r < self.latest_scan.range_max:
                sx = r * math.cos(a)
                sy = r * math.sin(a)
                bx = tx + r00 * sx + r01 * sy
                by = ty + r10 * sx + r11 * sy
                pts.append((bx, by))
            a += self.latest_scan.angle_increment
        return pts

    def _compute_cmd_to_target(self, x, y, yaw, tx, ty):
        heading = math.atan2(ty - y, tx - x)
        err = _norm_angle(heading - yaw)
        dist = math.hypot(tx - x, ty - y)

        cmd = Twist()
        desired_angular = _clamp(self.kp_angular * err, -self.max_angular, self.max_angular)
        cmd.angular.z = desired_angular

        if self.coverage_stop_turn_go:
            if self.coverage_turning_in_place:
                if abs(err) <= self.coverage_turn_exit_rad:
                    self.coverage_turning_in_place = False
            elif abs(err) >= self.coverage_turn_enter_rad:
                self.coverage_turning_in_place = True

            if self.coverage_turning_in_place:
                cmd.linear.x = 0.0
                if (
                    abs(cmd.angular.z) < self.coverage_turn_min_angular_rps
                    and abs(err) > self.coverage_turn_exit_rad
                ):
                    cmd.angular.z = math.copysign(
                        min(self.max_angular, self.coverage_turn_min_angular_rps),
                        err,
                    )
                self._dbg_turning_in_place = True
                return cmd, dist

        if abs(err) < self.rotate_in_place_threshold:
            cmd.linear.x = _clamp(
                max(self.min_linear, self.max_linear * math.cos(err)),
                0.0,
                min(self.max_linear, dist),
            )
        elif self.coverage_allow_in_place_turn:
            cmd.linear.x = 0.0
        else:
            # Use a finite-radius arc turn instead of in-place spin.
            # This better matches mowers that need turn radius near edges.
            cmd.linear.x = _clamp(
                self.coverage_turn_speed_mps,
                max(0.0, self.min_linear),
                min(self.max_linear, max(self.coverage_turn_speed_mps, dist)),
            )

        if not self.coverage_allow_in_place_turn and cmd.linear.x > 0.0:
            max_angular_for_radius = cmd.linear.x / max(0.05, self.coverage_turn_radius_m)
            cmd.angular.z = _clamp(
                cmd.angular.z,
                -max_angular_for_radius,
                max_angular_for_radius,
            )

            # Keep a meaningful turn command when error is large.
            if abs(err) > 0.35 and abs(cmd.angular.z) < 0.5 * max_angular_for_radius:
                cmd.angular.z = math.copysign(0.5 * max_angular_for_radius, err)
        return cmd, dist

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _set_state(self, state):
        self._dbg_state = state
        self.state_pub.publish(String(data=state))

    @staticmethod
    def _fmt_dist(d):
        if not math.isfinite(d):
            return "inf"
        return "%.2f" % d

    def _publish_debug_markers(self, cmd):
        m = MarkerArray()
        now = rospy.Time.now()

        cmd_arrow = Marker()
        cmd_arrow.header.stamp = rospy.Time(0)
        cmd_arrow.header.frame_id = self.perimeter_base_frame
        cmd_arrow.frame_locked = True
        cmd_arrow.ns = "coverage_cmd"
        cmd_arrow.id = 0
        cmd_arrow.type = Marker.ARROW
        cmd_arrow.action = Marker.ADD
        cmd_arrow.scale.x = 0.06
        cmd_arrow.scale.y = 0.12
        cmd_arrow.scale.z = 0.12
        cmd_arrow.color.a = 0.95
        if cmd.linear.x >= 0.0:
            cmd_arrow.color.r = 0.2
            cmd_arrow.color.g = 1.0
            cmd_arrow.color.b = 0.2
        else:
            cmd_arrow.color.r = 1.0
            cmd_arrow.color.g = 0.4
            cmd_arrow.color.b = 0.2
        p0 = Point()
        p0.x = 0.0
        p0.y = 0.0
        p0.z = 0.18
        p1 = Point()
        p1.x = _clamp(cmd.linear.x * 2.0, -1.0, 1.0)
        p1.y = _clamp(cmd.angular.z * 0.35, -0.8, 0.8)
        p1.z = 0.18
        if abs(p1.x) < 0.03 and abs(p1.y) < 0.03:
            p1.x = 0.03
        cmd_arrow.points = [p0, p1]
        m.markers.append(cmd_arrow)

        actual_arrow = Marker()
        actual_arrow.header.stamp = rospy.Time(0)
        actual_arrow.header.frame_id = self.perimeter_base_frame
        actual_arrow.frame_locked = True
        actual_arrow.ns = "coverage_cmd"
        actual_arrow.id = 2
        actual_arrow.type = Marker.ARROW
        actual_arrow.action = Marker.ADD
        actual_arrow.scale.x = 0.045
        actual_arrow.scale.y = 0.09
        actual_arrow.scale.z = 0.09
        actual_arrow.color.a = 0.95
        actual_arrow.color.r = 0.2
        actual_arrow.color.g = 0.8
        actual_arrow.color.b = 1.0
        ap0 = Point()
        ap0.x = 0.0
        ap0.y = 0.0
        ap0.z = 0.08
        ap1 = Point()
        ap1.x = _clamp(self._dbg_actual_linear * 2.0, -1.0, 1.0)
        ap1.y = _clamp(self._dbg_actual_angular * 0.35, -0.8, 0.8)
        ap1.z = 0.08
        if abs(ap1.x) < 0.03 and abs(ap1.y) < 0.03:
            ap1.x = 0.03
        actual_arrow.points = [ap0, ap1]
        m.markers.append(actual_arrow)

        text = Marker()
        text.header.stamp = rospy.Time(0)
        text.header.frame_id = self.perimeter_base_frame
        text.frame_locked = True
        text.ns = "coverage_cmd"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.orientation.w = 1.0
        text.pose.position.x = 0.0
        text.pose.position.y = 0.0
        text.pose.position.z = 0.8
        text.scale.z = 0.18
        text.color.a = 1.0
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        escape_suffix = ""
        if self.perimeter_escape_use_pivot:
            escape_suffix += " pivot"
        if rospy.Time.now() < self.perimeter_escape_reverse_until:
            escape_suffix += " rev_only"
        if self.perimeter_corner_turn_active:
            escape_suffix += " corner_turn"
        if rospy.Time.now() < self.perimeter_corner_backoff_until:
            escape_suffix += " corner_backoff"
        if rospy.Time.now() < self.perimeter_corner_retry_until:
            escape_suffix += " corner_retry"
        if rospy.Time.now() < self.motion_stuck_escape_until:
            escape_suffix += " unstick"
        text.text = (
            "state=%s nav=(%.2f,%.2f) raw=(%.2f,%.2f) out=(%.2f,%.2f) odom=(%.3f,%.3f) wl=%.3f wr=%.3f turn=%d front=%s fr=%s fl=%s right=%s left=%s stop=%s stall=%d%s"
            % (
                self._dbg_state,
                cmd.linear.x,
                cmd.angular.z,
                self._dbg_raw_linear,
                self._dbg_raw_angular,
                self._dbg_out_linear,
                self._dbg_out_angular,
                self._dbg_actual_linear,
                self._dbg_actual_angular,
                self._dbg_wheel_left_rps,
                self._dbg_wheel_right_rps,
                1 if self._dbg_turning_in_place else 0,
                self._fmt_dist(self._dbg_front),
                self._fmt_dist(self._dbg_front_right),
                self._fmt_dist(self._dbg_front_left),
                self._fmt_dist(self._dbg_right),
                self._fmt_dist(self._dbg_left),
                self._fmt_dist(self._dbg_dynamic_stop),
                self.perimeter_stall_count,
                escape_suffix,
            )
        )
        m.markers.append(text)

        self.debug_markers_pub.publish(m)

    def _transform_pose_to_frame(self, x, y, yaw, from_frame, to_frame):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time(0)
        pose.header.frame_id = from_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(0.5 * yaw)
        pose.pose.orientation.w = math.cos(0.5 * yaw)
        try:
            tf_pose = self.tf_buffer.transform(
                pose,
                to_frame,
                rospy.Duration(self.tf_timeout_s),
            )
            tx = tf_pose.pose.position.x
            ty = tf_pose.pose.position.y
            tyaw = _yaw_from_quat(tf_pose.pose.orientation)
            return tx, ty, tyaw
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ):
            return None

    def _nearest_waypoint_index(self, x, y):
        if self.path is None or not self.path.poses:
            return 0
        best_i = 0
        best_d2 = float("inf")
        for i, pose_stamped in enumerate(self.path.poses):
            p = pose_stamped.pose.position
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _reset_perimeter_progress(self, x, y, publish_reset):
        self.perimeter_start_xy = (x, y)
        self.perimeter_last_xy = (x, y)
        self.perimeter_travel_m = 0.0
        self.perimeter_went_far = False
        self.perimeter_started_stamp = rospy.Time.now()
        self.perimeter_min_x = x
        self.perimeter_max_x = x
        self.perimeter_min_y = y
        self.perimeter_max_y = y
        self.perimeter_progress_ref_xy = (x, y)
        self.perimeter_progress_ref_stamp = rospy.Time.now()
        self.perimeter_turn_ref_yaw = None
        self.perimeter_turn_ref_stamp = rospy.Time(0)
        self.perimeter_escape_until = rospy.Time(0)
        self.perimeter_escape_reverse_until = rospy.Time(0)
        self.perimeter_escape_turn_sign = -1.0
        self.perimeter_stall_count = 0
        self.perimeter_escape_use_pivot = False
        self.perimeter_wall_seen = False
        self.perimeter_corner_turn_active = False
        self.perimeter_corner_turn_target_yaw = 0.0
        self.perimeter_corner_turn_started_stamp = rospy.Time(0)
        self.perimeter_corner_turn_sign = -1.0
        self.perimeter_corner_anchor_yaw = None
        self.perimeter_corner_trigger_hits = 0
        self.perimeter_corner_backoff_until = rospy.Time(0)
        self.perimeter_corner_retry_until = rospy.Time(0)
        self.perimeter_corner_retry_required = False
        self._perimeter_timeout_no_wall_warned = False
        if publish_reset and self.perimeter_bootstrap_enabled:
            self.perimeter_done = False
            self.perimeter_done_pub.publish(Bool(data=False))
            rospy.loginfo("Coverage follower entering perimeter bootstrap mode.")

    def _update_perimeter_progress(self, x, y):
        if self.perimeter_start_xy is None:
            self._reset_perimeter_progress(x, y, publish_reset=False)

        if self.perimeter_last_xy is not None:
            self.perimeter_travel_m += math.hypot(
                x - self.perimeter_last_xy[0], y - self.perimeter_last_xy[1]
            )
        self.perimeter_last_xy = (x, y)

        dist_from_start = math.hypot(
            x - self.perimeter_start_xy[0], y - self.perimeter_start_xy[1]
        )
        self.perimeter_min_x = min(self.perimeter_min_x, x)
        self.perimeter_max_x = max(self.perimeter_max_x, x)
        self.perimeter_min_y = min(self.perimeter_min_y, y)
        self.perimeter_max_y = max(self.perimeter_max_y, y)
        extent_x = self.perimeter_max_x - self.perimeter_min_x
        extent_y = self.perimeter_max_y - self.perimeter_min_y
        if dist_from_start >= self.perimeter_far_distance_m:
            self.perimeter_went_far = True

        age_s = (rospy.Time.now() - self.perimeter_started_stamp).to_sec()
        has_wall_contact = (not self.perimeter_require_wall_contact) or self.perimeter_wall_seen
        completed_loop = (
            has_wall_contact
            and
            self.perimeter_went_far
            and age_s >= self.perimeter_min_duration_s
            and self.perimeter_travel_m >= self.perimeter_min_travel_m
            and extent_x >= self.perimeter_min_extent_x_m
            and extent_y >= self.perimeter_min_extent_y_m
            and dist_from_start <= self.perimeter_close_tolerance_m
        )
        timeout_hit = (
            self.perimeter_max_duration_s > 0.0
            and age_s >= self.perimeter_max_duration_s
            and has_wall_contact
        )
        timeout_quality_ok = (
            self.perimeter_went_far
            and self.perimeter_travel_m
            >= self.perimeter_min_travel_m * self.perimeter_timeout_min_progress_ratio
            and extent_x >= self.perimeter_min_extent_x_m * self.perimeter_timeout_min_progress_ratio
            and extent_y >= self.perimeter_min_extent_y_m * self.perimeter_timeout_min_progress_ratio
        )

        if (
            self.perimeter_max_duration_s > 0.0
            and age_s >= self.perimeter_max_duration_s
            and not has_wall_contact
            and not self._perimeter_timeout_no_wall_warned
        ):
            rospy.logwarn(
                "Perimeter timeout reached but no wall contact detected yet; continuing perimeter seek."
            )
            self._perimeter_timeout_no_wall_warned = True

        if timeout_hit and not completed_loop and not timeout_quality_ok and not self.perimeter_done:
            if self.perimeter_timeout_restart_count < self.perimeter_timeout_max_restarts:
                self.perimeter_timeout_restart_count += 1
                rospy.logwarn(
                    "Perimeter timeout quality gate failed (travel=%.1fm extents=(%.2f, %.2f)m, retry %d/%d). Restarting perimeter bootstrap.",
                    self.perimeter_travel_m,
                    extent_x,
                    extent_y,
                    self.perimeter_timeout_restart_count,
                    self.perimeter_timeout_max_restarts,
                )
                self._reset_perimeter_progress(x, y, publish_reset=False)
                return
            rospy.logwarn(
                "Perimeter timeout quality gate exceeded retry budget; accepting timeout fallback."
            )

        if (completed_loop or timeout_hit) and not self.perimeter_done:
            self.perimeter_done = True
            self.perimeter_done_pub.publish(Bool(data=True))
            self.perimeter_timeout_restart_count = 0
            if completed_loop:
                rospy.loginfo(
                    "Perimeter bootstrap complete: travel=%.1fm age=%.1fs dist_to_start=%.2fm extents=(%.2f, %.2f)m",
                    self.perimeter_travel_m,
                    age_s,
                    dist_from_start,
                    extent_x,
                    extent_y,
                )
            else:
                rospy.logwarn(
                    "Perimeter bootstrap timed out after %.1fs, continuing to coverage phase.",
                    age_s,
                )

    def _sector_min_range(self, center_rad, points):
        if not points:
            return float("inf")
        half = math.radians(self.perimeter_sector_half_width_deg)
        outside_distances = []
        any_distances = []
        for bx, by in points:
            d = math.hypot(bx, by)
            if d < self.perimeter_self_filter_m:
                continue
            a = math.atan2(by, bx)
            if abs(_norm_angle(a - center_rad)) > half:
                continue
            any_distances.append(d)
            if self._outside_robot_footprint(bx, by):
                outside_distances.append(d)

        def robust_distance(distances):
            if not distances:
                return float("inf")
            distances.sort()
            if len(distances) < self.perimeter_sector_min_points:
                # Do not hide valid near-wall readings when sectors are sparse.
                return distances[0]
            idx = min(len(distances) - 1, self.perimeter_sector_distance_rank - 1)
            return distances[idx]

        # Prefer filtered points, but fall back to any in-sector points so
        # contact-level walls cannot disappear when footprint filtering removes all points.
        best_outside = robust_distance(outside_distances)
        if math.isfinite(best_outside):
            return best_outside
        return robust_distance(any_distances)

    def _ultrasonic_front_distance(self):
        if not self.use_ultrasonic_front or self.latest_ultrasonic_m is None:
            return float("inf"), False
        age = (rospy.Time.now() - self.latest_ultrasonic_stamp).to_sec()
        if age > self.ultrasonic_timeout_s:
            return float("inf"), False
        return self.latest_ultrasonic_m, True

    def _perimeter_observations(self):
        points = self._scan_points_in_base()
        front_lidar = self._sector_min_range(0.0, points)
        front_right_lidar = self._sector_min_range(-math.pi / 4.0, points)
        front_left_lidar = self._sector_min_range(math.pi / 4.0, points)
        right_lidar = self._sector_min_range(-math.pi / 2.0, points)
        left_lidar = self._sector_min_range(math.pi / 2.0, points)
        ultra_front, ultra_valid = self._ultrasonic_front_distance()

        if ultra_valid and math.isfinite(front_lidar):
            front_dist = min(front_lidar, ultra_front)
        elif ultra_valid:
            front_dist = ultra_front
        else:
            front_dist = front_lidar

        sensor_valid = bool(points) or ultra_valid
        return (
            front_dist,
            front_right_lidar,
            front_left_lidar,
            right_lidar,
            left_lidar,
            sensor_valid,
        )

    def _cap_reverse_turn_rate(self, linear_x, angular_z):
        # Keep both drive wheels in reverse while backing out of contact.
        # For diff drive: v_r = v + w*sep/2, v_l = v - w*sep/2.
        # If v is negative and |w| is too large, one wheel flips forward.
        if linear_x >= -1e-4:
            return angular_z
        sep = max(0.05, self.perimeter_drive_wheel_separation_m)
        max_turn = (2.0 * abs(linear_x) / sep) * self.perimeter_reverse_turn_ratio
        return _clamp(angular_z, -max_turn, max_turn)

    def _perimeter_recovery_cmd(self, turn_sign=-1.0):
        now = rospy.Time.now()
        cmd = Twist()
        cmd.linear.x = -abs(self.perimeter_escape_reverse_speed_mps)

        if now < self.perimeter_escape_reverse_until:
            cmd.angular.z = 0.0
            return cmd

        if self.perimeter_escape_use_pivot:
            cmd.linear.x = 0.0
            cmd.angular.z = turn_sign * abs(self.perimeter_escape_pivot_turn_rps)
            return cmd
        else:
            cmd.angular.z = turn_sign * abs(self.perimeter_right_turn_rate_rps)
        cmd.angular.z = self._cap_reverse_turn_rate(cmd.linear.x, cmd.angular.z)
        return cmd

    def _perimeter_corner_backoff_cmd(self):
        cmd = Twist()
        cmd.linear.x = -abs(self.perimeter_corner_backoff_speed_mps)
        turn_sign = 1.0 if self.perimeter_corner_turn_sign >= 0.0 else -1.0
        cmd.angular.z = turn_sign * abs(self.perimeter_corner_backoff_turn_rps)
        cmd.angular.z = self._cap_reverse_turn_rate(cmd.linear.x, cmd.angular.z)
        return cmd

    def _choose_turn_sign(self, right_dist, left_dist):
        right_clear = right_dist if math.isfinite(right_dist) else float("inf")
        left_clear = left_dist if math.isfinite(left_dist) else float("inf")

        if right_clear < self.perimeter_right_close_m and left_clear > right_clear:
            return 1.0
        if left_clear < self.perimeter_right_close_m and right_clear > left_clear:
            return -1.0
        if left_clear > right_clear + self.perimeter_front_turn_bias_m:
            return 1.0
        return -1.0

    def _choose_corner_turn_sign(self, right_dist, left_dist, front_right_dist, front_left_dist):
        # During perimeter tracing we keep boundary on the robot's right side,
        # so convex corners should consistently be negotiated with a left turn.
        # Flipping signs near equal-distance readings causes dithering/180s.
        return 1.0

    def _start_perimeter_corner_turn(self, yaw, turn_sign):
        now = rospy.Time.now()
        if self.perimeter_corner_turn_active:
            return
        if self.perimeter_corner_retry_required:
            return
        if now < self.perimeter_corner_backoff_until or now < self.perimeter_corner_retry_until:
            return
        sign = 1.0 if turn_sign >= 0.0 else -1.0
        angle = math.pi / 2.0
        if self.perimeter_corner_anchor_yaw is None:
            self.perimeter_corner_anchor_yaw = yaw
        self.perimeter_corner_turn_active = True
        self.perimeter_corner_turn_sign = sign
        self.perimeter_corner_turn_target_yaw = _norm_angle(
            self.perimeter_corner_anchor_yaw + sign * angle
        )
        self.perimeter_corner_turn_started_stamp = now
        rospy.loginfo_throttle(
            1.0,
            "Perimeter corner turn start: sign=%.0f target_yaw=%.2frad angle=%.2frad",
            sign,
            self.perimeter_corner_turn_target_yaw,
            angle,
        )

    def _perimeter_corner_turn_cmd(self, yaw):
        if not self.perimeter_corner_turn_active:
            return None

        now = rospy.Time.now()
        if self.perimeter_corner_turn_started_stamp == rospy.Time(0):
            self.perimeter_corner_turn_started_stamp = now

        err = _norm_angle(self.perimeter_corner_turn_target_yaw - yaw)
        if abs(err) <= self.perimeter_corner_turn_exit_rad:
            self.perimeter_corner_turn_active = False
            self.perimeter_corner_retry_required = False
            self.perimeter_corner_anchor_yaw = None
            self.perimeter_corner_trigger_hits = 0
            return None

        if self.perimeter_corner_turn_max_s > 0.0:
            elapsed = (now - self.perimeter_corner_turn_started_stamp).to_sec()
            if elapsed >= self.perimeter_corner_turn_max_s:
                if abs(err) <= self.perimeter_corner_timeout_accept_rad:
                    rospy.logwarn(
                        "Perimeter corner turn timeout but near-complete (yaw err %.2frad <= %.2frad), accepting turn.",
                        abs(err),
                        self.perimeter_corner_timeout_accept_rad,
                    )
                    self.perimeter_corner_turn_active = False
                    self.perimeter_corner_retry_required = False
                    self.perimeter_corner_anchor_yaw = None
                    self.perimeter_corner_trigger_hits = 0
                    return None
                backoff_s = max(0.0, self.perimeter_corner_backoff_s)
                retry_s = max(0.0, self.perimeter_corner_retry_cooldown_s)
                self.perimeter_corner_backoff_until = now + rospy.Duration(backoff_s)
                self.perimeter_corner_retry_until = now + rospy.Duration(retry_s)
                self.perimeter_corner_retry_required = True
                rospy.logwarn(
                    "Perimeter corner turn timeout after %.2fs (yaw err %.2frad), backoff %.2fs then retry. nav=(0.00,%.2f) raw=(%.2f,%.2f) out=(%.2f,%.2f) odom=(%.3f,%.3f) wl=%.3f wr=%.3f",
                    elapsed,
                    err,
                    backoff_s,
                    math.copysign(
                        max(abs(self.perimeter_corner_turn_rate_rps), self.perimeter_turn_stall_min_cmd_rps),
                        err,
                    ),
                    self._dbg_raw_linear,
                    self._dbg_raw_angular,
                    self._dbg_out_linear,
                    self._dbg_out_angular,
                    self._dbg_actual_linear,
                    self._dbg_actual_angular,
                    self._dbg_wheel_left_rps,
                    self._dbg_wheel_right_rps,
                )
                self.perimeter_corner_turn_active = False
                return None

        cmd = Twist()
        cmd.linear.x = abs(self.perimeter_corner_turn_linear_mps)
        turn_rate = max(
            abs(self.perimeter_corner_turn_rate_rps), self.perimeter_turn_stall_min_cmd_rps
        )
        cmd.angular.z = math.copysign(turn_rate, err)
        self.perimeter_corner_turn_sign = 1.0 if cmd.angular.z >= 0.0 else -1.0
        self._dbg_turning_in_place = abs(cmd.linear.x) < 0.03
        return cmd

    def _apply_perimeter_stall_escape(self, cmd, x, y, yaw):
        now = rospy.Time.now()
        if now < self.perimeter_escape_until:
            return self._perimeter_recovery_cmd(self.perimeter_escape_turn_sign)

        cmd.angular.z = self._cap_reverse_turn_rate(cmd.linear.x, cmd.angular.z)

        if self.perimeter_progress_ref_xy is None or self.perimeter_progress_ref_stamp == rospy.Time(0):
            self.perimeter_progress_ref_xy = (x, y)
            self.perimeter_progress_ref_stamp = now
        if self.perimeter_turn_ref_yaw is None or self.perimeter_turn_ref_stamp == rospy.Time(0):
            self.perimeter_turn_ref_yaw = yaw
            self.perimeter_turn_ref_stamp = now

        if abs(cmd.linear.x) < 0.03 and abs(cmd.angular.z) < self.perimeter_turn_stall_min_cmd_rps:
            self.perimeter_progress_ref_xy = (x, y)
            self.perimeter_progress_ref_stamp = now
            self.perimeter_turn_ref_yaw = yaw
            self.perimeter_turn_ref_stamp = now
            return cmd

        if abs(cmd.linear.x) >= 0.03:
            self.perimeter_turn_ref_yaw = yaw
            self.perimeter_turn_ref_stamp = now
            elapsed = (now - self.perimeter_progress_ref_stamp).to_sec()
            if elapsed < self.perimeter_stall_window_s:
                return cmd

            reverse_mismatch = (
                cmd.linear.x <= -self.motion_stuck_min_cmd_linear_mps
                and self._dbg_actual_linear >= self.forward_alignment_reverse_detect_mps
            )
            if reverse_mismatch:
                self.perimeter_stall_count += 1
                self.perimeter_escape_use_pivot = self.perimeter_stall_count > 1
                self.perimeter_escape_turn_sign = (
                    1.0 if self.perimeter_corner_turn_sign >= 0.0 else -1.0
                )
                self.perimeter_escape_until = now + rospy.Duration(self.perimeter_escape_turn_s)
                self.perimeter_escape_reverse_until = now + rospy.Duration(
                    self.perimeter_escape_reverse_only_s
                )
                rospy.logwarn(
                    "Perimeter reverse mismatch x%d (cmd_v=%.2f odom_v=%.2f), forcing %s recovery.",
                    self.perimeter_stall_count,
                    cmd.linear.x,
                    self._dbg_actual_linear,
                    "pivot" if self.perimeter_escape_use_pivot else "reverse-turn",
                )
                return self._perimeter_recovery_cmd(self.perimeter_escape_turn_sign)

            progress = math.hypot(
                x - self.perimeter_progress_ref_xy[0], y - self.perimeter_progress_ref_xy[1]
            )
            self.perimeter_progress_ref_xy = (x, y)
            self.perimeter_progress_ref_stamp = now

            if progress >= self.perimeter_stall_min_progress_m:
                self.perimeter_stall_count = 0
                self.perimeter_escape_use_pivot = False
                self.perimeter_escape_reverse_until = rospy.Time(0)
                return cmd

            self.perimeter_stall_count += 1
            self.perimeter_escape_turn_sign = -1.0 if cmd.angular.z <= 0.0 else 1.0
            if self.perimeter_stall_count > self.perimeter_escape_max_reverse_attempts:
                self.perimeter_escape_use_pivot = True
                self.perimeter_escape_turn_sign *= -1.0
                self.perimeter_escape_until = now + rospy.Duration(self.perimeter_escape_pivot_turn_s)
                self.perimeter_escape_reverse_until = now + rospy.Duration(
                    self.perimeter_escape_reverse_only_s
                )
                rospy.logwarn(
                    "Perimeter translation stall x%d (%.3fm in %.1fs), escalating to pivot escape.",
                    self.perimeter_stall_count,
                    progress,
                    elapsed,
                )
            else:
                self.perimeter_escape_use_pivot = False
                self.perimeter_escape_until = now + rospy.Duration(self.perimeter_escape_turn_s)
                self.perimeter_escape_reverse_until = now + rospy.Duration(
                    self.perimeter_escape_reverse_only_s
                )
                rospy.logwarn(
                    "Perimeter translation stall x%d (%.3fm in %.1fs), reverse-turn recovery.",
                    self.perimeter_stall_count,
                    progress,
                    elapsed,
                )
            return self._perimeter_recovery_cmd(self.perimeter_escape_turn_sign)

        elapsed = (now - self.perimeter_turn_ref_stamp).to_sec()
        if elapsed < self.perimeter_stall_window_s:
            return cmd

        yaw_delta = abs(_norm_angle(yaw - self.perimeter_turn_ref_yaw))
        self.perimeter_turn_ref_yaw = yaw
        self.perimeter_turn_ref_stamp = now
        self.perimeter_progress_ref_xy = (x, y)
        self.perimeter_progress_ref_stamp = now

        if yaw_delta >= self.perimeter_turn_stall_min_yaw_rad:
            self.perimeter_stall_count = 0
            self.perimeter_escape_use_pivot = False
            self.perimeter_escape_reverse_until = rospy.Time(0)
            return cmd

        self.perimeter_stall_count += 1
        self.perimeter_escape_turn_sign = -1.0 if cmd.angular.z <= 0.0 else 1.0
        if self.perimeter_stall_count > self.perimeter_escape_max_reverse_attempts:
            self.perimeter_escape_use_pivot = True
            self.perimeter_escape_turn_sign *= -1.0
            self.perimeter_escape_until = now + rospy.Duration(self.perimeter_escape_pivot_turn_s)
            self.perimeter_escape_reverse_until = now + rospy.Duration(
                self.perimeter_escape_reverse_only_s
            )
            rospy.logwarn(
                "Perimeter turn stall x%d (%.3frad in %.1fs), pivot escape.",
                self.perimeter_stall_count,
                yaw_delta,
                elapsed,
            )
        else:
            self.perimeter_escape_use_pivot = False
            self.perimeter_escape_until = now + rospy.Duration(self.perimeter_escape_turn_s)
            self.perimeter_escape_reverse_until = now + rospy.Duration(
                self.perimeter_escape_reverse_only_s
            )
            rospy.logwarn(
                "Perimeter turn stall x%d (%.3frad in %.1fs), reverse-turn recovery.",
                self.perimeter_stall_count,
                yaw_delta,
                elapsed,
            )
        return self._perimeter_recovery_cmd(self.perimeter_escape_turn_sign)

    def _reset_motion_stuck_refs(self, x=None, y=None, yaw=None):
        self.motion_stuck_ref_xy = (x, y) if (x is not None and y is not None) else None
        self.motion_stuck_ref_yaw = yaw
        self.motion_stuck_ref_stamp = rospy.Time.now()
        self.motion_stuck_escape_until = rospy.Time(0)
        self.motion_stuck_escape_reverse_until = rospy.Time(0)
        self.motion_stuck_count = 0

    def _motion_stuck_escape_cmd(self):
        now = rospy.Time.now()
        cmd = Twist()
        cmd.linear.x = -abs(self.motion_stuck_escape_reverse_mps)
        if now < self.motion_stuck_escape_reverse_until:
            cmd.angular.z = 0.0
        else:
            cmd.angular.z = self.motion_stuck_escape_turn_sign * abs(
                self.motion_stuck_escape_turn_rps
            )
        return cmd

    def _apply_motion_stuck_recovery(self, cmd):
        if not self.motion_stuck_enabled or self.odom is None:
            return cmd

        now = rospy.Time.now()
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        yaw = _yaw_from_quat(self.odom.pose.pose.orientation)

        if now < self.motion_stuck_escape_until:
            return self._motion_stuck_escape_cmd()

        cmd_linear = abs(cmd.linear.x)
        cmd_angular = abs(cmd.angular.z)
        wheel_spin = max(abs(self._dbg_wheel_left_rps), abs(self._dbg_wheel_right_rps))

        cmd_active = (
            cmd_linear >= self.motion_stuck_min_cmd_linear_mps
            or cmd_angular >= self.motion_stuck_min_cmd_angular_rps
        )
        wheels_active = wheel_spin >= self.motion_stuck_min_wheel_rps

        if not (cmd_active and wheels_active):
            self.motion_stuck_ref_xy = (x, y)
            self.motion_stuck_ref_yaw = yaw
            self.motion_stuck_ref_stamp = now
            self.motion_stuck_count = 0
            return cmd

        if self.motion_stuck_ref_xy is None or self.motion_stuck_ref_stamp == rospy.Time(0):
            self.motion_stuck_ref_xy = (x, y)
            self.motion_stuck_ref_yaw = yaw
            self.motion_stuck_ref_stamp = now
            return cmd

        elapsed = (now - self.motion_stuck_ref_stamp).to_sec()
        if elapsed < self.motion_stuck_window_s:
            return cmd

        progress = math.hypot(x - self.motion_stuck_ref_xy[0], y - self.motion_stuck_ref_xy[1])
        yaw_delta = abs(_norm_angle(yaw - (self.motion_stuck_ref_yaw or yaw)))

        self.motion_stuck_ref_xy = (x, y)
        self.motion_stuck_ref_yaw = yaw
        self.motion_stuck_ref_stamp = now

        abs_actual_linear = abs(self._dbg_actual_linear)
        abs_actual_angular = abs(self._dbg_actual_angular)
        expected_progress = cmd_linear * elapsed
        expected_yaw = cmd_angular * elapsed
        progress_ratio = (
            progress / max(expected_progress, 1e-3)
            if cmd_linear >= self.motion_stuck_min_cmd_linear_mps
            else 1.0
        )
        yaw_ratio = (
            yaw_delta / max(expected_yaw, 1e-3)
            if cmd_angular >= self.motion_stuck_min_cmd_angular_rps
            else 1.0
        )
        low_motion = (
            progress < self.motion_stuck_min_progress_m
            and yaw_delta < self.motion_stuck_min_yaw_rad
            and abs_actual_linear < 0.05
            and abs_actual_angular < 0.20
        )
        linear_mismatch = (
            cmd_linear >= self.motion_stuck_min_cmd_linear_mps
            and progress_ratio < self.motion_stuck_min_effective_ratio
            and abs_actual_linear < (cmd_linear * self.motion_stuck_min_effective_ratio)
        )
        angular_mismatch = (
            cmd_angular >= self.motion_stuck_min_cmd_angular_rps
            and yaw_ratio < self.motion_stuck_min_effective_ratio
            and abs_actual_angular < (cmd_angular * self.motion_stuck_min_effective_ratio)
        )
        low_motion = low_motion or linear_mismatch or angular_mismatch
        if not low_motion:
            self.motion_stuck_count = 0
            return cmd

        self.motion_stuck_count += 1
        if cmd.angular.z != 0.0:
            if self.motion_stuck_keep_turn_direction:
                self.motion_stuck_escape_turn_sign = 1.0 if cmd.angular.z > 0.0 else -1.0
            else:
                self.motion_stuck_escape_turn_sign = -1.0 if cmd.angular.z > 0.0 else 1.0
        else:
            wheel_delta = self._dbg_wheel_left_rps - self._dbg_wheel_right_rps
            if abs(wheel_delta) < 0.05:
                self.motion_stuck_escape_turn_sign *= -1.0
            else:
                self.motion_stuck_escape_turn_sign = 1.0 if wheel_delta > 0.0 else -1.0
        self.motion_stuck_escape_until = now + rospy.Duration(self.motion_stuck_escape_duration_s)
        self.motion_stuck_escape_reverse_until = now + rospy.Duration(
            self.motion_stuck_escape_reverse_only_s
        )
        rospy.logwarn(
            "Global motion stall x%d (progress=%.3fm yaw=%.3frad in %.1fs, wl=%.2f wr=%.2f). Running reverse-turn unstick.",
            self.motion_stuck_count,
            progress,
            yaw_delta,
            elapsed,
            self._dbg_wheel_left_rps,
            self._dbg_wheel_right_rps,
        )
        return self._motion_stuck_escape_cmd()

    def _apply_forward_alignment_guard(self, cmd):
        if not self.forward_alignment_guard_enabled:
            return cmd
        if cmd.linear.x < self.forward_alignment_min_cmd_mps:
            return cmd
        # If the robot is actually moving backward while we command forward,
        # stop driving straight and enforce an in-place turn to realign.
        if self._dbg_actual_linear > -self.forward_alignment_reverse_detect_mps:
            return cmd

        guarded = Twist()
        guarded.linear.x = 0.0
        if abs(cmd.angular.z) >= self.forward_alignment_min_turn_rps:
            guarded.angular.z = cmd.angular.z
        elif abs(self._dbg_raw_angular) >= self.forward_alignment_min_turn_rps:
            guarded.angular.z = self._dbg_raw_angular
        else:
            sign = 1.0 if self._dbg_wheel_left_rps >= self._dbg_wheel_right_rps else -1.0
            guarded.angular.z = sign * self.forward_alignment_min_turn_rps
        rospy.logwarn_throttle(
            1.0,
            "Forward alignment guard active: cmd_v=%.2f actual_v=%.2f forcing turn-in-place w=%.2f",
            cmd.linear.x,
            self._dbg_actual_linear,
            guarded.angular.z,
        )
        return guarded

    def _compute_perimeter_cmd(self, x, y, yaw):
        now = rospy.Time.now()
        if now < self.perimeter_escape_until:
            return self._perimeter_recovery_cmd(self.perimeter_escape_turn_sign)

        (
            front_dist,
            front_right_dist,
            front_left_dist,
            right_dist,
            left_dist,
            sensor_valid,
        ) = self._perimeter_observations()
        dynamic_front_stop_m = max(
            self.perimeter_front_stop_m,
            abs(self.perimeter_forward_speed_mps) * self.perimeter_front_predict_time_s,
        )
        emergency_stop_m = min(dynamic_front_stop_m, self.perimeter_emergency_stop_m)
        emergency_votes = sum(
            1
            for d in (front_dist, front_right_dist, front_left_dist)
            if math.isfinite(d) and d < emergency_stop_m
        )
        self._dbg_front = front_dist
        self._dbg_front_right = front_right_dist
        self._dbg_front_left = front_left_dist
        self._dbg_right = right_dist
        self._dbg_left = left_dist
        self._dbg_dynamic_stop = dynamic_front_stop_m
        dynamic_front_clear_m = max(self.perimeter_front_clear_m, dynamic_front_stop_m + 0.35)

        if self.perimeter_corner_retry_required:
            if math.isfinite(front_dist) and front_dist >= self.perimeter_corner_retry_clearance_m:
                self.perimeter_corner_retry_required = False
                self.perimeter_corner_retry_until = rospy.Time(0)
                self.perimeter_corner_anchor_yaw = None
                self.perimeter_corner_trigger_hits = 0
            else:
                cmd = self._perimeter_corner_backoff_cmd()
                return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        if now < self.perimeter_corner_backoff_until:
            cmd = self._perimeter_corner_backoff_cmd()
            return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        if now < self.perimeter_corner_retry_until:
            if math.isfinite(front_dist) and front_dist >= self.perimeter_corner_retry_clearance_m:
                self.perimeter_corner_retry_until = rospy.Time(0)
                self.perimeter_corner_anchor_yaw = None
                self.perimeter_corner_trigger_hits = 0
            else:
                cmd = self._perimeter_corner_backoff_cmd()
                return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        if (
            not self.perimeter_wall_seen
            and math.isfinite(right_dist)
            and right_dist <= self.perimeter_right_acquire_m
        ):
            self.perimeter_wall_seen = True
            rospy.loginfo("Perimeter bootstrap acquired right boundary at %.2fm.", right_dist)
        if (
            not self.perimeter_corner_turn_active
            and not self.perimeter_corner_retry_required
            and now >= self.perimeter_corner_backoff_until
            and now >= self.perimeter_corner_retry_until
            and (
                not math.isfinite(front_dist)
                or front_dist >= dynamic_front_clear_m
            )
        ):
            self.perimeter_corner_anchor_yaw = None

        cmd = Twist()
        if self.perimeter_corner_turn_active:
            turn_cmd = self._perimeter_corner_turn_cmd(yaw)
            if turn_cmd is not None:
                return self._apply_perimeter_stall_escape(turn_cmd, x, y, yaw)
            if now < self.perimeter_corner_backoff_until:
                cmd = self._perimeter_corner_backoff_cmd()
                return self._apply_perimeter_stall_escape(cmd, x, y, yaw)
        if not sensor_valid:
            cmd.angular.z = -abs(self.perimeter_blind_turn_rps)
            return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        if not self.perimeter_wall_seen:
            if math.isfinite(front_dist) and front_dist < dynamic_front_stop_m:
                self.perimeter_wall_seen = True
                turn_sign = self._choose_corner_turn_sign(
                    right_dist, left_dist, front_right_dist, front_left_dist
                )
                self._start_perimeter_corner_turn(yaw, turn_sign)
                turn_cmd = self._perimeter_corner_turn_cmd(yaw)
                if turn_cmd is not None:
                    return self._apply_perimeter_stall_escape(turn_cmd, x, y, yaw)
            cmd.linear.x = self.perimeter_forward_speed_mps
            cmd.angular.z = -abs(self.perimeter_seek_turn_rps)
            return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        if emergency_votes >= self.perimeter_emergency_stop_votes:
            # Hard safety override near contact: no forward pre-turn at this range.
            turn_sign = self._choose_corner_turn_sign(
                right_dist, left_dist, front_right_dist, front_left_dist
            )
            self._start_perimeter_corner_turn(yaw, turn_sign)
            turn_cmd = self._perimeter_corner_turn_cmd(yaw)
            if turn_cmd is not None:
                return self._apply_perimeter_stall_escape(turn_cmd, x, y, yaw)

        corner_trigger = min(front_dist, front_right_dist)
        corner_candidate = (
            math.isfinite(corner_trigger)
            and corner_trigger < self.perimeter_corner_preturn_m
            and corner_trigger >= dynamic_front_stop_m
        )
        if corner_candidate:
            self.perimeter_corner_trigger_hits += 1
        else:
            self.perimeter_corner_trigger_hits = 0

        if self.perimeter_corner_trigger_hits >= self.perimeter_corner_trigger_count:
            turn_sign = self._choose_corner_turn_sign(
                right_dist, left_dist, front_right_dist, front_left_dist
            )
            self._start_perimeter_corner_turn(yaw, turn_sign)
            turn_cmd = self._perimeter_corner_turn_cmd(yaw)
            if turn_cmd is not None:
                rospy.loginfo_throttle(
                    1.0,
                    "Perimeter corner turn active: front=%.2f front_right=%.2f right=%.2f yaw_cmd=%.2f",
                    front_dist,
                    front_right_dist,
                    right_dist,
                    turn_cmd.angular.z,
                )
                return self._apply_perimeter_stall_escape(turn_cmd, x, y, yaw)

        if math.isfinite(front_dist) and front_dist < dynamic_front_clear_m:
            turn_sign = self._choose_corner_turn_sign(
                right_dist, left_dist, front_right_dist, front_left_dist
            )
            self._start_perimeter_corner_turn(yaw, turn_sign)
            turn_cmd = self._perimeter_corner_turn_cmd(yaw)
            if turn_cmd is not None:
                return self._apply_perimeter_stall_escape(turn_cmd, x, y, yaw)

        cmd.linear.x = self.perimeter_forward_speed_mps
        if not math.isfinite(right_dist):
            cmd.angular.z = -abs(min(self.perimeter_turn_rate_rps, 0.25))
            return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        if right_dist > self.perimeter_right_open_m:
            cmd.angular.z = -abs(self.perimeter_turn_rate_rps)
            return self._apply_perimeter_stall_escape(cmd, x, y, yaw)
        if right_dist < self.perimeter_right_close_m:
            cmd.angular.z = abs(self.perimeter_turn_rate_rps)
            return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

        right_err = right_dist - self.perimeter_right_target_m
        cmd.angular.z = _clamp(
            -self.perimeter_wall_kp * right_err,
            -self.perimeter_wall_max_turn_rps,
            self.perimeter_wall_max_turn_rps,
        )
        return self._apply_perimeter_stall_escape(cmd, x, y, yaw)

    def _step(self):
        if not self.enabled:
            if self._prev_enabled:
                self.returning_home = False
                self.home_reached = False
                self.coverage_turning_in_place = False
                self._needs_reseed = True
                if self.perimeter_bootstrap_enabled and self.perimeter_repeat_each_enable:
                    self.perimeter_done = False
                    self.perimeter_done_pub.publish(Bool(data=False))
                    self.perimeter_start_xy = None
                    self.perimeter_last_xy = None
                    self.perimeter_started_stamp = rospy.Time(0)
                    self.perimeter_travel_m = 0.0
                    self.perimeter_went_far = False
                    self.perimeter_min_x = 0.0
                    self.perimeter_max_x = 0.0
                    self.perimeter_min_y = 0.0
                    self.perimeter_max_y = 0.0
                    self.perimeter_progress_ref_xy = None
                    self.perimeter_progress_ref_stamp = rospy.Time(0)
                    self.perimeter_turn_ref_yaw = None
                    self.perimeter_turn_ref_stamp = rospy.Time(0)
                    self.perimeter_escape_until = rospy.Time(0)
                    self.perimeter_escape_reverse_until = rospy.Time(0)
                    self.perimeter_escape_turn_sign = -1.0
                    self.perimeter_stall_count = 0
                    self.perimeter_escape_use_pivot = False
                    self.perimeter_wall_seen = False
                    self.perimeter_corner_turn_active = False
                    self.perimeter_corner_turn_target_yaw = 0.0
                    self.perimeter_corner_turn_started_stamp = rospy.Time(0)
                    self.perimeter_corner_turn_sign = -1.0
                    self.perimeter_corner_anchor_yaw = None
                    self.perimeter_corner_trigger_hits = 0
                    self.perimeter_corner_backoff_until = rospy.Time(0)
                    self.perimeter_corner_retry_until = rospy.Time(0)
                    self.perimeter_corner_retry_required = False
                    self.perimeter_timeout_restart_count = 0
                    self._perimeter_timeout_no_wall_warned = False
                self.motion_stuck_ref_xy = None
                self.motion_stuck_ref_yaw = None
                self.motion_stuck_ref_stamp = rospy.Time(0)
                self.motion_stuck_escape_until = rospy.Time(0)
                self.motion_stuck_escape_reverse_until = rospy.Time(0)
                self.motion_stuck_count = 0
            self._prev_enabled = False
            self._set_state("disabled")
            return Twist()

        if self.odom is None:
            self._set_state("waiting_for_odom")
            return Twist()
        if (rospy.Time.now() - self.odom_stamp).to_sec() > self.odom_timeout_s:
            self._set_state("odom_timeout")
            return Twist()

        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        yaw = _yaw_from_quat(self.odom.pose.pose.orientation)
        self._dbg_turning_in_place = False

        if self.enabled and not self._prev_enabled:
            self.start_xy = (x, y)
            self.returning_home = False
            self.home_reached = False
            self.coverage_turning_in_place = False
            self._needs_reseed = True
            if self.perimeter_bootstrap_enabled and (
                self.perimeter_repeat_each_enable or not self.perimeter_done
            ):
                self.perimeter_timeout_restart_count = 0
                self._reset_perimeter_progress(x, y, publish_reset=True)
            rospy.loginfo("Coverage follower captured mission start at x=%.2f y=%.2f", x, y)
        self._prev_enabled = True

        # Phase 1: map perimeter before coverage planning/following.
        if self.perimeter_bootstrap_enabled and not self.perimeter_done:
            self._update_perimeter_progress(x, y)
            if not self.perimeter_done:
                self._set_state("mapping_perimeter")
                return self._compute_perimeter_cmd(x, y, yaw)
            self._set_state("perimeter_complete")

        if self.path is None or len(self.path.poses) == 0:
            self._set_state("waiting_for_path")
            return Twist()

        pose_frame = self.odom.header.frame_id.strip() if self.odom.header.frame_id else "odom"
        target_frame = self.path_frame or pose_frame
        if target_frame != pose_frame:
            transformed = self._transform_pose_to_frame(x, y, yaw, pose_frame, target_frame)
            if transformed is None:
                self._set_state("waiting_for_tf")
                return Twist()
            x, y, yaw = transformed

        if self.start_nearest_waypoint and self._needs_reseed and not self.returning_home:
            self.idx = self._nearest_waypoint_index(x, y)
            self._needs_reseed = False
            rospy.loginfo(
                "Coverage follower reseeded to nearest waypoint idx=%d/%d.",
                self.idx,
                len(self.path.poses),
            )

        if not self.returning_home:
            # While performing stop-turn-go at a fixed spot, do not advance path
            # index; otherwise near-waypoint tolerance can "skip" rows and flip
            # heading targets behind the robot.
            if not self.coverage_turning_in_place:
                advance_budget = max(1, self.coverage_max_waypoint_advance)
                while self.idx < len(self.path.poses) and advance_budget > 0:
                    p = self.path.poses[self.idx].pose.position
                    d = math.hypot(p.x - x, p.y - y)
                    tol = (
                        self.goal_tolerance
                        if self.idx == len(self.path.poses) - 1
                        else self.waypoint_tolerance
                    )
                    if d <= tol:
                        self.idx += 1
                        advance_budget -= 1
                    else:
                        break

            if self.idx >= len(self.path.poses):
                if self.return_to_start and self.start_xy is not None and not self.home_reached:
                    self.returning_home = True
                    rospy.loginfo("Coverage complete, returning to mission start.")
                else:
                    self._set_state("coverage_complete")
                    return Twist()

        if self.returning_home and self.start_xy is not None:
            cmd, home_dist = self._compute_cmd_to_target(x, y, yaw, self.start_xy[0], self.start_xy[1])
            if home_dist <= self.goal_tolerance:
                self.returning_home = False
                self.home_reached = True
                self._set_state("home_reached")
                rospy.loginfo("Return-to-start complete.")
                return Twist()
            self._set_state("returning_home")
            return cmd

        tgt = self.path.poses[self.idx].pose.position
        cmd, _ = self._compute_cmd_to_target(x, y, yaw, tgt.x, tgt.y)
        self._set_state("following_coverage")
        return cmd

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            cmd = self._step()
            cmd = self._apply_motion_stuck_recovery(cmd)
            cmd = self._apply_forward_alignment_guard(cmd)
            if (
                not self._dbg_turning_in_place
                and abs(cmd.linear.x) < 0.03
                and abs(cmd.angular.z) > 0.20
            ):
                self._dbg_turning_in_place = True
            self._publish_debug_markers(cmd)
            self.cmd_pub.publish(cmd)
            rate.sleep()
        self._publish_stop()


def main():
    rospy.init_node("coverage_path_follower")
    CoveragePathFollower().spin()


if __name__ == "__main__":
    main()

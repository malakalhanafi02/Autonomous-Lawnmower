#!/usr/bin/env python3
import math

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _quat_from_yaw(yaw):
    half = 0.5 * yaw
    return math.sin(half), math.cos(half)


def _norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class InPlaceTurnAssist:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "mower")
        raw_hints = rospy.get_param("~model_name_hints", "mower,robot,base")
        self.model_name_hints = [h.strip().lower() for h in str(raw_hints).split(",") if h.strip()]
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.mode_topic = rospy.get_param("~mode_topic", "/mower/mode")
        self.enabled_topic = rospy.get_param("~enabled_topic", "/mower/autonomy_enabled")
        self.model_states_topic = rospy.get_param("~model_states_topic", "/gazebo/model_states")
        self.set_model_state_service = rospy.get_param("~set_model_state_service", "/gazebo/set_model_state")
        self.min_turn_rate_rps = float(rospy.get_param("~min_turn_rate_rps", 0.2))
        self.max_linear_for_pivot_mps = float(rospy.get_param("~max_linear_for_pivot_mps", 0.08))
        self.angular_scale = float(rospy.get_param("~angular_scale", 1.0))
        self.max_yaw_step_rad = float(rospy.get_param("~max_yaw_step_rad", 0.03))
        self.yaw_tolerance_rad = float(rospy.get_param("~yaw_tolerance_rad", 0.012))
        self.cmd_timeout_s = float(rospy.get_param("~cmd_timeout_s", 0.35))
        self.loop_hz = float(rospy.get_param("~loop_hz", 40.0))
        self.require_auto_mode = _as_bool(rospy.get_param("~require_auto_mode", True))
        self.require_autonomy_enabled = _as_bool(
            rospy.get_param("~require_autonomy_enabled", True)
        )
        self.zero_twist_on_set = _as_bool(rospy.get_param("~zero_twist_on_set", True))
        self.reference_frame = rospy.get_param("~reference_frame", "world")

        self.latest_cmd = Twist()
        self.mode = "manual"
        self.autonomy_enabled = False
        self.latest_pose = None
        self.latest_cmd_stamp = rospy.Time(0)
        self._estimated_yaw = None
        self._target_yaw = None
        self._resolved_model_name = self.model_name

        self.active_pub = rospy.Publisher("~active", Bool, queue_size=5)
        rospy.Subscriber(self.cmd_topic, Twist, self._cmd_cb, queue_size=20)
        rospy.Subscriber(self.mode_topic, String, self._mode_cb, queue_size=10)
        rospy.Subscriber(self.enabled_topic, Bool, self._enabled_cb, queue_size=10)
        rospy.Subscriber(self.model_states_topic, ModelStates, self._model_states_cb, queue_size=5)

    def _cmd_cb(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_stamp = rospy.Time.now()

    def _mode_cb(self, msg):
        self.mode = msg.data.strip().lower()

    def _enabled_cb(self, msg):
        self.autonomy_enabled = _as_bool(msg.data)

    def _resolve_model_index(self, msg):
        names = msg.name
        if not names:
            return None

        # Fast path: current resolved name still valid.
        try:
            return names.index(self._resolved_model_name)
        except ValueError:
            pass

        # Exact configured name match.
        try:
            idx = names.index(self.model_name)
            self._resolved_model_name = names[idx]
            rospy.loginfo("in_place_turn_assist resolved model name: %s", self._resolved_model_name)
            return idx
        except ValueError:
            pass

        # Hint-based fuzzy match.
        for i, name in enumerate(names):
            lname = name.lower()
            if any(h in lname for h in self.model_name_hints):
                self._resolved_model_name = name
                rospy.loginfo("in_place_turn_assist auto-selected model name: %s", self._resolved_model_name)
                return i

        rospy.logwarn_throttle(
            2.0,
            "in_place_turn_assist cannot find model '%s' in /gazebo/model_states (available: %s)",
            self.model_name,
            ",".join(names[:8]) + ("..." if len(names) > 8 else ""),
        )
        return None

    def _model_states_cb(self, msg):
        idx = self._resolve_model_index(msg)
        if idx is None:
            return
        self.latest_pose = msg.pose[idx]
        self._estimated_yaw = _yaw_from_quat(self.latest_pose.orientation)
        if self._target_yaw is None:
            self._target_yaw = self._estimated_yaw

    def _assist_allowed(self):
        if self.require_auto_mode and self.mode != "auto":
            return False
        if self.require_autonomy_enabled and not self.autonomy_enabled:
            return False
        return True

    def _should_pivot_assist(self):
        if not self._assist_allowed():
            return False
        if self.latest_pose is None:
            return False
        age = (rospy.Time.now() - self.latest_cmd_stamp).to_sec()
        if age > self.cmd_timeout_s:
            return False
        if abs(self.latest_cmd.linear.x) > self.max_linear_for_pivot_mps:
            return False
        return abs(self.latest_cmd.angular.z) >= self.min_turn_rate_rps

    def spin(self):
        rospy.loginfo(
            "in_place_turn_assist waiting for %s (model=%s, cmd=%s)",
            self.set_model_state_service,
            self.model_name,
            self.cmd_topic,
        )
        rospy.wait_for_service(self.set_model_state_service, timeout=15.0)
        set_state = rospy.ServiceProxy(self.set_model_state_service, SetModelState)

        last = rospy.Time.now()
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - last).to_sec()
            last = now
            active = False
            if 0.0 < dt < 0.5 and self.latest_pose is not None and self._assist_allowed():
                if self._estimated_yaw is None:
                    self._estimated_yaw = _yaw_from_quat(self.latest_pose.orientation)
                if self._target_yaw is None:
                    self._target_yaw = self._estimated_yaw

                # Integrate commanded angular motion into a target yaw, but apply
                # only in small bounded steps for precise, non-repetitive pivots.
                if self._should_pivot_assist():
                    requested_delta = self.latest_cmd.angular.z * self.angular_scale * dt
                    requested_delta = _clamp(
                        requested_delta,
                        -self.max_yaw_step_rad,
                        self.max_yaw_step_rad,
                    )
                    self._target_yaw = _norm_angle(self._target_yaw + requested_delta)
                elif abs(self.latest_cmd.linear.x) > self.max_linear_for_pivot_mps:
                    # Linear drive should never keep residual pivot target.
                    self._target_yaw = self._estimated_yaw

                yaw_err = _norm_angle(self._target_yaw - self._estimated_yaw)
                if abs(yaw_err) > self.yaw_tolerance_rad:
                    step = _clamp(yaw_err, -self.max_yaw_step_rad, self.max_yaw_step_rad)
                    yaw_next = _norm_angle(self._estimated_yaw + step)
                    qz, qw = _quat_from_yaw(yaw_next)

                    state = ModelState()
                    state.model_name = self._resolved_model_name
                    state.reference_frame = self.reference_frame
                    state.pose = self.latest_pose
                    state.pose.orientation.x = 0.0
                    state.pose.orientation.y = 0.0
                    state.pose.orientation.z = qz
                    state.pose.orientation.w = qw
                    if self.zero_twist_on_set:
                        state.twist = Twist()

                    req = SetModelStateRequest(model_state=state)
                    try:
                        resp = set_state(req)
                        if resp.success:
                            self._estimated_yaw = yaw_next
                            self.latest_pose.orientation.x = 0.0
                            self.latest_pose.orientation.y = 0.0
                            self.latest_pose.orientation.z = qz
                            self.latest_pose.orientation.w = qw
                            active = True
                        else:
                            rospy.logwarn_throttle(
                                1.0, "in_place_turn_assist set_model_state failed: %s", resp.status_message
                            )
                    except rospy.ServiceException as exc:
                        rospy.logwarn_throttle(1.0, "in_place_turn_assist service error: %s", exc)

            self.active_pub.publish(Bool(data=active))
            rate.sleep()


def main():
    rospy.init_node("in_place_turn_assist")
    InPlaceTurnAssist().spin()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class CmdVelArbiter:
    def __init__(self):
        self.manual_topic = rospy.get_param("~manual_topic", "/cmd_vel_manual")
        self.nav_topic = rospy.get_param("~nav_topic", "/cmd_vel_nav")
        self.output_topic = rospy.get_param("~output_topic", "/cmd_vel_raw")
        self.mode_topic = rospy.get_param("~mode_topic", "/mower/mode")
        self.obstacle_alert_topic = rospy.get_param("~obstacle_alert_topic", "/camera/obstacle_alert")
        self.stop_on_obstacle_alert = _as_bool(
            rospy.get_param("~stop_on_obstacle_alert", True)
        )
        self.default_mode = rospy.get_param("~default_mode", "manual")
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.5))
        self.loop_hz = float(rospy.get_param("~loop_hz", 30.0))

        self.mode = self.default_mode
        self.manual_cmd = Twist()
        self.nav_cmd = Twist()
        self.manual_stamp = rospy.Time(0)
        self.nav_stamp = rospy.Time(0)
        self.obstacle_alert = False

        rospy.Subscriber(self.manual_topic, Twist, self._manual_cb, queue_size=20)
        rospy.Subscriber(self.nav_topic, Twist, self._nav_cb, queue_size=20)
        rospy.Subscriber(self.mode_topic, String, self._mode_cb, queue_size=5)
        rospy.Subscriber(self.obstacle_alert_topic, Bool, self._obstacle_alert_cb, queue_size=5)

        self.cmd_pub = rospy.Publisher(self.output_topic, Twist, queue_size=20)
        self.source_pub = rospy.Publisher("~active_source", String, queue_size=5)

    def _manual_cb(self, msg):
        self.manual_cmd = msg
        self.manual_stamp = rospy.Time.now()

    def _nav_cb(self, msg):
        self.nav_cmd = msg
        self.nav_stamp = rospy.Time.now()

    def _mode_cb(self, msg):
        self.mode = msg.data.strip().lower()

    def _obstacle_alert_cb(self, msg):
        self.obstacle_alert = msg.data

    def _is_fresh(self, stamp):
        if stamp == rospy.Time(0):
            return False
        return (rospy.Time.now() - stamp).to_sec() <= self.cmd_timeout

    @staticmethod
    def _zero_twist():
        return Twist()

    def _select_cmd(self):
        if self.stop_on_obstacle_alert and self.obstacle_alert:
            return self._zero_twist(), "camera_alert_stop"

        mode = self.mode
        if mode not in ("manual", "auto", "safe_stop"):
            mode = self.default_mode

        if mode == "safe_stop":
            return self._zero_twist(), "safe_stop"

        if mode == "auto":
            if self._is_fresh(self.nav_stamp):
                return self.nav_cmd, "nav"
            return self._zero_twist(), "nav_timeout"

        if self._is_fresh(self.manual_stamp):
            return self.manual_cmd, "manual"
        return self._zero_twist(), "manual_timeout"

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            cmd, source = self._select_cmd()
            self.cmd_pub.publish(cmd)
            self.source_pub.publish(String(data=source))
            rate.sleep()


def main():
    rospy.init_node("cmd_vel_arbiter")
    CmdVelArbiter().spin()


if __name__ == "__main__":
    main()

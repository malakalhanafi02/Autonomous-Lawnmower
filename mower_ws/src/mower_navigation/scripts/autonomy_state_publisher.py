#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, String


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class AutonomyStatePublisher:
    def __init__(self):
        self.mode = rospy.get_param("~mode", "manual").strip().lower()
        self.autonomy_enabled = _as_bool(rospy.get_param("~autonomy_enabled", False))
        self.rate_hz = float(rospy.get_param("~rate_hz", 1.0))
        self.mode_topic = rospy.get_param("~mode_topic", "/mower/mode")
        self.enabled_topic = rospy.get_param("~enabled_topic", "/mower/autonomy_enabled")

        self.mode_pub = rospy.Publisher(self.mode_topic, String, queue_size=5, latch=True)
        self.enabled_pub = rospy.Publisher(self.enabled_topic, Bool, queue_size=5, latch=True)
        rospy.Subscriber(self.mode_topic, String, self._mode_cb, queue_size=10)
        rospy.Subscriber(self.enabled_topic, Bool, self._enabled_cb, queue_size=10)

    def _mode_cb(self, msg):
        incoming = msg.data.strip().lower()
        if incoming and incoming != self.mode:
            self.mode = incoming
            self.mode_pub.publish(String(data=self.mode))

    def _enabled_cb(self, msg):
        incoming = _as_bool(msg.data)
        if incoming != self.autonomy_enabled:
            self.autonomy_enabled = incoming
            self.enabled_pub.publish(Bool(data=self.autonomy_enabled))

    def spin(self):
        # Publish initial configured state once (latched), then keep current
        # runtime state alive without forcing it back to launch defaults.
        # Latching retains these values for future subscribers, so a periodic
        # republish loop is unnecessary and can override runtime mode changes.
        self.mode_pub.publish(String(data=self.mode))
        self.enabled_pub.publish(Bool(data=self.autonomy_enabled))
        rospy.spin()


def main():
    rospy.init_node("autonomy_state_publisher")
    AutonomyStatePublisher().spin()


if __name__ == "__main__":
    main()

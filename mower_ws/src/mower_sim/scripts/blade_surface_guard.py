#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64


class BladeSurfaceGuard:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.command_topic = rospy.get_param("~command_topic", "/cutter_controller/command")
        self.surface_topic = rospy.get_param("~surface_topic", "/mower_sim/on_concrete")
        self.enabled_topic = rospy.get_param("~blade_enabled_topic", "/mower_sim/blade_enabled")

        self.blade_speed_on = float(rospy.get_param("~blade_speed_on", 45.0))
        self.blade_speed_off = float(rospy.get_param("~blade_speed_off", 0.0))
        self.concrete_x_min = float(rospy.get_param("~concrete_x_min", 0.0))
        self.concrete_x_max = float(rospy.get_param("~concrete_x_max", 4.0))
        self.concrete_y_min = float(rospy.get_param("~concrete_y_min", 0.5))
        self.concrete_y_max = float(rospy.get_param("~concrete_y_max", 3.5))
        self.loop_hz = float(rospy.get_param("~loop_hz", 20.0))

        self.on_concrete = False
        self._have_odom = False

        self.cmd_pub = rospy.Publisher(self.command_topic, Float64, queue_size=10)
        self.surface_pub = rospy.Publisher(self.surface_topic, Bool, queue_size=10)
        self.enabled_pub = rospy.Publisher(self.enabled_topic, Bool, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)

    def _odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.on_concrete = (self.concrete_x_min <= x <= self.concrete_x_max) and (
            self.concrete_y_min <= y <= self.concrete_y_max
        )
        self._have_odom = True

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            if self._have_odom and self.on_concrete:
                self.cmd_pub.publish(Float64(data=self.blade_speed_off))
                self.surface_pub.publish(Bool(data=True))
                self.enabled_pub.publish(Bool(data=False))
            else:
                self.cmd_pub.publish(Float64(data=self.blade_speed_on))
                self.surface_pub.publish(Bool(data=False))
                self.enabled_pub.publish(Bool(data=True))
            rate.sleep()


def main():
    rospy.init_node("blade_surface_guard")
    BladeSurfaceGuard().spin()


if __name__ == "__main__":
    main()

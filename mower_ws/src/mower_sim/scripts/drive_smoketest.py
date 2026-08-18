#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DriveSmokeTest:
    def __init__(self):
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.last_odom = None
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)

    def odom_cb(self, msg):
        self.last_odom = msg

    @staticmethod
    def yaw_from_quat(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def wrap_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def wait_for_odom(self, timeout=10.0):
        end = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < end:
            if self.last_odom is not None:
                return True
            rate.sleep()
        return False

    def send_for(self, linear_x, angular_z, seconds):
        rate = rospy.Rate(20)
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        ticks = int(seconds * 20)
        for _ in range(ticks):
            self.cmd_pub.publish(cmd)
            rate.sleep()

    def stop(self):
        self.cmd_pub.publish(Twist())

    def run(self):
        if not self.wait_for_odom():
            rospy.logerr("No odom received on %s.", self.odom_topic)
            return 2

        # Let Gazebo settle before measurements.
        rospy.sleep(2.0)

        start = self.last_odom
        x0 = start.pose.pose.position.x
        y0 = start.pose.pose.position.y
        yaw0 = self.yaw_from_quat(start.pose.pose.orientation)

        # Forward segment
        self.send_for(0.25, 0.0, 4.0)
        self.stop()
        rospy.sleep(0.5)

        mid = self.last_odom
        x1 = mid.pose.pose.position.x
        y1 = mid.pose.pose.position.y
        yaw1 = self.yaw_from_quat(mid.pose.pose.orientation)
        forward_dist = math.hypot(x1 - x0, y1 - y0)
        yaw_delta_forward = abs(self.wrap_angle(yaw1 - yaw0))

        # Pure turn segment (primary regression check for skid/diff tuning).
        self.send_for(0.0, 0.8, 4.0)
        self.stop()
        rospy.sleep(0.5)

        end = self.last_odom
        x2 = end.pose.pose.position.x
        y2 = end.pose.pose.position.y
        yaw2 = self.yaw_from_quat(end.pose.pose.orientation)
        yaw_delta_turn = abs(self.wrap_angle(yaw2 - yaw1))
        drift_during_turn = math.hypot(x2 - x1, y2 - y1)

        rospy.loginfo(
            "Forward distance=%.3f m, yaw_change_during_forward=%.3f rad, yaw_change_during_turn=%.3f rad, drift_during_turn=%.3f m",
            forward_dist,
            yaw_delta_forward,
            yaw_delta_turn,
            drift_during_turn,
        )

        pass_forward = forward_dist > 0.08
        pass_turn = yaw_delta_turn > 0.25 and drift_during_turn < 0.35

        if pass_forward and pass_turn:
            rospy.loginfo("DRIVE_SMOKETEST PASS")
            return 0

        if not pass_forward:
            rospy.logerr("DRIVE_SMOKETEST FAIL: forward motion too small.")
        if not pass_turn:
            rospy.logerr("DRIVE_SMOKETEST FAIL: turn response too small or excessive straight-line drift during turn.")
        return 1


def main():
    rospy.init_node("drive_smoketest")
    test = DriveSmokeTest()
    code = test.run()
    raise SystemExit(code)


if __name__ == "__main__":
    main()

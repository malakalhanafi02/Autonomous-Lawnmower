#!/usr/bin/env python3
import math
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus

# Simple local ENU meters -> lat/lon degrees approximation.
# Good for small areas (campus scale). Not a full geodesy transform.

def enu_to_latlon(east_m, north_m, lat0_deg, lon0_deg):
    lat0 = math.radians(lat0_deg)
    # meters per degree
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat0) + 1.175 * math.cos(4 * lat0)
    m_per_deg_lon = 111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)

    dlat = north_m / m_per_deg_lat
    dlon = east_m / m_per_deg_lon
    return lat0_deg + dlat, lon0_deg + dlon

class FakeGPS:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "gps_link")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.fix_topic = rospy.get_param("~fix_topic", "/fix")
        self.lat0 = float(rospy.get_param("~origin_lat", 43.0096))   # default: London, ON-ish
        self.lon0 = float(rospy.get_param("~origin_lon", -81.2737))
        self.alt0 = float(rospy.get_param("~origin_alt", 250.0))     # meters
        self.cov = float(rospy.get_param("~position_covariance", 1.0))  # m^2

        self.pub = rospy.Publisher(self.fix_topic, NavSatFix, queue_size=10)
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.cb, queue_size=10)

    def cb(self, msg: Odometry):
        # ROS REP-105: base_link x forward, y left. We'll treat:
        # east = +x, north = +y (simple convention for sim)
        east = msg.pose.pose.position.x
        north = msg.pose.pose.position.y

        lat, lon = enu_to_latlon(east, north, self.lat0, self.lon0)

        fix = NavSatFix()
        fix.header.stamp = rospy.Time.now()
        fix.header.frame_id = self.frame_id

        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS

        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = self.alt0

        # Covariance in meters^2 (NavSatFix expects m^2)
        fix.position_covariance = [
            self.cov, 0.0, 0.0,
            0.0, self.cov, 0.0,
            0.0, 0.0, self.cov
        ]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.pub.publish(fix)

if __name__ == "__main__":
    rospy.init_node("fake_gps_from_odom")
    FakeGPS()
    rospy.loginfo("fake_gps_from_odom publishing /fix from %s (origin lat=%.6f lon=%.6f)",
                  rospy.get_param("~odom_topic", "/odom"),
                  rospy.get_param("~origin_lat", 43.0096),
                  rospy.get_param("~origin_lon", -81.2737))
    rospy.spin()

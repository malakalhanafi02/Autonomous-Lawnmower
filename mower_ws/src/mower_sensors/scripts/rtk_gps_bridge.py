#!/usr/bin/env python3
import math

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix


class RtkGpsBridge:
    def __init__(self):
        self.fix_topic = rospy.get_param("~fix_topic", "/rtk/fix")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/gps")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")

        self.use_initial_datum = bool(rospy.get_param("~use_initial_datum", True))
        self.datum_lat = rospy.get_param("~datum_lat", None)
        self.datum_lon = rospy.get_param("~datum_lon", None)
        self.datum_alt = rospy.get_param("~datum_alt", 0.0)
        self.require_valid_status = bool(rospy.get_param("~require_valid_status", False))

        self._R = 6378137.0
        self._lat0 = None
        self._lon0 = None
        self._alt0 = None
        self._cos_lat0 = 1.0
        self._datum_ready = False

        self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)
        rospy.Subscriber(self.fix_topic, NavSatFix, self._fix_cb, queue_size=20)

        if (self.datum_lat is not None) and (self.datum_lon is not None):
            self._set_datum(float(self.datum_lat), float(self.datum_lon), float(self.datum_alt))

    def _set_datum(self, lat, lon, alt):
        self._lat0 = lat
        self._lon0 = lon
        self._alt0 = alt
        self._cos_lat0 = math.cos(math.radians(lat))
        self._datum_ready = True
        rospy.loginfo("RTK datum set lat=%.8f lon=%.8f alt=%.3f", lat, lon, alt)

    def _fix_cb(self, msg):
        if self.require_valid_status and msg.status.status < 0:
            return
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return

        if not self._datum_ready:
            if self.use_initial_datum:
                self._set_datum(msg.latitude, msg.longitude, msg.altitude)
            else:
                return

        dlat = math.radians(msg.latitude - self._lat0)
        dlon = math.radians(msg.longitude - self._lon0)
        x = self._R * dlon * self._cos_lat0
        y = self._R * dlat
        z = msg.altitude - self._alt0

        odom = Odometry()
        odom.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.w = 1.0

        # Use NavSatFix covariance when available, otherwise conservative defaults.
        if msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            c = msg.position_covariance
            odom.pose.covariance[0] = c[0]
            odom.pose.covariance[7] = c[4]
            odom.pose.covariance[14] = c[8]
        else:
            odom.pose.covariance[0] = 0.25
            odom.pose.covariance[7] = 0.25
            odom.pose.covariance[14] = 1.0
        odom.pose.covariance[21] = 9999.0
        odom.pose.covariance[28] = 9999.0
        odom.pose.covariance[35] = 9999.0

        self.pub.publish(odom)


def main():
    rospy.init_node("rtk_gps_bridge")
    RtkGpsBridge()
    rospy.spin()


if __name__ == "__main__":
    main()

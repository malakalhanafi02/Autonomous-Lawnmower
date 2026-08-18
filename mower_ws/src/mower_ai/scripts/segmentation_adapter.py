#!/usr/bin/env python3
import json

import rospy
from std_msgs.msg import String


class SegmentationAdapter:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/camera/segmentation_zones_json")
        self.output_topic = rospy.get_param("~output_topic", "/camera/detections_json")
        self.min_points = int(rospy.get_param("~min_points_per_polygon", 3))

        self.pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        rospy.Subscriber(self.input_topic, String, self._cb, queue_size=10)

    def _cb(self, msg):
        try:
            raw = json.loads(msg.data)
            # Supported incoming keys: no_cut_zones or polygons
            zones = raw.get("no_cut_zones", raw.get("polygons", []))
            out_zones = []
            for poly in zones:
                out_poly = []
                for pt in poly:
                    out_poly.append({"x": float(pt["x"]), "y": float(pt["y"])})
                if len(out_poly) >= self.min_points:
                    out_zones.append(out_poly)
            payload = {"no_cut_zones": out_zones}
            self.pub.publish(String(data=json.dumps(payload)))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Segmentation adapter parse failed: %s", exc)


def main():
    rospy.init_node("segmentation_adapter")
    SegmentationAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()

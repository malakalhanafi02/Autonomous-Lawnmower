#!/usr/bin/env python3
"""
YOLOv8 camera inference node for the autonomous mower.

Subscribes to camera images, runs obstacle detection, and publishes:
  /camera/detections_json   (String)  -> feeds cut_region_planner
  /camera/detection_image   (Image)   -> annotated visualization
  /camera/obstacle_alert    (Bool)    -> safety flag (person/dog nearby)

Detections are projected onto the ground plane via the camera model
and TF, then formatted as no-cut-zone polygons for the planner.
"""

import json
import math

import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 – registers PointStamped transform
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool

try:
    from ultralytics import YOLO
except ImportError:
    rospy.logfatal(
        "ultralytics is not installed. "
        "Run: pip3 install ultralytics"
    )
    raise


class CameraDetector:

    DEFAULT_RADII = {
        "Person": 0.5,
        "Dog": 0.4,
        "Tree": 0.6,
        "Bicycle": 0.8,
        "Electric pole": 0.3,
        "Uncovered manhole": 0.5,
    }

    def __init__(self):
        # ---- parameters ----
        model_path = rospy.get_param("~model_path", "best.pt")
        self.conf_threshold = float(rospy.get_param("~confidence_threshold", 0.40))
        self.inference_rate = float(rospy.get_param("~inference_rate", 5.0))

        self.image_topic = rospy.get_param("~image_topic", "/mower_camera/camera/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/mower_camera/camera/camera_info")
        self.detection_topic = rospy.get_param("~detection_topic", "/camera/detections_json")
        self.det_image_topic = rospy.get_param("~detection_image_topic", "/camera/detection_image")
        self.alert_topic = rospy.get_param("~alert_topic", "/camera/obstacle_alert")

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")
        self.ground_z = float(rospy.get_param("~ground_z", 0.0))
        self.max_proj_dist = float(rospy.get_param("~max_projection_distance", 15.0))

        self.safety_distance = float(rospy.get_param("~safety_distance", 2.0))
        self.safety_classes = set(rospy.get_param("~safety_classes", ["Person", "Dog"]))
        self.obstacle_radii = rospy.get_param("~obstacle_radii", self.DEFAULT_RADII)
        self.default_radius = float(self.obstacle_radii.get("default", 0.5))

        # ---- model ----
        rospy.loginfo("Loading YOLOv8 model from: %s", model_path)
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        rospy.loginfo("Model loaded. Classes: %s", self.class_names)

        # ---- internals ----
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None
        self.min_interval = 1.0 / max(self.inference_rate, 0.1)
        self.last_inference_time = rospy.Time(0)

        # ---- publishers ----
        self.det_pub = rospy.Publisher(self.detection_topic, String, queue_size=5)
        self.img_pub = rospy.Publisher(self.det_image_topic, Image, queue_size=1)
        self.alert_pub = rospy.Publisher(self.alert_topic, Bool, queue_size=10)

        # ---- subscribers ----
        rospy.Subscriber(self.camera_info_topic, CameraInfo,
                         self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image,
                         self._image_cb, queue_size=1, buff_size=2**24)

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def _camera_info_cb(self, msg):
        if self.camera_info is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            rospy.loginfo(
                "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                self.fx, self.fy, self.cx, self.cy,
            )
        self.camera_info = msg

    def _image_cb(self, msg):
        now = rospy.Time.now()
        if (now - self.last_inference_time).to_sec() < self.min_interval:
            return
        self.last_inference_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge error: %s", exc)
            return

        results = self.model.predict(
            source=cv_image,
            conf=self.conf_threshold,
            verbose=False,
        )
        result = results[0]

        self._publish_annotated(result, msg.header)
        self._publish_detections(result, msg.header)

    # ------------------------------------------------------------------
    #  Annotated image
    # ------------------------------------------------------------------
    def _publish_annotated(self, result, header):
        try:
            annotated = result.plot()
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header = header
            self.img_pub.publish(img_msg)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to publish detection image: %s", exc)

    # ------------------------------------------------------------------
    #  Structured detections + no-cut zones + safety alert
    # ------------------------------------------------------------------
    def _publish_detections(self, result, header):
        if result.boxes is None or len(result.boxes) == 0:
            self.alert_pub.publish(Bool(data=False))
            self.det_pub.publish(String(data=json.dumps(
                {"no_cut_zones": [], "detections": []}
            )))
            return

        detections = []
        no_cut_zones = []
        safety_alert = False

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.class_names.get(cls_id, "unknown")
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            det = {
                "class": cls_name,
                "confidence": round(conf, 3),
                "bbox_px": [round(x1), round(y1), round(x2), round(y2)],
            }

            map_pt = self._project_to_ground(
                px=(x1 + x2) / 2.0,
                py=y2,
                stamp=header.stamp,
            )
            if map_pt is not None:
                det["position"] = {"x": round(map_pt[0], 3),
                                   "y": round(map_pt[1], 3)}
                dist = math.hypot(map_pt[0], map_pt[1])
                det["distance_m"] = round(dist, 2)

                radius = float(self.obstacle_radii.get(cls_name, self.default_radius))
                no_cut_zones.append(self._square_polygon(map_pt[0], map_pt[1], radius))

                if cls_name in self.safety_classes and dist < self.safety_distance:
                    safety_alert = True

            detections.append(det)

        self.alert_pub.publish(Bool(data=safety_alert))
        self.det_pub.publish(String(data=json.dumps({
            "no_cut_zones": no_cut_zones,
            "detections": detections,
        })))

        if safety_alert:
            rospy.logwarn_throttle(
                1.0,
                "SAFETY ALERT: %s detected within %.1f m!",
                ", ".join(d["class"] for d in detections
                          if d["class"] in self.safety_classes),
                self.safety_distance,
            )

    # ------------------------------------------------------------------
    #  Ground-plane projection via camera model + TF
    # ------------------------------------------------------------------
    def _project_to_ground(self, px, py, stamp):
        """Project pixel (px, py) onto z = ground_z in the map frame."""
        if self.fx is None:
            return None

        # Pixel -> ray in camera optical convention, then into camera_link
        # optical: z forward, x right, y down
        # camera_link: x forward, y left, z up
        ray_cam_x = 1.0
        ray_cam_y = -(px - self.cx) / self.fx
        ray_cam_z = -(py - self.cy) / self.fy

        origin = PointStamped()
        origin.header.stamp = stamp
        origin.header.frame_id = self.camera_frame

        ray_end = PointStamped()
        ray_end.header.stamp = stamp
        ray_end.header.frame_id = self.camera_frame
        ray_end.point.x = ray_cam_x
        ray_end.point.y = ray_cam_y
        ray_end.point.z = ray_cam_z

        try:
            origin_map = self.tf_buffer.transform(origin, self.map_frame,
                                                  rospy.Duration(0.1))
            ray_end_map = self.tf_buffer.transform(ray_end, self.map_frame,
                                                   rospy.Duration(0.1))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5.0, "TF lookup failed: %s", exc)
            return None

        dx = ray_end_map.point.x - origin_map.point.x
        dy = ray_end_map.point.y - origin_map.point.y
        dz = ray_end_map.point.z - origin_map.point.z

        if abs(dz) < 1e-6:
            return None
        t = (self.ground_z - origin_map.point.z) / dz
        if t < 0:
            return None

        hit_x = origin_map.point.x + t * dx
        hit_y = origin_map.point.y + t * dy

        if math.hypot(hit_x - origin_map.point.x,
                       hit_y - origin_map.point.y) > self.max_proj_dist:
            return None

        return (hit_x, hit_y)

    @staticmethod
    def _square_polygon(cx, cy, radius):
        """Return a square no-cut zone as a list of {x, y} dicts."""
        return [
            {"x": round(cx - radius, 3), "y": round(cy - radius, 3)},
            {"x": round(cx + radius, 3), "y": round(cy - radius, 3)},
            {"x": round(cx + radius, 3), "y": round(cy + radius, 3)},
            {"x": round(cx - radius, 3), "y": round(cy + radius, 3)},
        ]


def main():
    rospy.init_node("camera_detector")
    CameraDetector()
    rospy.loginfo(
        "Camera detector ready.  image=%s  info=%s",
        rospy.get_param("~image_topic", "/mower_camera/camera/image_raw"),
        rospy.get_param("~camera_info_topic", "/mower_camera/camera/camera_info"),
    )
    rospy.spin()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import hashlib

import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class CoverageMissionManager:
    def __init__(self):
        self.enabled_topic = rospy.get_param("~enabled_topic", "/mower/autonomy_enabled")
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.plan_service = rospy.get_param("~plan_service", "/cut_region_planner/build_plan")
        self.map_timeout_s = float(rospy.get_param("~map_timeout_s", 5.0))
        self.allow_stale_map_for_bootstrap = _as_bool(
            rospy.get_param("~allow_stale_map_for_bootstrap", True)
        )
        self.stale_map_bootstrap_max_age_s = float(
            rospy.get_param("~stale_map_bootstrap_max_age_s", 600.0)
        )
        self.replan_on_enable = _as_bool(rospy.get_param("~replan_on_enable", True))
        self.replan_interval_s = float(rospy.get_param("~replan_interval_s", 0.0))
        self.plan_retry_interval_s = float(rospy.get_param("~plan_retry_interval_s", 2.0))
        self.map_min_known_cells = int(rospy.get_param("~map_min_known_cells", 500))
        self.map_min_known_ratio = float(rospy.get_param("~map_min_known_ratio", 0.0))
        self.replan_on_map_growth = _as_bool(rospy.get_param("~replan_on_map_growth", True))
        self.map_growth_known_cells = int(rospy.get_param("~map_growth_known_cells", 600))
        self.map_growth_min_plan_age_s = float(
            rospy.get_param("~map_growth_min_plan_age_s", 20.0)
        )
        self.detection_topic = rospy.get_param("~detection_topic", "/camera/detections_json")
        self.replan_on_detection_change = _as_bool(
            rospy.get_param("~replan_on_detection_change", True)
        )
        self.detection_replan_cooldown_s = float(
            rospy.get_param("~detection_replan_cooldown_s", 2.0)
        )
        self.wait_for_perimeter_complete = _as_bool(
            rospy.get_param("~wait_for_perimeter_complete", True)
        )
        self.perimeter_complete_topic = rospy.get_param(
            "~perimeter_complete_topic", "/mower/perimeter_complete"
        )
        self.reset_perimeter_on_enable = _as_bool(
            rospy.get_param("~reset_perimeter_on_enable", True)
        )
        self.loop_hz = float(rospy.get_param("~loop_hz", 2.0))

        self.enabled = False
        self._prev_enabled = False
        self.last_map_stamp = rospy.Time(0)
        self.last_plan_attempt_stamp = rospy.Time(0)
        self.last_plan_stamp = rospy.Time(0)
        self.last_detection_replan_stamp = rospy.Time(0)
        self.known_cells = 0
        self.total_cells = 0
        self.last_planned_known_cells = 0
        self.last_detection_hash = None
        self.detection_changed_pending = False
        self._waiting_for_service_logged = False
        self.perimeter_complete = not self.wait_for_perimeter_complete

        rospy.Subscriber(self.enabled_topic, Bool, self._enabled_cb, queue_size=5)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_cb, queue_size=1)
        rospy.Subscriber(self.detection_topic, String, self._detection_cb, queue_size=10)
        if self.wait_for_perimeter_complete:
            rospy.Subscriber(
                self.perimeter_complete_topic, Bool, self._perimeter_complete_cb, queue_size=5
            )

    def _enabled_cb(self, msg):
        self.enabled = msg.data

    def _perimeter_complete_cb(self, msg):
        self.perimeter_complete = _as_bool(msg.data)

    def _map_cb(self, msg):
        self.last_map_stamp = rospy.Time.now()
        total = len(msg.data)
        if total > 0:
            unknown = msg.data.count(-1)
            self.total_cells = total
            self.known_cells = total - unknown

    def _map_is_fresh(self):
        if self.last_map_stamp == rospy.Time(0):
            return False
        return (rospy.Time.now() - self.last_map_stamp).to_sec() <= self.map_timeout_s

    def _map_is_ready_for_plan(self):
        map_is_fresh = self._map_is_fresh()
        if not map_is_fresh:
            bootstrap_with_stale_map = False
            if (
                self.allow_stale_map_for_bootstrap
                and self.last_plan_stamp == rospy.Time(0)
                and self.last_map_stamp != rospy.Time(0)
            ):
                stale_age = (rospy.Time.now() - self.last_map_stamp).to_sec()
                if stale_age <= self.stale_map_bootstrap_max_age_s:
                    bootstrap_with_stale_map = True
                    rospy.logwarn_throttle(
                        3.0,
                        "Mission manager using stale /map for bootstrap plan (age=%.1fs).",
                        stale_age,
                    )
            if not bootstrap_with_stale_map:
                rospy.logwarn_throttle(3.0, "Mission manager waiting for fresh /map before planning.")
                return False

        if self.total_cells <= 0:
            rospy.logwarn_throttle(3.0, "Mission manager waiting for map cells before planning.")
            return False

        known_ratio = float(self.known_cells) / float(self.total_cells)
        if self.known_cells < self.map_min_known_cells:
            rospy.logwarn_throttle(
                3.0,
                "Mission manager waiting for map coverage (%d/%d known cells).",
                self.known_cells,
                self.map_min_known_cells,
            )
            return False
        if self.map_min_known_ratio > 0.0 and known_ratio < self.map_min_known_ratio:
            rospy.logwarn_throttle(
                3.0,
                "Mission manager waiting for map known ratio %.4f < %.4f.",
                known_ratio,
                self.map_min_known_ratio,
            )
            return False
        return True

    def _detection_cb(self, msg):
        h = hashlib.md5(msg.data.encode("utf-8")).hexdigest()
        if self.last_detection_hash is None:
            self.last_detection_hash = h
            return
        if h != self.last_detection_hash:
            self.last_detection_hash = h
            self.detection_changed_pending = True

    def _can_attempt_plan_now(self):
        if self.last_plan_attempt_stamp == rospy.Time(0):
            return True
        age = (rospy.Time.now() - self.last_plan_attempt_stamp).to_sec()
        return age >= self.plan_retry_interval_s

    def _try_build_plan(self, reason):
        if not self._can_attempt_plan_now():
            return False
        self.last_plan_attempt_stamp = rospy.Time.now()

        if self.wait_for_perimeter_complete and not self.perimeter_complete:
            rospy.logwarn_throttle(
                3.0,
                "Mission manager waiting for perimeter bootstrap completion.",
            )
            return False

        if not self._map_is_ready_for_plan():
            return False

        try:
            rospy.wait_for_service(self.plan_service, timeout=1.0)
            self._waiting_for_service_logged = False
        except rospy.ROSException:
            if not self._waiting_for_service_logged:
                rospy.logwarn("Mission manager waiting for planner service: %s", self.plan_service)
                self._waiting_for_service_logged = True
            return False

        try:
            srv = rospy.ServiceProxy(self.plan_service, Trigger)
            res = srv()
            if res.success:
                self.last_plan_stamp = rospy.Time.now()
                self.last_planned_known_cells = self.known_cells
                self.last_detection_replan_stamp = self.last_plan_stamp
                self.detection_changed_pending = False
                rospy.loginfo("Mission manager plan build succeeded: %s", res.message)
                rospy.loginfo("Mission manager trigger: %s", reason)
                return True
            rospy.logwarn("Mission manager plan build failed: %s", res.message)
            return False
        except rospy.ServiceException as exc:
            rospy.logwarn("Mission manager planner call failed: %s", exc)
            return False

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            rising_edge = self.enabled and not self._prev_enabled

            if (
                rising_edge
                and self.wait_for_perimeter_complete
                and self.reset_perimeter_on_enable
            ):
                self.perimeter_complete = False

            if self.enabled:
                reason = None
                if self.replan_on_enable and rising_edge:
                    reason = "enable_edge"
                elif self.last_plan_stamp == rospy.Time(0):
                    reason = "bootstrap"
                elif self.replan_interval_s > 0.0:
                    age = (rospy.Time.now() - self.last_plan_stamp).to_sec()
                    if age >= self.replan_interval_s:
                        reason = "periodic"

                if (
                    reason is None
                    and self.replan_on_map_growth
                    and self.last_plan_stamp != rospy.Time(0)
                ):
                    known_growth = self.known_cells - self.last_planned_known_cells
                    age = (rospy.Time.now() - self.last_plan_stamp).to_sec()
                    if (
                        known_growth >= self.map_growth_known_cells
                        and age >= self.map_growth_min_plan_age_s
                    ):
                        reason = "map_growth_%d" % known_growth

                if (
                    reason is None
                    and self.replan_on_detection_change
                    and self.detection_changed_pending
                ):
                    if self.last_detection_replan_stamp == rospy.Time(0):
                        reason = "detections_changed"
                    else:
                        det_age = (
                            rospy.Time.now() - self.last_detection_replan_stamp
                        ).to_sec()
                        if det_age >= self.detection_replan_cooldown_s:
                            reason = "detections_changed"

                if reason is not None:
                    self._try_build_plan(reason)

            self._prev_enabled = self.enabled
            rate.sleep()


def main():
    rospy.init_node("coverage_mission_manager")
    CoverageMissionManager().spin()


if __name__ == "__main__":
    main()

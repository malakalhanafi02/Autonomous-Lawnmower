#!/usr/bin/env python3
import heapq
import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(v)


class CutRegionPlanner:
    def __init__(self):
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.detection_topic = rospy.get_param("~detection_topic", "/camera/detections_json")
        self.output_topic = rospy.get_param("~output_topic", "/mower_ai/cut_plan")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.free_threshold = int(rospy.get_param("~free_threshold", 20))
        self.stripe_spacing_m = float(rospy.get_param("~stripe_spacing_m", 0.45))
        self.path_clearance_m = float(rospy.get_param("~path_clearance_m", 0.35))
        self.turnaround_margin_m = float(rospy.get_param("~turnaround_margin_m", 0.375))
        self.row_turn_radius_m = float(rospy.get_param("~row_turn_radius_m", 0.70))
        self.row_transition_step_m = float(rospy.get_param("~row_transition_step_m", 0.06))
        self.transition_astar_max_expansions = int(
            rospy.get_param("~transition_astar_max_expansions", 120000)
        )
        self.avoid_no_cut_zones = _as_bool(rospy.get_param("~avoid_no_cut_zones", False))
        self.sample_step_cells = int(rospy.get_param("~sample_step_cells", 2))
        self.min_waypoints = int(rospy.get_param("~min_waypoints", 200))
        self.start_corner = rospy.get_param("~start_corner", "lower_left").strip().lower()

        self.latest_map = None
        self.no_cut_polygons = []

        self.plan_pub = rospy.Publisher(self.output_topic, Path, queue_size=1, latch=True)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_cb, queue_size=1)
        rospy.Subscriber(self.detection_topic, String, self.detection_cb, queue_size=5)
        self.plan_srv = rospy.Service("~build_plan", Trigger, self.handle_build_plan)

    def map_cb(self, msg):
        self.latest_map = msg

    def detection_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            polygons = payload.get("no_cut_zones", [])
            parsed = []
            for poly in polygons:
                pts = []
                for p in poly:
                    pts.append((float(p["x"]), float(p["y"])))
                if len(pts) >= 3:
                    parsed.append(pts)
            self.no_cut_polygons = parsed
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Could not parse detection JSON: %s", exc)

    def handle_build_plan(self, _req):
        if self.latest_map is None:
            return TriggerResponse(success=False, message="No map received yet.")

        path = self.build_path(self.latest_map, self.no_cut_polygons)
        if len(path.poses) < self.min_waypoints:
            return TriggerResponse(
                success=False,
                message="Map not ready: only %d waypoints (need >= %d)."
                % (len(path.poses), self.min_waypoints),
            )

        self.plan_pub.publish(path)
        msg = "Published plan with %d waypoints and %d no-cut zones." % (
            len(path.poses),
            len(self.no_cut_polygons),
        )
        return TriggerResponse(success=True, message=msg)

    def build_path(self, occ, no_cut_polygons):
        width = occ.info.width
        height = occ.info.height
        resolution = occ.info.resolution
        origin_x = occ.info.origin.position.x
        origin_y = occ.info.origin.position.y
        data = occ.data

        stripe_spacing_cells = max(1, int(round(self.stripe_spacing_m / resolution)))
        clearance_cells = max(0, int(round(self.path_clearance_m / resolution)))
        turnaround_margin_cells = max(0, int(round(self.turnaround_margin_m / resolution)))
        blocked_prefix = self.build_blocked_prefix(data, width, height)
        row_paths = []
        start_right = self.start_corner.endswith("right")
        start_upper = self.start_corner.startswith("upper")
        left_to_right = not start_right

        if start_upper:
            ys = range(height - 1, -1, -stripe_spacing_cells)
        else:
            ys = range(0, height, stripe_spacing_cells)

        x_start = min(max(0, turnaround_margin_cells), max(0, width - 1))
        x_end_exclusive = max(x_start + 1, width - turnaround_margin_cells)
        base_xs = list(range(x_start, x_end_exclusive, self.sample_step_cells))

        for y in ys:
            xs = base_xs if left_to_right else list(reversed(base_xs))
            row_segments = []
            segment = []

            for x in xs:
                idx = y * width + x
                if idx < 0 or idx >= len(data):
                    if segment:
                        row_segments.append(segment)
                        segment = []
                    continue

                cell = data[idx]
                is_free = 0 <= cell <= self.free_threshold
                has_clearance = True
                if is_free and clearance_cells > 0:
                    has_clearance = self.cell_has_clearance(
                        x, y, blocked_prefix, clearance_cells, width, height
                    )

                if not is_free or not has_clearance:
                    if segment:
                        row_segments.append(segment)
                        segment = []
                    continue

                wx = origin_x + (x + 0.5) * resolution
                wy = origin_y + (y + 0.5) * resolution
                if self.avoid_no_cut_zones and self.in_any_polygon(wx, wy, no_cut_polygons):
                    if segment:
                        row_segments.append(segment)
                        segment = []
                    continue

                segment.append((wx, wy))

            if segment:
                row_segments.append(segment)

            if row_segments:
                row_paths.extend(row_segments)
                left_to_right = not left_to_right

        waypoints = []
        if row_paths:
            current_row = row_paths[0]
            waypoints.extend(current_row)
            for i in range(1, len(row_paths)):
                next_row = row_paths[i]
                prev_end = current_row[-1]
                next_start = next_row[0]
                if (
                    abs(prev_end[0] - next_start[0]) > 1e-3
                    or abs(prev_end[1] - next_start[1]) > 1e-3
                ):
                    prev_dir_sign = 1.0 if current_row[-1][0] >= current_row[0][0] else -1.0
                    # Turn outward relative to travel direction. With turnaround margin
                    # this creates clean, non-intersecting U-turns at row ends.
                    turn_outward_sign = prev_dir_sign
                    transition = self.build_row_transition_points(
                        prev_end,
                        next_start,
                        turn_outward_sign,
                        blocked_prefix,
                        clearance_cells,
                        width,
                        height,
                        resolution,
                        origin_x,
                        origin_y,
                        no_cut_polygons,
                    )
                    if not transition:
                        rospy.logwarn_throttle(
                            2.0,
                            "Planner could not find a safe transition between row segments; skipping one segment.",
                        )
                        continue
                    waypoints.extend(transition)
                waypoints.extend(next_row)
                current_row = next_row

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        for (x, y) in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def build_blocked_prefix(self, data, width, height):
        # Prefix sum over blocked cells (occupied or unknown). This lets us
        # query whether a square neighborhood is clear in O(1) per waypoint.
        prefix = [[0] * (width + 1) for _ in range(height + 1)]
        for y in range(height):
            row_acc = 0
            base = y * width
            dst = prefix[y + 1]
            src = prefix[y]
            for x in range(width):
                cell = data[base + x]
                blocked = 1 if (cell < 0 or cell > self.free_threshold) else 0
                row_acc += blocked
                dst[x + 1] = src[x + 1] + row_acc
        return prefix

    @staticmethod
    def rect_sum(prefix, x0, y0, x1, y1):
        return (
            prefix[y1 + 1][x1 + 1]
            - prefix[y0][x1 + 1]
            - prefix[y1 + 1][x0]
            + prefix[y0][x0]
        )

    def cell_has_clearance(self, x, y, blocked_prefix, clearance_cells, width, height):
        x0 = max(0, x - clearance_cells)
        y0 = max(0, y - clearance_cells)
        x1 = min(width - 1, x + clearance_cells)
        y1 = min(height - 1, y + clearance_cells)
        return self.rect_sum(blocked_prefix, x0, y0, x1, y1) == 0

    @staticmethod
    def _cubic_bezier(p0, c1, c2, p3, t):
        mt = 1.0 - t
        x = (
            mt * mt * mt * p0[0]
            + 3.0 * mt * mt * t * c1[0]
            + 3.0 * mt * t * t * c2[0]
            + t * t * t * p3[0]
        )
        y = (
            mt * mt * mt * p0[1]
            + 3.0 * mt * mt * t * c1[1]
            + 3.0 * mt * t * t * c2[1]
            + t * t * t * p3[1]
        )
        return (x, y)

    def point_clear_world(
        self,
        wx,
        wy,
        blocked_prefix,
        clearance_cells,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        no_cut_polygons,
    ):
        cx = int((wx - origin_x) / resolution)
        cy = int((wy - origin_y) / resolution)
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return False
        if clearance_cells > 0 and not self.cell_has_clearance(
            cx, cy, blocked_prefix, clearance_cells, width, height
        ):
            return False
        if self.avoid_no_cut_zones and self.in_any_polygon(wx, wy, no_cut_polygons):
            return False
        return True

    def _all_points_clear(
        self,
        points,
        blocked_prefix,
        clearance_cells,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        no_cut_polygons,
    ):
        for qx, qy in points:
            if not self.point_clear_world(
                qx,
                qy,
                blocked_prefix,
                clearance_cells,
                width,
                height,
                resolution,
                origin_x,
                origin_y,
                no_cut_polygons,
            ):
                return False
        return True

    def _sample_line_points(self, p0, p1):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-5:
            return []
        step = max(0.02, self.row_transition_step_m)
        n = max(1, int(math.ceil(dist / step)))
        out = []
        for i in range(1, n):
            t = float(i) / float(n)
            out.append((p0[0] + dx * t, p0[1] + dy * t))
        return out

    def _build_uturn_transition_points(self, p0, p1, outward_sign, resolution):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dy) < max(resolution, 1e-3):
            return []
        max_uturn_dx = max(0.60, 0.5 * self.row_turn_radius_m, 4.0 * resolution)
        if abs(dx) > max_uturn_dx:
            return []

        # Semicircle between rows plus optional straight "push out" for larger turn radius.
        half_sep = abs(dy) * 0.5
        push_out = max(0.0, self.row_turn_radius_m - half_sep)
        pre = (p0[0] + outward_sign * push_out, p0[1])
        post = (p1[0] + outward_sign * push_out, p1[1])
        center = (pre[0], 0.5 * (pre[1] + post[1]))
        radius = half_sep
        if radius < 1e-3:
            return self._sample_line_points(p0, p1)

        out = []
        out.extend(self._sample_line_points(p0, pre))

        arc_len = math.pi * radius
        step = max(0.02, self.row_transition_step_m)
        n_arc = max(2, int(math.ceil(arc_len / step)))
        for i in range(1, n_arc):
            u = float(i) / float(n_arc)
            if outward_sign >= 0.0:
                theta = (-math.pi / 2.0 + math.pi * u) if dy >= 0.0 else (math.pi / 2.0 - math.pi * u)
            else:
                theta = (-math.pi / 2.0 - math.pi * u) if dy >= 0.0 else (math.pi / 2.0 + math.pi * u)
            qx = center[0] + radius * math.cos(theta)
            qy = center[1] + radius * math.sin(theta)
            out.append((qx, qy))

        out.extend(self._sample_line_points(post, p1))
        return out

    def _build_cubic_transition_points(self, p0, p1, outward_sign):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return []

        r = max(0.05, self.row_turn_radius_m)
        c1 = (p0[0] + outward_sign * r, p0[1])
        c2 = (p1[0] + outward_sign * r, p1[1])
        est_len = abs(dy) + 2.0 * r + abs(dx)
        n = max(2, int(round(est_len / max(0.02, self.row_transition_step_m))))
        out = []
        for i in range(1, n):
            t = float(i) / float(n)
            out.append(self._cubic_bezier(p0, c1, c2, p1, t))
        return out

    @staticmethod
    def _world_to_cell(wx, wy, resolution, origin_x, origin_y):
        return int((wx - origin_x) / resolution), int((wy - origin_y) / resolution)

    @staticmethod
    def _cell_to_world(cx, cy, resolution, origin_x, origin_y):
        return origin_x + (cx + 0.5) * resolution, origin_y + (cy + 0.5) * resolution

    def _cell_traversable(
        self,
        cx,
        cy,
        blocked_prefix,
        clearance_cells,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        no_cut_polygons,
    ):
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return False
        if clearance_cells > 0 and not self.cell_has_clearance(
            cx, cy, blocked_prefix, clearance_cells, width, height
        ):
            return False
        if self.avoid_no_cut_zones:
            wx, wy = self._cell_to_world(cx, cy, resolution, origin_x, origin_y)
            if self.in_any_polygon(wx, wy, no_cut_polygons):
                return False
        return True

    def _nearest_traversable_cell(
        self,
        cx,
        cy,
        blocked_prefix,
        clearance_cells,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        no_cut_polygons,
        max_radius=20,
    ):
        if self._cell_traversable(
            cx,
            cy,
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        ):
            return (cx, cy)

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for tx, ty in ((cx + dx, cy - radius), (cx + dx, cy + radius)):
                    if self._cell_traversable(
                        tx,
                        ty,
                        blocked_prefix,
                        clearance_cells,
                        width,
                        height,
                        resolution,
                        origin_x,
                        origin_y,
                        no_cut_polygons,
                    ):
                        return (tx, ty)
            for dy in range(-radius + 1, radius):
                for tx, ty in ((cx - radius, cy + dy), (cx + radius, cy + dy)):
                    if self._cell_traversable(
                        tx,
                        ty,
                        blocked_prefix,
                        clearance_cells,
                        width,
                        height,
                        resolution,
                        origin_x,
                        origin_y,
                        no_cut_polygons,
                    ):
                        return (tx, ty)
        return None

    def _build_astar_transition_points(
        self,
        p0,
        p1,
        blocked_prefix,
        clearance_cells,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        no_cut_polygons,
    ):
        start = self._world_to_cell(p0[0], p0[1], resolution, origin_x, origin_y)
        goal = self._world_to_cell(p1[0], p1[1], resolution, origin_x, origin_y)
        start = self._nearest_traversable_cell(
            start[0],
            start[1],
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        )
        goal = self._nearest_traversable_cell(
            goal[0],
            goal[1],
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        )
        if start is None or goal is None:
            return []

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_heap = []
        heapq.heappush(open_heap, (h(start, goal), 0.0, start))
        g_cost = {start: 0.0}
        came_from = {}
        closed = set()
        expansions = 0

        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        found = False
        while open_heap and expansions < self.transition_astar_max_expansions:
            _, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                found = True
                break
            closed.add(cur)
            expansions += 1

            for dx, dy in neighbors:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in closed:
                    continue
                if not self._cell_traversable(
                    nxt[0],
                    nxt[1],
                    blocked_prefix,
                    clearance_cells,
                    width,
                    height,
                    resolution,
                    origin_x,
                    origin_y,
                    no_cut_polygons,
                ):
                    continue
                step = math.sqrt(2.0) if (dx != 0 and dy != 0) else 1.0
                ng = g + step
                if ng < g_cost.get(nxt, float("inf")):
                    g_cost[nxt] = ng
                    came_from[nxt] = cur
                    heapq.heappush(open_heap, (ng + h(nxt, goal), ng, nxt))

        if not found:
            return []

        path_cells = [goal]
        cur = goal
        while cur != start:
            cur = came_from[cur]
            path_cells.append(cur)
        path_cells.reverse()

        out = []
        for cx, cy in path_cells[1:-1]:
            out.append(self._cell_to_world(cx, cy, resolution, origin_x, origin_y))
        return out

    @staticmethod
    def _chaikin_smooth_polyline(points, iterations=2):
        if len(points) < 3 or iterations <= 0:
            return list(points)
        smoothed = list(points)
        for _ in range(iterations):
            nxt = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p = smoothed[i]
                q = smoothed[i + 1]
                q1 = (0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1])
                q2 = (0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1])
                nxt.extend([q1, q2])
            nxt.append(smoothed[-1])
            smoothed = nxt
        return smoothed

    def _densify_polyline(self, points):
        if len(points) < 2:
            return list(points)
        out = [points[0]]
        for i in range(len(points) - 1):
            out.extend(self._sample_line_points(points[i], points[i + 1]))
            out.append(points[i + 1])
        return out

    def build_row_transition_points(
        self,
        p0,
        p1,
        outward_sign,
        blocked_prefix,
        clearance_cells,
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        no_cut_polygons,
    ):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return []

        # Prefer explicit rounded U-turns at row ends when geometry matches.
        rounded = self._build_uturn_transition_points(p0, p1, outward_sign, resolution)
        if rounded and self._all_points_clear(
            rounded,
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        ):
            return rounded

        # Fall back to smooth cubic transition.
        cubic = self._build_cubic_transition_points(p0, p1, outward_sign)
        if cubic and self._all_points_clear(
            cubic,
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        ):
            return cubic

        # Last-resort: grid-safe transition around interior obstacles.
        astar = self._build_astar_transition_points(
            p0,
            p1,
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        )
        if not astar:
            return []

        # Smooth jagged A* corners into rounded transitions when clearance allows it.
        # Keep endpoints anchored to row endpoints and only return interior samples.
        polyline = [p0] + astar + [p1]
        smoothed = self._chaikin_smooth_polyline(polyline, iterations=2)
        smoothed = self._densify_polyline(smoothed)
        interior = smoothed[1:-1]
        if interior and self._all_points_clear(
            interior,
            blocked_prefix,
            clearance_cells,
            width,
            height,
            resolution,
            origin_x,
            origin_y,
            no_cut_polygons,
        ):
            return interior
        return astar

    @staticmethod
    def in_any_polygon(x, y, polygons):
        for poly in polygons:
            if CutRegionPlanner.point_in_poly(x, y, poly):
                return True
        return False

    @staticmethod
    def point_in_poly(x, y, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            denom = y2 - y1
            if abs(denom) < 1e-12:
                continue
            intersects = ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / denom + x1
            )
            if intersects:
                inside = not inside
        return inside


def main():
    rospy.init_node("cut_region_planner")
    CutRegionPlanner()
    rospy.loginfo("Cut region planner ready. Call ~/build_plan when map + detections are available.")
    rospy.spin()


if __name__ == "__main__":
    main()

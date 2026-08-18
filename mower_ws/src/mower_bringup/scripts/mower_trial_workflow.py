#!/usr/bin/env python3
import argparse
import ast
import math
import os
import sys
import time

import rospkg
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _default_map_base():
    try:
        pkg_path = rospkg.RosPack().get_path("mower_bringup")
        return os.path.join(pkg_path, "maps", "latest_scan")
    except Exception:
        return os.path.abspath("latest_scan")


def _save_occupancy_to_files(occ, output_base):
    output_dir = os.path.dirname(output_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pgm_path = output_base + ".pgm"
    yaml_path = output_base + ".yaml"

    width = occ.info.width
    height = occ.info.height
    data = occ.data

    with open(pgm_path, "wb") as f:
        f.write(("P5\n# CREATOR: mower_trial_workflow.py\n%d %d\n255\n" % (width, height)).encode("ascii"))
        for y in range(height):
            row = bytearray()
            map_y = height - y - 1
            base = map_y * width
            for x in range(width):
                v = data[base + x]
                if v < 0:
                    row.append(205)
                elif v >= 65:
                    row.append(0)
                elif v <= 25:
                    row.append(254)
                else:
                    row.append(int(round((100.0 - float(v)) * 254.0 / 100.0)))
            f.write(row)

    yaw = _yaw_from_quat(occ.info.origin.orientation)
    image_name = os.path.basename(pgm_path)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("image: %s\n" % image_name)
        f.write("resolution: %.6f\n" % occ.info.resolution)
        f.write(
            "origin: [%.6f, %.6f, %.6f]\n"
            % (occ.info.origin.position.x, occ.info.origin.position.y, yaw)
        )
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")

    return pgm_path, yaml_path


def _parse_simple_yaml(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _read_pgm_u8(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError("Unsupported PGM format in %s (expected P5)." % path)

        def read_token():
            tok = b""
            while True:
                ch = f.read(1)
                if ch == b"":
                    raise ValueError("Unexpected EOF in PGM header: %s" % path)
                if ch.isspace():
                    continue
                if ch == b"#":
                    f.readline()
                    continue
                tok += ch
                break
            while True:
                ch = f.read(1)
                if ch == b"" or ch.isspace():
                    break
                tok += ch
            return tok

        width = int(read_token())
        height = int(read_token())
        maxval = int(read_token())
        if maxval <= 0 or maxval > 255:
            raise ValueError("Unsupported PGM max value %d in %s" % (maxval, path))
        buf = f.read(width * height)
        if len(buf) != width * height:
            raise ValueError("PGM payload size mismatch in %s" % path)
        return width, height, maxval, buf


def _load_saved_occupancy(output_base):
    yaml_path = output_base + ".yaml"
    if not os.path.exists(yaml_path):
        return None
    meta = _parse_simple_yaml(yaml_path)
    image_name = meta.get("image", output_base + ".pgm")
    pgm_path = image_name
    if not os.path.isabs(pgm_path):
        pgm_path = os.path.join(os.path.dirname(yaml_path), pgm_path)
    if not os.path.exists(pgm_path):
        raise RuntimeError("Saved map image not found: %s" % pgm_path)

    width, height, maxval, pixels = _read_pgm_u8(pgm_path)
    occupied_thresh = float(meta.get("occupied_thresh", "0.65"))
    free_thresh = float(meta.get("free_thresh", "0.196"))
    negate = int(float(meta.get("negate", "0")))
    resolution = float(meta.get("resolution", "0.05"))
    origin = ast.literal_eval(meta.get("origin", "[0.0, 0.0, 0.0]"))
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise RuntimeError("Invalid origin in saved map yaml: %s" % yaml_path)

    data = [-1] * (width * height)
    for file_y in range(height):
        map_y = height - file_y - 1
        file_base = file_y * width
        map_base = map_y * width
        for x in range(width):
            px = pixels[file_base + x]
            occ = (float(maxval - px) / float(maxval))
            if negate:
                occ = 1.0 - occ
            if occ > occupied_thresh:
                val = 100
            elif occ < free_thresh:
                val = 0
            else:
                val = -1
            data[map_base + x] = int(val)

    return {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "data": data,
        "yaml_path": yaml_path,
        "pgm_path": pgm_path,
    }


def _merge_cell(prior, latest, conflict_mode):
    if latest < 0:
        return int(prior)
    if prior < 0:
        return int(latest)

    p = int(prior)
    n = int(latest)
    if conflict_mode == "newest":
        return n
    if conflict_mode == "average":
        return int(round(0.5 * float(p + n)))

    # conservative: occupied wins, free only when both are free.
    p_occ = p >= 65
    n_occ = n >= 65
    p_free = p <= 25
    n_free = n <= 25
    if p_occ or n_occ:
        return max(p, n)
    if p_free and n_free:
        return min(p, n)
    return max(p, n)


def _merge_occupancy_grid_with_saved(live_occ, saved, conflict_mode):
    width = int(live_occ.info.width)
    height = int(live_occ.info.height)
    if saved["width"] != width or saved["height"] != height:
        raise RuntimeError(
            "Saved map dimensions (%dx%d) != live map (%dx%d)."
            % (saved["width"], saved["height"], width, height)
        )
    if abs(saved["resolution"] - float(live_occ.info.resolution)) > 1e-6:
        raise RuntimeError(
            "Saved map resolution (%.6f) != live map resolution (%.6f)."
            % (saved["resolution"], float(live_occ.info.resolution))
        )
    if (
        abs(saved["origin_x"] - float(live_occ.info.origin.position.x)) > 1e-6
        or abs(saved["origin_y"] - float(live_occ.info.origin.position.y)) > 1e-6
    ):
        raise RuntimeError(
            "Saved map origin (%.3f, %.3f) != live map origin (%.3f, %.3f)."
            % (
                saved["origin_x"],
                saved["origin_y"],
                float(live_occ.info.origin.position.x),
                float(live_occ.info.origin.position.y),
            )
        )

    merged = OccupancyGrid()
    merged.header = live_occ.header
    merged.info = live_occ.info
    live_data = list(live_occ.data)
    prior_data = list(saved["data"])
    out = [0] * len(live_data)
    for i in range(len(live_data)):
        out[i] = _merge_cell(prior_data[i], live_data[i], conflict_mode)
    merged.data = out
    return merged


def _maps_compatible_for_compare(live_occ, saved):
    width = int(live_occ.info.width)
    height = int(live_occ.info.height)
    if saved["width"] != width or saved["height"] != height:
        return False
    if abs(saved["resolution"] - float(live_occ.info.resolution)) > 1e-6:
        return False
    if (
        abs(saved["origin_x"] - float(live_occ.info.origin.position.x)) > 1e-6
        or abs(saved["origin_y"] - float(live_occ.info.origin.position.y)) > 1e-6
    ):
        return False
    return True


def _map_stats(data):
    known = 0
    free = 0
    occ = 0
    for v in data:
        if v < 0:
            continue
        known += 1
        if v <= 25:
            free += 1
        elif v >= 65:
            occ += 1
    return known, free, occ


def _publish_mode(enabled, mode, repeats=8, hz=12.0):
    mode_pub = rospy.Publisher("/mower/mode", String, queue_size=5)
    enabled_pub = rospy.Publisher("/mower/autonomy_enabled", Bool, queue_size=5)
    deadline = time.time() + 2.0
    while time.time() < deadline and not rospy.is_shutdown():
        if mode_pub.get_num_connections() > 0 and enabled_pub.get_num_connections() > 0:
            break
        time.sleep(0.05)
    period = 1.0 / max(1.0, hz)
    for _ in range(max(1, repeats)):
        mode_pub.publish(String(data=mode))
        enabled_pub.publish(Bool(data=enabled))
        time.sleep(period)


def cmd_save_map(args):
    rospy.init_node("mower_trial_workflow_save_map", anonymous=True)
    rospy.loginfo("Waiting for map service: %s", args.map_service)
    rospy.wait_for_service(args.map_service, timeout=args.timeout_s)
    srv = rospy.ServiceProxy(args.map_service, GetMap)
    occ = srv().map
    base = os.path.abspath(args.output_base)
    map_to_save = occ
    saved = _load_saved_occupancy(base)
    if (
        saved is not None
        and not args.merge_existing
        and not args.force_overwrite
        and _maps_compatible_for_compare(occ, saved)
    ):
        old_known, _, _ = _map_stats(list(saved["data"]))
        new_known, _, _ = _map_stats(list(occ.data))
        min_known_ratio = max(0.0, float(args.min_known_ratio))
        if old_known > 0 and new_known < int(round(old_known * min_known_ratio)):
            raise RuntimeError(
                (
                    "Refusing to overwrite saved map: live known cells %d < %.0f%% of saved %d. "
                    "Use --merge-existing to fuse updates, or --force-overwrite to replace."
                )
                % (new_known, 100.0 * min_known_ratio, old_known)
            )
    if args.merge_existing:
        if saved is None:
            rospy.logwarn(
                "No existing saved map at %s.[yaml|pgm]; writing live map only.",
                base,
            )
        else:
            map_to_save = _merge_occupancy_grid_with_saved(
                occ, saved, args.merge_conflict_mode
            )
            l_known, l_free, l_occ = _map_stats(list(occ.data))
            m_known, m_free, m_occ = _map_stats(list(map_to_save.data))
            rospy.loginfo(
                "Merged live map into saved map (%s): known %d->%d, free %d->%d, occupied %d->%d",
                args.merge_conflict_mode,
                l_known,
                m_known,
                l_free,
                m_free,
                l_occ,
                m_occ,
            )
    pgm_path, yaml_path = _save_occupancy_to_files(map_to_save, base)
    rospy.loginfo("Saved map to: %s and %s", pgm_path, yaml_path)
    print(yaml_path)
    return 0


def cmd_build_plan(args):
    rospy.init_node("mower_trial_workflow_build_plan", anonymous=True)
    rospy.loginfo("Waiting for planner service: %s", args.plan_service)
    rospy.wait_for_service(args.plan_service, timeout=args.timeout_s)
    srv = rospy.ServiceProxy(args.plan_service, Trigger)
    res = srv()
    if res.success:
        rospy.loginfo("Plan build succeeded: %s", res.message)
        return 0
    rospy.logerr("Plan build failed: %s", res.message)
    return 2


def cmd_start_cut(_args):
    rospy.init_node("mower_trial_workflow_start_cut", anonymous=True)
    _publish_mode(enabled=True, mode="auto")
    rospy.loginfo("Autonomous cutting started (mode=auto, autonomy_enabled=true).")
    return 0


def cmd_stop_cut(_args):
    rospy.init_node("mower_trial_workflow_stop_cut", anonymous=True)
    _publish_mode(enabled=False, mode="manual")
    rospy.loginfo("Cutting stopped (mode=manual, autonomy_enabled=false).")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Two-phase mower workflow helper: save map, build plan, start/stop cut."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save-map", help="Save current SLAM map to mower_bringup/maps/latest_scan.*")
    p_save.add_argument("--map-service", default="/dynamic_map", help="GetMap service name")
    p_save.add_argument("--output-base", default=_default_map_base(), help="Output path without extension")
    p_save.add_argument("--timeout-s", type=float, default=10.0, help="Service wait timeout")
    p_save.add_argument(
        "--merge-existing",
        action="store_true",
        help="Fuse current map with existing saved map instead of replacing it.",
    )
    p_save.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow replacing saved map even if current map has much less known area.",
    )
    p_save.add_argument(
        "--min-known-ratio",
        type=float,
        default=0.70,
        help="Safety floor for overwrite: require current known cells >= this fraction of saved known cells.",
    )
    p_save.add_argument(
        "--merge-conflict-mode",
        choices=["occupied_wins", "newest", "average"],
        default="occupied_wins",
        help="How to resolve cell conflicts when both maps have known values.",
    )
    p_save.set_defaults(func=cmd_save_map)

    p_plan = sub.add_parser("build-plan", help="Call cut planner Trigger service")
    p_plan.add_argument(
        "--plan-service", default="/cut_region_planner/build_plan", help="Planner Trigger service"
    )
    p_plan.add_argument("--timeout-s", type=float, default=8.0, help="Service wait timeout")
    p_plan.set_defaults(func=cmd_build_plan)

    p_start = sub.add_parser("start-cut", help="Set mode=auto and autonomy_enabled=true")
    p_start.set_defaults(func=cmd_start_cut)

    p_stop = sub.add_parser("stop-cut", help="Set mode=manual and autonomy_enabled=false")
    p_stop.set_defaults(func=cmd_stop_cut)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except rospy.ROSException as exc:
        rospy.logerr(str(exc))
        return 3
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

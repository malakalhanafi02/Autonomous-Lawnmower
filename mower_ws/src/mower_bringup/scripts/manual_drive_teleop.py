#!/usr/bin/env python3
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


INSTRUCTIONS = """
Manual Drive Teleop (publishes to /cmd_vel_manual)
--------------------------------------------------
Drive:
  w/i : forward
  s/, : reverse
  a/j : turn left in place
  d/l : turn right in place
  q/u : forward + left
  e/o : forward + right
  z/m : reverse + left
  c/. : reverse + right

Other:
  space or k : stop
  r/f        : increase/decrease linear speed (10%)
  t/g        : increase/decrease angular speed (10%)
  Ctrl-C     : quit
"""


def _drain_escape_sequence():
    # Drain arrow-key escape sequence so it does not leak into command parsing.
    while True:
        ready_more, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready_more:
            break
        sys.stdin.read(1)


def _get_key(timeout_s):
    """
    Return the latest key currently buffered in stdin.
    This avoids delayed/buffered motion when the terminal auto-repeats keys.
    """
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return ""

    latest = ""
    while True:
        key = sys.stdin.read(1)
        if key == "\x1b":
            _drain_escape_sequence()
            key = ""
        if key:
            latest = key
        ready_more, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready_more:
            break
    return latest


def _publish_mode(mode_pub, enabled_pub, mode, enabled, repeats):
    for _ in range(max(1, repeats)):
        mode_pub.publish(String(data=mode))
        enabled_pub.publish(Bool(data=enabled))


def main():
    rospy.init_node("manual_drive_teleop")

    cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel_manual")
    mode_topic = rospy.get_param("~mode_topic", "/mower/mode")
    enabled_topic = rospy.get_param("~enabled_topic", "/mower/autonomy_enabled")
    linear_speed = float(rospy.get_param("~linear_speed_mps", 0.35))
    angular_speed = float(rospy.get_param("~angular_speed_rps", 0.9))
    loop_hz = float(rospy.get_param("~loop_hz", 20.0))
    force_manual = bool(rospy.get_param("~force_manual", True))
    mode_refresh_s = float(rospy.get_param("~mode_refresh_s", 0.5))
    mode_publish_burst = int(rospy.get_param("~mode_publish_burst", 4))
    command_hold_timeout_s = float(rospy.get_param("~command_hold_timeout_s", 1.0))
    key_release_timeout_s = float(rospy.get_param("~key_release_timeout_s", 0.18))

    cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
    mode_pub = rospy.Publisher(mode_topic, String, queue_size=5)
    enabled_pub = rospy.Publisher(enabled_topic, Bool, queue_size=5)

    key_to_factors = {
        "w": (1.0, 0.0),
        "W": (1.0, 0.0),
        "i": (1.0, 0.0),
        "I": (1.0, 0.0),
        "s": (-1.0, 0.0),
        "S": (-1.0, 0.0),
        ",": (-1.0, 0.0),
        "a": (0.0, 1.0),
        "A": (0.0, 1.0),
        "j": (0.0, 1.0),
        "J": (0.0, 1.0),
        "d": (0.0, -1.0),
        "D": (0.0, -1.0),
        "l": (0.0, -1.0),
        "L": (0.0, -1.0),
        "q": (1.0, 1.0),
        "Q": (1.0, 1.0),
        "u": (1.0, 1.0),
        "U": (1.0, 1.0),
        "e": (1.0, -1.0),
        "E": (1.0, -1.0),
        "o": (1.0, -1.0),
        "O": (1.0, -1.0),
        "z": (-1.0, 1.0),
        "Z": (-1.0, 1.0),
        "m": (-1.0, 1.0),
        "M": (-1.0, 1.0),
        "c": (-1.0, -1.0),
        "C": (-1.0, -1.0),
        ".": (-1.0, -1.0),
    }
    stop_keys = {" ", "k"}

    cmd = Twist()
    last_mode_pub = rospy.Time(0)
    last_cmd_set = rospy.Time(0)
    last_key_event = rospy.Time(0)
    active_motion_key = ""

    old_settings = termios.tcgetattr(sys.stdin)
    print(INSTRUCTIONS)
    print(
        "Current speeds: linear=%.2f m/s angular=%.2f rad/s"
        % (linear_speed, angular_speed)
    )
    print("Publishing manual commands to: %s" % cmd_topic)

    try:
        tty.setcbreak(sys.stdin.fileno())
        rate = rospy.Rate(loop_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if force_manual and (now - last_mode_pub).to_sec() >= mode_refresh_s:
                _publish_mode(
                    mode_pub,
                    enabled_pub,
                    mode="manual",
                    enabled=False,
                    repeats=mode_publish_burst,
                )
                last_mode_pub = now

            # Approximate key release in a terminal by waiting for a quiet gap.
            if (
                last_key_event != rospy.Time(0)
                and (now - last_key_event).to_sec() >= key_release_timeout_s
            ):
                active_motion_key = ""

            # Manual-only deadman: motion command expires unless re-armed by a fresh key press.
            if (
                command_hold_timeout_s > 0.0
                and last_cmd_set != rospy.Time(0)
                and (now - last_cmd_set).to_sec() >= command_hold_timeout_s
            ):
                cmd = Twist()
                last_cmd_set = rospy.Time(0)

            key = _get_key(0.0)
            if key:
                last_key_event = now
                if key == "\x03":
                    break
                if key in key_to_factors:
                    # Ignore terminal auto-repeat while key is held;
                    # require a brief key-up gap before accepting same key again.
                    if key != active_motion_key:
                        lin, ang = key_to_factors[key]
                        cmd = Twist()
                        cmd.linear.x = lin * linear_speed
                        cmd.angular.z = ang * angular_speed
                        last_cmd_set = now
                        active_motion_key = key
                elif key in stop_keys:
                    cmd = Twist()
                    last_cmd_set = rospy.Time(0)
                    active_motion_key = ""
                elif key == "r":
                    linear_speed *= 1.1
                    print("linear speed -> %.2f m/s" % linear_speed)
                elif key == "f":
                    linear_speed = max(0.02, linear_speed / 1.1)
                    print("linear speed -> %.2f m/s" % linear_speed)
                elif key == "t":
                    angular_speed *= 1.1
                    print("angular speed -> %.2f rad/s" % angular_speed)
                elif key == "g":
                    angular_speed = max(0.05, angular_speed / 1.1)
                    print("angular speed -> %.2f rad/s" % angular_speed)
                else:
                    # Unknown key: stop to avoid stale forward commands.
                    cmd = Twist()
                    last_cmd_set = rospy.Time(0)
                    active_motion_key = ""

            cmd_pub.publish(cmd)
            rate.sleep()
    finally:
        cmd_pub.publish(Twist())
        if force_manual:
            _publish_mode(
                mode_pub,
                enabled_pub,
                mode="manual",
                enabled=False,
                repeats=mode_publish_burst,
            )
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()

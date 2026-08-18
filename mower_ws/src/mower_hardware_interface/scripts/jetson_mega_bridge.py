#!/usr/bin/env python3
import math
import re
import threading

import rospy
import serial
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


class JetsonMegaBridge:
    def __init__(self):
        self.port = rospy.get_param("~port", "/dev/ttyACM0")
        self.baud = int(rospy.get_param("~baud", 115200))
        self.write_timeout = float(rospy.get_param("~write_timeout", 0.05))
        self.command_timeout = float(rospy.get_param("~command_timeout", 0.5))
        self.loop_hz = float(rospy.get_param("~loop_hz", 30.0))

        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 1.0))
        self.max_steer_rad = float(rospy.get_param("~max_steer_rad", 0.6))
        self.wheelbase_m = float(rospy.get_param("~wheelbase_m", 0.50))
        self.min_speed_for_steer = float(rospy.get_param("~min_speed_for_steer", 0.08))
        self.in_place_turn_speed_mps = float(rospy.get_param("~in_place_turn_speed_mps", 0.20))
        self.min_publish_delta = float(rospy.get_param("~min_publish_delta", 0.01))
        self.read_loop_enable = bool(rospy.get_param("~read_loop_enable", True))
        self.read_timeout = float(rospy.get_param("~read_timeout", 0.0))

        self._us_re = re.compile(r"UScm=([0-9]*\.?[0-9]+).*blk=([01])")

        self.serial_lock = threading.Lock()
        self.serial_conn = None
        self.last_cmd_time = rospy.Time(0)
        self.last_velocity = 0.0
        self.last_angle = 0.0
        self.stop_sent = True

        self.last_serial_payload = ""
        self.debug_pub = rospy.Publisher("~last_serial_command", String, queue_size=10)
        self.last_line_pub = rospy.Publisher("~last_serial_line", String, queue_size=10)
        self.us_cm_pub = rospy.Publisher("~ultrasonic_distance_cm", Float32, queue_size=10)
        self.blocked_pub = rospy.Publisher("~collision_blocked", Bool, queue_size=10)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_cb, queue_size=20)

        self.connect_serial()

    def connect_serial(self):
        while not rospy.is_shutdown():
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    timeout=self.read_timeout,
                    write_timeout=self.write_timeout,
                )
                rospy.loginfo("Connected to Arduino Mega on %s @ %d", self.port, self.baud)
                return
            except serial.SerialException as exc:
                rospy.logwarn_throttle(5.0, "Serial connect failed on %s: %s", self.port, exc)
                rospy.sleep(1.0)

    def cmd_cb(self, msg):
        velocity = self.clamp(msg.linear.x, -self.max_speed_mps, self.max_speed_mps)
        omega = msg.angular.z
        angle = self.omega_to_steering(velocity, omega)
        now = rospy.Time.now()

        should_send = (
            abs(velocity - self.last_velocity) >= self.min_publish_delta
            or abs(angle - self.last_angle) >= self.min_publish_delta
            or self.stop_sent
        )
        if should_send:
            self.send_drive_command(velocity, angle)

        self.last_cmd_time = now
        self.last_velocity = velocity
        self.last_angle = angle
        self.stop_sent = False

    def omega_to_steering(self, velocity, omega):
        if abs(omega) < 1e-4:
            return 0.0

        speed_for_calc = velocity
        if abs(speed_for_calc) < self.min_speed_for_steer:
            speed_for_calc = math.copysign(self.in_place_turn_speed_mps, omega)

        angle = math.atan2(self.wheelbase_m * omega, speed_for_calc)
        return self.clamp(angle, -self.max_steer_rad, self.max_steer_rad)

    def send_drive_command(self, velocity, angle):
        payload = f"V:{velocity:.3f},A:{angle:.3f}\n"
        self.send_serial(payload)

    def send_stop_once(self):
        if not self.stop_sent:
            self.send_drive_command(0.0, 0.0)
            self.stop_sent = True
            rospy.logwarn("Watchdog timeout hit. Sent one-shot stop command.")

    def send_serial(self, payload):
        if rospy.is_shutdown():
            return

        with self.serial_lock:
            if self.serial_conn is None or not self.serial_conn.is_open:
                self.connect_serial()
            if self.serial_conn is None:
                return
            try:
                self.serial_conn.write(payload.encode("ascii"))
                self.last_serial_payload = payload.strip()
                self.debug_pub.publish(String(data=self.last_serial_payload))
            except (serial.SerialException, OSError) as exc:
                rospy.logerr_throttle(2.0, "Serial write failed: %s", exc)
                self.close_serial()

    def poll_serial(self):
        if not self.read_loop_enable:
            return
        with self.serial_lock:
            if self.serial_conn is None or not self.serial_conn.is_open:
                return
            try:
                raw = self.serial_conn.readline()
            except (serial.SerialException, OSError) as exc:
                rospy.logerr_throttle(2.0, "Serial read failed: %s", exc)
                self.close_serial()
                return
        if not raw:
            return
        try:
            line = raw.decode("ascii", errors="ignore").strip()
        except Exception:
            return
        if not line:
            return

        self.last_line_pub.publish(String(data=line))
        m = self._us_re.search(line)
        if m:
            self.us_cm_pub.publish(Float32(data=float(m.group(1))))
            self.blocked_pub.publish(Bool(data=(m.group(2) == "1")))

    def close_serial(self):
        with self.serial_lock:
            if self.serial_conn is not None:
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
                self.serial_conn = None

    @staticmethod
    def clamp(val, low, high):
        return max(low, min(high, val))

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            if self.last_cmd_time != rospy.Time(0):
                elapsed = (rospy.Time.now() - self.last_cmd_time).to_sec()
                if elapsed > self.command_timeout:
                    self.send_stop_once()
            self.poll_serial()
            rate.sleep()

        self.send_drive_command(0.0, 0.0)
        self.close_serial()


def main():
    rospy.init_node("jetson_mega_bridge")
    bridge = JetsonMegaBridge()
    bridge.spin()


if __name__ == "__main__":
    main()

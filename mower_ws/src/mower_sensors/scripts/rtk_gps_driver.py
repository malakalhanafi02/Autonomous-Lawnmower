#!/usr/bin/env python3
"""
ROS driver for the Waveshare LC29H dual-band RTK GPS HAT.

Reads NMEA sentences from the module over serial and publishes:
  /fix              (NavSatFix)    – latitude, longitude, altitude, fix type
  /fix/velocity     (TwistStamped) – ground speed and course
  /gps/satellites   (UInt16)       – visible satellite count

Optionally connects to an NTRIP caster to receive RTCM3 corrections,
which are forwarded to the GPS module for RTK Fixed positioning
(centimetre-level accuracy).

Supported fix types reported via NavSatStatus:
  0 = No fix
  1 = GPS SPS        (~2.5 m)
  2 = DGPS / SBAS    (~1.0 m)
  4 = RTK Fixed       (~0.02 m)
  5 = RTK Float       (~0.30 m)
"""

import base64
import math
import socket
import threading
import time

import rospy
import serial
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import UInt16, String


# ---------------------------------------------------------------------------
#  NMEA helpers (no external dependency needed)
# ---------------------------------------------------------------------------

def _nmea_checksum_ok(sentence):
    """Validate NMEA checksum (*XX at the end)."""
    if "*" not in sentence:
        return False
    body, cksum_hex = sentence.rsplit("*", 1)
    body = body.lstrip("$")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(cksum_hex[:2], 16)
    except ValueError:
        return False


def _nmea_field(fields, idx, default=""):
    return fields[idx] if idx < len(fields) else default


def _parse_lat(raw, hemi):
    if not raw:
        return float("nan")
    deg = int(raw[:2])
    minutes = float(raw[2:])
    val = deg + minutes / 60.0
    if hemi == "S":
        val = -val
    return val


def _parse_lon(raw, hemi):
    if not raw:
        return float("nan")
    deg = int(raw[:3])
    minutes = float(raw[3:])
    val = deg + minutes / 60.0
    if hemi == "W":
        val = -val
    return val


def _knots_to_mps(knots_str):
    try:
        return float(knots_str) * 0.514444
    except (ValueError, TypeError):
        return 0.0


# GGA fix-quality → (NavSatStatus.status, covariance m²)
_FIX_MAP = {
    0: (NavSatStatus.STATUS_NO_FIX, 999.0),
    1: (NavSatStatus.STATUS_FIX, 6.25),       # SPS ~2.5 m
    2: (NavSatStatus.STATUS_SBAS_FIX, 1.0),   # DGPS ~1.0 m
    4: (NavSatStatus.STATUS_GBAS_FIX, 0.0004), # RTK Fixed ~0.02 m
    5: (NavSatStatus.STATUS_GBAS_FIX, 0.09),   # RTK Float ~0.30 m
    6: (NavSatStatus.STATUS_NO_FIX, 999.0),    # Dead reckoning
}

_FIX_LABEL = {0: "No fix", 1: "SPS", 2: "DGPS", 4: "RTK Fixed", 5: "RTK Float", 6: "DR"}


# ---------------------------------------------------------------------------
#  NTRIP client (runs in a background thread)
# ---------------------------------------------------------------------------

class NTRIPClient:
    """Minimal NTRIP 1.0/2.0 client that feeds RTCM3 to a serial port."""

    RECONNECT_DELAY = 5.0
    READ_SIZE = 4096

    def __init__(self, host, port, mountpoint, user, password,
                 serial_conn, serial_lock,
                 send_gga=True, gga_interval=10.0):
        self.host = host
        self.port = port
        self.mountpoint = mountpoint
        self.auth = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.serial_conn = serial_conn
        self.serial_lock = serial_lock
        self.send_gga = send_gga
        self.gga_interval = gga_interval

        self.latest_gga = None
        self._sock = None
        self._running = True
        self._bytes_rx = 0

    def set_gga(self, gga_sentence):
        self.latest_gga = gga_sentence

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def run(self):
        while self._running and not rospy.is_shutdown():
            try:
                self._connect_and_stream()
            except Exception as exc:
                rospy.logwarn_throttle(10.0, "NTRIP error: %s", exc)
            if self._running:
                rospy.sleep(self.RECONNECT_DELAY)

    def _connect_and_stream(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10.0)
        self._sock.connect((self.host, self.port))

        request = (
            f"GET /{self.mountpoint} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Ntrip-Version: Ntrip/2.0\r\n"
            f"User-Agent: NTRIP MowerROS/1.0\r\n"
            f"Authorization: Basic {self.auth}\r\n"
            f"\r\n"
        )
        self._sock.sendall(request.encode("ascii"))

        header = self._sock.recv(4096).decode("ascii", errors="ignore")
        if "200" not in header.split("\n")[0]:
            rospy.logerr("NTRIP caster rejected connection: %s",
                         header.split("\n")[0].strip())
            self._sock.close()
            return

        rospy.loginfo("NTRIP connected to %s:%d/%s",
                      self.host, self.port, self.mountpoint)

        self._sock.settimeout(30.0)
        last_gga_time = 0.0

        while self._running and not rospy.is_shutdown():
            # Send GGA to caster periodically (needed for VRS)
            now = time.time()
            if (self.send_gga and self.latest_gga
                    and now - last_gga_time >= self.gga_interval):
                try:
                    self._sock.sendall((self.latest_gga + "\r\n").encode("ascii"))
                    last_gga_time = now
                except Exception:
                    break

            # Receive RTCM3 data
            try:
                data = self._sock.recv(self.READ_SIZE)
            except socket.timeout:
                continue
            except Exception:
                break

            if not data:
                break

            self._bytes_rx += len(data)

            # Forward RTCM to the GPS module
            with self.serial_lock:
                try:
                    if self.serial_conn and self.serial_conn.is_open:
                        self.serial_conn.write(data)
                except Exception as exc:
                    rospy.logwarn_throttle(5.0, "RTCM serial write error: %s", exc)

        self._sock.close()
        rospy.logwarn("NTRIP stream disconnected. Bytes received: %d", self._bytes_rx)


# ---------------------------------------------------------------------------
#  Main ROS node
# ---------------------------------------------------------------------------

class RTKGPSDriver:

    def __init__(self):
        # Serial
        self.port = rospy.get_param("~port", "/dev/ttyS0")
        self.baud = int(rospy.get_param("~baud", 115200))
        self.frame_id = rospy.get_param("~frame_id", "gps_link")

        # NTRIP
        self.ntrip_enabled = rospy.get_param("~ntrip/enabled", False)

        # Publishers
        self.fix_pub = rospy.Publisher("~fix", NavSatFix, queue_size=10)
        self.vel_pub = rospy.Publisher("~velocity", TwistStamped, queue_size=10)
        self.sat_pub = rospy.Publisher("~satellites", UInt16, queue_size=10)
        self.raw_pub = rospy.Publisher("~nmea_sentence", String, queue_size=50)

        # Remap /fix for the rest of the stack
        self.global_fix_pub = rospy.Publisher("/fix", NavSatFix, queue_size=10)

        # Serial
        self.serial_lock = threading.Lock()
        self.serial_conn = None
        self._connect_serial()

        # NTRIP client
        self.ntrip = None
        if self.ntrip_enabled:
            self._start_ntrip()

        # State
        self.last_fix_quality = -1

    def _connect_serial(self):
        while not rospy.is_shutdown():
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    timeout=1.0,
                )
                rospy.loginfo("GPS serial connected: %s @ %d", self.port, self.baud)
                return
            except serial.SerialException as exc:
                rospy.logwarn_throttle(5.0, "GPS serial connect failed: %s", exc)
                rospy.sleep(2.0)

    def _start_ntrip(self):
        host = rospy.get_param("~ntrip/host", "")
        port = int(rospy.get_param("~ntrip/port", 2101))
        mount = rospy.get_param("~ntrip/mountpoint", "")
        user = rospy.get_param("~ntrip/user", "")
        pw = rospy.get_param("~ntrip/password", "")
        send_gga = rospy.get_param("~ntrip/send_gga", True)
        gga_interval = float(rospy.get_param("~ntrip/gga_interval", 10.0))

        if not host or not mount:
            rospy.logwarn("NTRIP enabled but host/mountpoint not set. Skipping.")
            return

        self.ntrip = NTRIPClient(
            host, port, mount, user, pw,
            self.serial_conn, self.serial_lock,
            send_gga, gga_interval,
        )
        t = threading.Thread(target=self.ntrip.run, daemon=True)
        t.start()
        rospy.loginfo("NTRIP client started -> %s:%d/%s", host, port, mount)

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def spin(self):
        while not rospy.is_shutdown():
            if self.serial_conn is None or not self.serial_conn.is_open:
                self._connect_serial()
                if self.serial_conn is None:
                    continue

            try:
                raw_line = self.serial_conn.readline()
            except serial.SerialException as exc:
                rospy.logerr_throttle(5.0, "GPS serial read error: %s", exc)
                with self.serial_lock:
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass
                    self.serial_conn = None
                continue

            if not raw_line:
                continue

            try:
                sentence = raw_line.decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            if not sentence.startswith("$"):
                continue
            if not _nmea_checksum_ok(sentence):
                continue

            self.raw_pub.publish(String(data=sentence))

            msg_type = sentence.split(",")[0]
            if msg_type in ("$GNGGA", "$GPGGA"):
                self._handle_gga(sentence)
            elif msg_type in ("$GNRMC", "$GPRMC"):
                self._handle_rmc(sentence)

        # Shutdown
        if self.ntrip:
            self.ntrip.stop()
        if self.serial_conn:
            self.serial_conn.close()

    # ------------------------------------------------------------------
    #  GGA – position, fix quality, altitude, satellites
    # ------------------------------------------------------------------
    def _handle_gga(self, sentence):
        fields = sentence.split(",")
        # $GxGGA,time,lat,N/S,lon,E/W,quality,numSV,HDOP,alt,M,geoid,M,...*cs

        try:
            fix_quality = int(_nmea_field(fields, 6, "0"))
        except ValueError:
            fix_quality = 0

        lat = _parse_lat(_nmea_field(fields, 2), _nmea_field(fields, 3))
        lon = _parse_lon(_nmea_field(fields, 4), _nmea_field(fields, 5))

        try:
            alt = float(_nmea_field(fields, 9, "0"))
        except ValueError:
            alt = 0.0

        try:
            num_sats = int(_nmea_field(fields, 7, "0"))
        except ValueError:
            num_sats = 0

        try:
            hdop = float(_nmea_field(fields, 8, "99"))
        except ValueError:
            hdop = 99.0

        status_val, base_cov = _FIX_MAP.get(fix_quality, _FIX_MAP[0])
        cov = base_cov * max(hdop, 1.0)

        # Log fix type changes
        if fix_quality != self.last_fix_quality:
            label = _FIX_LABEL.get(fix_quality, "Unknown")
            rospy.loginfo("GPS fix type changed: %s (quality=%d, sats=%d)",
                          label, fix_quality, num_sats)
            self.last_fix_quality = fix_quality

        # NavSatFix
        fix = NavSatFix()
        fix.header.stamp = rospy.Time.now()
        fix.header.frame_id = self.frame_id
        fix.status.status = status_val
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = alt
        fix.position_covariance = [
            cov, 0.0, 0.0,
            0.0, cov, 0.0,
            0.0, 0.0, cov * 4.0,
        ]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        self.fix_pub.publish(fix)
        self.global_fix_pub.publish(fix)
        self.sat_pub.publish(UInt16(data=num_sats))

        # Forward GGA to NTRIP caster
        if self.ntrip:
            self.ntrip.set_gga(sentence)

    # ------------------------------------------------------------------
    #  RMC – ground speed and course
    # ------------------------------------------------------------------
    def _handle_rmc(self, sentence):
        fields = sentence.split(",")
        # $GxRMC,time,status,lat,N/S,lon,E/W,speed_kn,course,date,...*cs

        status = _nmea_field(fields, 2)
        if status != "A":
            return

        speed_mps = _knots_to_mps(_nmea_field(fields, 7))

        try:
            course_deg = float(_nmea_field(fields, 8, "0"))
        except ValueError:
            course_deg = 0.0

        course_rad = math.radians(course_deg)

        vel = TwistStamped()
        vel.header.stamp = rospy.Time.now()
        vel.header.frame_id = self.frame_id
        vel.twist.linear.x = speed_mps * math.cos(course_rad)
        vel.twist.linear.y = speed_mps * math.sin(course_rad)
        self.vel_pub.publish(vel)


def main():
    rospy.init_node("rtk_gps_driver")
    driver = RTKGPSDriver()
    rospy.loginfo(
        "RTK GPS driver ready on %s @ %d  (NTRIP: %s)",
        rospy.get_param("~port", "/dev/ttyS0"),
        rospy.get_param("~baud", 115200),
        "enabled" if rospy.get_param("~ntrip/enabled", False) else "disabled",
    )
    driver.spin()


if __name__ == "__main__":
    main()

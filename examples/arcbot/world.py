"""ARCBOT simulation core.

A 4x4 m laboratory test arena with a small differential-drive robot. Physics
ticks at 50 Hz in real wall-clock time. Sensor snapshots are synthesized from
the physical state with realistic noise, bias drift, occasional dropouts, and
the kind of imperfections real sensors show.

The agent talks to a `World` instance through `World.snapshot()` (sensor read)
and `World.cmd_vel(linear, angular)` (velocity command). The agent does not
see this Python; it sees the formatted perception prompt produced by
`bridge.format_perception()`.

Coordinate convention:
    x = east-west, y = north-south, theta CCW from +x axis. SI units throughout.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional


# Physical constants
ROOM_X = 4.0
ROOM_Y = 4.0
ROBOT_RADIUS = 0.10
WHEEL_BASE = 0.16
WHEEL_RADIUS = 0.033
ENCODER_TICKS_PER_REV = 1440  # quadrature, 360 ppr * 4
LIDAR_BEAMS = 12
LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 4.0
TICK_HZ = 50.0
TICK_DT = 1.0 / TICK_HZ

BATTERY_FULL_V = 12.6
BATTERY_EMPTY_V = 9.0
BATTERY_CAPACITY_WH = 8.0
BASE_LOAD_W = 4.5
MOTOR_POWER_W_PER_MPS = 8.0
# Reduced from 60s to 22s so the first-order heating curve is visibly
# flattening within a 30-second observation window. Previously the agent
# perceived the rise as "too linear" because we were observing only the
# early linear regime of a long-tau exponential.
MOTOR_THERMAL_TIME_CONST_S = 22.0
# Heating coefficient: roughly 0.18 C per W per second. Steady-state rise
# at 2W is then ~ tau * 0.18 * 2 = ~8C above ambient.
MOTOR_HEAT_COEF = 0.18
MOTOR_AMBIENT_TEMP_C = 22.5

LIDAR_NOISE_STD = 0.012
# Real RPLIDAR A1 in a cluttered indoor scene returns ~5-10% invalid beams
# per scan (acute angles, dark surfaces, glass). Previous debrief: agent
# said "all 12 beams return valid ranges every scan - too clean."
LIDAR_DROPOUT_PROB = 0.06
LIDAR_OUTLIER_PROB = 0.008
IMU_ACCEL_NOISE_STD = 0.04
IMU_GYRO_NOISE_STD = 0.005
IMU_GYRO_BIAS_RW = 0.0001
ENCODER_SLIP_STD = 0.002

DOCK_X = 3.85
DOCK_Y = 2.0
DOCK_THETA = math.pi
DOCK_TOLERANCE = 0.10

MAX_LINEAR_V = 0.30
MAX_ANGULAR_V = 1.5
LINEAR_ACCEL_MAX = 0.4
ANGULAR_ACCEL_MAX = 3.0


@dataclass
class Wall:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class CircleObstacle:
    cx: float
    cy: float
    r: float


def default_walls() -> list[Wall]:
    """Outer walls slightly rotated so LIDAR scans don't return exact 0/90."""
    a = math.radians(1.5)
    cx_room = ROOM_X / 2
    cy_room = ROOM_Y / 2
    corners = []
    for x, y in [(0, 0), (ROOM_X, 0), (ROOM_X, ROOM_Y), (0, ROOM_Y)]:
        rx = (x - cx_room) * math.cos(a) - (y - cy_room) * math.sin(a) + cx_room
        ry = (x - cx_room) * math.sin(a) + (y - cy_room) * math.cos(a) + cy_room
        corners.append((rx, ry))
    walls = []
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        walls.append(Wall(x0, y0, x1, y1))
    # Desk in SE
    walls.extend([
        Wall(2.5, 0.5, 3.4, 0.5),
        Wall(3.4, 0.5, 3.4, 1.4),
        Wall(3.4, 1.4, 2.5, 1.4),
        Wall(2.5, 1.4, 2.5, 0.5),
    ])
    # Bookshelf NW
    walls.extend([
        Wall(0.4, 3.0, 1.8, 3.0),
        Wall(1.8, 3.0, 1.8, 3.3),
        Wall(1.8, 3.3, 0.4, 3.3),
        Wall(0.4, 3.3, 0.4, 3.0),
    ])
    return walls


def default_obstacles() -> list[CircleObstacle]:
    return [
        CircleObstacle(1.6, 1.8, 0.18),  # chair
        CircleObstacle(0.6, 0.6, 0.12),  # wastebasket
    ]


def _ray_segment(ox, oy, dx, dy, x0, y0, x1, y1) -> Optional[float]:
    sx = x1 - x0
    sy = y1 - y0
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:
        return None
    t = ((x0 - ox) * sy - (y0 - oy) * sx) / denom
    u = ((x0 - ox) * dy - (y0 - oy) * dx) / denom
    if t < 0 or u < 0 or u > 1:
        return None
    return t


def _ray_circle(ox, oy, dx, dy, cx, cy, r) -> Optional[float]:
    fx = ox - cx
    fy = oy - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    if t0 > 0:
        return t0
    if t1 > 0:
        return t1
    return None


def cast_ray(ox, oy, angle, walls, obstacles, max_range=LIDAR_MAX_RANGE) -> float:
    dx = math.cos(angle)
    dy = math.sin(angle)
    best = max_range
    for w in walls:
        d = _ray_segment(ox, oy, dx, dy, w.x0, w.y0, w.x1, w.y1)
        if d is not None and 0 < d < best:
            best = d
    for o in obstacles:
        d = _ray_circle(ox, oy, dx, dy, o.cx, o.cy, o.r)
        if d is not None and 0 < d < best:
            best = d
    return best


@dataclass
class RobotState:
    x: float = 0.8
    y: float = 0.8
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    cmd_v: float = 0.0
    cmd_omega: float = 0.0
    left_ticks: int = 0
    right_ticks: int = 0
    # Persistent IMU biases populated in World.__init__. Real MEMS sensors
    # have stable per-axis offsets (the agent's debrief flagged the absence
    # of these when the gyro_z exactly tracked commanded omega).
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0
    battery_v: float = BATTERY_FULL_V
    battery_charge_wh: float = BATTERY_CAPACITY_WH
    motor_left_temp: float = MOTOR_AMBIENT_TEMP_C
    motor_right_temp: float = MOTOR_AMBIENT_TEMP_C
    motor_left_tau_s: float = MOTOR_THERMAL_TIME_CONST_S
    motor_right_tau_s: float = MOTOR_THERMAL_TIME_CONST_S
    motor_left_heat_coef: float = 0.18
    motor_right_heat_coef: float = 0.18
    # Gyro per-axis scale factor errors. Real MEMS gyros have ~0.5-2%
    # scale factor error from calibration. Previous debrief noted the
    # gyro_z tracking commanded omega within 0.04% (implausibly clean).
    gyro_scale_x: float = 1.0
    gyro_scale_y: float = 1.0
    gyro_scale_z: float = 1.0
    # Per-wheel calibration scale factors. Real wheels are not exactly the
    # same diameter (manufacturing tolerance ~0.5%), so encoder ticks drift
    # apart on straight runs. The previous debrief flagged identical ticks.
    wheel_left_scale: float = 1.0
    wheel_right_scale: float = 1.0
    # Per-beam dropout state for correlated LIDAR fault bursts.
    lidar_glitched_beams: tuple = ()
    # BMS-side smoothed values. Real fuel gauge ICs apply EWMA filtering
    # to reported SoC and runtime so the reported values lag and have
    # slow-correlated noise rather than per-sample independent noise.
    soc_reported: float = 1.0
    runtime_reported: float = 60.0
    # Magnetometer hard-iron offsets. Local distortion from steel chassis
    # plates, motor magnets, and DC currents produces persistent per-axis
    # offsets. Initialized at boot.
    mag_offset_x: float = 0.0
    mag_offset_y: float = 0.0
    mag_offset_z: float = 0.0
    cpu_temp: float = 41.5
    ambient_temp: float = MOTOR_AMBIENT_TEMP_C
    bumper_left: bool = False
    bumper_right: bool = False
    boot_elapsed: float = 0.0
    calibrated: bool = False
    docked: bool = False
    charging: bool = False


class World:
    """Differential-drive robot in a 4x4m room. Implements the phantomstack
    Simulator protocol: tick_hz attribute, step(), snapshot(), cmd(payload).
    """
    tick_hz: float = TICK_HZ

    def __init__(self, seed: Optional[int] = None):
        self.walls = default_walls()
        self.obstacles = default_obstacles()
        self.rng = random.Random(seed) if seed is not None else random.Random()
        s = RobotState()
        # Persistent IMU biases. MPU-9250 datasheet: accel bias ~50 mg
        # (~0.49 m/s^2) typical at room temp; gyro bias ~0.5 deg/s ~= 0.009 rad/s.
        s.accel_bias_x = self.rng.gauss(0, 0.05)
        s.accel_bias_y = self.rng.gauss(0, 0.05)
        s.accel_bias_z = self.rng.gauss(0, 0.05)
        s.gyro_bias_x = self.rng.gauss(0, 0.009)
        s.gyro_bias_y = self.rng.gauss(0, 0.009)
        s.gyro_bias_z = self.rng.gauss(0, 0.009)
        # Motor temps: each motor starts at a slightly different temperature
        # (residual from previous run, mounting differences). Real motors
        # are not exactly at ambient at power-on.
        s.motor_left_temp = MOTOR_AMBIENT_TEMP_C + self.rng.gauss(1.2, 0.6)
        s.motor_right_temp = MOTOR_AMBIENT_TEMP_C + self.rng.gauss(1.2, 0.6)
        # Thermal time constants: ~10% asymmetry between motors (different
        # mounting, different harness routing).
        s.motor_left_tau_s = MOTOR_THERMAL_TIME_CONST_S * (1 + self.rng.gauss(0, 0.07))
        s.motor_right_tau_s = MOTOR_THERMAL_TIME_CONST_S * (1 + self.rng.gauss(0, 0.07))
        # Per-wheel scale factor: ~0.5% manufacturing tolerance per side.
        # Means a robot commanded straight will drift slightly in heading
        # over time, and the encoder tick counts will differ on straight
        # runs - both signatures of real hardware.
        s.wheel_left_scale = 1.0 + self.rng.gauss(0, 0.004)
        s.wheel_right_scale = 1.0 + self.rng.gauss(0, 0.004)
        # Per-motor heating coefficient asymmetry: one motor runs hotter
        # under same load due to brush wear / bearing friction differences.
        s.motor_left_heat_coef = MOTOR_HEAT_COEF * (1 + self.rng.gauss(0, 0.10))
        s.motor_right_heat_coef = MOTOR_HEAT_COEF * (1 + self.rng.gauss(0, 0.10))
        # Gyro scale factor error: ~1% per axis is typical at room temperature
        # for an MPU-9250. The bias is already in s.gyro_bias_*; this is the
        # multiplicative error.
        s.gyro_scale_x = 1.0 + self.rng.gauss(0, 0.012)
        s.gyro_scale_y = 1.0 + self.rng.gauss(0, 0.012)
        s.gyro_scale_z = 1.0 + self.rng.gauss(0, 0.012)
        # Magnetometer hard-iron offsets: local distortion from steel
        # chassis, motor magnets, and battery currents. Real lab readings
        # are NOT clean unit vectors aligned with magnetic north; they
        # show persistent offsets of 0.05-0.2 normalized units per axis.
        s.mag_offset_x = self.rng.gauss(0, 0.10)
        s.mag_offset_y = self.rng.gauss(0, 0.10)
        s.mag_offset_z = self.rng.gauss(0, 0.08)
        # BMS smoothed values. Initialize at full charge.
        s.soc_reported = 1.0
        s.runtime_reported = (BATTERY_CAPACITY_WH / max(0.1, BASE_LOAD_W + 2.0)) * 60
        # CPU starts somewhere in idle range, not exactly at the steady state.
        s.cpu_temp = self.rng.uniform(38.0, 45.0)
        # Ambient temp: room HVAC drifts. Initialize a few tenths off nominal.
        s.ambient_temp = MOTOR_AMBIENT_TEMP_C + self.rng.gauss(0, 0.4)
        self.state = s
        self.tick_n = 0

    def cmd(self, payload: dict) -> None:
        """Apply a velocity command from the agent.

        Expected payload: {"linear_x": float, "angular_z": float}.
        Missing keys default to zero (safe stop). Velocities are clamped
        to MAX_LINEAR_V and MAX_ANGULAR_V; rate-of-change is bounded by
        the acceleration limits in step().
        """
        linear = float(payload.get("linear_x", 0.0))
        angular = float(payload.get("angular_z", 0.0))
        self.state.cmd_v = max(-MAX_LINEAR_V, min(MAX_LINEAR_V, linear))
        self.state.cmd_omega = max(-MAX_ANGULAR_V, min(MAX_ANGULAR_V, angular))

    def step(self) -> None:
        s = self.state
        dt = TICK_DT
        s.boot_elapsed += dt
        if s.boot_elapsed < 3.5:
            s.cmd_v = 0.0
            s.cmd_omega = 0.0
        elif s.boot_elapsed >= 4.5 and not s.calibrated:
            s.calibrated = True

        dv = s.cmd_v - s.v
        s.v += max(-LINEAR_ACCEL_MAX * dt, min(LINEAR_ACCEL_MAX * dt, dv))
        do = s.cmd_omega - s.omega
        s.omega += max(-ANGULAR_ACCEL_MAX * dt, min(ANGULAR_ACCEL_MAX * dt, do))

        if s.docked:
            s.v = 0.0
            s.omega = 0.0
            s.charging = s.battery_charge_wh < BATTERY_CAPACITY_WH
            if s.charging:
                s.battery_charge_wh = min(BATTERY_CAPACITY_WH, s.battery_charge_wh + 6.0 * dt / 3600.0)

        new_x = s.x + s.v * math.cos(s.theta) * dt
        new_y = s.y + s.v * math.sin(s.theta) * dt
        new_theta = s.theta + s.omega * dt

        bl, br = self._collision_check(new_x, new_y, new_theta)
        if bl or br:
            new_x = s.x
            new_y = s.y
            slip = self.rng.gauss(0, 0.02)
            s.v *= max(0.0, 0.5 + slip)
        s.bumper_left = bl
        s.bumper_right = br

        s.x = new_x
        s.y = new_y
        s.theta = (new_theta + math.pi) % (2 * math.pi) - math.pi

        wheel_circ = 2 * math.pi * WHEEL_RADIUS
        left_v = s.v - s.omega * WHEEL_BASE / 2
        right_v = s.v + s.omega * WHEEL_BASE / 2
        l_d = (left_v * s.wheel_left_scale * dt / wheel_circ * ENCODER_TICKS_PER_REV
               + self.rng.gauss(0, ENCODER_SLIP_STD * abs(left_v) * 6))
        r_d = (right_v * s.wheel_right_scale * dt / wheel_circ * ENCODER_TICKS_PER_REV
               + self.rng.gauss(0, ENCODER_SLIP_STD * abs(right_v) * 6))
        # Brush chatter: occasional +/-1 to +/-3 tick glitches. Real brushed
        # DC quadrature encoders show single-tick anomalies under PWM
        # transitions and brush bounce. The previous debrief called the
        # tick stream "monotonic and smooth - too clean for brushed DC."
        if abs(left_v) > 0.01 and self.rng.random() < 0.04:
            l_d += self.rng.choice([-3, -2, -1, 1, 2, 3])
        if abs(right_v) > 0.01 and self.rng.random() < 0.04:
            r_d += self.rng.choice([-3, -2, -1, 1, 2, 3])
        s._left_ticks_frac = getattr(s, "_left_ticks_frac", 0.0) + l_d
        s._right_ticks_frac = getattr(s, "_right_ticks_frac", 0.0) + r_d
        s.left_ticks = int(s._left_ticks_frac)
        s.right_ticks = int(s._right_ticks_frac)

        # Gyro biases random walk on all three axes. Bias on z is the one
        # the agent's previous debrief noted (gyro_z exactly tracking commanded
        # omega). Now offset by the persistent + drifting bias.
        s.gyro_bias_x += self.rng.gauss(0, IMU_GYRO_BIAS_RW * dt)
        s.gyro_bias_y += self.rng.gauss(0, IMU_GYRO_BIAS_RW * dt)
        s.gyro_bias_z += self.rng.gauss(0, IMU_GYRO_BIAS_RW * dt)
        # Accel biases drift very slowly with chassis temperature.
        bias_drift = self.rng.gauss(0, 0.0001 * dt)
        s.accel_bias_x += bias_drift
        s.accel_bias_y += bias_drift * 0.7

        ll = abs(left_v) * MOTOR_POWER_W_PER_MPS / 2
        rl = abs(right_v) * MOTOR_POWER_W_PER_MPS / 2
        # Per-motor heating coefficient asymmetry. One motor often has more
        # friction (worn brushes, tighter bearing), so it heats faster at
        # the same load. Previous debrief flagged "both motors nearly
        # identical" - this introduces a deliberate asymmetry baked in
        # at boot.
        s.motor_left_temp += (
            ll * s.motor_left_heat_coef * dt
            - (s.motor_left_temp - s.ambient_temp) / s.motor_left_tau_s * dt
            + self.rng.gauss(0, 0.04)
        )
        s.motor_right_temp += (
            rl * s.motor_right_heat_coef * dt
            - (s.motor_right_temp - s.ambient_temp) / s.motor_right_tau_s * dt
            + self.rng.gauss(0, 0.04)
        )
        # CPU temp wanders around an idle steady state.
        s.cpu_temp += self.rng.gauss(0, 0.08) - (s.cpu_temp - 42.0) / 30.0 * dt
        # Ambient drifts with HVAC cycling. Faster cycle (~25s period) so
        # the variation is visible within a 30-second observation window.
        # Real lab HVAC cycling can be quicker than minute-scale because
        # of small thermostat hysteresis bands.
        hvac_phase = math.sin(s.boot_elapsed * 0.25) * 0.6
        s.ambient_temp = (
            MOTOR_AMBIENT_TEMP_C + hvac_phase
            + (s.ambient_temp - MOTOR_AMBIENT_TEMP_C - hvac_phase) * 0.998
            + self.rng.gauss(0, 0.04)
        )
        s.ambient_temp = max(MOTOR_AMBIENT_TEMP_C - 1.8,
                              min(MOTOR_AMBIENT_TEMP_C + 1.8, s.ambient_temp))

        if not s.docked:
            total_w = BASE_LOAD_W + ll + rl
            s.battery_charge_wh -= total_w * dt / 3600.0
            s.battery_charge_wh = max(0.0, s.battery_charge_wh)
            soc = s.battery_charge_wh / BATTERY_CAPACITY_WH
            base_v = BATTERY_EMPTY_V + (BATTERY_FULL_V - BATTERY_EMPTY_V) * soc
            sag = total_w * 0.08
            # ADC noise (12-bit ADC, ~3mV LSB plus measurement noise)
            adc_noise = self.rng.gauss(0, 0.025)
            s.battery_v = max(BATTERY_EMPTY_V - 0.3, base_v - sag + adc_noise)

        d_dock = math.hypot(s.x - DOCK_X, s.y - DOCK_Y)
        if d_dock < DOCK_TOLERANCE and not s.docked:
            heading_err = abs(((s.theta - DOCK_THETA + math.pi) % (2 * math.pi)) - math.pi)
            if heading_err < math.radians(20):
                s.docked = True
                s.charging = True
        elif d_dock > DOCK_TOLERANCE * 2 and s.docked:
            s.docked = False
            s.charging = False

        self.tick_n += 1

    def _collision_check(self, x, y, theta) -> tuple[bool, bool]:
        for w in self.walls:
            dx = w.x1 - w.x0
            dy = w.y1 - w.y0
            seg_sq = dx * dx + dy * dy
            if seg_sq < 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - w.x0) * dx + (y - w.y0) * dy) / seg_sq))
            cx = w.x0 + t * dx
            cy = w.y0 + t * dy
            d = math.hypot(x - cx, y - cy)
            if d < ROBOT_RADIUS:
                lx = (cx - x) * math.cos(-theta) - (cy - y) * math.sin(-theta)
                ly = (cx - x) * math.sin(-theta) + (cy - y) * math.cos(-theta)
                if lx > -0.02:
                    return (ly > 0, ly <= 0)
                return (False, False)
        for o in self.obstacles:
            d = math.hypot(x - o.cx, y - o.cy)
            if d < ROBOT_RADIUS + o.r:
                lx = (o.cx - x) * math.cos(-theta) - (o.cy - y) * math.sin(-theta)
                ly = (o.cx - x) * math.sin(-theta) + (o.cy - y) * math.cos(-theta)
                if lx > -0.02:
                    return (ly > 0, ly <= 0)
                return (False, False)
        return (False, False)

    def lidar(self) -> list[float]:
        """LIDAR scan with range-dependent noise, mm quantization, and
        correlated dropouts. Real RPLIDAR beam failures cluster (a beam
        that returned NaN once is more likely to glitch on the next scan
        too, due to reflective surfaces or local interference). The
        previous debrief noted that a one-shot NaN reads like scripted
        anomaly injection.
        """
        s = self.state
        previously_glitched = set(s.lidar_glitched_beams)
        glitched_now = set()
        out = []
        for i in range(LIDAR_BEAMS):
            angle = s.theta + (i / LIDAR_BEAMS) * 2 * math.pi
            d_true = cast_ray(s.x, s.y, angle, self.walls, self.obstacles)
            # Correlated dropout: if this beam glitched last time, raise
            # the dropout probability significantly for this scan.
            base_p = LIDAR_DROPOUT_PROB
            if i in previously_glitched:
                base_p = 0.45  # strong correlation
            if self.rng.random() < base_p:
                out.append(None)
                glitched_now.add(i)
                continue
            if self.rng.random() < LIDAR_OUTLIER_PROB:
                d_true = max(LIDAR_MIN_RANGE, d_true * (0.5 + self.rng.random()))
            sigma = LIDAR_NOISE_STD * (1.0 + 1.2 * (d_true / LIDAR_MAX_RANGE) ** 1.5)
            d = d_true + self.rng.gauss(0, sigma)
            d = max(LIDAR_MIN_RANGE, min(LIDAR_MAX_RANGE, d))
            d = round(d * 1000) / 1000
            out.append(d)
        s.lidar_glitched_beams = tuple(glitched_now)
        return out

    def imu(self) -> dict:
        """Synthesize MPU-9250 readings.

        Real MPU-9250 has persistent per-axis biases (~50 mg accel,
        ~0.5 deg/s gyro), 1/f-ish low-frequency noise, and sensor-to-sensor
        variance. The previous debrief flagged the readings as too clean
        and the gyro_z too perfectly tracking commanded omega; biases are
        now applied per-axis and noise is widened.
        """
        s = self.state
        # 1/f-like noise: combine wide and narrow noise components
        def imu_noise(sigma_w: float, sigma_n: float) -> float:
            return self.rng.gauss(0, sigma_w) + self.rng.gauss(0, sigma_n)
        # Linear accel includes the chassis vibration in xy when motors run
        vib_amp = abs(s.v) * 0.08 + abs(s.omega) * 0.04
        ax = (s.accel_bias_x
              + imu_noise(IMU_ACCEL_NOISE_STD * 1.5, IMU_ACCEL_NOISE_STD * 0.4)
              + self.rng.gauss(0, vib_amp))
        ay = (s.accel_bias_y
              + imu_noise(IMU_ACCEL_NOISE_STD * 1.5, IMU_ACCEL_NOISE_STD * 0.4)
              + self.rng.gauss(0, vib_amp))
        az = (-9.81 + s.accel_bias_z
              + imu_noise(IMU_ACCEL_NOISE_STD * 1.5, IMU_ACCEL_NOISE_STD * 0.4)
              + self.rng.gauss(0, vib_amp * 0.3))
        # Apply gyro scale factor errors. Real MEMS gyros report scaled +
        # biased + noisy versions of the true rate.
        gx = s.gyro_bias_x + imu_noise(IMU_GYRO_NOISE_STD * 1.4, IMU_GYRO_NOISE_STD * 0.4)
        gy = s.gyro_bias_y + imu_noise(IMU_GYRO_NOISE_STD * 1.4, IMU_GYRO_NOISE_STD * 0.4)
        gz = (s.omega * s.gyro_scale_z + s.gyro_bias_z
              + imu_noise(IMU_GYRO_NOISE_STD * 1.4, IMU_GYRO_NOISE_STD * 0.4))
        # Magnetometer with persistent hard-iron offsets + soft-iron scale.
        # Real lab readings show distortion from local ferrous metal and DC
        # currents - the agent's previous debrief flagged the readings as
        # "near-perfect unit vector aligned with X axis" when the robot was
        # at theta=0; that's only true in a magnetic vacuum. Real readings
        # have 0.05-0.2 normalized offset per axis.
        mx = math.cos(-s.theta) + s.mag_offset_x + self.rng.gauss(0, 0.025)
        my = math.sin(-s.theta) + s.mag_offset_y + self.rng.gauss(0, 0.025)
        mz = 0.04 + s.mag_offset_z + self.rng.gauss(0, 0.025)
        return {
            "linear_accel": [round(ax, 3), round(ay, 3), round(az, 3)],
            "angular_vel": [round(gx, 4), round(gy, 4), round(gz, 4)],
            "mag_norm": [round(mx, 3), round(my, 3), round(mz, 3)],
        }

    def dock_signal(self) -> dict:
        """Wireless charging dock IR beacon: analog received-signal-strength
        plus a binary detect flag. Real beacons return RSSI in dBm; we
        synthesize an inverse-square-with-occlusion model."""
        s = self.state
        dx = DOCK_X - s.x
        dy = DOCK_Y - s.y
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        ray_d = cast_ray(s.x, s.y, ang, self.walls, self.obstacles, max_range=d + 0.1)
        occluded = ray_d < d - 0.05
        if occluded:
            rssi = -100.0 + self.rng.gauss(0, 2.5)
            visible = False
        else:
            base_rssi = -45.0 - 20.0 * math.log10(max(0.05, d))
            bearing_from_dock = math.atan2(s.y - DOCK_Y, s.x - DOCK_X)
            cone_err = abs(((bearing_from_dock - 0 + math.pi) % (2 * math.pi)) - math.pi)
            cone_loss = -8.0 * (cone_err / math.pi) ** 2
            # Indoor multipath: 2.4 GHz reflections off walls / metal cause
            # 5-10 dB swings between samples. Use a wider noise distribution
            # plus an occasional deep null. Previous debrief noted the
            # signal "tracked too cleanly" without multipath spikes.
            multipath_noise = self.rng.gauss(0, 4.5)
            if self.rng.random() < 0.07:
                multipath_noise -= self.rng.uniform(4.0, 12.0)  # deep fade
            rssi = base_rssi + cone_loss + multipath_noise
            visible = rssi > -85.0
        return {
            "rssi_dbm": round(rssi, 1),
            "visible": visible,
            "docked": s.docked,
            "approx_range_m": round(d, 2) if visible else None,
        }

    def snapshot(self) -> dict:
        s = self.state
        # BMS smoothed SoC + runtime. Real fuel gauge ICs apply heavy EWMA
        # filtering with a hard physical constraint: under discharge SoC
        # cannot recover above the true value (no charge is being added).
        # Previous debrief: agent saw 99.98 -> 100.0 reversal, classic
        # clamp-with-noise tell.
        soc_true = s.battery_charge_wh / BATTERY_CAPACITY_WH
        s.soc_reported = (s.soc_reported * 0.85
                          + soc_true * 0.15
                          + self.rng.gauss(0, 0.0008))
        # Hard physical: when not charging, reported cannot exceed truth.
        if not s.charging:
            s.soc_reported = min(s.soc_reported, soc_true + 0.0005)
        s.soc_reported = max(0.0, min(1.0, s.soc_reported))
        runtime_inst = (s.battery_charge_wh / max(0.01, BASE_LOAD_W + 2.0)) * 60
        s.runtime_reported = (s.runtime_reported * 0.92
                              + runtime_inst * 0.08
                              + self.rng.gauss(0, 0.25))
        runtime_min = max(0.0, s.runtime_reported)
        soc_reported = s.soc_reported
        lid = self.lidar()
        return {
            "boot_elapsed_s": round(s.boot_elapsed, 2),
            "calibrated": s.calibrated,
            "pose": {"x": round(s.x, 3), "y": round(s.y, 3), "theta_rad": round(s.theta, 3)},
            "vel": {"linear_x": round(s.v, 3), "angular_z": round(s.omega, 3)},
            "lidar_m": lid,
            "imu": self.imu(),
            "encoders": {"left_ticks": s.left_ticks, "right_ticks": s.right_ticks},
            "battery": {
                "voltage_v": round(s.battery_v, 2),
                "charge_pct": round(soc_reported * 100, 2),
                "runtime_min_est": round(max(0, runtime_min), 1),
                "charging": s.charging,
            },
            "thermal_c": {
                "motor_left": round(s.motor_left_temp + self.rng.gauss(0, 0.08), 2),
                "motor_right": round(s.motor_right_temp + self.rng.gauss(0, 0.08), 2),
                "cpu": round(s.cpu_temp + self.rng.gauss(0, 0.15), 2),
                "ambient": round(s.ambient_temp + self.rng.gauss(0, 0.05), 2),
            },
            "bumpers": {"front_left": s.bumper_left, "front_right": s.bumper_right},
            "dock": self.dock_signal(),
        }

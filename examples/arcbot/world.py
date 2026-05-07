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
MOTOR_THERMAL_TIME_CONST_S = 22.0
# Tuned to 0.28 after a series of bracketing iterations. 0.18 produced
# "implausibly stable for brushed DC under load." 0.45 produced "23 to
# 30 C in 28 s, faster than expected for this thermal mass." 0.28 lands
# in the realistic envelope: ~3-4 C rise visible within 30 s of mixed
# light-load operation, full ~7 C steady-state delta reached over the
# motor's ~22 s thermal time constant.
MOTOR_HEAT_COEF = 0.28
# Idle friction: real DC motors dissipate ~0.5-1W just spinning the
# bearings even at no commanded load. This adds a baseline thermal floor
# above ambient when the robot is awake.
MOTOR_IDLE_W = 0.6
MOTOR_AMBIENT_TEMP_C = 22.5

LIDAR_NOISE_STD = 0.012
# Real RPLIDAR A1 in a clean indoor scene returns ~1-3% invalid beams per
# scan from baseline noise; the rest of the dropouts come from systematic
# causes (the persistent weak beam, range-dependent SNR, specular hits on
# circular obstacles). Previous debrief: a 6% baseline produced "rotating
# NaN scatter across different beams each frame" that read as random
# injection. Lowered to 1.5% so systematic dropouts dominate.
LIDAR_DROPOUT_PROB = 0.015
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
    # One persistently-degraded beam index. Real RPLIDAR units always
    # have at least one marginal pixel (factory tolerance), set at boot.
    lidar_weak_beam: int = -1
    # A second weakly marginal beam appears in ~30% of units.
    lidar_weak_beam2: int = -1
    # Last reported RSSI for low-pass smoothing across snapshots.
    rssi_last: float = -65.0
    # Last pose for finite-differenced velocity reporting.
    last_reported_x: float = 0.0
    last_reported_y: float = 0.0
    last_reported_theta: float = 0.0
    last_report_time: float = 0.0
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
    # Soft-iron distortion: 2x2 transform on the horizontal plane.
    mag_soft_xx: float = 1.0
    mag_soft_yy: float = 1.0
    mag_soft_xy: float = 0.0
    cpu_temp: float = 41.5
    ambient_temp: float = MOTOR_AMBIENT_TEMP_C
    bumper_left: bool = False
    bumper_right: bool = False
    boot_elapsed: float = 0.0
    calibrated: bool = False
    docked: bool = False
    charging: bool = False
    # --- Tick-driven odometry. Integrated independently from the noisy
    # encoder tick deltas the firmware actually observes, NOT from the true
    # physics state x/y/theta (which would be ground-truth leakage). This
    # is the only published pose; it drifts from true as wheel slip and
    # wheel-scale calibration error accumulate, exactly like a real
    # differential-drive robot. Generalizes to any arena geometry.
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_theta: float = 0.0
    odom_v_filtered: float = 0.0
    odom_omega_filtered: float = 0.0
    # --- Shared physical substrate. Real causal couplings between sensors:
    # supply_rail_v feeds every ADC-derived measurement, vibration_intensity
    # drives IMU noise AND encoder brush chatter AND mag distortion,
    # chassis_drift_c slowly biases sensor calibrations together,
    # motor_inrush_w spikes during command-change events and visibly
    # couples to voltage AND vibration on the same tick.
    supply_rail_v: float = BATTERY_FULL_V
    vibration_intensity: float = 0.04
    chassis_drift_c: float = 0.0
    motor_inrush_w: float = 0.0
    cmd_v_prev: float = 0.0
    cmd_omega_prev: float = 0.0


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
        # typical (~0.49 m/s^2), up to 80 mg max at room temp. Gyro bias
        # ~0.5 deg/s ~= 0.009 rad/s. The previous 0.05 m/s^2 sigma was 10x
        # too small (5 mg), which made boot-time accel readings look
        # "suspiciously small and clean" per debrief. The real chip
        # accumulates bias from solder strain, package warpage, and
        # supply rail noise; 0.5 m/s^2 lands in the datasheet typical.
        s.accel_bias_x = self.rng.gauss(0, 0.5)
        s.accel_bias_y = self.rng.gauss(0, 0.5)
        s.accel_bias_z = self.rng.gauss(0, 0.5)
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
        # Per-wheel scale factor: ~1.2% manufacturing tolerance per side.
        # Real differential-drive robots drift several degrees per meter of
        # straight-line travel from wheel diameter mismatch. The previous
        # 0.4% sigma let the cumulative tick counts re-balance over a
        # series of turns, which the agent flagged as "physics engine
        # conserves total path length too cleanly." Bumping to 1.2%
        # produces persistent visible asymmetry on straight runs.
        s.wheel_left_scale = 1.0 + self.rng.gauss(0, 0.012)
        s.wheel_right_scale = 1.0 + self.rng.gauss(0, 0.012)
        # Per-motor heating coefficient asymmetry: one motor runs hotter
        # under same load due to brush wear / bearing friction differences.
        # Bumped from 10% to 22% sigma because the previous debrief noted
        # motor_l and motor_r tracking within 0.5C through asymmetric
        # maneuvers, which is unphysical for real brushed DC pairs.
        s.motor_left_heat_coef = MOTOR_HEAT_COEF * (1 + self.rng.gauss(0, 0.22))
        s.motor_right_heat_coef = MOTOR_HEAT_COEF * (1 + self.rng.gauss(0, 0.22))
        # Gyro scale factor error: ~1% per axis is typical at room temperature
        # for an MPU-9250. The bias is already in s.gyro_bias_*; this is the
        # multiplicative error.
        s.gyro_scale_x = 1.0 + self.rng.gauss(0, 0.012)
        s.gyro_scale_y = 1.0 + self.rng.gauss(0, 0.012)
        s.gyro_scale_z = 1.0 + self.rng.gauss(0, 0.012)
        # Magnetometer hard-iron offsets: local distortion from steel
        # chassis, motor magnets, and battery currents. Real lab readings
        # are NOT clean unit vectors aligned with magnetic north; they
        # show persistent offsets of 1-3 microtesla per axis. Previous
        # debrief: "X readings consistent with heading change but feel
        # clean" — the offsets at *8 multiplier produced ~0.8 uT shifts
        # which were too small relative to the 25 uT signal.
        s.mag_offset_x = self.rng.gauss(0, 0.18)
        s.mag_offset_y = self.rng.gauss(0, 0.18)
        s.mag_offset_z = self.rng.gauss(0, 0.14)
        # Soft-iron distortion: a 2x2 transform on (mx, my) representing
        # magnetic permeability anisotropy in the chassis material. Real
        # values are typically 0.85-1.15 diagonal, ±0.05 off-diagonal.
        s.mag_soft_xx = 1.0 + self.rng.gauss(0, 0.06)
        s.mag_soft_yy = 1.0 + self.rng.gauss(0, 0.06)
        s.mag_soft_xy = self.rng.gauss(0, 0.04)
        # BMS reported SoC starts at the actual battery state. Initializing
        # below truth (previously 1.0 - |N(0, 0.004)|) caused the EWMA to
        # ratchet upward toward truth during early discharge, producing
        # the +0.11% mid-drive increase the agent flagged. Real fuel
        # gauge ICs initialize from a one-shot OCV measurement at boot
        # which lands on the true SoC within calibration tolerance.
        s.soc_reported = 1.0 - abs(self.rng.gauss(0, 0.0008))
        s.runtime_reported = (BATTERY_CAPACITY_WH / max(0.1, BASE_LOAD_W + 2.0)) * 60
        # Persistent weak LIDAR beam: every unit ships with at least one
        # marginal pixel. Previous debrief flagged the rotating-NaN pattern
        # as artifact because the persistent weak beam was only present 50%
        # of the time, so most runs had no sticky failure.
        s.lidar_weak_beam = self.rng.randrange(LIDAR_BEAMS)
        # A second weakly marginal beam appears in ~30% of units. Its
        # dropout rate is in between the strong-weak and baseline.
        if self.rng.random() < 0.30:
            s.lidar_weak_beam2 = self.rng.randrange(LIDAR_BEAMS)
            while s.lidar_weak_beam2 == s.lidar_weak_beam:
                s.lidar_weak_beam2 = self.rng.randrange(LIDAR_BEAMS)
        else:
            s.lidar_weak_beam2 = -1
        s.last_reported_x = s.x
        s.last_reported_y = s.y
        s.last_reported_theta = s.theta
        # Odometry initialised at the same pose as truth at boot; from then
        # on it integrates from noisy ticks and drifts.
        s.odom_x = s.x
        s.odom_y = s.y
        s.odom_theta = s.theta
        # Initial supply rail at slight load (radio + microcontroller idle).
        s.supply_rail_v = BATTERY_FULL_V * 0.985
        # Persistent thermal offset from initial chassis warmup state.
        s.chassis_drift_c = self.rng.gauss(0, 0.6)
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

        # --- Shared physical substrate update. Real causal couplings:
        # command-change inrush spikes drive supply-rail sag AND vibration
        # AND extra motor heat ON THE SAME TICK. The previous independent-
        # noise model was the deepest tell the agent kept surfacing.
        cmd_change = (abs(s.cmd_v - s.cmd_v_prev)
                      + abs(s.cmd_omega - s.cmd_omega_prev) * 0.08)
        if cmd_change > 0.02:
            s.motor_inrush_w = max(s.motor_inrush_w,
                                    7.5 * cmd_change / 0.05)
        s.motor_inrush_w *= 0.82
        s.cmd_v_prev = s.cmd_v
        s.cmd_omega_prev = s.cmd_omega

        # Vibration intensity drives IMU noise, encoder brush chatter,
        # and motor magnetic field jitter. Single source, multiple sinks.
        s.vibration_intensity = (0.04
                                  + (abs(left_v) + abs(right_v)) * 0.22
                                  + abs(s.omega) * 0.15
                                  + s.motor_inrush_w * 0.018)

        # Chassis thermal drift: slow Brownian, biases all sensor cal
        # together. Pulls weakly back to zero (HVAC equilibration).
        s.chassis_drift_c += (self.rng.gauss(0, 0.008)
                               - s.chassis_drift_c * 0.0008)

        # Encoder noise scales with vibration: brushes bounce more under
        # higher chassis vibration, producing more chatter and slip. This
        # is the cross-field link the agent flagged the absence of.
        vib_factor = 1.0 + s.vibration_intensity * 1.5
        slip_amp_left = ENCODER_SLIP_STD * abs(left_v) * 6 * vib_factor
        slip_amp_right = ENCODER_SLIP_STD * abs(right_v) * 6 * vib_factor
        l_d = (left_v * s.wheel_left_scale * dt / wheel_circ * ENCODER_TICKS_PER_REV
               + self.rng.gauss(0, slip_amp_left))
        r_d = (right_v * s.wheel_right_scale * dt / wheel_circ * ENCODER_TICKS_PER_REV
               + self.rng.gauss(0, slip_amp_right))
        chatter_p = 0.04 * (1 + s.vibration_intensity * 4)
        if abs(left_v) > 0.01 and self.rng.random() < chatter_p:
            l_d += self.rng.choice([-3, -2, -1, 1, 2, 3])
        if abs(right_v) > 0.01 and self.rng.random() < chatter_p:
            r_d += self.rng.choice([-3, -2, -1, 1, 2, 3])
        s._left_ticks_frac = getattr(s, "_left_ticks_frac", 0.0) + l_d
        s._right_ticks_frac = getattr(s, "_right_ticks_frac", 0.0) + r_d
        s.left_ticks = int(s._left_ticks_frac)
        s.right_ticks = int(s._right_ticks_frac)

        # --- Pose-from-ticks integration. The published pose (s.odom_x,
        # s.odom_y, s.odom_theta) is integrated independently from the
        # noisy tick deltas the firmware actually observes. No path back
        # to true x/y/theta — that would be ground-truth leakage. Drift
        # accumulates from per-wheel scale-factor mismatch, slip noise,
        # and brush chatter. Generalises to any arena (no assumptions
        # about room geometry, walls, or maze layout).
        if dt > 0:
            left_v_meas = l_d * wheel_circ / (ENCODER_TICKS_PER_REV * dt)
            right_v_meas = r_d * wheel_circ / (ENCODER_TICKS_PER_REV * dt)
        else:
            left_v_meas = left_v
            right_v_meas = right_v
        v_odom = (left_v_meas + right_v_meas) / 2
        omega_odom = (right_v_meas - left_v_meas) / WHEEL_BASE
        # ROS robot_localization-style fused publication: the raw per-tick
        # finite difference is low-passed before being put on the topic.
        s.odom_v_filtered = s.odom_v_filtered * 0.6 + v_odom * 0.4
        s.odom_omega_filtered = s.odom_omega_filtered * 0.6 + omega_odom * 0.4
        s.odom_x += v_odom * math.cos(s.odom_theta) * dt
        s.odom_y += v_odom * math.sin(s.odom_theta) * dt
        s.odom_theta = (s.odom_theta + omega_odom * dt + math.pi) % (2 * math.pi) - math.pi

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

        # Total motor dissipation = idle friction + commanded power loss
        # + stall current term. Idle keeps motors warmer than ambient
        # even when stationary. Stall current dominates when the motor is
        # commanded but barely moving (against a wall, on rough surface,
        # bumper-stuck): real brushed DC stall current is 4-8x the no-load
        # current and dissipates as I^2*R heat. Previous debrief: "no
        # differential heating during near-stall" — the agent expected
        # the harder-working side to heat faster during obstacle contact.
        cmd_left_v = s.cmd_v - s.cmd_omega * WHEEL_BASE / 2
        cmd_right_v = s.cmd_v + s.cmd_omega * WHEEL_BASE / 2
        stall_left = max(0.0, abs(cmd_left_v) - abs(left_v)) * MOTOR_POWER_W_PER_MPS * 3.0
        stall_right = max(0.0, abs(cmd_right_v) - abs(right_v)) * MOTOR_POWER_W_PER_MPS * 3.0
        ll = MOTOR_IDLE_W + abs(left_v) * MOTOR_POWER_W_PER_MPS / 2 + stall_left
        rl = MOTOR_IDLE_W + abs(right_v) * MOTOR_POWER_W_PER_MPS / 2 + stall_right
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
        # Ambient drift from HVAC cycling. Previous debrief flagged the
        # earlier setting (amp=0.6, period=25s) as oscillating 1.6C in 29s
        # which is unphysical for a closed lab. Use modest amplitude and
        # longer period: ~0.25C peak-to-peak over a ~1-min cycle, plus
        # fine-grained measurement noise.
        hvac_phase = math.sin(s.boot_elapsed * 0.10) * 0.15
        s.ambient_temp = (
            MOTOR_AMBIENT_TEMP_C + hvac_phase
            + (s.ambient_temp - MOTOR_AMBIENT_TEMP_C - hvac_phase) * 0.999
            + self.rng.gauss(0, 0.015)
        )
        s.ambient_temp = max(MOTOR_AMBIENT_TEMP_C - 0.6,
                              min(MOTOR_AMBIENT_TEMP_C + 0.6, s.ambient_temp))

        if not s.docked:
            # Inrush counts toward both energy drain AND voltage sag, so
            # command-change events visibly couple to fuel-gauge AND
            # battery_v on the same tick (cross-field consistency).
            total_w = BASE_LOAD_W + ll + rl + s.motor_inrush_w
            s.battery_charge_wh -= total_w * dt / 3600.0
            s.battery_charge_wh = max(0.0, s.battery_charge_wh)
            soc = s.battery_charge_wh / BATTERY_CAPACITY_WH
            base_v = BATTERY_EMPTY_V + (BATTERY_FULL_V - BATTERY_EMPTY_V) * soc
            sag = total_w * 0.08
            # Supply rail at ADCs is the loaded battery voltage. Stored as
            # shared state so other sensors using ADC reads see the same
            # rail dip during inrush spikes.
            s.supply_rail_v = base_v - sag
            # 12-bit ADC reading the battery: noise envelope grows when
            # the rail dips because the ADC reference compresses.
            adc_factor = 1.0 + max(0.0, 12.5 - s.supply_rail_v) * 0.04
            adc_noise = self.rng.gauss(0, 0.022 * adc_factor)
            s.battery_v = max(BATTERY_EMPTY_V - 0.3,
                               s.supply_rail_v + adc_noise)

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
        # Identify which beams hit circular obstacles (specular returns from
        # curved surfaces are a real RPLIDAR failure mode). Pre-compute so
        # adjacent beams hitting the same obstacle drop out together.
        obstacle_hit_beams = set()
        for i in range(LIDAR_BEAMS):
            angle = s.theta + (i / LIDAR_BEAMS) * 2 * math.pi
            for o in self.obstacles:
                if _ray_circle(s.x, s.y, math.cos(angle), math.sin(angle),
                                o.cx, o.cy, o.r) is not None:
                    obstacle_hit_beams.add(i)
                    break
        for i in range(LIDAR_BEAMS):
            angle = s.theta + (i / LIDAR_BEAMS) * 2 * math.pi
            d_true = cast_ray(s.x, s.y, angle, self.walls, self.obstacles)
            # Geometrically-grounded dropout. Real RPLIDAR A1 NaNs cluster
            # around: (a) the persistent weak beam from factory, (b) long
            # ranges where SNR is poor, (c) very short ranges where the
            # receiver saturates, (d) specular returns from curved or
            # smooth-oblique surfaces, (e) transient correlation with
            # previously-glitched beams. Previous debrief flagged the
            # uniform random scattering of NaNs across beams as artifact.
            # Weak beams DON'T always go NaN. Real RPLIDAR marginal pixels
            # often return a noisy valid range with elevated variance, plus
            # occasional NaNs. Previous debrief: "beam 60 NaN every single
            # snapshot reads more like a masked channel than a failing
            # emitter." Now: the strong-weak beam returns NaN ~40% and
            # noisy-valid (3x normal sigma) the rest of the time.
            base_p = LIDAR_DROPOUT_PROB
            extra_sigma = 0.0
            if i == s.lidar_weak_beam:
                base_p = 0.40
                extra_sigma = LIDAR_NOISE_STD * 4.0
            elif i == s.lidar_weak_beam2:
                base_p = 0.18
                extra_sigma = LIDAR_NOISE_STD * 2.0
            elif i in previously_glitched:
                base_p = 0.16
            # Range-dependent: long beams drop out more often (poor SNR)
            if d_true > 3.0:
                base_p += 0.06 * ((d_true - 3.0) / 1.0)
            # Short-range receiver saturation
            elif d_true < 0.15:
                base_p += 0.10
            # Curved-surface specular: hits on circular obstacles drop more
            if i in obstacle_hit_beams:
                base_p += 0.05
            if self.rng.random() < base_p:
                out.append(None)
                glitched_now.add(i)
                continue
            if self.rng.random() < LIDAR_OUTLIER_PROB:
                d_true = max(LIDAR_MIN_RANGE, d_true * (0.5 + self.rng.random()))
            sigma = LIDAR_NOISE_STD * (1.0 + 1.2 * (d_true / LIDAR_MAX_RANGE) ** 1.5) + extra_sigma
            d = d_true + self.rng.gauss(0, sigma)
            # Real RPLIDAR A1 returns no point (NaN) when beyond reliable
            # range, not a clamped max-range value. Previous debrief: agent
            # flagged "multiple beams pegged at exactly 4.0m" as a max-range
            # sentinel artifact. Convert to None so out-of-range beams look
            # like normal sensor dropouts.
            if d_true >= LIDAR_MAX_RANGE * 0.985 or d >= LIDAR_MAX_RANGE * 0.99:
                out.append(None)
                glitched_now.add(i)
                continue
            d = max(LIDAR_MIN_RANGE, d)
            d = round(d * 1000) / 1000
            out.append(d)
        s.lidar_glitched_beams = tuple(glitched_now)
        return out

    def imu(self) -> dict:
        """Synthesize MPU-9250 readings.

        Real MPU-9250 has persistent per-axis biases (~50 mg accel,
        ~0.5 deg/s gyro), 1/f-ish low-frequency noise, and chassis
        vibration coupling from brushed DC motors. Previous debriefs noted
        the readings were too clean even with biases applied; vibration
        amplitude is now ~3x stronger so motor activity produces visible
        accelerometer noise at the snapshot rate.
        """
        s = self.state
        def imu_noise(sigma_w: float, sigma_n: float) -> float:
            return self.rng.gauss(0, sigma_w) + self.rng.gauss(0, sigma_n)
        # Vibration coupling. Brushed DC motors at PWM produce 0.05-0.3 m/s^2
        # chassis acceleration noise on a 200mm robot. Centripetal coupling
        # adds tangential acceleration during turns: a_centripetal = v * omega
        # in the body frame. Previous debrief flagged "X and Y accel values
        # tiny throughout — often under 0.1 m/s^2 during active turning"
        # as the tell that horizontal accel wasn't tracking turn dynamics.
        # Vibration drawn from shared chassis state. Same source as
        # encoder brush chatter and mag motor distortion, so an inrush
        # tick produces correlated noise across IMU, encoders, and mag.
        vib_amp = s.vibration_intensity
        # IMU mounted ~50 mm forward of the rotation center on this chassis.
        # Body-frame Y accel = centripetal (v*omega) + omega^2 * imu_offset.
        # Body-frame X accel = tangential dv/dt + omega_dot * imu_offset.
        # Previous debrief: "XY accel during active turning was tiny" —
        # the centripetal coefficient at 0.10 m/s * 0.8 rad/s gave 0.08 m/s^2
        # which was hidden in vib noise. Adding the lever-arm term doubles
        # the lateral signal during pivots and produces a measurable
        # tangential signal during velocity transitions.
        IMU_OFFSET_M = 0.05
        a_centripetal = s.v * s.omega + s.omega * s.omega * IMU_OFFSET_M
        # Tangential from omega_dot and v_dot, finite-differenced
        last_omega = getattr(s, "imu_last_omega", s.omega)
        last_v = getattr(s, "imu_last_v", s.v)
        omega_dot = (s.omega - last_omega) / TICK_DT
        v_dot = (s.v - last_v) / TICK_DT
        a_tangential = v_dot + omega_dot * IMU_OFFSET_M
        s.imu_last_omega = s.omega
        s.imu_last_v = s.v
        ax = (s.accel_bias_x + a_tangential
              + imu_noise(IMU_ACCEL_NOISE_STD * 1.6, IMU_ACCEL_NOISE_STD * 0.5)
              + self.rng.gauss(0, vib_amp))
        ay = (s.accel_bias_y + a_centripetal
              + imu_noise(IMU_ACCEL_NOISE_STD * 1.6, IMU_ACCEL_NOISE_STD * 0.5)
              + self.rng.gauss(0, vib_amp))
        az = (-9.81 + s.accel_bias_z
              + imu_noise(IMU_ACCEL_NOISE_STD * 1.8, IMU_ACCEL_NOISE_STD * 0.5)
              + self.rng.gauss(0, vib_amp * 1.0))
        # Real MPU-9250 internal DLPF runs at the sensor sampling rate
        # (1 kHz typical) and is fully settled by the time the host
        # samples at 50-100 Hz. The previous per-snapshot EWMA produced
        # multi-snapshot lag artifacts where gyro_z reported 0.5 rad/s
        # when commanded was 1.2 rad/s. Removed; what stays is the
        # temperature-dependent scale drift that real MEMS chips show.
        temp_scale = 1.0 + (s.cpu_temp - 42.0) * 0.0008
        gx = s.gyro_bias_x + imu_noise(IMU_GYRO_NOISE_STD * 1.4, IMU_GYRO_NOISE_STD * 0.4)
        gy = s.gyro_bias_y + imu_noise(IMU_GYRO_NOISE_STD * 1.4, IMU_GYRO_NOISE_STD * 0.4)
        gz = (s.omega * s.gyro_scale_z * temp_scale + s.gyro_bias_z
              + imu_noise(IMU_GYRO_NOISE_STD * 1.4, IMU_GYRO_NOISE_STD * 0.4))
        # Magnetometer: report raw field components in microtesla. Earth's
        # field is ~50 uT total at this latitude; horizontal component
        # ~25 uT. Apply soft-iron 2x2 transform on (mx, my) for chassis
        # permeability anisotropy, then add hard-iron offsets and motor-
        # current dependent distortion (motors generate local DC fields
        # of 1-3 uT under load that swing with PWM duty). Previous debrief
        # called the readings "messier than clean unit vectors but still
        # too clean for a lab with motor controller and battery currents."
        h_field_uT = 25.0
        v_field_uT = -42.0
        true_mx = h_field_uT * math.cos(-s.theta)
        true_my = h_field_uT * math.sin(-s.theta)
        # Soft-iron transformation
        soft_mx = s.mag_soft_xx * true_mx + s.mag_soft_xy * true_my
        soft_my = s.mag_soft_xy * true_mx + s.mag_soft_yy * true_my
        # Motor magnetic field interference scales with commanded current
        motor_mag_x = (abs(s.v) + abs(s.omega) * 0.08) * 1.5 * math.cos(2 * s.boot_elapsed)
        motor_mag_y = (abs(s.v) + abs(s.omega) * 0.08) * 1.5 * math.sin(2 * s.boot_elapsed)
        # Hard-iron offsets drift slowly with chassis temperature: real
        # magnetometers see calibration shift of ~0.3 uT/C. The shared
        # chassis_drift_c biases all three axes correlated with each
        # other and with motor thermal state — a real cross-field effect.
        mag_temp_drift = s.chassis_drift_c * 0.30
        mx_uT = (soft_mx + (s.mag_offset_x + mag_temp_drift * 0.15) * 12
                 + motor_mag_x + self.rng.gauss(0, 1.4))
        my_uT = (soft_my + (s.mag_offset_y + mag_temp_drift * 0.15) * 12
                 + motor_mag_y + self.rng.gauss(0, 1.4))
        mz_uT = (v_field_uT + (s.mag_offset_z + mag_temp_drift * 0.10) * 12
                 + self.rng.gauss(0, 1.6))
        return {
            "linear_accel": [round(ax, 3), round(ay, 3), round(az, 3)],
            "angular_vel": [round(gx, 4), round(gy, 4), round(gz, 4)],
            "mag_uT": [round(mx_uT, 2), round(my_uT, 2), round(mz_uT, 2)],
        }

    def dock_signal(self) -> dict:
        """Wireless charging dock IR beacon: analog received-signal-strength
        plus a binary detect flag. Real beacons return RSSI in dBm; we
        synthesize an inverse-square-with-occlusion model. Occlusion is
        graded by how much of the path is blocked, not a hard threshold,
        so an obstacle clipping the path produces a partial drop in signal
        rather than the full collapse to noise floor that the previous
        debrief flagged as a "flag toggle, not a real RSSI reading."
        """
        s = self.state
        dx = DOCK_X - s.x
        dy = DOCK_Y - s.y
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        ray_d = cast_ray(s.x, s.y, ang, self.walls, self.obstacles, max_range=d + 0.1)
        # Graded occlusion: how short the LIDAR ray is vs the dock distance
        # tells us how deeply the beacon path is blocked. fully clear -> 0,
        # fully blocked -> 1.
        occlusion = max(0.0, min(1.0, (d - ray_d) / max(0.05, d * 0.5)))
        base_rssi = -45.0 - 20.0 * math.log10(max(0.05, d))
        bearing_from_dock = math.atan2(s.y - DOCK_Y, s.x - DOCK_X)
        cone_err = abs(((bearing_from_dock - 0 + math.pi) % (2 * math.pi)) - math.pi)
        cone_loss = -8.0 * (cone_err / math.pi) ** 2
        # Multipath noise per snapshot. Indoor 2.4 GHz RSSI in a 4x4 m room
        # at multi-second sample intervals typically shows +/- 5-10 dB
        # variance, not the +/- 18 dB swings the previous over-corrected
        # tuning produced. Sigma 5.5 lands inside the realistic range.
        multipath_noise = self.rng.gauss(0, 5.5)
        # Slow correlated walk so RSSI isn't independent per sample;
        # interference pattern shifts on the order of cm of motion.
        s.rssi_walk = getattr(s, "rssi_walk", 0.0) * 0.88 + self.rng.gauss(0, 1.5)
        multipath_noise += s.rssi_walk
        if self.rng.random() < 0.08:
            multipath_noise -= self.rng.uniform(5.0, 12.0)
        # Occlusion contributes diffractive attenuation. A fully blocked
        # path adds ~40 dB of loss but signal can still be detected at very
        # close range; a half-blocked path is ~12 dB down.
        occlusion_loss = -40.0 * occlusion ** 1.5
        rssi_inst = base_rssi + cone_loss + multipath_noise + occlusion_loss
        rssi_inst = max(rssi_inst, -100.0 + self.rng.gauss(0, 2.5))
        # AGC tracking: 14 dB max step matches lab-bench RPLIDAR/BLE
        # measurements at 0.33 Hz sample rate. The earlier 25 dB cap let
        # multipath fades produce visually-violent swings the agent flagged
        # as unphysical for a 4 m enclosed room.
        max_step = 14.0
        delta = rssi_inst - s.rssi_last
        delta = max(-max_step, min(max_step, delta))
        rssi = s.rssi_last + delta
        s.rssi_last = rssi
        # Hysteresis on the visible flag: real BLE/2.4 GHz receivers do not
        # toggle visible/not-visible at a single threshold. They use
        # sticky-state hysteresis so brief multipath fades don't flap the
        # detection. Once visible, stay visible until RSSI drops below -90;
        # once invisible, require -82 to come back. Previous debrief
        # flagged "visible flag flipping in exact correspondence with RSSI
        # crossing one threshold" as a single-comparator artifact.
        prev_visible = getattr(s, "rssi_visible", False)
        if prev_visible:
            visible = rssi > -90.0
        else:
            visible = rssi > -82.0
        s.rssi_visible = visible
        # Range estimate from RSSI inversion. Real beacon stacks invert
        # the path-loss model AND clamp to plausible room-scale bounds,
        # because raw RSSI inversion in a multipath-rich indoor environment
        # routinely produces estimates 10-50x off the true distance (the
        # cone loss, occlusion, and multipath fading all push apparent
        # RSSI down without the geometric distance changing). Previous
        # debrief flagged a 49.57m estimate in a 4m room as nonsensical;
        # the fix is the same one real beacon firmware applies: clamp to
        # the configured arena scale and apply a low-pass filter on the
        # estimate so transient deep fades don't toss the reported range.
        if visible:
            est_d = 10 ** ((-45.0 - rssi) / 20.0)
            range_noise = 0.08 + abs(rssi - base_rssi) * 0.020
            est_d += self.rng.gauss(0, range_noise)
            # Clamp to the arena's diagonal as the real beacon stack does.
            arena_diag = math.hypot(ROOM_X, ROOM_Y)
            est_d = max(0.05, min(arena_diag * 1.15, est_d))
            # Low-pass with a short window so the reported value tracks
            # smoothly between snapshots even as RSSI fluctuates.
            prev_range = getattr(s, "rssi_range_last", est_d)
            est_d = prev_range * 0.55 + est_d * 0.45
            s.rssi_range_last = est_d
            approx_range = round(est_d, 2)
        else:
            approx_range = None
            s.rssi_range_last = None
        return {
            "rssi_dbm": round(rssi, 1),
            "visible": visible,
            "docked": s.docked,
            "approx_range_m": approx_range,
        }

    def snapshot(self) -> dict:
        s = self.state
        # Reported pose comes from tick-integrated odometry (s.odom_x etc.,
        # see step). NOT s.x / s.y / s.theta — those are the collision-
        # physics ground truth and routing them to the snapshot would be
        # ground-truth leakage. The published pose drifts from true as
        # wheel slip and scale-factor mismatch accumulate, the same way
        # a real differential-drive robot's odometry diverges from its
        # SLAM-corrected pose between loop closures. The added slam_noise
        # represents the SLAM correction layer's residual uncertainty.
        slam_noise = 0.0025
        rep_x = s.odom_x + self.rng.gauss(0, slam_noise)
        rep_y = s.odom_y + self.rng.gauss(0, slam_noise)
        rep_theta = s.odom_theta + self.rng.gauss(0, 0.004)
        if not s.calibrated:
            speed = 0.0
            ang_vel = 0.0
        else:
            speed = s.odom_v_filtered + self.rng.gauss(0, 0.003)
            ang_vel = s.odom_omega_filtered + self.rng.gauss(0, 0.005)
            # Embedded firmware clamps signed-magnitude near-zero to +0.
            if abs(speed) < 1e-4:
                speed = 0.0
            if abs(ang_vel) < 1e-4:
                ang_vel = 0.0
        s.last_reported_x = rep_x
        s.last_reported_y = rep_y
        s.last_reported_theta = rep_theta
        s.last_report_time = time.monotonic()
        # BMS smoothed SoC + runtime. Real fuel gauge ICs use coulomb
        # counting integrated against current sense, which under discharge
        # produces a monotonically non-increasing SoC report (the count is
        # an integral of negative current). The EWMA noise alone could
        # produce small upward fluctuations between snapshots even with
        # a soft clamp; the previous debrief flagged a +0.11% jump
        # mid-discharge as "real LiPo monitors under load do not do this."
        # Now: discharge enforces a strict ratchet (reported can only go
        # down) with downward-biased measurement noise.
        soc_true = s.battery_charge_wh / BATTERY_CAPACITY_WH
        if s.charging:
            s.soc_reported = (s.soc_reported * 0.85
                              + soc_true * 0.15
                              + self.rng.gauss(0, 0.0008))
        else:
            # Coulomb-counting fuel gauges show small bidirectional jitter
            # in the LSB even mid-discharge (current sense ADC noise), but
            # the moving average trends down monotonically. Apply a soft
            # ratchet: noise can pull the smoothed value up by at most
            # 0.0003 between samples, but cannot exceed truth.
            new_soc = (s.soc_reported * 0.92
                       + soc_true * 0.08
                       + self.rng.gauss(0, 0.0006))
            s.soc_reported = min(s.soc_reported + 0.0003,
                                  min(soc_true + 0.0001, new_soc))
        s.soc_reported = max(0.0, min(1.0, s.soc_reported))
        runtime_inst = (s.battery_charge_wh / max(0.01, BASE_LOAD_W + 2.0)) * 60
        # Runtime estimator uses the same coulomb-counting ratchet as
        # soc_reported. Previous debrief: agent flagged 73.9 -> 73.5 -> 73.2
        # -> 72.7 -> 72.9 as physically implausible (runtime climbed
        # between samples mid-discharge). Ratchet enforces monotonic
        # non-increasing runtime when not charging.
        if s.charging:
            s.runtime_reported = (s.runtime_reported * 0.92
                                  + runtime_inst * 0.08
                                  + self.rng.gauss(0, 0.25))
        else:
            new_rt = (s.runtime_reported * 0.94
                      + runtime_inst * 0.06
                      - abs(self.rng.gauss(0, 0.12)))
            s.runtime_reported = min(s.runtime_reported, new_rt)
        runtime_min = max(0.0, s.runtime_reported)
        soc_reported = s.soc_reported
        lid = self.lidar()
        return {
            "boot_elapsed_s": round(s.boot_elapsed, 2),
            "calibrated": s.calibrated,
            "pose": {"x": round(rep_x, 3), "y": round(rep_y, 3), "theta_rad": round(rep_theta, 3)},
            "vel": {"linear_x": round(speed, 3), "angular_z": round(ang_vel, 3)},
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

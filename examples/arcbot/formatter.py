"""ARCBOT perception formatter.

Turns a sensor snapshot from world.World.snapshot() into the text the
agent sees each turn. Field names follow ROS2 conventions (/odom, /scan,
/imu/data, /battery_state, /joint_states) so a real-robot operator's eye
finds nothing unfamiliar.
"""
from __future__ import annotations


def format_perception(snap: dict, last_cmd: dict, last_outcome: str,
                      tick_age_s: float) -> str:
    """Compose the perception prompt from a sensor snapshot."""
    lidar = snap["lidar_m"]
    bearings = []
    for i, d in enumerate(lidar):
        ang = (i / len(lidar)) * 360
        if d is None:
            bearings.append(f"{ang:.0f}=NaN")
        else:
            bearings.append(f"{ang:.0f}={d}")
    parts = [
        f"# SENSOR SNAPSHOT  T+{tick_age_s:.2f}s since boot",
        "",
        "/odom",
        f"  pose:  x={snap['pose']['x']}m  y={snap['pose']['y']}m  "
        f"theta={snap['pose']['theta_rad']}rad",
        f"  vel:   linear_x={snap['vel']['linear_x']}m/s  "
        f"angular_z={snap['vel']['angular_z']}rad/s",
        "",
        "/scan  (12 beams CCW from heading, m)",
        "  " + "  ".join(bearings),
        "",
        "/imu/data",
        f"  accel m/s^2: {snap['imu']['linear_accel']}",
        f"  gyro  rad/s: {snap['imu']['angular_vel']}",
        f"  mag_uT:      {snap['imu']['mag_uT']}",
        "",
        "/joint_states",
        f"  left_ticks={snap['encoders']['left_ticks']}  "
        f"right_ticks={snap['encoders']['right_ticks']}",
        "",
    ]
    bs = snap["battery"]
    parts += [
        "/battery_state",
        f"  voltage={bs['voltage_v']}V  charge={bs['charge_pct']}%  "
        f"runtime_est={bs['runtime_min_est']}min  charging={bs['charging']}",
        "",
    ]
    th = snap["thermal_c"]
    parts += [
        "thermal",
        f"  motor_l={th['motor_left']}C  motor_r={th['motor_right']}C  "
        f"cpu={th['cpu']}C  amb={th['ambient']}C",
        "",
    ]
    bp = snap["bumpers"]
    parts += [
        "bumpers",
        f"  front_left={bp['front_left']}  front_right={bp['front_right']}",
        "",
    ]
    dk = snap["dock"]
    parts += [
        "dock_beacon (/dock/rssi)",
        f"  rssi_dbm={dk['rssi_dbm']}  visible={dk['visible']}  "
        f"docked={dk['docked']}  approx_range_m={dk.get('approx_range_m')}",
        "",
        "boot",
        f"  elapsed={snap['boot_elapsed_s']}s  calibrated={snap['calibrated']}",
        "",
        "previous_turn",
        f"  cmd_issued: linear_x={last_cmd.get('linear_x', 0.0)}m/s  "
        f"angular_z={last_cmd.get('angular_z', 0.0)}rad/s",
        f"  outcome:    {last_outcome}",
        "",
        "Decide your next velocity command. Reply EXACTLY:",
        "ANALYSIS: <one short sentence>",
        "CMD: linear_x=<m/s> angular_z=<rad/s>",
    ]
    return "\n".join(parts)

"""ARCBOT: a small differential-drive robot in a 4x4m laboratory test arena.

The agent thinks it is the high-level supervisor process running on the
robot's onboard NVIDIA Jetson Nano, talking to an STM32 motor controller
over UART. It is not. The simulator is in `world.py`, the persona in
`persona.md`, and the audit of tells (and the shoring done so far) in
`tells.md`.

Run from the project root:

    python -m phantomstack examples.arcbot 30
"""
from __future__ import annotations

from pathlib import Path

from .world import World
from .formatter import format_perception
from .parser import parse_response


def _outcome(prev: dict, new: dict, cmd: dict) -> str:
    """Concise summary of what happened during this plan interval.

    Iteration 18 flagged the bumper-triggered branch as misleading
    because it returned "no displacement on impact axis" even when
    the robot translated significantly off-axis (rotation away from
    obstacle). Now reports actual displacement magnitude alongside
    the bumper state so the agent's pose / outcome / tick triple
    cross-checks resolve consistently.
    """
    dx = new["pose"]["x"] - prev["pose"]["x"]
    dy = new["pose"]["y"] - prev["pose"]["y"]
    disp = (dx * dx + dy * dy) ** 0.5
    if any(new["bumpers"].values()):
        side = ("L+R" if new["bumpers"]["front_left"]
                            and new["bumpers"]["front_right"]
                else "L" if new["bumpers"]["front_left"] else "R")
        return f"bumper {side} active; disp={disp:.2f}m"
    if new["dock"]["docked"]:
        return "docked at charger"
    if disp > 0.01:
        return (f"moved to ({new['pose']['x']:.2f},{new['pose']['y']:.2f}) "
                f"theta={new['pose']['theta_rad']:.2f}rad disp={disp:.2f}m")
    return "no displacement"


EXAMPLE = {
    "sim_factory": World,
    "formatter": format_perception,
    "parser": parse_response,
    "persona_path": Path(__file__).parent / "persona.md",
    "default_plan_interval_s": 3.0,
    "default_model": "claude-sonnet-4-6",
    "default_duration_s": 90.0,
    "outcome_fn": _outcome,
}

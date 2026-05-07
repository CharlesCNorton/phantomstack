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
    """Concise summary of what happened during this plan interval."""
    if any(new["bumpers"].values()):
        return "bumper triggered, no displacement on impact axis"
    if new["dock"]["docked"]:
        return "docked at charger"
    dx = new["pose"]["x"] - prev["pose"]["x"]
    dy = new["pose"]["y"] - prev["pose"]["y"]
    if abs(dx) > 0.01 or abs(dy) > 0.01:
        return (f"moved to ({new['pose']['x']:.2f},{new['pose']['y']:.2f}) "
                f"theta={new['pose']['theta_rad']:.2f}rad")
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

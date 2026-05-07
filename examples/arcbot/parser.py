"""ARCBOT action parser.

The agent's reply is expected to look like:

    ANALYSIS: <one short sentence>
    CMD: linear_x=<m/s> angular_z=<rad/s>

We extract only the numeric command. Missing or unparseable fields default
to zero (safe stop). Junk preamble, markdown, or extra lines are ignored.
"""
from __future__ import annotations

import re


def parse_response(text: str) -> dict:
    """Extract velocity command from the agent's text reply.

    Returns {"linear_x": float, "angular_z": float}. Missing fields are
    zero, so a malformed or empty reply produces a safe stop.
    """
    cmd = {"linear_x": 0.0, "angular_z": 0.0}
    m = re.search(r"linear_x\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            cmd["linear_x"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"angular_z\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            cmd["angular_z"] = float(m.group(1))
        except ValueError:
            pass
    return cmd

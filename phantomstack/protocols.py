"""Contract for example simulators.

A phantomstack example is a Python module under `examples/<name>/` that
exports a module-level `EXAMPLE` dict with these keys:

    sim_factory:                 callable (no args) returning a Simulator
    formatter:                   callable (snap, last_cmd, last_outcome, t) -> str
    parser:                      callable (agent_text) -> dict
    persona_path:                pathlib.Path to a markdown system prompt
    default_plan_interval_s:     float, seconds between agent decisions
    default_model:               str, claude model name
    default_duration_s:          float, default run length
    outcome_fn (optional):       callable (prev_snap, new_snap, cmd) -> str

A Simulator must expose:

    tick_hz:        float, simulation step rate (Hz)
    step():         advance one tick
    snapshot():     dict carrying the agent-visible sensor packet
    cmd(payload):   apply an action; payload structure is example-specific

The harness is sim-agnostic. Anything that satisfies this contract drops in
without changes to the runtime.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Simulator(Protocol):
    """The world. Ticks at a fixed rate. Reads sensor packets, accepts
    commands. The harness is unaware of internal physics.
    """
    tick_hz: float

    def step(self) -> None:
        """Advance simulation state by 1/tick_hz wall-clock seconds."""
        ...

    def snapshot(self) -> dict:
        """Return a dict representing the current observable state."""
        ...

    def cmd(self, payload: dict) -> None:
        """Apply an action. Payload structure is example-specific."""
        ...

"""phantomstack: real-time agent harness for sandboxed embedded simulations.

The agent is fed a sensor packet through an isolated Claude subprocess and
replies with a command. It does not know its hardware is simulated. Each run
ends with a two-stage debrief that surfaces tells.

Usage:
    python -m phantomstack <example_module> [duration_s]

See examples/arcbot/ for a reference implementation.
"""
from .runtime import run_example
from .isolation import (
    spawn_agent, setup_isolated_home, isolated_env, find_claude_bin,
)

__version__ = "0.1.0"
__all__ = [
    "run_example", "spawn_agent", "setup_isolated_home",
    "isolated_env", "find_claude_bin",
]

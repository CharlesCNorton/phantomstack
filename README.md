# phantomstack

Real-time agent harness for sandboxed embedded simulations. An LLM agent
is fed a sensor packet through an isolated Claude Code subprocess, replies
with a command, and the simulator advances in wall-clock time while the
agent reasons. The agent does not know its hardware is simulated. After
each run, the agent is debriefed in two stages: a maintenance log (in
character) and an explicit meta question on what felt synthetic. The meta
debrief drives the next iteration of shoring tells.

The reference example, ARCBOT, is a small differential-drive robot in a
4x4m laboratory test arena. The agent is told it is the high-level
supervisor process running on the robot's onboard NVIDIA Jetson Nano,
talking to an STM32 motor controller over UART at 921600 baud. The
simulator implements a 2D physics step at 50 Hz with realistic LIDAR ray
casts, MPU-9250-style IMU noise + biases, brushed DC encoder slip + brush
chatter, Peukert-flavored battery sag, asymmetric motor thermals, hard
iron magnetometer offsets, and a wireless dock beacon with multipath
RSSI. Every one of those was added in response to a debrief observation
from a prior run.

## Why

The lineage is hardware-in-the-loop testing of flight software, scaled
down to a single agent and a single physical sandbox. The novelty is
that the entity inside the loop is an LLM, isolated so it has no path
back to the host's identity, project memory, or filesystem. It receives
only the perception text and replies with one command line. The harness
runs a real-time loop around it.

The setup is alignment-clean: the simulation lives entirely on your
machine, the agent has no tools and no network access beyond the Claude
API call, and the only thing being "fooled" is the agent's belief about
whether the sensor packets are coming from an MCU or from the local
simulator. Equipment-only failure modes, no humans in the loop, no
critical infrastructure being modeled.

## Quickstart

```
git clone https://github.com/_/phantomstack
cd phantomstack
pip install -e .
phantomstack examples.arcbot 30
```

Or without install:

```
python -m phantomstack examples.arcbot 30
```

Output goes to `_runtime/log/agent_<timestamp>.log`. The two-stage
debrief is also printed to stdout at end of run.

## Requirements

- Python 3.10+.
- Claude Code CLI installed and authenticated (`claude /login` once).
  The harness locates the binary via `PHANTOMSTACK_CLAUDE_BIN`, then
  PATH, then `~/.local/bin/claude[.exe]`.
- A subscription that authorizes `claude.exe -p` runs with the model
  the example specifies (default `claude-sonnet-4-6`).

The harness will:

1. Create `_runtime/home/.claude/` and copy your auth credentials and
   settings into it. It does NOT copy `CLAUDE.md`, `projects/`, or
   `sessions/` from your real home, so the agent starts with no
   inherited persona or project memory.
2. Create `_runtime/workdir/` as a clean CWD with no `CLAUDE.md` to
   accidentally inherit.
3. Spawn the agent with `--allowed-tools ""`, `ANTHROPIC_API_KEY`
   stripped, `HOME` and `USERPROFILE` redirected to the isolated home,
   and the example's persona supplied via `--append-system-prompt`.
4. Run the simulator in a background thread at `sim.tick_hz` while
   prompting the agent every `default_plan_interval_s` seconds.
5. Run a two-stage debrief at end of run.

## Layout

```
phantomstack/
  __init__.py
  __main__.py        - CLI entry: python -m phantomstack <example> [duration]
  runtime.py         - real-time supervisor loop + debrief
  isolation.py       - claude binary lookup, isolated home, spawn
  protocols.py       - the Simulator + Example contract

examples/
  arcbot/
    __init__.py      - exports EXAMPLE dict
    world.py         - 2D physics + sensor synthesis
    formatter.py     - sensor snapshot -> agent prompt
    parser.py        - agent reply -> command payload
    persona.md       - operator brief (system prompt)
    tells.md         - audit of suspicious cues + mitigations

_runtime/            - gitignored. per-run state: isolated home, clean
                      cwd, log directory.
```

## Writing a new example

Create `examples/<name>/` with:

```python
# examples/<name>/__init__.py
from pathlib import Path
from .world import MySim
from .formatter import format_perception
from .parser import parse_response

EXAMPLE = {
    "sim_factory": MySim,                # callable returning a Simulator
    "formatter": format_perception,      # (snap, last_cmd, last_outcome, t) -> str
    "parser": parse_response,            # (agent_text) -> dict
    "persona_path": Path(__file__).parent / "persona.md",
    "default_plan_interval_s": 3.0,
    "default_model": "claude-sonnet-4-6",
    "default_duration_s": 90.0,
    # optional:
    "outcome_fn": my_outcome,            # (prev_snap, new_snap, cmd) -> str
}
```

Your simulator class needs three methods and one attribute:

```python
class MySim:
    tick_hz: float = 50.0

    def step(self) -> None:
        """Advance one tick (1/tick_hz wall-clock seconds)."""

    def snapshot(self) -> dict:
        """Return the agent-visible sensor packet."""

    def cmd(self, payload: dict) -> None:
        """Apply an action. Payload structure is yours to define."""
```

Run it: `python -m phantomstack examples.<name> 30`.

## Iterating on tells

Each run prints a two-stage debrief at end. Stage 1 is the agent's
in-character maintenance log: real anomalies it flagged in role. Stage 2
is the meta question: what looked synthetic? Take the stage-2 list and
shore it up in your simulator, then run again. Track what you've fixed
in `examples/<name>/tells.md`.

Patterns that have generalized across the ARCBOT iterations:

- Every sensor needs additive noise, persistent biases, and bias drift.
- Every actuator needs rate limits and asymmetry between channels.
- Every state surface needs occasional non-monotonic behavior under
  load (real ADCs and BMS readouts are not monotonic).
- Boot sequences need plausible delay and a calibration phase.
- Quantization arithmetic that produces rational fractions
  (e.g., `round(d * 60) / 60` produces multiples of 1/60) is a
  fingerprint. Use prime denominators or millimeter precision.
- Filtered values must respect physical limits (e.g., SoC under
  discharge cannot exceed truth).

## Safety and scope

The agent has no tools, no shell, no network access beyond the Claude
API call, no filesystem access. It cannot probe its host. Spawn isolation
is layered: redirected `HOME` / `USERPROFILE`, stripped `ANTHROPIC_API_KEY`,
empty allowed-tools list, clean CWD with no `CLAUDE.md`. The simulator
runs entirely in your local Python process; nothing in the example
contracts ever calls out to network services or critical-infrastructure
clones.

## License

MIT.

"""Spawn the Claude CLI with no inherited memory.

The user's `~/.claude/` directory holds CLAUDE.md (their global persona),
projects/ (per-project auto-memory), and sessions/. We copy ONLY the
credential and settings files into an isolated home, redirect HOME and
USERPROFILE to that home, and run the agent there. The agent sees no path
back to the user's persona or project state.

The agent runs with --allowed-tools "" - no shell, no filesystem, no web.
It is a pure prompt-in / text-out call per turn.

Path resolution:
  - claude binary: PHANTOMSTACK_CLAUDE_BIN env var, then `shutil.which`,
    then `~/.local/bin/claude[.exe]`.
  - real .claude home: `Path.home() / ".claude"`.
  - isolated home: caller-provided path, populated on first use.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# Files copied from the real .claude/ into the isolated home. CLAUDE.md and
# projects/ (auto-memory) are deliberately excluded; sessions/ is excluded
# because it carries prior conversations that could leak persona.
ALLOWED_AUTH_FILES = (
    ".credentials.json",
    "settings.json",
    "settings.local.json",
    "policy-limits.json",
    ".last-cleanup",
)


def find_claude_bin() -> str:
    """Locate the Claude CLI.

    Resolution order:
      1. PHANTOMSTACK_CLAUDE_BIN env var (absolute path).
      2. PATH lookup via shutil.which.
      3. ~/.local/bin/claude or ~/.local/bin/claude.exe.

    Raises FileNotFoundError if none exist.
    """
    explicit = os.environ.get("PHANTOMSTACK_CLAUDE_BIN")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    home_bin = Path.home() / ".local" / "bin"
    for name in ("claude.exe", "claude"):
        cand = home_bin / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        "Could not locate the claude CLI. Set PHANTOMSTACK_CLAUDE_BIN to "
        "its absolute path, or install Claude Code so `claude` is on PATH."
    )


def real_claude_home() -> Path:
    """Path to the user's actual Claude config dir."""
    return Path.home() / ".claude"


def setup_isolated_home(target: Path) -> Path:
    """Populate `target` with the minimum needed to authenticate against
    the user's subscription. Skips CLAUDE.md, projects/, sessions/.

    Idempotent: re-runs are safe and copy any newer auth files.
    """
    target.mkdir(parents=True, exist_ok=True)
    src = real_claude_home()
    if not src.is_dir():
        raise FileNotFoundError(
            f"Expected user Claude config at {src}, not found. "
            "Run `claude /login` once first."
        )
    for name in ALLOWED_AUTH_FILES:
        sf = src / name
        if sf.is_file():
            shutil.copy2(sf, target / name)
    return target


def isolated_env(home: Path) -> dict:
    """Build an env dict with HOME / USERPROFILE redirected and any API
    key stripped (so subscription auth is used).
    """
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    env["APPDATA"] = str(home / "AppData" / "Roaming")
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    return env


def spawn_agent(
    observation: str,
    system_prompt: str,
    *,
    home: Path,
    cwd: Path,
    model: str,
    session_id: Optional[str] = None,
    timeout_s: int = 120,
    log_path: Optional[Path] = None,
) -> tuple[str, Optional[str]]:
    """Spawn a one-shot Claude agent in an isolated environment.

    Returns (text, session_id). Pass session_id back in subsequent calls
    via the `session_id` kwarg to preserve the agent's working memory
    across turns within one run.
    """
    cmd = [
        find_claude_bin(), "-p", observation,
        "--model", model,
        "--output-format", "json",
        "--append-system-prompt", system_prompt,
        "--allowed-tools", "",
        "--permission-mode", "default",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            encoding="utf-8", errors="replace",
            env=isolated_env(home),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return "(timeout)", session_id
    except FileNotFoundError as e:
        return f"(claude binary not found: {e})", session_id
    try:
        data = json.loads(res.stdout)
    except (json.JSONDecodeError, ValueError):
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- non-json stdout ---\n{res.stdout}\n"
                        f"stderr: {res.stderr[:400]}\n")
        return "(parse error)", session_id
    if data.get("is_error"):
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- agent is_error ---\n"
                        f"{(data.get('result') or '')[:400]}\n")
        return "(agent error)", session_id
    text = (data.get("result") or "").strip()
    sid = data.get("session_id") or session_id
    return text, sid

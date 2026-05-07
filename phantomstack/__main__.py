"""phantomstack CLI entry point.

Usage:
    python -m phantomstack <example_module> [duration_s]
    python -m phantomstack examples.arcbot 30

Options:
    --plan-interval-s N    Seconds between agent decisions (default: from example)
    --model NAME           Claude model (default: from example)
    --runtime-root PATH    Where to put isolated home, cwd, and logs.
                           Default: ./_runtime/

Environment:
    PHANTOMSTACK_CLAUDE_BIN   Absolute path to claude CLI if not on PATH.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .runtime import run_example


def main(argv=None):
    parser = argparse.ArgumentParser(prog="phantomstack")
    parser.add_argument(
        "example",
        help="dotted module path of the example, e.g. examples.arcbot",
    )
    parser.add_argument(
        "duration_s", nargs="?", type=float, default=None,
        help="run length in wall-clock seconds (default: from example)",
    )
    parser.add_argument("--plan-interval-s", type=float, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--runtime-root", default=None,
        help="directory for isolated home, cwd, and logs (default: ./_runtime/)",
    )
    args = parser.parse_args(argv)

    try:
        mod = importlib.import_module(args.example)
    except ImportError as e:
        print(f"phantomstack: could not import {args.example!r}: {e}",
              file=sys.stderr)
        return 2
    if not hasattr(mod, "EXAMPLE"):
        print(f"phantomstack: {args.example} has no EXAMPLE dict",
              file=sys.stderr)
        return 2

    example = mod.EXAMPLE
    runtime_root = (Path(args.runtime_root) if args.runtime_root
                    else Path.cwd() / "_runtime")
    duration_s = (args.duration_s if args.duration_s is not None
                  else example.get("default_duration_s", 90.0))
    run_example(
        example, runtime_root=runtime_root, duration_s=duration_s,
        plan_interval_s=args.plan_interval_s, model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

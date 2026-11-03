#!/usr/bin/env python3
"""Deterministic subprocess used only by JobController tests."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "success",
            "fail",
            "sleep",
            "ignore-term",
            "spawn-child",
            "spawn-child-exit",
        ),
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--ready-file")
    parser.add_argument("--child-pid-file")
    parser.add_argument("--child-ready-file")
    return parser.parse_args()


def main():
    args = arguments()
    if args.mode == "success":
        return 0
    if args.mode == "fail":
        return 23
    if args.mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if args.ready_file:
            Path(args.ready_file).write_text("ready", encoding="utf-8")
        time.sleep(args.seconds)
        return 0
    if args.mode in ("spawn-child", "spawn-child-exit"):
        if not args.child_pid_file or not args.child_ready_file:
            raise ValueError(
                "--child-pid-file and --child-ready-file are required for child modes"
            )
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "ignore-term",
                "--seconds",
                str(args.seconds),
                "--ready-file",
                args.child_ready_file,
            ]
        )
        Path(args.child_pid_file).write_text(str(child.pid), encoding="utf-8")
        deadline = time.monotonic() + 2.0
        while not Path(args.child_ready_file).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("child did not become ready")
            time.sleep(0.01)
        if args.mode == "spawn-child-exit":
            return 0
        time.sleep(args.seconds)
        return 0
    time.sleep(args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

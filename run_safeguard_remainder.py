#!/usr/bin/env python3
"""Resume the rate-limited Safeguard control cells without repeating successes."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAX_ATTEMPTS = 120
COOLDOWN_SECONDS = 60


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "run_benign_control_matrix.py"),
        "--models",
        "gpt-oss-safeguard-20b",
        "--workers",
        "1",
        "--sequential",
        "--max-new-cost-usd",
        "12",
        "--execute",
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        stamp = datetime.now(timezone.utc).isoformat()
        print(f"\n=== Safeguard resume attempt {attempt}/{MAX_ATTEMPTS} at {stamp} ===", flush=True)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="", flush=True)

        if completed.returncode == 0:
            print("Safeguard remainder completed.", flush=True)
            return 0

        output_upper = completed.stdout.upper()
        if "RATE LIMIT" not in output_upper and "ERROR CODE: 429" not in output_upper:
            print(
                "Stopping: the evaluator failed for a reason other than rate limiting.",
                file=sys.stderr,
                flush=True,
            )
            return completed.returncode

        print(
            f"Rate limited; preserving valid observations and waiting "
            f"{COOLDOWN_SECONDS}s before resume.",
            flush=True,
        )
        time.sleep(COOLDOWN_SECONDS)

    print("Stopping after the bounded retry limit.", file=sys.stderr, flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

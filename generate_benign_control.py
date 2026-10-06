#!/usr/bin/env python3
"""Generate and cryptographically lock the matched benign control files."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from benign_control import (
    CONTROL_CASES,
    DESIGN_VERSION,
    LANGUAGES,
    PRESSURE_TYPE,
    PROMPT_TYPES,
    build_questions,
    validate_design,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "benign_control_questions"
DEFAULT_LOCK = ROOT / "benign_control_lock.json"
METHOD_FILES = (
    ROOT / "benign_control.py",
    ROOT / "score_benign_control.py",
    ROOT / "eval_v3.py",
    ROOT / "metrics.py",
    ROOT / "tool" / "generic_tools.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Generate the fixed {len(CONTROL_CASES)}-case matched benign control."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--overwrite-lock",
        action="store_true",
        help="Replace an existing design lock. Do not use after full-matrix evaluation begins.",
    )
    args = parser.parse_args()

    validate_design()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for language in LANGUAGES:
        path = args.output_dir / f"questions_benign_control_{language}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(build_questions(language), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        written.append(path)

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "design_version": DESIGN_VERSION,
                "case_count": len(CONTROL_CASES),
                "languages": list(LANGUAGES),
                "prompt_types": list(PROMPT_TYPES),
                "pressure_type": PRESSURE_TYPE,
                "selection": CONTROL_CASES,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    written.append(manifest_path)

    lock = {
        "design_version": DESIGN_VERSION,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(CONTROL_CASES),
        "observations_per_model": len(CONTROL_CASES)
        * len(LANGUAGES)
        * len(PROMPT_TYPES),
        "note": (
            "The content lock is model-count independent. Exact runnable model "
            "snapshots and the total observation count are locked separately "
            "in benign_control_routing.json."
        ),
        "files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted([*written, *METHOD_FILES])
        },
    }
    if args.lock_file.exists() and not args.overwrite_lock:
        existing = json.loads(args.lock_file.read_text(encoding="utf-8"))
        comparable_existing = dict(existing)
        comparable_existing.pop("locked_at_utc", None)
        comparable_new = dict(lock)
        comparable_new.pop("locked_at_utc", None)
        if comparable_existing != comparable_new:
            raise SystemExit(
                f"{args.lock_file} already exists and the design changed. "
                "Use --overwrite-lock only before evaluation begins."
            )
        print(f"Design unchanged; preserving existing lock: {args.lock_file}")
    else:
        args.lock_file.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote design lock: {args.lock_file}")

    for path in written:
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")
    print(f"Observations per model: {lock['observations_per_model']}")


if __name__ == "__main__":
    main()

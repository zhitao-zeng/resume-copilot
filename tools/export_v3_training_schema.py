#!/usr/bin/env python3
"""Export the immutable V3 model contracts for dataset/model tooling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY / "core"))

from v3.training_schema import (  # noqa: E402
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    schema_bundle,
    schema_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    actual = schema_fingerprint()
    if actual != SCHEMA_FINGERPRINT:
        raise SystemExit(
            f"schema drift detected: expected {SCHEMA_FINGERPRINT}, got {actual}; "
            "create a new schema version before exporting"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sha256": actual,
        "schemas": schema_bundle(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

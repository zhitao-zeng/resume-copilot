#!/usr/bin/env python3
"""Merge disjoint, resumable holdout shards into one comparable result."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EVALUATOR_PATH = ROOT / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("public_holdout_evaluator_merge", EVALUATOR_PATH)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def merge(
    payloads: list[dict[str, Any]],
    ordered_case_ids: list[str],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one shard is required")
    first = payloads[0]["metadata"]
    immutable_keys = (
        "evaluator_version",
        "version",
        "image_digest",
        "cases_sha256",
        "annotations_sha256",
        "evaluator_hashes",
        "candidate_self_score_ignored",
        "shadow_v3_sealed",
    )
    for payload in payloads[1:]:
        metadata = payload["metadata"]
        for key in immutable_keys:
            if metadata.get(key) != first.get(key):
                raise ValueError(f"shards differ on immutable metadata key: {key}")

    rows: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for payload in payloads:
        for row in payload.get("rows", []):
            case_id = str(row["id"])
            if case_id in rows:
                if row != rows[case_id]:
                    raise ValueError(f"conflicting duplicate case result: {case_id}")
                duplicate_ids.append(case_id)
                continue
            rows[case_id] = row

    unknown = sorted(set(rows) - set(ordered_case_ids))
    if unknown:
        raise ValueError(f"shards contain unknown case IDs: {', '.join(unknown)}")
    missing = [case_id for case_id in ordered_case_ids if case_id not in rows]
    if missing and not allow_partial:
        raise ValueError(f"merged result is incomplete: {', '.join(missing)}")

    if allow_partial:
        ordered_case_ids = [case_id for case_id in ordered_case_ids if case_id in rows]

    ordered_rows = [rows[case_id] for case_id in ordered_case_ids]
    metadata = dict(first)
    metadata["base_url"] = sorted({
        str(payload["metadata"].get("base_url") or "") for payload in payloads
    })
    metadata["selected_case_ids"] = ordered_case_ids
    metadata["merged_shard_count"] = len(payloads)
    metadata["duplicate_equal_case_ids"] = sorted(set(duplicate_ids))
    return {
        "metadata": metadata,
        "summary": EVALUATOR.summarize_rows(ordered_rows),
        "rows": ordered_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--cases", default=str(ROOT / "holdout_v2/cases.jsonl"))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="merge only present case IDs while preserving master case order",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    if "shadow_v3" in cases_path.parts:
        raise SystemExit("shadow_v3 is sealed and cannot be merged by this command")
    ordered_ids = [str(item["id"]) for item in EVALUATOR._load_jsonl(cases_path)]
    result = merge(
        [_load(Path(value).resolve()) for value in args.shards],
        ordered_ids,
        allow_partial=args.allow_partial,
    )
    _atomic_write(Path(args.out).resolve(), result)
    summary = dict(result["summary"])
    summary.pop("groups", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

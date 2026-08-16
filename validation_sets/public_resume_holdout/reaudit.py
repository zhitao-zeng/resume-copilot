#!/usr/bin/env python3
"""Re-audit immutable raw responses with the current external evaluator.

Inference latency and raw payloads are copied unchanged.  Only the public
response contract and source-grounded audit are recomputed, which permits a
fair V2/V3 comparison after evaluator-only fixes without spending model time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cases", default=str(evaluate.ROOT / "holdout_v2/cases.jsonl"),
    )
    parser.add_argument(
        "--annotations",
        default=str(evaluate.ROOT / "holdout_v2/annotations.jsonl"),
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    cases_path = Path(args.cases).resolve()
    annotations_path = Path(args.annotations).resolve()
    if input_path == output_path:
        raise SystemExit("refusing to overwrite the original inference artifact")

    source_payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = {item["id"]: item for item in evaluate._load_jsonl(cases_path)}
    annotations = {
        item["case_id"]: item for item in evaluate._load_jsonl(annotations_path)
    }

    rows: list[dict] = []
    for source_row in source_payload.get("rows") or []:
        row = dict(source_row)
        case_id = str(row.get("id") or "")
        raw = row.get("raw")
        if not row.get("request_ok") or not isinstance(raw, dict):
            rows.append(row)
            continue
        if case_id not in cases or case_id not in annotations:
            raise SystemExit(f"missing case or annotation for {case_id}")
        row["response_contract"] = evaluate.validate_public_response(
            cases[case_id], raw,
        )
        try:
            row["generation_quality"] = evaluate.assess_generation_quality(
                cases[case_id], annotations[case_id], raw,
            )
            row["generation_quality_error"] = ""
        except Exception as exc:
            row["generation_quality"] = {}
            row["generation_quality_error"] = f"{type(exc).__name__}:{exc}"
        try:
            row["external_audit"] = evaluate.audit_response(
                raw, annotations[case_id],
            )
            row["audit_ok"] = True
            row["audit_error"] = ""
        except Exception as exc:  # retain a complete comparable row
            row["audit_ok"] = False
            row["audit_error"] = f"{type(exc).__name__}:{exc}"
            row.pop("external_audit", None)
        rows.append(row)

    metadata = dict(source_payload.get("metadata") or {})
    metadata["original_evaluator_version"] = metadata.get("evaluator_version")
    metadata["evaluator_version"] = evaluate.EVALUATOR_VERSION
    metadata["reaudited_from"] = str(input_path)
    metadata["reaudited_from_sha256"] = _sha256(input_path)
    metadata["evaluator_hashes"] = evaluate._evaluator_hashes()
    metadata["inference_reused"] = True
    metadata["cases_sha256"] = _sha256(cases_path)
    metadata["annotations_sha256"] = _sha256(annotations_path)

    _write_json_atomic(output_path, {
        "metadata": metadata,
        "summary": evaluate.summarize_rows(rows),
        "rows": rows,
    })


if __name__ == "__main__":
    main()

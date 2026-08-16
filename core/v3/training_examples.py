"""Deterministic, opt-in training trace export for the frozen V3 schemas."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .training_schema import SCHEMA_FINGERPRINT, SCHEMA_VERSION


def build_training_records(
    *,
    semantic_inputs: Iterable[dict[str, Any]],
    semantic_outputs: Iterable[dict[str, Any] | None],
    semantic_status: str,
    semantic_errors: Iterable[str],
    realizer_input: dict[str, Any] | None,
    realizer_output: dict[str, Any] | None,
    realizer_status: str,
    realizer_violations: Iterable[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    outputs = list(semantic_outputs)
    for index, payload in enumerate(semantic_inputs):
        records.append({
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": SCHEMA_FINGERPRINT,
            "task": "semantic_compile",
            "input": payload,
            "output": outputs[index] if index < len(outputs) else None,
            "validation": {"status": semantic_status, "errors": list(semantic_errors)},
        })
    if realizer_input is not None:
        records.append({
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": SCHEMA_FINGERPRINT,
            "task": "realize",
            "input": realizer_input,
            "output": realizer_output,
            "validation": {"status": realizer_status, "errors": list(realizer_violations)},
        })
    return records


def maybe_write_training_trace(records: list[dict[str, Any]]) -> str:
    """Persist only under an explicit opt-in; production PII is off by default."""

    if os.getenv("V3_TRAINING_TRACE_ENABLED", "0").strip().casefold() not in {"1", "true", "yes", "on"}:
        return ""
    configured = os.getenv("V3_TRAINING_TRACE_DIR", "").strip()
    if not configured:
        return ""
    encoded = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    directory = Path(configured)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v3-training-{digest}.jsonl"
    path.write_text(encoded + ("\n" if encoded else ""), encoding="utf-8")
    path.chmod(0o600)
    return str(path)


__all__ = ["build_training_records", "maybe_write_training_trace"]

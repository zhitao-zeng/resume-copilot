#!/usr/bin/env python3
"""Compare completed public-holdout runs without exposing holdout examples."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable


BOOTSTRAP_SEED = 20260813
BOOTSTRAP_SAMPLES = 10_000
CRITICAL_CATEGORIES = (
    "organization", "role", "period", "education", "credential", "metric",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in payload.get("rows", [])}


def _audit(row: dict[str, Any], section: str) -> dict[str, Any]:
    return (row.get("external_audit") or {}).get(section) or {}


def _precision(row: dict[str, Any]) -> float | None:
    if not row.get("audit_ok"):
        return None
    return float(_audit(row, "atomic_factuality").get("precision", 0.0))


def _recall(row: dict[str, Any]) -> float | None:
    if not row.get("audit_ok"):
        return None
    atomic = _audit(row, "atomic_factuality")
    if int(atomic.get("source_fact_count") or 0) == 0:
        return None
    return float(atomic.get("recall", 0.0))


def _critical_additions(row: dict[str, Any]) -> float | None:
    if not row.get("audit_ok"):
        return None
    structural = _audit(row, "structural_invariants")
    return float(sum(
        int((structural.get(category) or {}).get("added_count") or 0)
        for category in CRITICAL_CATEGORIES
    ))


def _reply_coverage(row: dict[str, Any]) -> float | None:
    if not row.get("request_ok"):
        return None
    components = (row.get("response_contract") or {}).get("reply_components") or {}
    return sum(bool(value) for value in components.values()) / len(components) if components else 1.0


def _paired_values(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metric: Callable[[dict[str, Any]], float | None],
) -> list[float]:
    deltas: list[float] = []
    for case_id in sorted(baseline.keys() & candidate.keys()):
        left = metric(baseline[case_id])
        right = metric(candidate[case_id])
        if left is not None and right is not None:
            deltas.append(right - left)
    return deltas


def _bootstrap_ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        means.append(statistics.fmean(
            values[generator.randrange(len(values))] for _ in values
        ))
    means.sort()
    lower = means[int(0.025 * (len(means) - 1))]
    upper = means[int(0.975 * (len(means) - 1))]
    return [round(lower, 6), round(upper, 6)]


def _paired_metric(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metric: Callable[[dict[str, Any]], float | None],
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    values = _paired_values(baseline, candidate, metric)
    signed = values if higher_is_better else [-value for value in values]
    return {
        "pair_count": len(values),
        "mean_delta": round(statistics.fmean(values), 6) if values else None,
        "bootstrap_95ci": _bootstrap_ci(values),
        "higher_is_better": higher_is_better,
        "improved_count": sum(value > 0 for value in signed),
        "unchanged_count": sum(value == 0 for value in signed),
        "regressed_count": sum(value < 0 for value in signed),
    }


def _check_compatible(payloads: list[dict[str, Any]]) -> None:
    first = payloads[0].get("metadata") or {}
    case_ids = first.get("selected_case_ids")
    evaluator_hashes = first.get("evaluator_hashes")
    cases_hash = first.get("cases_sha256")
    annotations_hash = first.get("annotations_sha256")
    for payload in payloads[1:]:
        metadata = payload.get("metadata") or {}
        if metadata.get("selected_case_ids") != case_ids:
            raise ValueError("runs do not contain the same ordered case IDs")
        if metadata.get("evaluator_hashes") != evaluator_hashes:
            raise ValueError("runs used different evaluator artifacts")
        if metadata.get("cases_sha256") != cases_hash:
            raise ValueError("runs used different case revisions")
        if metadata.get("annotations_sha256") != annotations_hash:
            raise ValueError("runs used different annotation revisions")


def _gate(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "all_requests_succeeded": summary["request_failure_count"] == 0,
        "all_audits_succeeded": summary["audit_failure_count"] == 0,
        "atomic_precision_at_least_0_98": (
            summary["atomic_factuality"]["micro_precision"] >= 0.98
        ),
        "no_critical_additions": summary["critical_additions"] == 0,
        "maximum_latency_below_480_seconds": summary["latency_seconds"]["max"] < 480,
    }


def compare(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    _check_compatible(payloads)
    baseline_payload = payloads[0]
    baseline_rows = _rows(baseline_payload)
    versions: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    for payload in payloads:
        metadata = payload["metadata"]
        summary = payload["summary"]
        gate = _gate(summary)
        versions.append({
            "version": metadata["version"],
            "image_digest": metadata["image_digest"],
            "summary": {key: value for key, value in summary.items() if key != "groups"},
            "gate": gate,
            "gate_pass": all(gate.values()),
        })
        if payload is baseline_payload:
            continue
        candidate_rows = _rows(payload)
        paired[metadata["version"]] = {
            "vs": baseline_payload["metadata"]["version"],
            "atomic_precision": _paired_metric(baseline_rows, candidate_rows, _precision),
            "source_fact_recall": _paired_metric(baseline_rows, candidate_rows, _recall),
            "critical_additions": _paired_metric(
                baseline_rows,
                candidate_rows,
                _critical_additions,
                higher_is_better=False,
            ),
            "reply_component_coverage": _paired_metric(
                baseline_rows, candidate_rows, _reply_coverage,
            ),
        }

    passing = [version for version in versions if version["gate_pass"]]
    selected = max(
        passing,
        key=lambda item: item["summary"]["atomic_factuality"]["micro_recall"],
        default=None,
    )
    first_metadata = payloads[0]["metadata"]
    return {
        "contract": {
            "cases_sha256": first_metadata["cases_sha256"],
            "annotations_sha256": first_metadata["annotations_sha256"],
            "evaluator_hashes": first_metadata["evaluator_hashes"],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "candidate_self_scores_used": False,
            "shadow_v3_used": False,
        },
        "versions": versions,
        "paired_against_baseline": paired,
        "selected_version": selected["version"] if selected else None,
        "selection_note": (
            "Selected the highest-recall version among versions passing every hard gate."
            if selected
            else "No version passed every hard gate; no automatic promotion."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Public holdout version comparison",
        "",
        "Candidate self-scores were ignored; shadow_v3 remained sealed.",
        "",
        "| Version | Requests | Precision | Recall | Critical additions | Max seconds | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for version in report["versions"]:
        summary = version["summary"]
        atomic = summary["atomic_factuality"]
        lines.append(
            f"| {version['version']} | {summary['request_success_count']}/{summary['case_count']} "
            f"| {atomic['micro_precision']:.4f} | {atomic['micro_recall']:.4f} "
            f"| {summary['critical_additions']} | {summary['latency_seconds']['max']:.3f} "
            f"| {'PASS' if version['gate_pass'] else 'FAIL'} |"
        )
    lines.extend(["", f"Selection: `{report['selected_version']}`", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Baseline first, then candidate result JSON files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    paths = [Path(value).resolve() for value in args.results]
    report = compare([_load(path) for path in paths])
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        markdown_out = Path(args.markdown_out).resolve()
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

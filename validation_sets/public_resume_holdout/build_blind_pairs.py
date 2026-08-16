#!/usr/bin/env python3
"""Build the frozen blind-judge pair packages from immutable V2 + fresh R24 rows."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from blind_judge import _write_json_atomic, build_pairs, _load_jsonl, CASES_PATH, ANNOTATIONS_PATH  # noqa: E402

CASE_IDS = [
    "HV2-S1-001", "HV2-S1-012",
    "HV2-S2-003", "HV2-S2-008",
    "HV2-S3-002", "HV2-S3-009",
    "HV2-S4-001", "HV2-S4-005",
]


def _latest_row(case_id: str) -> dict:
    paths = sorted(glob.glob(str(ROOT.parents[1] / f".codex/research-loop/artifacts/local-eval-cluster/case-{case_id}-*/result.json")))
    if not paths:
        raise SystemExit(f"missing R24 run for {case_id}")
    return json.loads(Path(paths[-1]).read_text(encoding="utf-8"))["rows"][0]


def main() -> None:
    v2_payload = json.loads(
        (ROOT.parents[1] / ".codex/research-loop/artifacts/darvin-aligned-quality-20260816/v2-full60-reaudit12.json")
        .read_text(encoding="utf-8")
    )
    v2_rows = {row["id"]: row for row in v2_payload["rows"]}
    r24_rows = {case_id: _latest_row(case_id) for case_id in CASE_IDS}
    cases = {item["id"]: item for item in _load_jsonl(CASES_PATH)}
    annotations = {item["case_id"]: item for item in _load_jsonl(ANNOTATIONS_PATH)}
    pairs = build_pairs(CASE_IDS, r24_rows, v2_rows, cases, annotations)
    out = ROOT.parents[1] / ".codex/research-loop/artifacts/darvin-aligned-quality-20260816/blind-judge-pairs.json"
    _write_json_atomic(out, {
        "case_ids": CASE_IDS,
        "r24_runs": {cid: r24_rows[cid].get("elapsed_s") for cid in CASE_IDS},
        "pairs": pairs,
    })
    print(f"wrote {len(pairs)} pairs -> {out}")
    for pair in pairs:
        print(pair["pair_id"], pair["kind"], f"left={len(pair['left'])}chars right={len(pair['right'])}chars")


if __name__ == "__main__":
    main()

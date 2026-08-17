#!/usr/bin/env python3
"""R27 task 0: retire inspected holdout cases and promote shadow reserves.

Moves the six inspected cases from holdout_v2 into regression fixtures
(tests/fixtures/holdout_retired/) and promotes shadow_v3 reserves to keep
holdout_v2 at 60 cases (15 per scenario).  Rebuilds both split manifests
and split_manifest.json counts; verify.py is the gate.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "validation_sets/public_resume_holdout"
RETIRED = ROOT / "tests/fixtures/holdout_retired"

RETIRE = ["HV2-S1-009", "HV2-S1-012", "HV2-S2-008", "HV2-S2-010", "HV2-S2-012", "HV2-S3-009"]
PROMOTE = {  # scenario -> count
    "scenario1": 2,
    "scenario2": 3,
    "scenario3": 1,
}
NEW_IDS = {
    "scenario1": ["HV2-S1-016", "HV2-S1-017"],
    "scenario2": ["HV2-S2-016", "HV2-S2-017", "HV2-S2-018"],
    "scenario3": ["HV2-S3-016"],
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _move_file(rel_from: str, split_from: str, split_to: str, dry: bool) -> str:
    """Move a split-relative file to the other split, return new rel path."""

    if not rel_from:
        return rel_from
    marker = f"{split_from}/"
    if marker not in rel_from:
        return rel_from
    rel_to = rel_from.replace(marker, f"{split_to}/", 1)
    src = HOLDOUT / split_from / rel_from.split(marker, 1)[1].split("/", 1)[0]  # unused
    src = HOLDOUT / rel_from.replace("../", "").replace(f"validation_sets/public_resume_holdout/", "")
    dst = HOLDOUT / rel_to.replace("../", "").replace("validation_sets/public_resume_holdout/", "")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))
    return rel_to


def main() -> None:
    hv_cases = _load_jsonl(HOLDOUT / "holdout_v2/cases.jsonl")
    hv_anns = _load_jsonl(HOLDOUT / "holdout_v2/annotations.jsonl")
    sv_cases = _load_jsonl(HOLDOUT / "shadow_v3/cases.jsonl")
    sv_anns = _load_jsonl(HOLDOUT / "shadow_v3/annotations.jsonl")

    # 1. Retire contaminated cases into regression fixtures.
    RETIRED.mkdir(parents=True, exist_ok=True)
    retired_cases = [c for c in hv_cases if c["id"] in RETIRE]
    retired_anns = [a for a in hv_anns if a["case_id"] in RETIRE]
    assert len(retired_cases) == len(RETIRE) == len(retired_anns)
    _write_jsonl(RETIRED / "cases.jsonl", retired_cases)
    _write_jsonl(RETIRED / "annotations.jsonl", retired_anns)

    hv_cases = [c for c in hv_cases if c["id"] not in RETIRE]
    hv_anns = [a for a in hv_anns if a["case_id"] not in RETIRE]

    # 2. Promote shadow reserves with fresh holdout IDs.
    promoted_cases: list[dict] = []
    promoted_anns: list[dict] = []
    used_shadow_ids: list[str] = []
    for scenario, count in PROMOTE.items():
        pool = [c for c in sv_cases if c["scenario"] == scenario and c["id"] not in used_shadow_ids]
        pool.sort(key=lambda c: c["id"])
        for case, new_id in zip(pool[:count], NEW_IDS[scenario]):
            old_id = case["id"]
            used_shadow_ids.append(old_id)
            case = dict(case)
            ann = dict(next(a for a in sv_anns if a["case_id"] == old_id))
            for key in ("cv_path", "target_jd_file_path"):
                if case.get(key):
                    case[key] = _move_file(case[key], "shadow_v3", "holdout_v2", dry=False)
            case["id"] = new_id
            ann_sources = []
            for source in ann["sources"]:
                source = dict(source)
                rel = source["canonical_text_path"]
                if rel.startswith("shadow_v3/"):
                    new_rel = rel.replace("shadow_v3/", "holdout_v2/", 1)
                    src = HOLDOUT / rel
                    dst = HOLDOUT / new_rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.exists():
                        shutil.move(str(src), str(dst))
                    source["canonical_text_path"] = new_rel
                ann_sources.append(source)
            ann = {**ann, "case_id": new_id, "sources": ann_sources}
            promoted_cases.append(case)
            promoted_anns.append(ann)

    hv_cases.extend(promoted_cases)
    hv_anns.extend(promoted_anns)
    sv_cases = [c for c in sv_cases if c["id"] not in used_shadow_ids]
    sv_anns = [a for a in sv_anns if a["case_id"] not in used_shadow_ids]

    for rows, name in ((hv_cases, "holdout cases"), (hv_anns, "holdout annotations")):
        ids = [r.get("id") or r.get("case_id") for r in rows]
        assert len(ids) == len(set(ids)), name
    _write_jsonl(HOLDOUT / "holdout_v2/cases.jsonl", hv_cases)
    _write_jsonl(HOLDOUT / "holdout_v2/annotations.jsonl", hv_anns)
    _write_jsonl(HOLDOUT / "shadow_v3/cases.jsonl", sv_cases)
    _write_jsonl(HOLDOUT / "shadow_v3/annotations.jsonl", sv_anns)

    # 3. Rebuild per-split manifests (file hashes) and split_manifest counts.
    for split in ("holdout_v2", "shadow_v3"):
        root = HOLDOUT / split
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append({
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                })
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = files
        split_cases = _load_jsonl(root / "cases.jsonl")
        manifest["case_count"] = len(split_cases)
        manifest["scenario_counts"] = dict(sorted(Counter(c["scenario"] for c in split_cases).items()))
        manifest["industry_counts"] = dict(sorted(Counter(c["industry"] for c in split_cases).items()))
        manifest["input_profile_counts"] = dict(sorted(Counter(c["input_profile"] for c in split_cases).items()))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sm = json.loads((HOLDOUT / "split_manifest.json").read_text(encoding="utf-8"))
    for split in ("holdout_v2", "shadow_v3"):
        split_cases = _load_jsonl(HOLDOUT / split / "cases.jsonl")
        entry = sm["splits"][split]
        entry["case_count"] = len(split_cases)
        entry["scenario_counts"] = dict(sorted(Counter(c["scenario"] for c in split_cases).items()))
        entry["industry_counts"] = dict(sorted(Counter(c["industry"] for c in split_cases).items()))
        entry["input_profile_counts"] = dict(sorted(Counter(c["input_profile"] for c in split_cases).items()))
        entry["source_resume_ids"] = sorted(
            str(c["provenance"].get("resume_id")) for c in split_cases if c["provenance"].get("resume_id")
        )
        entry["source_jd_ids"] = sorted(
            str(c["provenance"].get("jd_id")) for c in split_cases if c["provenance"].get("jd_id")
        )
    (HOLDOUT / "split_manifest.json").write_text(json.dumps(sm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"retired {len(retired_cases)} -> {RETIRED}")
    print(f"promoted {len(promoted_cases)}: {[c['id'] for c in promoted_cases]}")
    print(f"holdout_v2={len(hv_cases)} shadow_v3={len(sv_cases)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run controlled content-profile ablations on the managed GPU 3-6 cluster."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
CLUSTER = ROOT / "tools" / "local_eval_cluster.sh"
RUNNER = ROOT / "acceptance_testset" / "run_api_testset.py"
FACTUAL_EVALUATOR = ROOT / "tools" / "evaluate_development_results.py"
NARRATIVE_EVALUATOR = ROOT / "tools" / "evaluate_narrative_quality.py"
DEFAULT_CASES = ROOT / "validation_sets" / "narrative_development" / "cases.jsonl"
DEFAULT_PROFILES = (
    "f507_compatible",
    "ledger_shadow",
    "local_repair",
    "fact_compiler",
    "candidate",
)
KNOWN_PROFILES = (*DEFAULT_PROFILES, "quality_v2")
PORTS = (18085, 18086, 18087, 18088)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _profile_env(profile: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "LOCAL_EVAL_GPU_IDS": "3,4,5,6",
        "LOCAL_EVAL_PIPELINE_PROFILE": profile,
        # The named profile resolves the effective compiler mode.  This value
        # remains explicit because the cluster verifies the process contract.
        "LOCAL_EVAL_FACT_COMPILER_MODE": "on",
    })
    return env


def _merge_shards(shards: list[Path], output: Path) -> None:
    rows: list[dict] = []
    for shard in shards:
        rows.extend(json.loads(shard.read_text(encoding="utf-8")).get("rows", []))
    elapsed = [float(row.get("elapsed_s") or 0.0) for row in rows]
    ok = [row for row in rows if row.get("ok")]
    payload = {
        "summary": {
            "case_count": len(rows),
            "success_count": len(ok),
            "failure_count": len(rows) - len(ok),
            "average_elapsed_s": round(statistics.mean(elapsed), 3) if elapsed else 0.0,
            "max_elapsed_s": round(max(elapsed), 3) if elapsed else 0.0,
        },
        "rows": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_profile(profile: str, cases_path: Path, output_root: Path) -> None:
    env = _profile_env(profile)
    profile_dir = output_root / profile
    profile_dir.mkdir(parents=True, exist_ok=False)
    _run(["bash", str(CLUSTER), "restart"], env=env)

    cases = _load_cases(cases_path)
    shards: list[Path] = []
    processes: list[tuple[subprocess.Popen, Path]] = []
    for index, port in enumerate(PORTS):
        case_ids = [str(case["id"]) for position, case in enumerate(cases) if position % len(PORTS) == index]
        if not case_ids:
            continue
        shard = profile_dir / f"shard-{index}.json"
        log_path = profile_dir / f"shard-{index}.log"
        shards.append(shard)
        command = [
            str(PYTHON), str(RUNNER),
            "--base-url", f"http://127.0.0.1:{port}",
            "--cases", str(cases_path),
            "--out", str(shard),
            "--timeout", "480",
            "--case-id", ",".join(case_ids),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        process._ablation_log_handle = log_handle  # type: ignore[attr-defined]
        processes.append((process, log_path))

    failures: list[str] = []
    for process, log_path in processes:
        return_code = process.wait()
        process._ablation_log_handle.close()  # type: ignore[attr-defined]
        if return_code:
            failures.append(f"{log_path.name}: exit {return_code}")
    if failures:
        raise RuntimeError(f"{profile} failed: {', '.join(failures)}")

    merged = profile_dir / "merged.json"
    _merge_shards(shards, merged)
    factual = profile_dir / "factual-report.json"
    narrative = profile_dir / "narrative-report.json"
    factual_command = [str(PYTHON), str(FACTUAL_EVALUATOR), "--cases", str(cases_path)]
    narrative_command = [str(PYTHON), str(NARRATIVE_EVALUATOR)]
    for shard in shards:
        factual_command.extend(["--input", str(shard)])
        narrative_command.extend(["--input", str(shard)])
    factual_command.extend(["--output", str(factual)])
    narrative_command.extend(["--output", str(narrative)])
    _run(factual_command, env=env)
    _run(narrative_command, env=env)

    manifest = {
        "profile": profile,
        "cases": str(cases_path),
        "case_ids": [str(case["id"]) for case in cases],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts": [str(path.name) for path in [merged, factual, narrative]],
    }
    (profile_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = sorted(set(profiles) - set(KNOWN_PROFILES))
    if unknown:
        raise SystemExit(f"unknown profiles: {', '.join(unknown)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    for profile in profiles:
        run_profile(profile, args.cases.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()

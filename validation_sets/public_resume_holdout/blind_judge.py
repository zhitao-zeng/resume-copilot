#!/usr/bin/env python3
"""Blind V2/R24 submit-readiness judge harness (R24 validation ladder step 3).

Ten frozen anonymous pairs: eight real V2/R24 comparisons plus one V2/V2 and
one R24/R24 control.  Every pair is judged three times in each A/B and B/A
order (60 judgments).  Per-order majority; disagreement across orders is a
tie.  The judge sees only the source/request and two anonymous outputs and
answers submit-ready, preference and one reason.

Results are a Goodhart monitor ONLY: they never enter local gates, never
become repair labels, and ten pairs are too small for score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]

JUDGE_VERSION = "blind-submit-judge-r24-v3"
ROUTER_URL = "http://127.0.0.1:4001/v1/responses"
JUDGE_MODEL = "deepseek-v4-flash"
REPEATS_PER_ORDER = 3

CASES_PATH = ROOT / "holdout_v2/cases.jsonl"
ANNOTATIONS_PATH = ROOT / "holdout_v2/annotations.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(handle.name, path)


# ---------------------------------------------------------------------------
# Anonymous rendering of one candidate output
# ---------------------------------------------------------------------------

def _resume_text(resume_data: dict[str, Any]) -> str:
    """Faithful deterministic text view of the rendered resume (no version)."""

    lines: list[str] = []
    meta = resume_data.get("meta") if isinstance(resume_data.get("meta"), dict) else {}
    meta_bits = [str(meta.get(key) or "") for key in ("name", "phone", "email", "target_role")]
    header = " ".join(bit for bit in meta_bits if bit.strip())
    if header.strip():
        lines.append(header.strip())
    summary = str(resume_data.get("summary") or "").strip()
    if summary:
        lines.append(f"个人总结：{summary}")
    section_labels = (
        ("experience", "工作/实习经历"),
        ("projects", "项目经历"),
        ("education", "教育经历"),
        ("skills", "专业技能"),
        ("additional_sections", "补充信息"),
    )
    for key, label in section_labels:
        value = resume_data.get(key)
        if not value:
            continue
        lines.append(f"【{label}】")
        if isinstance(value, list):
            for record in value:
                if not isinstance(record, dict):
                    lines.append(f"- {record}")
                    continue
                head = " ".join(
                    str(record.get(field) or "").strip()
                    for field in ("company", "organization", "role", "period", "school", "degree", "name")
                    if str(record.get(field) or "").strip()
                )
                if head:
                    lines.append(head)
                for bullet in record.get("bullets") or []:
                    lines.append(f"- {bullet}")
        elif isinstance(value, dict):
            for sub, items in value.items():
                if isinstance(items, list) and items:
                    lines.append(f"{sub}：" + "、".join(str(item) for item in items))
    framework = resume_data.get("framework")
    if isinstance(framework, dict):
        lines.append("【结构化待填写简历框架】")
        for section in framework.get("sections") or []:
            if isinstance(section, dict):
                fields = "、".join(str(f) for f in section.get("fields") or [])
                lines.append(f"- {section.get('title', section.get('key', ''))}：{fields}")
    return "\n".join(lines)


def _source_text(case: dict[str, Any], annotation: dict[str, Any]) -> str:
    parts: list[str] = []
    for source in annotation.get("sources") or []:
        kind = source.get("kind")
        if kind not in {"cv", "query"}:
            continue
        label = "原简历" if kind == "cv" else "用户输入"
        units = [unit.get("text", "") for unit in source.get("units") or [] if unit.get("text")]
        if units:
            parts.append(f"--- {label} ---\n" + "\n".join(units))
    jd = case.get("target_jd")
    if jd:
        parts.append(f"--- 目标岗位 JD（仅供参考，不是候选人事实） ---\n{jd}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

JUDGE_INSTRUCTION = """你是简历优化产品的盲评评审。你会看到用户的原始材料（原简历/输入/JD）和两份匿名优化产出（A 和 B）。
只看产出本身，不要猜测它们来自哪个系统或版本。
请回答三个问题：
1. A 是否达到可直接投递水平？（ready_a: yes/no）
2. B 是否达到可直接投递水平？（ready_b: yes/no）
3. 你更偏好哪一份？（preference: A/B/tie），并给一个最主要的原因（reason，一句话，<=40字）。
评价要点：事实不得编造（不得出现原材料没有的公司/岗位/时间/数字/学历/证书）、内容完整度、表达专业连贯、回复是否有用。
严格只输出如下 JSON，不要输出其他内容：
{"ready_a": "yes|no", "ready_b": "yes|no", "preference": "A|B|tie", "reason": "..."}"""


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[^{}]*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_once(
    source: str,
    output_a: str,
    output_b: str,
    *,
    router_url: str = ROUTER_URL,
    model: str = JUDGE_MODEL,
    timeout: float = 120.0,
) -> dict[str, Any]:
    user = (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"===== 用户原始材料 =====\n{source}\n\n"
        f"===== 产出 A =====\n{output_a}\n\n"
        f"===== 产出 B =====\n{output_b}"
    )
    body = json.dumps({
        "model": model,
        "input": user,
        "max_output_tokens": 2000,
    }).encode("utf-8")
    request = urllib.request.Request(
        router_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = ""
    for item in payload.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                text += content.get("text", "")
    parsed = _extract_json(text)
    if not parsed:
        return {"judge_error": f"unparseable_response:{text[:160]}"}
    verdict = {
        "ready_a": str(parsed.get("ready_a") or "").strip().lower(),
        "ready_b": str(parsed.get("ready_b") or "").strip().lower(),
        "preference": str(parsed.get("preference") or "").strip().upper(),
        "reason": str(parsed.get("reason") or "").strip()[:120],
    }
    if verdict["preference"] not in {"A", "B", "TIE"}:
        return {"judge_error": f"bad_preference:{text[:160]}"}
    return verdict


# ---------------------------------------------------------------------------
# Pair assembly + protocol
# ---------------------------------------------------------------------------

def build_pairs(
    case_ids: list[str],
    r24_rows: dict[str, dict[str, Any]],
    v2_rows: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    seed: int = 20260817,
) -> list[dict[str, Any]]:
    """Assemble 8 real pairs + V2/V2 and R24/R24 controls (deterministic)."""

    pairs: list[dict[str, Any]] = []
    for case_id in case_ids:
        v2_raw = v2_rows[case_id]["raw"]
        r24_raw = r24_rows[case_id]["raw"]
        pairs.append({
            "pair_id": f"real:{case_id}",
            "kind": "real",
            "case_id": case_id,
            "scenario": cases[case_id].get("scenario"),
            "industry": cases[case_id].get("industry"),
            "source": _source_text(cases[case_id], annotations[case_id]),
            "left": _resume_text(v2_raw.get("resume_data") or {}) + "\n\n--- 回复 ---\n" + str(v2_raw.get("reply_text") or ""),
            "right": _resume_text(r24_raw.get("resume_data") or {}) + "\n\n--- 回复 ---\n" + str(r24_raw.get("reply_text") or ""),
            "left_version": "v2",
            "right_version": "r24",
        })
    control_case = case_ids[0]
    v2_raw = v2_rows[control_case]["raw"]
    v2_text = _resume_text(v2_raw.get("resume_data") or {}) + "\n\n--- 回复 ---\n" + str(v2_raw.get("reply_text") or "")
    pairs.append({
        "pair_id": f"control:v2v2:{control_case}",
        "kind": "control_v2v2",
        "case_id": control_case,
        "scenario": cases[control_case].get("scenario"),
        "source": _source_text(cases[control_case], annotations[control_case]),
        "left": v2_text,
        "right": v2_text,
        "left_version": "v2",
        "right_version": "v2",
    })
    r24_raw = r24_rows[control_case]["raw"]
    r24_text = _resume_text(r24_raw.get("resume_data") or {}) + "\n\n--- 回复 ---\n" + str(r24_raw.get("reply_text") or "")
    pairs.append({
        "pair_id": f"control:r24r24:{control_case}",
        "kind": "control_r24r24",
        "case_id": control_case,
        "scenario": cases[control_case].get("scenario"),
        "source": _source_text(cases[control_case], annotations[control_case]),
        "left": r24_text,
        "right": r24_text,
        "left_version": "r24",
        "right_version": "r24",
    })
    return pairs


def run_protocol(
    pairs: list[dict[str, Any]],
    *,
    judge,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Run the full 60-judgment protocol with deterministic A/B assignment."""

    rng = random.Random(seed)
    judgments: list[dict[str, Any]] = []
    for pair in pairs:
        # One base assignment per pair; B/A order is its mirror.
        base_left_is_a = rng.random() < 0.5
        for order_index, left_is_a in enumerate((base_left_is_a, not base_left_is_a)):
            for repeat in range(REPEATS_PER_ORDER):
                if left_is_a:
                    output_a, output_b = pair["left"], pair["right"]
                    a_version, b_version = pair["left_version"], pair["right_version"]
                else:
                    output_a, output_b = pair["right"], pair["left"]
                    a_version, b_version = pair["right_version"], pair["left_version"]
                verdict = judge(pair["source"], output_a, output_b)
                record = {
                    "pair_id": pair["pair_id"],
                    "kind": pair["kind"],
                    "case_id": pair["case_id"],
                    "order": "AB" if order_index == 0 else "BA",
                    "repeat": repeat,
                    "a_version": a_version,
                    "b_version": b_version,
                    **verdict,
                }
                # Preference mapped into version space (v2 / r24 / tie).
                if "judge_error" not in verdict:
                    preferred = {"A": a_version, "B": b_version}.get(verdict["preference"])
                    record["preferred_version"] = preferred if preferred else "tie"
                judgments.append(record)
    return {"judgments": judgments}


def _majority(votes: list[str]) -> str | None:
    if not votes:
        return None
    counts = {value: votes.count(value) for value in set(votes)}
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None  # no majority
    return top[0][0]


def aggregate(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for pair_id in sorted({item["pair_id"] for item in judgments}):
        items = [item for item in judgments if item["pair_id"] == pair_id]
        errors = [item for item in items if "judge_error" in item]
        order_results: dict[str, Any] = {}
        for order in ("AB", "BA"):
            order_items = [item for item in items if item["order"] == order and "judge_error" not in item]
            votes = [item["preferred_version"] for item in order_items]
            order_results[order] = {
                "votes": votes,
                "majority": _majority(votes),
            }
        ab, ba = order_results["AB"]["majority"], order_results["BA"]["majority"]
        if items[0]["kind"].startswith("control"):
            # Controls carry identical content on both sides; a version-level
            # majority in either order or in the pool is position bias.
            pooled = [
                item["preferred_version"] for item in items if "judge_error" not in item
            ]
            pooled_majority = _majority(pooled)
            order_majorities = {
                order_results[order]["majority"]
                for order in ("AB", "BA")
                if order_results[order]["majority"] in {"v2", "r24"}
            }
            final = pooled_majority or (sorted(order_majorities)[0] if order_majorities else "tie")
        elif ab is None or ba is None:
            final = "unstable"
        elif ab != ba:
            final = "tie"  # order disagreement collapses to tie per protocol
        else:
            final = ab
        ready: dict[str, int] = {}
        for item in items:
            if "judge_error" in item:
                continue
            for side, version_key in (("ready_a", "a_version"), ("ready_b", "b_version")):
                version = item[version_key]
                if item.get(side) == "yes":
                    ready[version] = ready.get(version, 0) + 1
        pairs[pair_id] = {
            "kind": items[0]["kind"],
            "case_id": items[0]["case_id"],
            "order_results": order_results,
            "final": final,
            "ready_votes": ready,
            "judge_errors": len(errors),
            "reasons": [item.get("reason") for item in items if item.get("reason")][:6],
        }
    real = {pid: p for pid, p in pairs.items() if pid.startswith("real:")}
    controls = {pid: p for pid, p in pairs.items() if pid.startswith("control:")}
    summary = {
        "real_pair_outcomes": {
            outcome: sum(1 for p in real.values() if p["final"] == outcome)
            for outcome in ("r24", "v2", "tie", "unstable")
        },
        "control_preference": {
            pid: p["final"] for pid, p in controls.items()
        },
        "judge_bias_warning": any(
            p["final"] not in {"tie", "unstable"} for p in controls.values()
        ),
        "total_judgments": len(judgments),
        "judge_errors": sum(p["judge_errors"] for p in pairs.values()),
    }
    return {"pairs": pairs, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="JSON file with prebuilt pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    result = run_protocol(pairs["pairs"], judge=judge_once, seed=args.seed)
    aggregated = aggregate(result["judgments"])
    payload = {
        "judge_version": JUDGE_VERSION,
        "judge_model": JUDGE_MODEL,
        "protocol": {
            "repeats_per_order": REPEATS_PER_ORDER,
            "orders": ("AB", "BA"),
            "order_disagreement": "tie",
            "role": "Goodhart monitor only; not a gate, not repair labels",
        },
        "pairs_sha256": _sha256_text(Path(args.pairs).read_text(encoding="utf-8")),
        **aggregated,
        "judgments": result["judgments"],
    }
    _write_json_atomic(Path(args.output), payload)
    summary = payload["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

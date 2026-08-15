#!/usr/bin/env python3
"""Aggregate expression and reply diagnostics from API evaluation results.

This evaluator intentionally does not assign a synthetic resume score.  It
measures observable presentation properties and is paired with the independent
atomic-factuality evaluator for promotion decisions.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EVALUATOR_VERSION = "narrative-quality-1.0"
RECORD_SECTIONS = (
    "experience",
    "projects",
    "research",
    "campus_experience",
    "activities",
    "teaching",
    "training",
)
ACTION = re.compile(
    r"(?:主导|统筹|牵头|独立负责|负责|组织|推动|管理|设计|开发|构建|实现|制定|"
    r"参与|协助|支持|配合|开展|执行|处理|分析|研究|维护|跟进|诊疗|授课|复核|"
    r"运营|完成|建立|带领|监控|安排|合作|改进|优化|交付|撰写|编制|协调|"
    r"led|managed|built|developed|designed|implemented|analyzed|created|"
    r"coordinated|supported|delivered|conducted)"
)
METHOD = re.compile(
    r"(?:通过|使用|采用|基于|借助|运用|利用|结合|按照|依据|协同|协调|"
    r"\b(?:using|through|with|via|by)\b)",
    re.IGNORECASE,
)
DELIVERABLE = re.compile(
    r"(?:输出|形成|产出|撰写|编制|制定|设计|开发|构建|搭建|建立|交付|上线|"
    r"发布|完成|报告|方案|清单|系统|模型|课程|流程|制度|"
    r"\b(?:report|plan|system|model|course|process|program|product)\b)",
    re.IGNORECASE,
)
RESULT = re.compile(
    r"(?:(?:提升|提高|增长|降低|减少|缩短|节省|达到|达成|获得|获奖|录用|覆盖|"
    r"支持)[^，。；;]{0,32}(?:\d+(?:\.\d+)?\s*(?:%|万|人|名|次|项|个|条|"
    r"篇|例|台|套|元|万元)|目标|要求|标准)|"
    r"\d+(?:\.\d+)?\s*(?:%|万|人|名|次|项|个|条|篇|例|台|套|元|万元)|"
    r"\b(?:increased|reduced|improved|saved|grew|achieved)\b)",
    re.IGNORECASE,
)
CONTEXT_ONLY = re.compile(
    r"^(?:行业|业务领域|业务范围|项目背景|工作地点|部门|团队|客户类型)\s*[:：]"
)
INSTRUCTION_TARGET = re.compile(
    r"(?:优化|修改|调整|改写|完善|生成|制作|处理).{0,8}(?:简历|CV)|"
    r"(?:简历|CV).{0,8}(?:优化|修改|调整|改写|完善|生成|制作|处理)",
    re.IGNORECASE,
)
REPLY_COMPONENTS = {
    "generation_direction": ("生成方向总结", "生成方向"),
    "missing_information": ("缺失信息", "待补充信息"),
    "job_advice": ("岗位匹配与建议", "针对岗位的建议", "岗位建议"),
    "conflicts": ("时间或内容冲突", "冲突检查"),
}


def _compact_length(value: str) -> int:
    return len(re.sub(r"\s+", "", str(value or "")))


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _response_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get("resume_data"), dict):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("raw"), dict):
        nested = raw["raw"]
        if isinstance(nested.get("resume_data"), dict):
            return nested
    if isinstance(row.get("resume_data"), dict):
        return row
    return None


def _load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("rows", []) if isinstance(payload, dict) else []
        if not isinstance(values, list):
            raise ValueError(f"{path}: rows must be a list")
        rows.extend(item for item in values if isinstance(item, dict))
    return rows


def _record_bullets(resume_data: dict[str, Any]) -> list[tuple[str, int, list[str]]]:
    result: list[tuple[str, int, list[str]]] = []
    for section in RECORD_SECTIONS:
        records = resume_data.get(section) or []
        if not isinstance(records, list):
            continue
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            bullets = [
                str(value).strip()
                for value in (record.get("bullets") or [])
                if str(value or "").strip()
            ]
            result.append((section, index, bullets))
    return result


def _bullet_features(value: str) -> dict[str, bool | int]:
    length = _compact_length(value)
    action = bool(ACTION.search(value))
    method = bool(METHOD.search(value))
    deliverable = bool(DELIVERABLE.search(value))
    result = bool(RESULT.search(value))
    context_only = bool(CONTEXT_ONLY.match(value.strip()))
    # A short action-led responsibility may be thin, but it is still a valid
    # standalone resume unit. Track it through ``lt18`` and accomplishment
    # density; reserve ``fragment`` for noun/layout pieces that cannot stand as
    # a claim by themselves.
    fragment = length < 18 and not action and not context_only
    accomplishment = action and (method or deliverable or result)
    return {
        "length": length,
        "action": action,
        "method": method,
        "deliverable": deliverable,
        "result": result,
        "context_only": context_only,
        "fragment": fragment,
        "accomplishment": accomplishment,
    }


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    bullet_lengths: list[int] = []
    summary_lengths: list[int] = []
    reply_lengths: list[int] = []
    missing_counts: list[int] = []
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        response = _response_from_row(row)
        if response is None:
            counters["invalid_response_count"] += 1
            continue
        counters["case_count"] += 1
        scenario = str(response.get("scenario") or row.get("scenario") or "unknown")
        scenario_counter = by_scenario[scenario]
        scenario_counter["case_count"] += 1
        resume_data = response.get("resume_data") or {}
        framework_mode = isinstance(resume_data.get("framework"), dict)
        if framework_mode:
            counters["framework_case_count"] += 1
            scenario_counter["framework_case_count"] += 1

        for _section, _index, bullets in _record_bullets(resume_data):
            counters["record_count"] += 1
            scenario_counter["record_count"] += 1
            if len(bullets) > 6:
                counters["records_over_six_bullets"] += 1
                scenario_counter["records_over_six_bullets"] += 1
            for bullet in bullets:
                features = _bullet_features(bullet)
                counters["bullet_count"] += 1
                scenario_counter["bullet_count"] += 1
                bullet_lengths.append(int(features["length"]))
                for key in (
                    "action", "method", "deliverable", "result", "context_only",
                    "fragment", "accomplishment",
                ):
                    if features[key]:
                        counters[f"bullet_{key}_count"] += 1
                        scenario_counter[f"bullet_{key}_count"] += 1
                if int(features["length"]) < 18:
                    counters["bullet_lt18_count"] += 1
                    scenario_counter["bullet_lt18_count"] += 1

        summary = str(resume_data.get("summary") or "").strip()
        summary_lengths.append(len(summary))
        if not summary and not framework_mode:
            counters["nonframework_empty_summary_count"] += 1
            scenario_counter["nonframework_empty_summary_count"] += 1
        if len(summary) > 260:
            counters["summary_over260_count"] += 1
            scenario_counter["summary_over260_count"] += 1
        target_role = str((resume_data.get("meta") or {}).get("target_role") or "")
        if INSTRUCTION_TARGET.search(target_role) or INSTRUCTION_TARGET.search(summary):
            counters["instruction_as_target_count"] += 1
            scenario_counter["instruction_as_target_count"] += 1

        reply = str(response.get("reply_text") or "").strip()
        reply_lengths.append(len(reply))
        if len(reply) > 2000:
            counters["reply_over2000_count"] += 1
            scenario_counter["reply_over2000_count"] += 1
        for component, markers in REPLY_COMPONENTS.items():
            if any(marker in reply for marker in markers):
                counters[f"reply_{component}_count"] += 1
                scenario_counter[f"reply_{component}_count"] += 1
        match = re.search(r"缺失或待补充信息（(\d+)项）", reply)
        missing_counts.append(int(match.group(1)) if match else 0)

    bullet_count = counters["bullet_count"]
    case_count = counters["case_count"]

    def rate(numerator: str, denominator: int) -> float:
        return round(counters[numerator] / max(denominator, 1), 4)

    scenario_summary: dict[str, Any] = {}
    for scenario, values in sorted(by_scenario.items()):
        scenario_bullets = values["bullet_count"]
        scenario_cases = values["case_count"]
        scenario_summary[scenario] = {
            **dict(values),
            "bullet_fragment_rate": round(
                values["bullet_fragment_count"] / max(scenario_bullets, 1), 4,
            ),
            "bullet_lt18_rate": round(
                values["bullet_lt18_count"] / max(scenario_bullets, 1), 4,
            ),
            "bullet_action_rate": round(
                values["bullet_action_count"] / max(scenario_bullets, 1), 4,
            ),
            "bullet_accomplishment_rate": round(
                values["bullet_accomplishment_count"] / max(scenario_bullets, 1), 4,
            ),
            "reply_contract_rate": round(
                sum(
                    values[f"reply_{component}_count"]
                    for component in REPLY_COMPONENTS
                ) / max(scenario_cases * len(REPLY_COMPONENTS), 1),
                4,
            ),
        }

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "summary": {
            **dict(counters),
            "bullet_length_p25": round(_percentile(bullet_lengths, 0.25), 2),
            "bullet_length_median": round(statistics.median(bullet_lengths), 2) if bullet_lengths else 0.0,
            "bullet_length_p75": round(_percentile(bullet_lengths, 0.75), 2),
            "bullet_fragment_rate": rate("bullet_fragment_count", bullet_count),
            "bullet_lt18_rate": rate("bullet_lt18_count", bullet_count),
            "bullet_action_rate": rate("bullet_action_count", bullet_count),
            "bullet_method_rate": rate("bullet_method_count", bullet_count),
            "bullet_deliverable_rate": rate("bullet_deliverable_count", bullet_count),
            "bullet_result_rate": rate("bullet_result_count", bullet_count),
            "bullet_accomplishment_rate": rate("bullet_accomplishment_count", bullet_count),
            "record_over_six_rate": rate("records_over_six_bullets", counters["record_count"]),
            "summary_length_median": round(statistics.median(summary_lengths), 2) if summary_lengths else 0.0,
            "reply_length_median": round(statistics.median(reply_lengths), 2) if reply_lengths else 0.0,
            "reply_length_p95": round(_percentile(reply_lengths, 0.95), 2),
            "missing_fields_mean": round(statistics.mean(missing_counts), 2) if missing_counts else 0.0,
            "reply_contract_rate": round(
                sum(counters[f"reply_{component}_count"] for component in REPLY_COMPONENTS)
                / max(case_count * len(REPLY_COMPONENTS), 1),
                4,
            ),
        },
        "by_scenario": scenario_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = evaluate_rows(_load_rows(Path(value) for value in args.input))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

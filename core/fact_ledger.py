"""Fact Ledger: immutable fact layer extracted from parse output.

After parse (structured_resume_from_text), this module extracts a FactLedger —
entities, bullets, meta — that all subsequent steps reference but never modify.
The extraction includes the recovery logic previously in _repair_common_parse_errors
so that the ledger is as complete as parse+repair combined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from server_runtime import (
    ACTION_WORDS,
    RESPONSIBILITY_WORDS,
    TECH_KEYWORDS,
    logger,
)


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FactEntity:
    """An immutable entity (company/school/degree/etc.) from the CV."""
    kind: str          # "company" | "school" | "degree" | "major" | "role" | "project_name"
    value: str         # verbatim from CV text
    source_span: str   # surrounding context snippet for verification


@dataclass(frozen=True)
class FactBullet:
    """One bullet from an experience or project entry."""
    id: str             # "exp_0_b0" or "proj_1_b0"
    source_text: str    # original bullet text (immutable)
    context: str        # company/name + period of the parent entry
    entities: tuple[str, ...]   # entity values appearing in this bullet
    metrics: tuple[str, ...]    # numbers/percentages/metrics already present
    has_action: bool
    has_result: bool
    missing_info: bool  # lacks quantification → optimizer must not add numbers


@dataclass(frozen=True)
class FactLedger:
    """The single source of truth after parse. All downstream steps reference this."""
    entities: dict[tuple[str, str], FactEntity]   # (kind, value_lower) → entity
    bullets: list[FactBullet]
    meta: dict[str, str]   # name, email, phone, target_role
    raw_text: str          # original CV text for fact tracing


# ── Entity Extraction ─────────────────────────────────────────────────────────

def _extract_named_entities(text: str) -> dict[str, set[str]]:
    """Extract potential company/school names from text using heuristics.
    Mirrors resume_validator._extract_named_entities but returns (kind, value) style sets.
    """
    companies: set[str] = set()
    schools: set[str] = set()

    name_patterns = [
        r"[A-Za-z一-鿿]{2,40}(?:有限公司|集团|公司|大学|学院|研究所|医院|中心|学校|教育|科技)",
        r"[一-鿿]{3,30}(?:大学|学院|医院|学校|集团|公司|研究院|实验室)",
    ]

    _verb_prefixes = frozenset({
        "实习", "参与", "负责", "完成", "撰写", "协助", "推动", "推进", "主导",
        "开发", "构建", "建立", "管理", "运营", "分析", "设计", "优化", "提供",
        "组织", "筹备", "开展",
    })

    for pattern in name_patterns:
        for m in re.finditer(pattern, text):
            cleaned = m.group().strip()
            if len(cleaned) > 30:
                continue
            if cleaned and any(cleaned.startswith(vp) or vp in cleaned[:6] for vp in _verb_prefixes):
                continue
            if len(cleaned) >= 2:
                cleaned_lower = cleaned.lower()
                if any(kw in cleaned_lower for kw in ("大学", "学院", "学校", "研究院")):
                    schools.add(cleaned_lower)
                else:
                    companies.add(cleaned_lower)

    return {"companies": companies, "schools": schools}


def _strip_company_suffix(name: str) -> str:
    """Remove common suffixes for fuzzy matching, e.g. 中国银行股份有限公司 → 中国银行."""
    return re.sub(
        r"(股份有限公司|有限责任公司|有限公司|集团|实验室|研究院|中心|（.*）|\(.*\))$",
        "", name
    ).strip()


# ── Bullet Collection ──────────────────────────────────────────────────────────

def _normalize_text_items(value: Any) -> list[str]:
    """Convert any value shape (str/list/dict) to a flat list of text strings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_text_items(item))
        return result
    return []


def _collect_bullets_from_entry(
    entry: dict[str, Any], entry_type: str, entry_idx: int
) -> list[dict[str, Any]]:
    """Collect bullets from one experience or project entry.

    Returns list of {text, bullet_index} dicts.
    """
    bullets: list[str] = []
    for key in ("bullets", "function_description", "result_description",
                "responsibilities", "achievements", "description"):
        val = entry.get(key)
        bullets.extend(_normalize_text_items(val))

    # Deduplicate by first 60 chars
    deduped: list[str] = []
    seen: set[str] = set()
    for b in bullets:
        b = re.sub(r"\s+", " ", b).strip()
        if not b:
            continue
        prefix = b[:60]
        if prefix not in seen:
            seen.add(prefix)
            deduped.append(b)

    prefix = "exp" if entry_type == "experience" else "proj"
    return [
        {"text": b, "bullet_index": i + 1, "id": f"{prefix}_{entry_idx}_b{i}"}
        for i, b in enumerate(deduped)
    ]


# ── Metrics + Action/Result Detection ─────────────────────────────────────────

_METRIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|ms|s|分钟|小时|天|倍|w|万|k|qps|tps|fps|mb|gb|"
    r"条|次|人|个|万元|客户|学生|病例|日活|月活|元|分|篇|项|台|套)",
    re.IGNORECASE,
)


def _extract_metrics(text: str) -> tuple[str, ...]:
    """Extract all metric expressions from a text."""
    return tuple(m.group() for m in _METRIC_PATTERN.finditer(text))


def _has_action_verb(text: str, extra_action: frozenset[str] = frozenset()) -> bool:
    """Check if text contains an action verb."""
    text_lower = text.lower()
    if any(w in text_lower for w in ACTION_WORDS):
        return True
    if any(w in text_lower for w in RESPONSIBILITY_WORDS):
        return True
    return False


def _has_result_indicator(text: str) -> bool:
    """Check if text contains a result/accomplishment indicator."""
    if _METRIC_PATTERN.search(text):
        return True
    result_verbs = {
        "提升", "降低", "减少", "增加", "改善", "完成", "交付", "上线",
        "通过", "覆盖", "解决", "输出", "达成", "实现", "缩短", "提高",
        "improved", "reduced", "increased", "achieved", "delivered",
        "缩短", "节省", "从", "优化至",
    }
    text_set = set(text.lower().split()) if text else set()
    if text_set & {w.lower() for w in result_verbs}:
        return True
    # Chinese result patterns
    result_patterns = [
        r"(?:提升|降低|减少|缩短|节省|提高|改善|完成|实现|达成|从\d)[^，。；;]{3,40}",
        r"(?:improved|reduced|increased|achieved|delivered|from|by)\s[^\.]{3,40}",
    ]
    for pat in result_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# ── Ledger Construction ────────────────────────────────────────────────────────

def build_ledger(
    resume_data: dict[str, Any],
    raw_text: str,
    *,
    run_repair: bool = True,
) -> FactLedger:
    """Build a FactLedger from parsed resume_data and the original CV text.

    Args:
        resume_data: output from structured_resume_from_text (or after normalize)
        raw_text: the original CV text (generation_text)
        run_repair: if True, run _repair_common_parse_errors before extraction
    """
    if not isinstance(resume_data, dict):
        return FactLedger(entities={}, bullets=[], meta={}, raw_text=raw_text)

    # ── Run repair (merge parse+repair recovery) ──
    if run_repair:
        try:
            from resume_copilot_service import _repair_common_parse_errors
            _repair_common_parse_errors(resume_data, raw_text)
        except Exception as exc:
            logger.warning("FactLedger repair skipped: %s", exc)

    # ── Entity extraction ──
    # Rule-based extraction from raw_text
    named = _extract_named_entities(raw_text)
    orig_companies = named.get("companies", set())
    orig_schools = named.get("schools", set())

    entities: dict[tuple[str, str], FactEntity] = {}

    def _add_entity(kind: str, value: str, source_text: str) -> None:
        value = str(value or "").strip()
        if not value or len(value) < 2:
            return
        key = (kind, value.lower())
        if key not in entities:
            # Verify entity has a source anchor in raw_text.
            # Entities parsed by LLM but absent from the original CV must
            # NOT be registered as facts — they are at risk of fabrication.
            if value.lower() not in raw_text.lower():
                return
            # Source span: find the value in raw_text
            idx = source_text.lower().find(value.lower())
            span = ""
            if idx >= 0:
                start = max(0, idx - 20)
                end = min(len(source_text), idx + len(value) + 40)
                span = source_text[start:end]
            else:
                span = value  # fallback, though this shouldn't happen now
            entities[key] = FactEntity(kind=kind, value=value, source_span=span)

    # Collect entities from resume_data fields
    for exp in resume_data.get("experience", []) or []:
        if not isinstance(exp, dict):
            continue
        for field, kind in (("company", "company"), ("role", "role")):
            val = str(exp.get(field, "")).strip()
            if val:
                _add_entity(kind, val, raw_text)

    for edu in resume_data.get("education", []) or []:
        if not isinstance(edu, dict):
            continue
        for field, kind in (("school", "school"), ("degree", "degree"), ("major", "major")):
            val = str(edu.get(field, "")).strip()
            if val:
                _add_entity(kind, val, raw_text)

    for proj in resume_data.get("projects", []) or []:
        if not isinstance(proj, dict):
            continue
        for field, kind in (("name", "project_name"), ("company", "company")):
            val = str(proj.get(field, "")).strip()
            if val:
                _add_entity(kind, val, raw_text)

    # ── Entity validation (spot-check ~5 entities against raw_text) ──
    _NON_ENTITY_VALUES = {"未明确", "不限", "无", "暂无", "未知", "其他", "其它", "未提供", ""}
    _entity_values = list(entities.values())
    _sample = _entity_values[:min(5, len(_entity_values))]
    missed = 0
    for ent in _sample:
        if ent.value in _NON_ENTITY_VALUES:
            continue
        norm_val = ent.value.lower()
        norm_src = raw_text.lower()
        # Strip suffixes for fuzzy match
        stripped = _strip_company_suffix(norm_val)
        if norm_val not in norm_src and (not stripped or stripped not in norm_src):
            # Try partial match (e.g. core 3 chars of company name)
            if len(norm_val) >= 3 and norm_val[:3] not in norm_src:
                missed += 1
                logger.warning(
                    "FactLedger entity may not exist in raw text: kind=%s value=%s",
                    ent.kind, ent.value,
                )
    if missed > 0:
        logger.warning(
            "FactLedger entity validation: %d/%d sampled entities not found in raw_text",
            missed, len(_sample),
        )

    # ── Bullet extraction ──
    bullets: list[FactBullet] = []

    # From experience
    exp_list = resume_data.get("experience", [])
    if isinstance(exp_list, list):
        for i, exp in enumerate(exp_list):
            if not isinstance(exp, dict):
                continue
            company = str(exp.get("company", "")).strip()
            role = str(exp.get("role", "")).strip()
            period = str(exp.get("period", "")).strip()
            context = f"{company} | {role} | {period}".strip(" |")
            if not context:
                context = f"experience_{i}"

            exp_bullets = _collect_bullets_from_entry(exp, "experience", i)
            for bd in exp_bullets:
                text = bd["text"]
                metrics = _extract_metrics(text)
                b_entities: list[str] = []
                for ent in entities.values():
                    if ent.value.lower() in text.lower():
                        b_entities.append(ent.value)

                bullets.append(FactBullet(
                    id=bd["id"],
                    source_text=text,
                    context=context,
                    entities=tuple(b_entities),
                    metrics=metrics,
                    has_action=_has_action_verb(text),
                    has_result=_has_result_indicator(text),
                    missing_info=len(metrics) == 0,
                ))

    # From standalone projects
    proj_list = resume_data.get("projects", [])
    if isinstance(proj_list, list):
        for i, proj in enumerate(proj_list):
            if not isinstance(proj, dict):
                continue
            name = str(proj.get("name", "")).strip()
            company = str(proj.get("company", "")).strip()
            period = str(proj.get("period", "")).strip()
            context = f"{name} | {company} | {period}".strip(" |")
            if not context:
                context = f"project_{i}"

            proj_bullets = _collect_bullets_from_entry(proj, "project", i)
            for bd in proj_bullets:
                text = bd["text"]
                metrics = _extract_metrics(text)
                b_entities: list[str] = []
                for ent in entities.values():
                    if ent.value.lower() in text.lower():
                        b_entities.append(ent.value)

                bullets.append(FactBullet(
                    id=bd["id"],
                    source_text=text,
                    context=context,
                    entities=tuple(b_entities),
                    metrics=metrics,
                    has_action=_has_action_verb(text),
                    has_result=_has_result_indicator(text),
                    missing_info=len(metrics) == 0,
                ))

    # ── Meta extraction ──
    meta_raw = resume_data.get("meta", {})
    meta: dict[str, str] = {}
    if isinstance(meta_raw, dict):
        for k, v in meta_raw.items():
            if isinstance(v, str) and v.strip():
                meta[k] = v.strip()

    # ── Coverage self-check ──
    _edu_count = len([e for e in resume_data.get("education", []) or [] if isinstance(e, dict)])
    _exp_count = len([e for e in resume_data.get("experience", []) or [] if isinstance(e, dict)])
    _proj_count = len([p for p in resume_data.get("projects", []) or [] if isinstance(p, dict)])
    logger.info(
        "FactLedger built | entities=%d | bullets=%d | meta_fields=%d | "
        "edu=%d | exp=%d | proj=%d | raw_text_chars=%d",
        len(entities), len(bullets), len(meta),
        _edu_count, _exp_count, _proj_count, len(raw_text),
    )

    return FactLedger(
        entities=entities,
        bullets=bullets,
        meta=meta,
        raw_text=raw_text,
    )

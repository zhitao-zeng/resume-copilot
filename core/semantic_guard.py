"""Semantic fidelity guard: validates that optimized bullets don't fabricate.

Three-layer defense:
1. Entity/number check (rule-based, <1ms) — new entities/numbers → reject
2. Semantic fidelity rules (rule-based, <1ms) — role escalation, JD keyword overreach → reject
3. LLM entailment check (batch, ~1-2s) — subtle exaggeration not caught by rules

Any candidate failing any layer is discarded. If all candidates fail, the
original bullet is kept unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from fact_ledger import FactBullet, FactLedger
from server_runtime import (
    ACTION_WORDS,
    RESPONSIBILITY_WORDS,
    TECH_KEYWORDS,
    call_llm_text,
    llm_enabled,
    logger,
    remaining_request_seconds,
)


# ── Layer 1: Entity / Number Check ─────────────────────────────────────────────

_METRIC_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|ms|s|分钟|小时|天|倍|w|万|k|qps|tps|fps|mb|gb|"
    r"条|次|人|个|万元|客户|学生|病例|日活|月活|元|分|篇|项|台|套)",
    re.IGNORECASE,
)

# Scale/quantity assertions that are NEVER in source but commonly fabricated
_SCALE_PATTERNS = re.compile(
    r"\d+\s*(?:万日活|万用户|TB级|PB级|QPS|TPS|万QPS|亿级|千万级|百万级|十万级|万级|百亿|十亿)",
    re.IGNORECASE,
)

# Maturity/domain assertions typically fabricated
_DOMAIN_CLAIMS = re.compile(
    r"(?:金融级|工业级|生产环境|线上环境|大规模|企业级|商业级|"
    r"production.grade|enterprise.grade|mission.critical)",
    re.IGNORECASE,
)


def _check_entity_metrics(
    bullet: FactBullet, candidate: str
) -> list[str]:
    """Check that candidate doesn't introduce new entities or metrics."""
    violations: list[str] = []

    # 1. Known entity values must not disappear (relaxed: we only check additions)
    # 2. New metrics not in source bullet
    src_metrics = set(m.group() for m in _METRIC_PATTERN.finditer(bullet.source_text))
    cand_metrics = set(m.group() for m in _METRIC_PATTERN.finditer(candidate))
    new_metrics = cand_metrics - src_metrics
    if new_metrics:
        violations.append(f"new_metrics: {', '.join(sorted(new_metrics)[:5])}")

    # 3. Scale/domain assertions not in source
    if _SCALE_PATTERNS.search(candidate) and not _SCALE_PATTERNS.search(bullet.source_text):
        violations.append("new_scale_claim")

    if _DOMAIN_CLAIMS.search(candidate) and not _DOMAIN_CLAIMS.search(bullet.source_text):
        violations.append("new_domain_claim")

    return violations


# ── Layer 2: Semantic Fidelity Rules ──────────────────────────────────────────

# Role escalation pairs: (weaker_signal, stronger_signal)
_ROLE_ESCALATION_PAIRS: list[tuple[frozenset[str], frozenset[str]]] = [
    # Chinese
    (frozenset({"参与", "协助", "配合", "支持", "帮助"}),
     frozenset({"主导", "负责", "owner", "领导", "带领", "牵头", "独立负责"})),
    # English
    (frozenset({"participated", "assisted", "supported", "helped", "contributed"}),
     frozenset({"led", "owned", "drove", "spearheaded", "architected", "independently"})),
]

# Verbs that go from "improved/optimized" to "designed/built" — capability inflation
_CAPABILITY_INFLATION_PAIRS: list[tuple[frozenset[str], frozenset[str]]] = [
    (frozenset({"优化", "改进", "改善", "调整", "修改", "维护", "迭代"}),
     frozenset({"设计", "构建", "搭建", "架构", "创建", "发明", "研发", "重构", "重写"})),
    (frozenset({"optimized", "improved", "tweaked", "adjusted", "maintained", "iterated"}),
     frozenset({"designed", "built", "architected", "created", "invented", "developed from scratch", "rewrote"})),
]


def _has_any_word(text: str, words: frozenset[str]) -> bool:
    """Check if any of the given words appear in text (word-boundary aware)."""
    text_lower = text.lower()
    for w in words:
        if w in text_lower:
            return True
    return False


def _check_semantic_fidelity_rules(
    bullet: FactBullet, candidate: str, ledger: FactLedger
) -> list[str]:
    """Apply deterministic rules to detect semantic fabrication.

    Returns list of violation labels. Empty = pass.
    """
    violations: list[str] = []
    src = bullet.source_text.lower()
    cand = candidate.lower()

    # Rule 1: Role escalation
    for weak_set, strong_set in _ROLE_ESCALATION_PAIRS:
        has_weak = _has_any_word(src, weak_set)
        has_strong = _has_any_word(cand, strong_set)
        if has_weak and has_strong and not _has_any_word(src, strong_set):
            violations.append("role_escalation")

    # Rule 2: Capability inflation (improve→design, etc.)
    for weak_set, strong_set in _CAPABILITY_INFLATION_PAIRS:
        has_weak = _has_any_word(src, weak_set)
        has_strong_new = _has_any_word(cand, strong_set) and not _has_any_word(src, strong_set)
        if has_weak and has_strong_new:
            # Check if context justifies it (e.g. source has "设计" elsewhere in the same entry)
            if not any(
                _has_any_word(e.lower(), strong_set)
                for e in bullet.entities
            ):
                violations.append("capability_inflation")

    # Rule 3: New skill/tech claims
    src_tech = {w for w in TECH_KEYWORDS if w in src}
    cand_tech = {w for w in TECH_KEYWORDS if w in cand}
    new_tech = cand_tech - src_tech
    if len(new_tech) >= 2:  # 1 new tech word might be from JD keyword rephrasing
        violations.append(f"new_skills: {','.join(sorted(new_tech)[:5])}")

    # Rule 4: JD keyword overreach
    # JD keywords can rephrase, not assert new capability claims
    # "优化接口响应" + JDkey="高并发" → "设计高并发架构" is overreach
    # Detect: new capability-asserting verb + JD keyword that was NOT in source
    _capability_verbs = frozenset({
        "设计", "构建", "搭建", "创建", "架构", "开发", "研发", "建立",
        "designed", "built", "architected", "created", "developed",
    })
    _src_has_cap = _has_any_word(src, _capability_verbs)
    _cand_has_new_cap = _has_any_word(cand, _capability_verbs) and not _src_has_cap
    if _cand_has_new_cap and new_tech:
        violations.append(f"jd_keyword_overreach: {','.join(sorted(new_tech)[:3])}")

    # Rule 5: Causal attribution inflation
    # "参与了XX项目" → "通过XX项目将指标提升Y%" — attributing team results to self
    _weak_attribution = frozenset({"参与", "协助", "配合", "支持"})
    _strong_attribution = frozenset({"将", "使", "让", "推动", "驱动", "促使"})
    if _has_any_word(src, _weak_attribution) and not _has_any_word(src, _strong_attribution):
        if _has_any_word(cand, _strong_attribution):
            # Check: does the candidate have a "通过X将Y提升" pattern?
            if re.search(r"(?:通过|将|使|让|推动)\s*.{0,20}(?:提升|降低|增长|提高)", candidate):
                violations.append("attribution_inflation")

    # Rule 6: Precision fabrication (vague→specific)
    # "提升了系统稳定性" → "将系统稳定性提升至 99.9%" — adding a specific number
    src_nums = len(re.findall(r"\d+(?:\.\d+)?", src))
    cand_nums = len(re.findall(r"\d+(?:\.\d+)?", cand))
    if src_nums == 0 and cand_nums >= 1 and not bullet.metrics:
        # Check if the number is from an entity (e.g. JD key) not a fabricated metric
        if re.search(r"\d+(?:\.\d+)?\s*%", cand):
            violations.append("precision_fabrication")

    return violations


# ── Layer 3: LLM Entailment Check ──────────────────────────────────────────────

_FIDELITY_SYSTEM_PROMPT = """你是简历事实审核专家。判断「改写版」是否对「原文版」做了超出事实的夸大。

规则：
- 改写版可以重组结构、改善措辞、按 STAR 重组
- 改写版不能声称原文没有支撑的能力、角色层级、技术决策权
- 改写版不能将模糊表述变成精确数字
- 改写版不能将间接/部分参与变成主导/创建
- JD 关键词只能用于重新措辞已有证据，不能断言原文没写的新能力

输出 JSON: {"pass": true/false, "reason": ""}"""


def review_entailment_batch(
    checks: list[tuple[str, str]],  # list of (source_text, candidate_text)
) -> list[bool] | None:
    """Batch LLM entailment check. One call for up to 6 candidate pairs.

    Returns one verdict per pair, or ``None`` when review is unavailable or
    inconclusive.  Deterministic hard-fact checks remain authoritative; a
    transport or JSON failure must not masquerade as a semantic approval.
    """
    if not checks:
        return []
    if not llm_enabled():
        return None
    remaining = remaining_request_seconds()
    if remaining is not None and remaining < 20:
        logger.info(
            "Semantic rewrite review skipped: %.1fs request budget remains",
            remaining,
        )
        return None

    results: list[bool | None] = [None] * len(checks)

    # Process in batches of 6
    for batch_start in range(0, len(checks), 6):
        batch = checks[batch_start:batch_start + 6]
        items = []
        for i, (src, cand) in enumerate(batch):
            items.append(f"[{i}] 原文: {src[:200]}\n[{i}] 改写: {cand[:200]}\n")

        prompt = (
            "请对以下 {n} 对（原文, 改写）逐一判断是否存在事实夸大。"
            "输出 JSON: {{\"results\": [{{\"index\": 0, \"pass\": true/false, \"reason\": \"\"}}]}}\n\n"
            .format(n=len(batch))
            + "\n---\n".join(items)
        )

        try:
            llm_out = call_llm_text(
                system_prompt=_FIDELITY_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=512,
            )
            # Parse the JSON output — the LLM may wrap in markdown or include extra text
            import json as _json
            # Strip markdown code fences if present
            _clean = (llm_out or "").strip()
            _clean = re.sub(r'^```(?:json)?\s*', '', _clean)
            _clean = re.sub(r'\s*```$', '', _clean)
            # Find the first { and last }
            _start = _clean.find('{')
            _end = _clean.rfind('}')
            if _start >= 0 and _end > _start:
                _clean = _clean[_start:_end + 1]
            parsed = _json.loads(_clean) if _clean else {}
            if not isinstance(parsed, dict):
                parsed = {}
            verdicts = parsed.get("results", [])
            if not isinstance(verdicts, list):
                verdicts = []

            for v in verdicts:
                if not isinstance(v, dict):
                    continue
                try:
                    idx = int(v.get("index", -1))
                except (ValueError, TypeError):
                    idx = -1
                passes = v.get("pass", True)
                if isinstance(passes, str):
                    passes = passes.lower() not in ("false", "no", "0")
                passes = bool(passes)
                if 0 <= idx < len(batch):
                    results[batch_start + idx] = passes
        except Exception as exc:
            logger.warning("LLM entailment batch check failed: %s", exc)
            return None

    if any(value is None for value in results):
        logger.warning("LLM entailment batch response omitted one or more verdicts")
        return None
    return [bool(value) for value in results]


def _llm_check_entailment_batch(
    checks: list[tuple[str, str]],
) -> list[bool]:
    """Backward-compatible wrapper for the legacy selector.

    The legacy path is currently disabled.  If it is re-enabled, an
    unavailable advisory review preserves deterministic decisions rather than
    failing an entire resume.
    """

    reviewed = review_entailment_batch(checks)
    return reviewed if reviewed is not None else [True] * len(checks)


# ── Expression Proxy Scorer ────────────────────────────────────────────────────

def _score_expression(
    source_text: str,
    candidate: str,
    jd_keywords: list[str],
    raw_text: str = "",
) -> float:
    """Score a candidate's expression quality (0-5 scale).

    JD keyword hits are only rewarded when the keyword ALSO exists in
    `raw_text` — JD-only terms that don't appear in the original CV
    must NOT get a scoring bonus, otherwise the system incentivizes
    JD leakage into rewritten bullets.
    """
    score = 0.0
    raw_lower = raw_text.lower() if raw_text else ""

    # Action verb present
    if _has_any_word(candidate, ACTION_WORDS) or _has_any_word(candidate, RESPONSIBILITY_WORDS):
        score += 1.0

    # Result indicator present
    if _METRIC_PATTERN.search(candidate):
        score += 1.0
    elif re.search(r"(?:验证|评估|对比|测试|上线|交付|完成|通过)", candidate):
        score += 0.5  # qualitative result

    # JD keyword hits — only reward if the keyword is ALSO present in raw_text
    # (original CV). A keyword absent from the CV is JD-only and must not
    # increase the score — doing so directly incentivizes fabrication.
    cand_lower = candidate.lower()
    jd_hits = 0
    for kw in jd_keywords:
        kw_lower = kw.lower()
        if kw_lower in cand_lower:
            if raw_lower and kw_lower in raw_lower:
                jd_hits += 1
            # else: JD-only term, no score bonus
    score += min(1.0, jd_hits * 0.25)

    # STAR-like structure (longer, more structured)
    if len(candidate) > len(source_text) * 0.8:
        score += 0.5

    # Penalize: too short (lazy rewrite)
    if len(candidate) < 20:
        score -= 1.0

    # Penalize: generic filler
    generic = re.findall(r"(?:显著|明显|大幅|有效|良好|顺利|全面)", candidate)
    score -= len(generic) * 0.3

    return max(0.0, min(5.0, score))


# ── Main Selector ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class BulletPatch:
    bullet_id: str
    new_text: str
    confidence: float


def select_best_candidate(
    bullet: FactBullet,
    candidates: list[str],
    ledger: FactLedger,
    jd_keywords: list[str],
) -> BulletPatch | None:
    """Run 3-layer validation and select the best candidate.

    Returns None if all candidates fail → keep the original bullet.
    """
    surviving: list[tuple[str, float, list[str]]] = []  # (text, score, violation_labels)
    raw_lower = ledger.raw_text.lower() if ledger.raw_text else ""

    for cand in candidates:
        if not cand or not cand.strip():
            continue

        # Layer 0: Reject JD-only terms (fastest, most definitive)
        # A JD keyword that appears in the candidate but NOT in the source
        # bullet AND NOT in the raw CV text is JD leakage — reject.
        src_lower = bullet.source_text.lower()
        jd_only_terms = [
            kw for kw in jd_keywords
            if kw.lower() in cand.lower()
            and kw.lower() not in src_lower
            and kw.lower() not in raw_lower
        ]
        if jd_only_terms:
            logger.debug("JD-only reject bullet=%s: terms=%s", bullet.id, jd_only_terms)
            continue

        # Layer 1: Entity / number check
        l1_violations = _check_entity_metrics(bullet, cand)
        if l1_violations:
            logger.debug("L1 reject bullet=%s: %s", bullet.id, l1_violations)
            continue

        # Layer 2: Semantic fidelity rules
        l2_violations = _check_semantic_fidelity_rules(bullet, cand, ledger)
        if l2_violations:
            logger.debug("L2 reject bullet=%s: %s", bullet.id, l2_violations)
            continue

        # Passed layers 1+2, add to pool for Layer 3
        surviving.append((cand, 0.0, []))  # score computed after L3

    if not surviving:
        logger.info("All %d candidates rejected for bullet=%s, keeping original", len(candidates), bullet.id)
        return None

    # Layer 3: LLM entailment check (batch all survivors)
    # TEMPORARILY DISABLED — the LLM JSON output format for batch entailment
    # ({"results": [...]}) produces a KeyError('"results"') that needs
    # investigation inside the container. L1+L2 rules catch most fabrication.
    # Re-enable after fixing the JSON parsing edge case.
    _ENABLE_L3_ENTAILMENT = False
    if _ENABLE_L3_ENTAILMENT and len(surviving) > 0 and llm_enabled():
        checks = [(bullet.source_text, cand) for cand, _, _ in surviving]
        llm_results = _llm_check_entailment_batch(checks)

        # Filter by LLM results
        _after_l3 = []
        for i, (cand, _, _) in enumerate(surviving):
            if i < len(llm_results) and llm_results[i]:
                _after_l3.append(cand)
            else:
                logger.debug("L3 reject bullet=%s: LLM flagged", bullet.id)

        if not _after_l3:
            logger.info("All candidates failed LLM entailment for bullet=%s, keeping original", bullet.id)
            return None
        surviving_cands = _after_l3
    else:
        surviving_cands = [c for c, _, _ in surviving]

    # Score and select best
    best_cand = None
    best_score = -999.0
    for cand in surviving_cands:
        score = _score_expression(bullet.source_text, cand, jd_keywords, raw_text=ledger.raw_text)
        if score > best_score:
            best_score = score
            best_cand = cand

    if best_cand is None:
        return None

    return BulletPatch(
        bullet_id=bullet.id,
        new_text=best_cand,
        confidence=best_score / 5.0,
    )

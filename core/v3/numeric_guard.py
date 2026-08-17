"""OCR numeric sanity guard (R25).

OCR digit loss turns ``2004`` into ``204`` and ``50`` into ``5``; preserving
such facts verbatim ships a fabrication against the truth (the R24 full60
gate failed on exactly this class).  The guard quarantines suspect numeric
facts fail-closed: they are not written into the resume and are surfaced in
the reply for user confirmation instead of silently dropped or guessed.

Detection (deterministic, no model):
- implausible/reversed/over-long year ranges (``204-2011``)
- short three-digit "years" in date context
- low-confidence OCR numerics (PP-Structure confidence below threshold)

A quarantined fact keeps its audit trail: ``eligible=False`` with the reason
recorded in the pipeline report and the user-facing reply.
"""
from __future__ import annotations

import re
from typing import Any

from .contracts import FactGraph, FactUnit
from .realizer import _NUMBER_RE


LOW_CONFIDENCE_THRESHOLD = 0.80

# A year-range pair like 2004-2011 / 2019年1月 - 2021年6月 / 2019.1-2021.6
_RANGE_RE = re.compile(
    r"(?P<y1>\d{3,4})\s*(?:年|[./-]\s*\d{1,2}\s*月?)?\s*[-—~至到–]+\s*"
    r"(?P<y2>\d{4}|至今|现在|今)(?:\s*(?:年|[./-]\s*\d{1,2}\s*月?))?"
)
# A bare short "year": 3 digits directly followed by a date unit or range sep
_SHORT_YEAR_RE = re.compile(r"(?<![\d.])(\d{3})(?=\s*(?:年|[./-]\d|[-—~至到–]))")
_VALID_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_MAX_PERIOD_YEARS = 45
# A "date" with all digits lost to OCR is a degenerate shell, not a fact.
# It must contain at least one date unit glyph; a bare "-" or "·" separator
# is punctuation, not a date shell.
_EMPTY_DATE_SHELL_RE = re.compile(r"^(?=.*[年月日])[\s年月日·./\-—~至到—]+$")


def _year_ok(value: str) -> bool:
    return bool(_VALID_YEAR_RE.fullmatch(value)) and 1900 <= int(value) <= 2099


def _suspicion_reasons(fact: FactUnit) -> list[str]:
    text = fact.text
    reasons: list[str] = []
    if _EMPTY_DATE_SHELL_RE.fullmatch(text.strip()) and not re.search(r"\d", text):
        reasons.append("empty_date_shell")
    for match in _RANGE_RE.finditer(text):
        y1, y2 = match.group("y1"), match.group("y2")
        if not _year_ok(y1):
            reasons.append(f"implausible_year:{y1}")
        if y2 not in {"至今", "现在", "今"}:
            if not _year_ok(y2):
                reasons.append(f"implausible_year:{y2}")
            elif _year_ok(y1):
                span = int(y2) - int(y1)
                if span < 0:
                    reasons.append(f"reversed_period:{y1}-{y2}")
                elif span > _MAX_PERIOD_YEARS:
                    reasons.append(f"overlong_period:{y1}-{y2}")
    for match in _SHORT_YEAR_RE.finditer(text):
        token = match.group(1)
        # 3-digit tokens adjacent to date markers are truncated years.
        if not _year_ok(token) and f"implausible_year:{token}" not in reasons:
            reasons.append(f"truncated_year:{token}")
    if (
        fact.confidence < LOW_CONFIDENCE_THRESHOLD
        and fact.source_type == "cv"
        and _NUMBER_RE.findall(text)
    ):
        reasons.append(f"low_confidence_numeric:{fact.confidence:.2f}")
    return reasons


def find_suspect_numeric_facts(graph: FactGraph) -> list[dict[str, Any]]:
    suspects: list[dict[str, Any]] = []
    for fact in graph.facts:
        if not fact.eligible or fact.classification != "fact":
            continue
        if fact.source_type not in {"cv", "query"}:
            continue
        reasons = _suspicion_reasons(fact)
        if reasons:
            suspects.append({
                "fact_id": fact.fact_id,
                "text": fact.text,
                "reasons": reasons,
            })
    return suspects


def quarantine_suspect_numeric_facts(graph: FactGraph) -> list[dict[str, Any]]:
    """Fail closed: suspect numeric facts leave the resume, not the audit.

    The semantic stage may already have split a suspect fact into atoms, so
    quarantine cascades to every atom derived from a suspect base fact.
    """

    suspects = find_suspect_numeric_facts(graph)
    suspect_ids = {item["fact_id"] for item in suspects}
    for fact in graph.facts:
        if fact.fact_id in suspect_ids or (fact.base_fact_id or "") in suspect_ids:
            fact.eligible = False
    return suspects


__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "find_suspect_numeric_facts",
    "quarantine_suspect_numeric_facts",
]

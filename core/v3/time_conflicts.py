"""Cross-record period overlap detection (rubric 确认类：时间重合).

The audit's conflict hints only compare facts inside one record; two
experiences whose date ranges overlap were never checked.  Detection is
deterministic date parsing — calendar tokens (year/month digits, month
names, 至今/present) are domain-inherent closed sets, not observed badcase
words.  A hit only adds a confirmation prompt; no content is altered.

Overlap is strict (shared span beyond a single boundary month): adjacent
jobs where one ends the month the next begins are normal, and parallel
part-time work is legitimate — hence the wording asks, never accuses.
"""
from __future__ import annotations

import re
from typing import Any

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# (?!\d) keeps the month group from swallowing a second year's digits
# ("2019 - 2021" must parse as two year points, not month 20).
_CN_POINT = re.compile(r"(\d{4})\s*[年./\-]\s*(\d{1,2}(?!\d))?")
_EN_POINT = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})",
    re.IGNORECASE,
)
_ONGOING = re.compile(r"至今|现在|当前|present|now|current", re.IGNORECASE)
_ONGOING_KEY = 999912
_MAX_REPORTED = 3


def _date_points(text: str) -> list[int]:
    """Every yyyymm date token in positional order (missing month -> 01)."""

    points: list[tuple[int, int]] = []
    for match in _CN_POINT.finditer(text):
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        if 1900 <= year <= 2100 and 1 <= month <= 12:
            points.append((match.start(), year * 100 + month))
    for match in _EN_POINT.finditer(text):
        year = int(match.group(2))
        month = _MONTH_NAMES[match.group(1)[:3].lower()]
        if 1900 <= year <= 2100:
            points.append((match.start(), year * 100 + month))
    return [value for _, value in sorted(points)]


def parse_period(text: str) -> tuple[int, int] | None:
    """(start_yyyymm, end_yyyymm) or None when the range cannot be trusted.

    Two or more date tokens: first is the start, last is the end.  A single
    token counts only with an explicit ongoing marker.  Reversed ranges are
    OCR suspects and return None rather than a guess.
    """

    value = str(text or "")
    points = _date_points(value)
    if len(points) >= 2:
        start, end = points[0], points[-1]
        return (start, end) if start <= end else None
    if len(points) == 1 and _ONGOING.search(value):
        return (points[0], _ONGOING_KEY)
    return None


def experience_overlaps(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Confirmation prompts for experience pairs with overlapping periods."""

    parsed: list[tuple[int, int, str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        period = str(record.get("period") or "")
        bounds = parse_period(period)
        if bounds is None:
            continue
        label = str(record.get("company") or record.get("role") or f"第{index + 1}段经历")
        parsed.append((bounds[0], bounds[1], label, period))
    overlaps: list[dict[str, str]] = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            start = max(parsed[i][0], parsed[j][0])
            end = min(parsed[i][1], parsed[j][1])
            if start < end:
                overlaps.append({
                    "field": "time_overlap",
                    "description": (
                        f"「{parsed[i][2]}」（{parsed[i][3]}）与「{parsed[j][2]}」（{parsed[j][3]}）"
                        "时间重叠，请确认是否为并行或兼职经历。"
                    ),
                })
            if len(overlaps) >= _MAX_REPORTED:
                return overlaps
    return overlaps


__all__ = ["experience_overlaps", "parse_period"]

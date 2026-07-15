"""JSON candidate extraction and tolerant parsing helpers."""

import ast
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from json_repair import repair_json as _repair_json_lib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _repair_json_lib = None


def _extract_json_candidates(content: str) -> list[str]:
    text = str(content or "")
    candidates: list[str] = []

    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL):
        payload = match.group(1).strip()
        if payload:
            candidates.append(payload)

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append(text[start_obj : end_obj + 1].strip())

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidates.append(text[start_arr : end_arr + 1].strip())

    # Extract balanced top-level JSON objects/arrays from mixed text.
    stack: list[str] = []
    in_string = False
    escaped = False
    start_idx: Optional[int] = None
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            if not stack:
                start_idx = idx
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                continue
            top = stack[-1]
            if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                stack.pop()
                if not stack and start_idx is not None:
                    segment = text[start_idx : idx + 1].strip()
                    if segment:
                        candidates.append(segment)
                    start_idx = None
            else:
                # Reset on mismatched bracket
                stack.clear()
                start_idx = None

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item[:2000]
        if not item or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _repair_json_candidate(candidate: str) -> str:
    text = str(candidate or "").strip().replace("\ufeff", "")
    # Keep full-width quotes as-is. Replacing them with `"` can break valid JSON strings.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # Heuristic: escape likely unescaped double quotes inside JSON strings.
    # This targets common model outputs like: "foo "bar" baz".
    buf: list[str] = []
    in_string = False
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if not in_string:
            if ch == '"':
                in_string = True
            buf.append(ch)
            i += 1
            continue

        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\":
            buf.append(ch)
            escaped = True
            i += 1
            continue

        if ch == '"':
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            next_ch = text[j] if j < n else ""
            # Valid close quote usually followed by one of these delimiters.
            if next_ch in {",", "}", "]", ":", ""}:
                in_string = False
                buf.append(ch)
            else:
                buf.append('\\"')
            i += 1
            continue

        buf.append(ch)
        i += 1

    text = "".join(buf)
    return text.strip()


def _json_error_with_context(payload: str, exc: Exception) -> str:
    if not isinstance(exc, json.JSONDecodeError):
        return str(exc)
    pos = max(0, min(exc.pos, len(payload)))
    left = max(0, pos - 80)
    right = min(len(payload), pos + 80)
    snippet = payload[left:right].replace("\n", "\\n")
    return f"{exc} | near: {snippet}"


def _load_json_object(candidate: str) -> Optional[dict[str, Any]]:
    text = str(candidate or "").strip()
    if not text:
        return None

    # Accept the first valid JSON object even when the model appends
    # trailing content (e.g. duplicated JSON blocks or explanations).
    try:
        parsed_prefix, end = json.JSONDecoder().raw_decode(text)
        if isinstance(parsed_prefix, dict):
            trailing = text[end:].strip()
            if trailing:
                logger.warning(
                    "Model JSON output contains trailing content (%d chars); using the first JSON object",
                    len(trailing),
                )
            return parsed_prefix
    except Exception:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback for Python-like dict strings emitted by models.
    try:
        python_like = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
        python_like = re.sub(r"\bfalse\b", "False", python_like, flags=re.IGNORECASE)
        python_like = re.sub(r"\bnull\b", "None", python_like, flags=re.IGNORECASE)
        parsed = ast.literal_eval(python_like)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def parse_json_content(content: str) -> dict[str, Any]:
    candidates = _extract_json_candidates(content)
    last_error: Optional[str] = None

    for candidate in candidates:
        parsed = _load_json_object(candidate)
        if parsed is not None:
            return parsed

        repaired = _repair_json_candidate(candidate)
        parsed = _load_json_object(repaired)
        if parsed is not None:
            return parsed

        if _repair_json_lib is not None:
            try:
                repaired_by_lib = _repair_json_lib(repaired, return_objects=False)
                parsed = _load_json_object(str(repaired_by_lib))
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        try:
            json.loads(repaired)
        except Exception as exc:
            last_error = _json_error_with_context(repaired, exc)

    if last_error:
        logger.warning("Failed to parse model JSON output: %s", last_error)
    else:
        logger.warning("Failed to parse model JSON output: no JSON candidate found")
    return {}

"""Input normalization helpers shared by the resume entry points."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlsplit


_URL_PREFIX_RE = re.compile(r"^(https?://[^\s，。；;、]+)(.*)$", re.IGNORECASE | re.DOTALL)
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
    "p", "pre", "section", "table", "td", "th", "tr", "ul",
}
_IGNORED_TAGS = {
    "script", "style", "svg", "noscript", "template", "nav", "footer",
    "header", "aside", "form",
}


def split_url_and_text(value: str) -> tuple[Optional[str], str]:
    """Split a pure URL or a common ``URL + pasted JD`` input.

    Returning ``(None, original)`` means that the value is ordinary text.  A
    URL is accepted only when it has an HTTP(S) scheme and a real hostname.
    Chinese punctuation is deliberately treated as a separator because it is
    not valid evidence that the entire form field is a URL.
    """

    raw = str(value or "").strip()
    match = _URL_PREFIX_RE.match(raw)
    if not match:
        return None, raw

    candidate = match.group(1).rstrip("),]}>\"'")
    remainder = raw[len(candidate):].lstrip(" \t\r\n，。；;、:：-")
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None, raw
    return candidate, remainder.strip()


def is_pure_http_url(value: str) -> bool:
    url, trailing = split_url_and_text(value)
    return bool(url and not trailing)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif not self._ignored_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data)


def html_to_visible_text(payload: bytes, *, max_chars: int = 20_000) -> str:
    """Extract useful visible text from an HTML response without CSS/JS noise."""

    decoded = payload.decode("utf-8", errors="ignore")
    parser = _VisibleTextParser()
    try:
        parser.feed(decoded)
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<(?:script|style|svg|noscript)\b[^>]*>.*?</(?:script|style|svg|noscript)>", "", decoded, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", "\n", text)

    clean_lines: list[str] = []
    for raw_line in html.unescape(text).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        # CSS/JS fragments that escaped malformed HTML are not useful JD text.
        punctuation = sum(line.count(token) for token in ("{", "}", ";", "=>"))
        if punctuation >= 3 and punctuation * 4 > len(line):
            continue
        if re.match(r"^[.#][\w-]+\s*[{,]", line):
            continue
        if clean_lines and clean_lines[-1] == line:
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines)[:max_chars].strip()


def merge_fetched_jd(fetched_text: str, user_text: str) -> str:
    """Combine fetched and inline JD content with explicit precedence."""

    fetched = str(fetched_text or "").strip()
    supplied = str(user_text or "").strip()
    if supplied and fetched:
        return (
            "【用户补充的岗位信息（优先）】\n"
            f"{supplied}\n\n"
            "【链接页面正文（仅作补充）】\n"
            f"{fetched}"
        )
    return supplied or fetched

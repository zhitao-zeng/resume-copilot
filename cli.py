#!/usr/bin/env python3
"""Local CLI for the current resume-copilot pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from resume_copilot_service import resume_copilot_service
from server_runtime import DEFAULT_TEMPLATE, MAX_FILE_SIZE


class _MemoryUpload:
    def __init__(self, path: Path):
        self.filename = path.name
        self._content = path.read_bytes()
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._content) - self._offset
        chunk = self._content[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


def _text(value: str | None, file_value: str | None = None) -> str:
    if file_value:
        return Path(file_value).read_text(encoding="utf-8", errors="replace").strip()
    return str(value or "").strip()


def _print_result(response, as_json: bool) -> None:
    payload = response.model_dump()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"score: {response.score:.1f}")
    print(response.reply_text)
    for output_format, path in response.files.items():
        if path:
            print(f"{output_format}: {path}")
    if response.missing_fields:
        print("missing: " + "、".join(item.get("label", item.get("field", "")) for item in response.missing_fields))


async def _run(args: argparse.Namespace) -> None:
    query = _text(getattr(args, "profile", None) or getattr(args, "content", None))
    target = _text(getattr(args, "target", None))
    if target:
        query = "\n".join(part for part in (query, f"目标说明：{target}") if part)
    if not query and args.command == "generate" and not sys.stdin.isatty():
        query = sys.stdin.read().strip()

    cv_upload = None
    if getattr(args, "file", None):
        path = Path(args.file).resolve()
        if not path.is_file():
            raise SystemExit(f"file not found: {path}")
        if path.stat().st_size > MAX_FILE_SIZE:
            raise SystemExit(f"file exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB")
        cv_upload = _MemoryUpload(path)

    if not query and cv_upload is None:
        raise SystemExit("provide --profile/--content/--file or pipe profile text via stdin")

    response = await resume_copilot_service(
        query=query,
        cv=cv_upload,
        cv_template=None,
        target_jd=_text(getattr(args, "jd", None), getattr(args, "jd_file", None)),
        target_jd_file=None,
        target_jd_url=None,
        jd_text=None,
        jd_url=None,
        template=getattr(args, "template", None) or DEFAULT_TEMPLATE,
    )
    _print_result(response, args.json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or optimize a resume")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate from a personal profile")
    generate.add_argument("--scenario", choices=["profile_only", "jd_only"], help=argparse.SUPPRESS)
    generate.add_argument("--profile", "-p")
    generate.add_argument("--jd", "-j")
    generate.add_argument("--jd-file")
    generate.add_argument("--target", "-t")
    generate.add_argument("--template", default=DEFAULT_TEMPLATE)
    generate.add_argument("--json", action="store_true")

    audit = subparsers.add_parser("audit", help="optimize an existing resume")
    audit.add_argument("--scenario", choices=["optimize_with_jd", "profile_with_target"], help=argparse.SUPPRESS)
    audit.add_argument("--file", "-f")
    audit.add_argument("--content", "-c")
    audit.add_argument("--jd", "-j")
    audit.add_argument("--jd-file")
    audit.add_argument("--target", "-t")
    audit.add_argument("--template", default=DEFAULT_TEMPLATE)
    audit.add_argument("--json", action="store_true")

    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()

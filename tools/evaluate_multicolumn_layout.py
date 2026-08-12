#!/usr/bin/env python3
"""Evaluate OCR reading-order reconstruction on geometry fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

from resume_io import _reconstruct_ocr_reading_order  # noqa: E402


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _adjacency_accuracy(expected: list[str], actual: list[str]) -> float:
    pairs = list(zip(expected, expected[1:]))
    if not pairs:
        return 1.0
    positions = {value: index for index, value in enumerate(actual)}
    correct = sum(
        positions.get(left, -2) + 1 == positions.get(right, -1)
        for left, right in pairs
    )
    return correct / len(pairs)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    boxes = []
    texts = []
    for block in case["blocks"]:
        x1, y1, x2, y2 = block["bbox"]
        boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        texts.append((block["text"], 1.0))
    expected = list(case["expected"])
    actual = _reconstruct_ocr_reading_order(
        np.asarray(boxes, dtype=np.float32),
        tuple(texts),
        img_width=int(case["width"]),
        img_height=int(case["height"]),
    )
    conserved = Counter(expected) == Counter(actual)
    return {
        "id": case["id"],
        "group": case["group"],
        "industry": case["industry"],
        "exact": actual == expected,
        "conserved": conserved,
        "order_accuracy": _lcs_length(expected, actual) / max(1, len(expected)),
        "adjacency_accuracy": _adjacency_accuracy(expected, actual),
        "expected": expected,
        "actual": actual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "multicolumn_layout_cases.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = [evaluate_case(case) for case in fixture["cases"]]
    summary = {
        "dataset_revision": fixture["revision"],
        "case_count": len(cases),
        "exact_cases": sum(case["exact"] for case in cases),
        "text_conservation_cases": sum(case["conserved"] for case in cases),
        "mean_order_accuracy": round(
            sum(case["order_accuracy"] for case in cases) / max(1, len(cases)), 4
        ),
        "mean_adjacency_accuracy": round(
            sum(case["adjacency_accuracy"] for case in cases) / max(1, len(cases)), 4
        ),
        "groups": {
            group: {
                "cases": sum(case["group"] == group for case in cases),
                "exact": sum(case["group"] == group and case["exact"] for case in cases),
                "mean_order_accuracy": round(
                    sum(case["order_accuracy"] for case in cases if case["group"] == group)
                    / max(1, sum(case["group"] == group for case in cases)),
                    4,
                ),
            }
            for group in sorted({case["group"] for case in cases})
        },
        "cases": cases,
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

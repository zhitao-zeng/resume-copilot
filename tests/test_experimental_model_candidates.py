from __future__ import annotations

import hashlib
import json

from experimental_model_candidates import (
    OCR_LAYOUT_REGION_SEPARATOR,
    cached_layout_regions,
    cached_query_spans,
    region_aware_ocr_order,
)
from resume_io import _reconstruct_ocr_reading_order
from source_adapter import build_source_bundle, candidate_blocks


def _write_cache(path, *, kind: str, items: dict) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "kind": kind, "items": items}),
        encoding="utf-8",
    )


def test_ppuie_cache_can_enable_an_exact_query_fact(monkeypatch, tmp_path) -> None:
    query = "Profile payload: Northwind Analytics"
    start = query.index("Northwind")
    end = len(query)
    digest = hashlib.sha256(query.encode()).hexdigest()
    cache = tmp_path / "ppuie.json"
    _write_cache(cache, kind="ppuie-query-spans", items={digest: {
        "query_sha256": digest,
        "spans": [{
            "start": start,
            "end": end,
            "text": query[start:end],
            "groups": ["工作经历"],
        }],
    }})

    baseline = build_source_bundle("", query, "")
    assert candidate_blocks(baseline) == []

    monkeypatch.setenv("QUERY_EXTRACTOR", "ppuie")
    monkeypatch.setenv("QUERY_EXTRACTOR_CACHE_FILE", str(cache))
    candidate = build_source_bundle("", query, "")
    assert [block.text for block in candidate_blocks(candidate)] == [query]
    assert all(fact.verbatim_text in query for fact in candidate.fact_units)


def test_ppuie_cache_rejects_non_exact_or_direction_spans(monkeypatch, tmp_path) -> None:
    query = "Please generate my resume"
    digest = hashlib.sha256(query.encode()).hexdigest()
    cache = tmp_path / "ppuie-directions.json"
    _write_cache(cache, kind="ppuie-query-spans", items={digest: {
        "query_sha256": digest,
        "spans": [
            {"start": 0, "end": len(query), "text": query, "groups": ["工作经历"]},
            {"start": 0, "end": 6, "text": "wrong!", "groups": ["工作经历"]},
        ],
    }})
    monkeypatch.setenv("QUERY_EXTRACTOR", "ppuie")
    monkeypatch.setenv("QUERY_EXTRACTOR_CACHE_FILE", str(cache))

    assert cached_query_spans(query) == []
    assert candidate_blocks(build_source_bundle("", query, "")) == []


def test_layout_cache_requires_exact_bytes_and_dimensions(monkeypatch, tmp_path) -> None:
    content = b"frozen-image-bytes"
    digest = hashlib.sha256(content).hexdigest()
    cache = tmp_path / "layout.json"
    _write_cache(cache, kind="ppstructure-regions", items={digest: {
        "content_sha256": digest,
        "prepared_width": 100,
        "prepared_height": 200,
        "regions": [{"bbox": [0, 0, 40, 200], "confidence": 0.9}],
    }})
    monkeypatch.setenv("LAYOUT_ORDER_ENGINE", "ppstructure")
    monkeypatch.setenv("LAYOUT_ORDER_CACHE_FILE", str(cache))

    assert len(cached_layout_regions(content, width=100, height=200)) == 1
    assert cached_layout_regions(content + b"x", width=100, height=200) == []
    assert cached_layout_regions(content, width=101, height=200) == []


def test_candidate_hooks_are_disabled_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUERY_EXTRACTOR", raising=False)
    monkeypatch.delenv("LAYOUT_ORDER_ENGINE", raising=False)
    monkeypatch.setenv("QUERY_EXTRACTOR_CACHE_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("LAYOUT_ORDER_CACHE_FILE", str(tmp_path / "missing.json"))

    assert cached_query_spans("anything") == []
    assert cached_layout_regions(b"anything", width=1, height=1) == []


def test_structure_order_reorders_regions_without_replacing_ocr_text() -> None:
    boxes = [
        [[20, 20], [180, 20], [180, 50], [20, 50]],
        [[20, 80], [180, 80], [180, 110], [20, 110]],
        [[620, 20], [780, 20], [780, 50], [620, 50]],
        [[620, 80], [780, 80], [780, 110], [620, 110]],
    ]
    texts = ["左一", "左二", "右一", "右二"]
    regions = [
        {"bbox": [0, 0, 400, 900], "confidence": 0.9, "order": 2},
        {"bbox": [600, 0, 1000, 900], "confidence": 0.9, "order": 1},
    ]

    ordered = region_aware_ocr_order(
        boxes,
        texts,
        width=1000,
        height=1000,
        regions=regions,
        baseline_order=_reconstruct_ocr_reading_order,
    )

    assert ordered == ["右一", "右二", "左一", "左二"]
    assert sorted(ordered) == sorted(texts)

    grouped = region_aware_ocr_order(
        boxes,
        texts,
        width=1000,
        height=1000,
        regions=regions,
        baseline_order=_reconstruct_ocr_reading_order,
        group_separator=OCR_LAYOUT_REGION_SEPARATOR,
    )
    assert grouped == [
        "右一", "右二", OCR_LAYOUT_REGION_SEPARATOR, "左一", "左二",
    ]

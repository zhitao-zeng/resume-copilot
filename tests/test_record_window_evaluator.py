from __future__ import annotations

import pytest

from tools.evaluate_record_window_shadow import (
    MAX_CHARACTERS,
    MAX_SEGMENTS,
    MAX_TOKENS,
    OVERLAP_SEGMENTS,
    predeclared_shadow_probe_cases,
)


def test_w1_parameters_match_frozen_contract() -> None:
    assert (MAX_SEGMENTS, MAX_CHARACTERS, OVERLAP_SEGMENTS, MAX_TOKENS) == (
        20,
        1_600,
        5,
        2_048,
    )


def test_predeclared_probe_uses_lexicographically_first_id_per_scenario() -> None:
    cases = [
        {"id": "S2-B", "scenario": "scenario2"},
        {"id": "S1-B", "scenario": "scenario1"},
        {"id": "S2-A", "scenario": "scenario2"},
        {"id": "S1-A", "scenario": "scenario1"},
    ]

    selected = predeclared_shadow_probe_cases(cases)

    assert [case["id"] for case in selected] == ["S1-A", "S2-A"]


def test_predeclared_probe_requires_scenario_field() -> None:
    with pytest.raises(KeyError):
        predeclared_shadow_probe_cases([{"id": "missing"}])

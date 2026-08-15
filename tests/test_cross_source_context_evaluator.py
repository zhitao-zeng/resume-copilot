from __future__ import annotations

from tools.evaluate_cross_source_context_shadow import (
    MAX_CHARACTERS,
    MAX_SEGMENTS,
    MAX_TOKENS,
    OVERLAP_SEGMENTS,
)


def test_c1_parameters_match_frozen_contract() -> None:
    assert (
        MAX_SEGMENTS,
        MAX_CHARACTERS,
        OVERLAP_SEGMENTS,
        MAX_TOKENS,
    ) == (20, 1_600, 5, 3_072)

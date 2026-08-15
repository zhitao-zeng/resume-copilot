# Segment-grounded parser development set

This is a synthetic, hand-authored development set for the output-neutral
segment/span extraction experiment. It is not a leaderboard proxy and must not
be reported as held-out evidence.

The set deliberately contains adjacent records, repeated role families,
query-only facts, a no-profile negative case, Chinese and English text, and a
three-member metamorphic family. `expected_fields[].quote` must occur exactly
once in its named source; the evaluator resolves and verifies the exact source
span before any model output is scored.

The `two_work_records` variants change entities, date notation, and line
wrapping while keeping the same semantic graph. Candidate-specific keywords or
rules are not allowed.

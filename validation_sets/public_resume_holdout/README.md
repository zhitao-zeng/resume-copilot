# Public Resume Holdout

This directory builds a frozen, public-data holdout for the resume generation
pipeline. It is deliberately separate from `eval_cases.jsonl` and
`acceptance_testset/`, which are development and regression suites.

## Source

The primary source is the Chinese split of
[JobResQA](https://github.com/Avature/jobresqa-benchmark), a synthetic and
anonymized resume/JD benchmark. A small supplemental source,
[Candidate-Job Matching Synthetic](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic),
adds explicit zero-to-five-year, junior, and mid-career profiles from ten
business domains. It is used only in query-only and CV-only cases because its
experience bullets are intentionally sparse.

- Upstream revision: `1dbe6ffcf82cb06ec00f29bb0e4aeebde556addf`
- Source file: `data/jobresqa.zh.tsv`
- Source SHA256: `a749770f4064e3723fb5f7a712f66b88f28a2b5545c589a8d3998f83c8852814`
- Upstream license: CC BY-SA 2.0
- Personal data: upstream uses placeholders such as `[姓名]`, `[邮箱]`, and
  `[电话]`; no scraped real-person resumes are included here.

Supplemental source:

- Upstream revision: `178ab864dcad9910c5670d43e4bdbbb901a11f18`
- Selected profiles: 18 of 10,000 synthetic profiles
- Upstream license: MIT
- Local selected-profile SHA256: recorded in `source_manifest.json`

Generated dataset content derived from JobResQA remains subject to CC BY-SA
2.0. Supplemental profiles retain their MIT license. The builder and verifier
are project code.

## Splits

- `holdout_v2`: 60 cases, 15 for each of the four product scenarios.
- `shadow_v3`: 24 reserve cases, 6 for each scenario. Do not run this split
  during ordinary development.

The four scenarios are:

1. Resume plus JD.
2. Query-only personal information.
3. Resume without a JD.
4. JD-only with no personal information; the expected result is a structured
   framework with explicit missing-information guidance.

All resume roots and JD roots are disjoint between `holdout_v2` and
`shadow_v3`. The holdout contains 48 Chinese JobResQA cases and 12 English
supplemental cases; the shadow contains 18 and 6 respectively. Twelve holdout
cases are rendered as DOCX, PDF, or PNG to exercise layout/OCR. The remaining
resume inputs use plain text to isolate content quality from document parsing.

## Build and verify

Use the project environment; no additional packages are required.

```bash
.venv/bin/python validation_sets/public_resume_holdout/build.py
.venv/bin/python validation_sets/public_resume_holdout/verify.py
```

The builder downloads only the pinned 3 MB TSV, verifies its SHA256 before
parsing, and writes derived fixtures atomically. The generated split
directories and verification report are ignored by Git and can always be
regenerated; only code and small provenance/split manifests belong in source
control.

To run a small subset against a local API:

```bash
.venv/bin/python acceptance_testset/run_api_testset.py \
  --cases validation_sets/public_resume_holdout/holdout_v2/cases.jsonl \
  --case-id HV2-S1-001,HV2-S2-001,HV2-S3-001,HV2-S4-001
```

## Holdout policy

- Never optimize prompts or rules against aggregate holdout output.
- Record only promotion metrics unless a release fails its gate.
- Once a case is inspected and used to implement a fix, move it to the normal
  regression suite and replace it from `shadow_v3` before the next release.
- Keep all representations of one source profile in the same split.
- JD and user instructions are never candidate personal facts.
- Do not impose a one-page limit; verified facts may produce multiple pages.

`annotations.jsonl` stores canonical source text hashes and exact source spans.
The automatic line-level annotations are suitable for deterministic integrity
checks. Atomic semantic labels and subjective expression/STAR judgments still
require blinded human review and are explicitly marked as such.

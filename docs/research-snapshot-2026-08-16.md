# V2 research snapshot — 2026-08-16

This snapshot preserves the complete post-`e0bfc9b` V2 research working tree
before development of Resume Evidence Compiler V3.

## Scope

- Exact-span source documents, fact units, evidence bindings, atomic factuality,
  source preservation, ownership, and structural audits.
- Source-first fact compiler, pipeline profiles, deterministic recovery, bounded
  narrative optimization, and final-output/reply consistency guards.
- Candidate-first and segment/window semantic extraction research surfaces.
- PP-OCRv6 TensorRT recognition, PP-StructureV3 layout/OCR integration,
  native-document fast paths, and bounded fallbacks.
- Uploaded-template shell preservation and rendering safeguards.
- Frozen development/public-holdout manifests, evaluation tools, and regression
  tests required to reproduce the research comparisons.

The deployment default remains `PIPELINE_PROFILE=f507_compatible`. Experimental
content-mutating profiles remain opt-in; this snapshot is a research checkpoint,
not a claim that every experimental module is production-ready.

## Verification

Executed from the repository virtual environment before committing:

```text
.venv/bin/python -m pytest -m 'not integration' -q
650 passed, 43 deselected, 2 warnings in 22.47s
```

The deselected tests are explicitly marked integration tests requiring live
model services. No model checkpoints, generated resumes, platform logs, or
research-loop artifacts are included in Git.

## Recovery contract

- Archive branch: `archive/v2-research-20260816`
- Archive tag: `v2-research-snapshot-20260816`
- Parent production-history commit: `e0bfc9b1e0bc3428baffafc0145dc7027f55c71d`

V3 work must start from the tagged snapshot on a separate feature branch and
remain selectable behind a version flag until grouped, held-out, deployment,
and platform gates pass.

# Resume Evidence Compiler V3

V3 compiles heterogeneous career inputs into an evidence-bound resume.  It is
wired into the service behind `RESUME_PIPELINE_VERSION=v3`; the default remains
`v2` until held-out and platform gates pass, so the production baseline is not
silently replaced.

## Why PP-Structure is conditional rather than universal

PP-StructureV3 is the authoritative parser for a raster branch: it supplies a
layout detector, OCR recognition and reading-order metadata.  V3 keeps its
`page`, `column_id`, `region_id`, `bbox`, `block_order`, `confidence` and
`label` fields on each `LayoutNode`; it does not flatten the result to a plain
string.

It is not lossless for every input, however:

* A native DOCX already has paragraphs, table rows/cells, heading styles,
  numbering, headers/footers and editable runs.  Rasterizing it for
  PP-Structure loses those relationships and introduces OCR substitutions.
* A native PDF can expose exact characters, font runs and character/page
  coordinates without recognition errors.  OCR is a fallback for scanned or
  low-quality pages, not an improvement over reliable native spans.
* Layout detection does not establish that a duty belongs to employer A rather
  than employer B.  Ownership needs section, table, region and source-span
  constraints in the semantic compiler.
* Keeping the complete PP-Structure stack resident beside a 27B LLM is costly
  under the 40GB GPU limit.  Conditional routing avoids paying that cost for
  text-native uploads.

The intended policy is therefore:

```text
native DOCX AST --------------------┐
native PDF character spans ---------├─> common DocumentGraph -> FactGraph
scan/image/ambiguous page -> PP-V3 -┘
```

Passing PP-Structure blocks to `build_document_graph` does not override a
reliable native graph.  `force_ppstructure=True` is required for an explicit
A/B takeover; `shadow_ppstructure=True` only records that a comparison result
was available.

“Use PP-Structure completely” is useful within the raster branch, meaning all
layout metadata is consumed.  It is not useful as a universal input parser.

## Compiler stages

1. **Source policy** separates CV/Resume and factual Query clauses from Query
   intent.  JD and template text is ineligible as candidate evidence.
2. **DocumentGraph** stores exact spans and layout hierarchy.  Native adapters
   use source character offsets; PP-Structure adapters create stable spans in
   a normalized parser document while preserving the original layout fields.
3. **FactGraph** creates generic facts, section and record nodes.  Ownership
   precedence is explicit structure (heading/table/container), same region,
   same record, then adjacency.  If a boundary is not established, ownership
   is left unknown rather than guessed across experiences.
4. **RequirementGraph** parses JD requirements for prioritization only.  Its
   spans are exact, but JD/template sources are structurally impossible to
   mark candidate-eligible.
5. **ResumePlan** groups facts by immutable source record and maintains a
   coverage ledger.  It has no one-page constraint.  With no candidate facts,
   it emits a structured framework with `[待补充：...]` placeholders.
6. **Semantic compiler** uses the frozen `resume_compiler_v3.4` schema to split
   exact source quotes into generic fact atoms and auditable context spans.
   Labels, separators, placeholders, intent and instructions must cover their
   exact source characters without hiding hard anchors.  Semantic errors are
   retained as model-training evidence; they are not patched with occupation
   keyword rules. Independent batches may execute with bounded concurrency,
   but results are reassembled in source order and inherit the same request
   deadline. Query and placeholder-bearing batches fail closed when no valid
   semantic decision exists; ordinary CV facts retain exact-source fallback.
   Batch shape is explicit as `V3_SEMANTIC_BATCH_FACTS` (default `14`,
   R24-accepted after measuring fallback 53 -> 16 on the representative long
   case), `V3_SEMANTIC_BATCH_CHARS` (default `9000`) and
   `V3_SEMANTIC_MAX_TOKENS` (default `6144`, removing the measured 4096
   truncation). Cross-model A/B
   runs record both values because batch shape can change JSON-schema
   adherence even when the semantic policy and prompt are frozen.
   Every material schema/compiler candidate is evaluated in two paired views:
   Qwen27B versus DeepSeek Local Flash on the identical frozen inputs and
   batch shape, plus a provider-optimized DeepSeek ceiling probe when its best
   batch differs. DeepSeek is a recurring second-model and schema-stress view,
   not an automatic replacement or a guaranteed teacher. Promotion is based
   on audited facts, ownership, response contract and latency rather than model
   identity.
7. **Realizer** receives fixed fact IDs through the same versioned contract.
   Every source atom must remain verbatim, while the model may order and join
   already-present STAR dimensions.  Invalid output falls back to source text.
   A remaining-time admission gate skips this optional call when it cannot
   finish safely inside the end-to-end deadline.
   Since R24 Phase 3, realization is **record-local** (`core/v3/realizer_records.py`):
   the plan is partitioned into per-record/per-section isolation units; a
   semantic fallback fact degrades only its own unit, clean units keep
   constrained LLM realization, and a unit failing the hard verifier (or a
   failed physical pack) restores exact record-local source sentences.
   Clean units may pack under `V3_REALIZER_PACK_CHARS` for one physical call,
   but validation and fallback stay per-unit.  Label-split atoms advertise
   their source-line `label_prefix`; claims that strip it are rejected
   (`label_not_preserved`) and the deterministic path restores the label, so
   a bare value never ships unmoored.  The optional profile summary is
   single-pack and clean-graph only until the Phase 4 summary compiler.
8. **Atomic verifier/repair** checks source text, critical entity anchors,
   record ownership and eligibility.  Repair retains supported atoms or falls
   back to a source sentence from the same record; it never imports JD/template
   content or silently crosses records.
9. **Reply builder** is derived from the frozen audit, so written facts cannot
   simultaneously be reported as missing.

Record ownership uses the following strict precedence: explicit
record/container/table-row metadata, then same region/column/parent, then
same-record adjacency.  A new region or column without a clear header causes
abstention (`record_id=None`); it never inherits the previous experience.
Classification is section/structure based and does not contain a technology or
industry dictionary.

## Fixed model contracts

The model-facing schemas live in `core/v3/training_schema.py`:

* `SemanticCompilationResponse`: exact quote, fact type, destination section,
  destination field, classification and constrained record assignment.
* `ConstrainedRealizerResponse`: flat claims containing only text, fixed fact
  IDs and their section/field/record/group destinations.  Internal anchors and
  verification state are filled by trusted code, not by the model.

The version is `resume_compiler_v3.4`; its canonical JSON Schema SHA256 is
`e5292be47ea8d84f43c1f0fc58e0835756a89c648f2552ef29f67906c48fb36f`.
`tools/export_v3_training_schema.py` exports the bundle and refuses to run if
the code drifts without a version/fingerprint update.  Training traces are off
by default and require both `V3_TRAINING_TRACE_ENABLED=1` and
`V3_TRAINING_TRACE_DIR`, avoiding accidental production PII retention.

## Contracts and service API

Internal contracts live under `core/v3/contracts.py`.  The foundation entry
point remains `core.v3.orchestrator.run_v3(...)`; the binary/service entry point
is `core.v3.pipeline.run_v3_pipeline(...)`.  The latter accepts original CV
bytes plus filename, Query, JD and template mode, invokes the schema-driven
compiler, then adapts frozen claims into the established editable renderer.

The initial tests cover exact source spans, JD/template contamination,
cross-record ownership, critical anchors, PP-Structure metadata preservation,
the four input scenarios, and the no-personal-information framework path.

## Current implementation boundary

This branch implements the opt-in end-to-end path but is intentionally not a
default production cut-over yet:

* The live PaddleX worker can return complete JSON block payloads.  DOCX body,
  table and style hierarchy plus native-PDF line/bbox pages are concrete
  adapters; scan/image pages use PP-Structure and retain its metadata.
* The 27B semantic compiler and realizer are untrusted schema clients.  Exact
  spans, source eligibility, ownership, hard anchors and full fact-ID coverage
  are validated before freezing; failures use source-preserving fallbacks.
  Deterministic fallback groups only atoms from the same transport fact rather
  than concatenating an entire record into one oversized bullet.
* Frozen claims are adapted to the existing tagged/anchored/style-only DOCX
  implementation, retaining headers, footers, geometry and editable content.
* Model language quality still requires held-out evaluation and later
  fine-tuning.  This stage does not claim a Darvin score improvement.

These boundaries keep the existing V2 service unchanged while the V3
invariants are evaluated on held-out documents.  They must not be presented as
evidence that V3 has already improved the Darvin score.

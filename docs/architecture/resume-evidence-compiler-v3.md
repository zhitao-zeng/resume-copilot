# Resume Evidence Compiler V3

V3 is an isolated, shadow-only pipeline for compiling heterogeneous career
inputs into an evidence-bound resume.  The current V2 request path is not
imported or changed by this prototype.

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
6. **Realizer** receives fixed fact IDs.  The prototype uses a deterministic
   connector fallback; an LLM implementation must pass the same protocol.
7. **Atomic verifier/repair** checks source text, critical entity anchors,
   record ownership and eligibility.  Repair retains supported atoms or falls
   back to a source sentence from the same record; it never imports JD/template
   content or silently crosses records.
8. **Reply builder** is derived from the frozen audit, so written facts cannot
   simultaneously be reported as missing.

Record ownership uses the following strict precedence: explicit
record/container/table-row metadata, then same region/column/parent, then
same-record adjacency.  A new region or column without a clear header causes
abstention (`record_id=None`); it never inherits the previous experience.
Classification is section/structure based and does not contain a technology or
industry dictionary.

## Contracts and shadow API

The contracts live under `core/v3/contracts.py`.  The experiment entry point is
`core.v3.orchestrator.run_v3(...)`; it accepts `cv_text`, `query_text`,
`jd_text`, and an optional `TemplateAST`.  This is deliberately not wired into
V2 or the HTTP production path.  `core/v3/render` receives a frozen content
model and can render sections without becoming a content authority.

The initial tests cover exact source spans, JD/template contamination,
cross-record ownership, critical anchors, PP-Structure metadata preservation,
the four input scenarios, and the no-personal-information framework path.

## Current implementation boundary (foundation)

This branch implements and validates the compiler contracts and deterministic
shadow path.  It is intentionally not a production cut-over yet:

* `from_ppstructure_blocks` consumes the complete raster parser block payload,
  but the live PaddleX worker still needs a V3 adapter that returns those
  blocks instead of the V2 flattened text result.
* `from_native_text` proves exact-span behavior; concrete DOCX paragraph/table
  AST and native-PDF character/bbox adapters are the next integration layer.
* The realizer currently uses a deterministic connector.  A live 27B response
  must be treated as untrusted and pass `validate_realizer_response` with
  planner-owned `allowed_fact_ids` before it can be frozen.
* The renderer is a content-authority boundary, not yet the tagged/anchored
  DOCX fidelity implementation.

These boundaries keep the existing V2 service unchanged while the V3
invariants are evaluated on held-out documents.  They must not be presented as
evidence that V3 has already improved the Darvin score.

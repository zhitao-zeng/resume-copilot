# External SOTA scan — 2026-08-17

A bounded external scan of published work and open code that is comparable to
the Resume Evidence Compiler V3 line.  It records what exists outside this
repository and where V3 already differs; it is not a claim that any external
result has been reproduced here, and no number below was measured locally.

Confidence is marked per item: `abstract-verified` means the arXiv abstract was
read directly; `search-reported` means the figure comes from a secondary
summary and must be re-read at the source before it is used in any decision.

## Directly comparable systems

### Resume Tailor — career-aware tailoring with provenance tracking
arXiv 2605.05257 (2026-05-06), Kumar Abhinav.  `abstract-verified`

The closest published analogue to V3's evidence discipline.  A 12-node LangGraph
pipeline over a longitudinal *career vault* held in a vector database, using
multi-source RAG to assemble job-specific content from historical resumes and
structured career records.  It reports typed state management, hybrid
semantic-lexical confidence scoring, provenance-aware fallback generation,
anti-hallucination guardrails and a conditional review loop.

Pilot scale is small — nine JDs, one candidate's history — and the result is
two-sided: enabling the vault improved ATS-style fit by an average of 7.8 points
on the six JDs where the candidate had a prior role in the same occupational
category, but *cost* an average of 8.0 points on the two JDs whose domain was
absent from the vault.  The authors' own conclusion is that retrieval must be
confidence-gated when domain overlap is weak.

Relevance: V3 is single-document by construction.  Its `SourcePolicy` and
ownership precedence would extend to a multi-document vault without weakening,
because eligibility and record ownership are already source-scoped.  The
negative-transfer result is the important part — it says a vault must ship with
a gate, not that a vault is free.

### ResumeFlow
arXiv 2402.06221, SIGIR 2024 demo.  `abstract-verified`

Three stages: extract JD details, extract role-specific resume details, generate
the tailored resume.  Off-the-shelf GPT-4/Gemini, no fine-tuning.  Notably it
proposes task-specific metrics intended to control for alignment *and*
hallucination.  It is a baseline rather than a competitor: there is no span-level
grounding, no ownership model and no verifier, so nothing prevents a rewritten
bullet from migrating between employers.

### ConFit v3 — LLM re-ranking for person-job fit
arXiv 2605.09760 (2026-05-10), Yu et al.  `abstract-verified`

Matching, not generation, but the training recipe is transferable.  A systematic
study of the re-ranker pipeline for person-job fit — inference algorithm, RL
algorithm, data processing, SFT distillation — finding that multi-pass
re-ranking, listwise RL objectives, noisy-sample removal, and distilling from a
stronger LLM before RL each help.  ConFit v3 fine-tunes Qwen3-8B/32B and reports
beating prior person-job-fit systems as well as GPT-5 and Claude Opus-4.5.

Relevance: same model family as this project's 27B deployment.  V3's
`RequirementGraph` currently drives prioritization only; the `jd_analysis`
subdimension of the reply component is the concrete place a stronger,
explainable match signal would land.  ConFit v2 (arXiv 2502.12361) covers the
embedding-side predecessor.

## Grounding and attribution methods

- **Attribute First, then Generate** (arXiv 2403.17104, ACL 2024).
  `search-reported`.  Decomposes generation into content selection, sentence
  planning and sequential sentence generation, where the selection and planning
  stages *exactly copy* source spans, enforced with constrained decoding.  This
  is structurally the same commitment as V3's plan-then-realize split with
  verbatim atoms and source-sentence fallback; V3 additionally carries record
  ownership, which this line does not model.
- **Think&Cite** (arXiv 2412.14860).  `search-reported`.  Self-guided tree search
  with progress reward modeling, motivated by the argument that single-pass
  attributed generation is "System 1" and propagates intermediate citation
  errors.  Relevant to V3's single-pass realize-verify-repair loop, but the
  latency cost is hard to reconcile with the 480s end-to-end budget.
- **Are Finer Citations Always Better?** (arXiv 2604.01432).  `search-reported`.
  Controls for citation volume and reports that fine-grained attribution
  constraints can *degrade* output quality for capable models by fracturing
  semantic dependencies.  This is the most direct external challenge to V3's
  atom-level design and should be read against the local expression component,
  which is the lowest-scoring and highest-weighted dimension in the Darvin
  breakdown.
- Background: FactScore and ALCE as the founding decomposition/attribution
  benchmarks; FaithLens (arXiv 2512.20182) for detect-and-explain; arXiv
  2501.00269 for a faithfulness-metric survey.

## Document parsing

OmniDocBench (CVPR 2025) is the current common evaluation surface.  Reported
OmniDocBench standings, all `search-reported` and unverified here:

| System | Note |
| --- | --- |
| MinerU2.5 (arXiv 2509.22186) | reported overall 90.67; decoupled layout/recognition; text edit distance 0.047 |
| dots.ocr | single 1.7B VLM, reported SOTA text/table/reading-order, strong multilingual |
| PP-StructureV3 (arXiv 2507.05595) | current production parser here; reported text edit distance 0.073 |

The trend is unified VLMs displacing multi-stage pipelines.  For this project the
trade is specific rather than obvious: V3's raster branch consumes `page`,
`column_id`, `region_id`, `bbox`, `block_order`, `confidence` and `label` from
PP-StructureV3, and ownership precedence depends on region/column metadata.  A
VLM that emits markdown without stable layout fields would remove the evidence
that abstention currently relies on.  Any swap is a layout-metadata question
first and an accuracy question second.

Adjacent and relevant to a known local failure mode: *When Good OCR Is Not
Enough* (arXiv 2605.00911) benchmarks OCR robustness for downstream RAG.  The
R25/R26 OCR digit incidents (`204-2011`, `5` versus `50`) are exactly this class
of error — a parser-level corruption that a faithful downstream stage will
preserve verbatim unless a numeric guard intercepts it.

## Benchmarks

- **JobResQA** — already the upstream source of this repository's
  `public_resume_holdout` (Chinese split, CC BY-SA 2.0).  No action; noted so the
  scan does not propose adopting something already in use.
- **TalentCLEF 2026** (arXiv 2606.31692) — multilingual job-person/job-skill
  matching shared task.  `search-reported`.
- **ResuméAtlas** (arXiv 2406.18125) — large-scale resume classification.
- **LiveCareer** — 2,484 human-written resumes across 24 occupations, authored
  before widespread LLM use; useful as a non-synthetic contrast to the current
  holdout, which is entirely synthetic/anonymized.

## Open-source implementations

None of these enforce span-level grounding; they are useful for interface and
failure-mode reference, not for method transfer.

- `olyaiy/resume-lm` — Next.js open-source builder, multi-provider.
- `interviewstreet/hiring-agent` — scoring agent.  Two findings worth importing
  as tests rather than code: a 100-run variance study separating stable
  categories (technical skills) from noisy ones (project-quality judgments), and
  a report that invisible text embedded in a PDF can inflate scores.  The latter
  is a prompt-injection surface for any uploaded-CV pipeline, including this one.
- `busycaesar/LLM-Powered_Resume_Optimizer`, `AlaGrine/CV_Improver_with_LLMs`,
  `ThomasHoooo/ResumeOptimizer` — LangChain/RAG-shaped tools.

## Reading of the landscape

Published work is converging on the position V3 already takes: select and bind
evidence before generating, and keep the binding auditable.  V3 is ahead of the
open-source field on enforcement — atom-level verification with record ownership
and source-preserving fallback is stricter than anything listed here — and
behind the published field on two axes that are both about *inputs* rather than
guarantees: no longitudinal evidence store (2605.05257) and no trained,
explainable JD match signal (2605.09760).

The one external result that argues against the current direction is arXiv
2604.01432 on citation granularity.  It should not be dismissed: the local
expression component carries weight 40 and scores lowest of the four, and
`star_richness` in particular sits near 0.18, which is the pattern that paper
predicts when atomic constraints fracture sentence-level semantics.  Whether that
is caused by the constraint or merely coincident with it is not established by
anything in this scan and would need a controlled local ablation.

## Provenance

Compiled 2026-08-17 from arXiv abstracts and web search summaries.  Three
abstracts were read directly at arxiv.org (2402.06221, 2605.05257, 2605.09760);
every other figure is second-hand and marked `search-reported`.  No external
model, dataset or code was downloaded, and no local evaluation was run for this
document.

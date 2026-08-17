# Output-quality defect inventory — 2026-08-17

A content-level read of what V3 actually produces, as opposed to what the
component metrics report.  Every defect below was observed in stored run
artifacts or in rendered DOCX; none was inferred from metrics alone.

The motivation is the R24 blind monitor result (`v2:4 r24:0 tie:1`): local
Darvin components rank R24 above V2 on all four dimensions, while the blind
judge prefers V2.  This inventory is an attempt to name the difference.

## Evidence base

| Source | Scope |
| --- | --- |
| `.codex/research-loop/artifacts/local-eval-cluster/r26-representative6-qwen/merged.json` | 6 cases, image `qwen27b-ca24ceb6a02d`, evaluator `public-holdout-evaluator-1.2` |
| `.codex/research-loop/artifacts/darvin-aligned-quality-20260816/quality-gate-r24-v3/r24-full60-reaudit.json` | 60 cases |
| `.../darvin-aligned-quality-20260816/v2-full60-reaudit12.json` | 60 cases, V2 control |
| `.codex/research-loop/runtime/local-eval-cluster/api-output-*/*.docx` | 40 rendered documents, mixed runs |

The 40-document DOCX sample spans several runs and pipeline versions, so DOCX
counts below are indicative of prevalence, not a per-version rate.

Confidence labels: **code-confirmed** — the defect was traced to a specific
code path; **data-confirmed** — reproduced in stored output, root cause
inferred; **observed** — seen in output, root cause not established.

## P0 — Contract and trust

### D1. Written facts are simultaneously reported as missing
6/6 cases.  **code-confirmed.**

`core/v3/pipeline.py::_missing_fields` tests only top-level keys:

```python
("summary", "个人总结", bool(str(resume_data.get("summary") or "").strip())),
("education", "教育经历", bool(resume_data.get("education"))),
```

Content routed into `resume_data["additional_sections"]` is invisible to this
check.  In every observed case the profile summary was written into
`补充经历` and reported missing in the same response.

`HV2-S1-009` — written into the document:

```
"注重细节的专业人士，在优化流程以提升绩效、效率和质量方面具有扎实的基础。寻求一个职位来发挥行政管理专"
"长，并在充满活力的工作环境中提升技能水平。"
```

Reported in the same payload:

```json
{"field": "summary", "reason": "个人材料中未提供该信息，未进行推断"}
```

Education behaves identically.  `HV2-S1-012` carries five credentials under
`教育补充信息` (`国际投资认证`, `资本市场文凭`, `已通过CFA—级考试`,
`认证财务管理分析师`, `执业市场分析师`) while reporting
`教育经历：个人材料中未提供该信息`.

This contradicts the invariant stated in
`docs/architecture/resume-evidence-compiler-v3.md`: *"Reply builder is derived
from the frozen audit, so written facts cannot simultaneously be reported as
missing."*  The reply builder honours it; `_missing_fields` does not, and it is
`_missing_fields` that feeds the user-visible list.

### D2. The missing-field reason asserts a cause it never checked
6/6 cases.  **code-confirmed.**

The reason string is a fixed `个人材料中未提供该信息，未进行推断`, but the
predicate only asks whether *we produced* the field.  For `HV2-S2-010` the
source query contains an explicit summary:

```
Summary
Data Analyst with 0 years of experience in Healthcare.
```

The real state is `summary_status: dropped_unverifiable`.  The product told the
user they had not supplied something they had supplied.  "We could not verify
it" and "you did not provide it" must not share a message.

### D3. Internal status codes surface as user-facing conflicts
**data-confirmed.**

```json
"conflicts": [{"field": "source_conflict", "description": "unassigned:metric"}]
```

renders as `存在多处不一致的数字表述，请核对确认`.  The case's
`expected_conflicts` is `[]`, so this is both a false positive and an internal
token reaching the user.

## P1 — Fragmentation

The V2 control is decisive here.  On the identical case `HV2-S1-009`, V2
produced `"协助组织社区活动并为团队提供行政支持"` intact, with
`company="当地社区中心"` and `role="志愿者"`.  V3 emitted the same sentence
split across two bullets.  The fragmentation is introduced downstream of
parsing, not by OCR.

### D4. Summary length repair empties the summary whenever it triggers
`summary_exceeds_100_chars` 3 occurrences, `summary_empty_after_length_repair`
3 occurrences — a 100% coupling.  **code-confirmed**,
`core/v3/summary_compiler.py:236`.

```python
total_chars += _compact_len(text)   # counted before the rejection test
if sentence_violations:
    violations.extend(sentence_violations)
    continue                        # sentence rejected, its length retained
verified.append({"text": text, "fact_ids": fact_ids})
```

Rejected sentences never enter `verified` but permanently inflate
`total_chars`.  The repair loop can only pop from `verified`, so it drains
every legitimate sentence and still fails the budget:

```python
while verified and total_chars > MAX_COMPACT_CHARS:
    removed = verified.pop()
    total_chars -= _compact_len(removed["text"])
if not verified:
    violations.append("summary_empty_after_length_repair")
```

Moving the accumulation below `verified.append(...)` is sufficient.

### D5. Sentences split mid-word across bullets
6 fragment-initial bullets in 60 cases.  **data-confirmed.**

```
"当地社区中心志愿者：协助组织社区活动并为团队提供行"   +   "政支持。"
"…寻求一个职位来发挥行政管理专"                      +   "长，并在…"
"，顾问及"
"，高级客户关系经理，"
```

`行政支持` and `行政管理专长` are cut mid-word, leaving `政支持。` and `长，`
as standalone bullets.  The atomic verifier accepts them because each fragment
is verbatim source text; nothing tests whether a bullet is a well-formed
sentence.  This is the concrete mechanism behind micro precision `0.996`
coexisting with unreadable output.

### D6. Quarantined numerals leave a mutilated sentence
2 occurrences in 60 cases.  **data-confirmed.**

```
"日均处理超过个预约"
"监督价值万美元的业务运营，整合桌面解决方案和数据源。"
"管理英国境内个卖方桌面许可证"
```

The numeric guard correctly refuses a suspect OCR digit, then ships the
sentence without it.  `超过个预约` reads as broken software and is worse than
omitting the bullet, which the reply already reports under `待确认数字`.

### D7. Latin spacing lost in extracted spans, then shown to the user
**data-confirmed.**  Source: `Delivered results using structured workflows and
clear communication`.  Reply text:

```
待确认归属：「Deliveredresults」、「usingstructuredworkflows」、「andclearcommunication」
```

Spaces are dropped and one sentence is presented to the user as three
unattributed fragments.

### D8. Unbalanced and mismatched brackets
4 occurrences in 60 cases.  **data-confirmed.**

```
"副总裁，投资策略，亚洲（收入1.1亿美元>副总裁，全球客户管理（收入(万美元）"
"（公司】"
"区域营销经理，B2B传播（临时职位"
```

`（公司】` mixes a full-width parenthesis with a full-width bracket.

### D9. Duplicate content alongside its own fragment
1 case.  **observed.**

```json
"补充经历": ["冲浪、创意设计、烹饪艺术", "、创意设计、"]
```

## P2 — Content quality

### D10. English sources can never produce a summary
`escape_cjk` is the single largest failure reason (23 occurrences).  12 of 60
holdout cases (20%) are English.  **structural.**

Summary verification requires that, once cited facts are removed, the residual
consist only of connectors or words appearing verbatim in the cited facts.  A
Chinese summary over an English source leaves every Chinese token as residual.
`HV2-S2-010` violations include `escape_cjk:初级数据分析师`,
`escape_cjk:医疗领域`, `escape_cjk:数据可视化` against a source reading
`Data Analyst`, `Healthcare`, `Data Visualization`.  Translation is
categorically rejected, so the 20% English slice cannot score this component
under the current rule.

A related over-reach: `computed_tenure:0年` fired on `0 years of experience`,
which is present in the source.

### D11. Summary absent in a third of cases
19/60.  **data-confirmed.**  Aggregate consequence of D4, D10 and
`dropped_atomic_audit`.

Observed `summary_status` distribution over the 6-case set: `generated` 1,
`dropped_atomic_audit` 3, `dropped_unverifiable` 2.

### D12. Semicolon-stacked bullets suppress STAR extraction
24 occurrences in 60 cases.  **data-confirmed.**

```
"通过有效的招聘和培训流程管理团队；制定和维护人员排班表；确保设施的清洁和维护标准。"
```

`HV2-S3-009` is the highest-quality case in the set — complete records, intact
sentences, populated `company`/`period` — yet scores `star_complete_rate 0.0`.
Three actions in one bullet leave no room for a result dimension.
`star_richness` is `0.178` at weight 20, the largest single subdimension in the
expression component.

### D13. `补充经历` is an untyped dumping ground
6/6 cases.  **data-confirmed.**  A single section mixes five unrelated kinds of
content:

- profile summary (belongs in `summary`)
- interests — `热衷于经济预测和国际象棋`
- resume boilerplate — `推荐信可按需提供。`, `如需参考资料，可提供`
- credentials (belong in `certifications`) — routed to `教育补充信息`
- genuine experience — `担任人体工程学倡导者，推广人体工程学安全并简化流程`

### D14. Resume boilerplate is preserved as content
**data-confirmed.**  `推荐信可按需提供。` is a References-available-upon-request
formula.  A faithful pipeline preserves it; a good resume deletes it.

### D15. Input metadata leaks into experience
**data-confirmed.**  `Years of Experience: 0`, `Career Level: Junior`,
`Industry: Healthcare` are profile metadata lines from the query, rendered as
experience bullets and as `补充经历` entries.

### D16. The planner drops the candidate's positioning facts
**data-confirmed.**  `HV2-S2-010` unwritten facts, all with reason
`not selected by deterministic planner`:

```
"Data Analyst"           target direction
"Junior"                 seniority
"Healthcare"             industry
"0 years of experience"  tenure
```

These are the first four things a recruiter reads.  Case recall `0.7273`.

### D17. Skills taxonomy regressed against V2
6/6 cases.  **data-confirmed.**  V2 populates `natural_languages`, `tools`,
`domains`, `certifications`; V3 places everything in `others`, including full
sentences such as
`使用人力资源系统处理薪资，并为团队领导提供系统功能西班牙语流利，法语会话水平`.

### D18. JD match evidence is degenerate
`job_alignment.support_rate` `0.1` — 1 of 10 requirements supported.
**data-confirmed.**

```
- 匹配：职位名称：助理副总裁-分析（依据：分析）
```

A two-character overlap is offered as evidence.

### D19. Missing fields over-reported
**observed.**  `HV2-S3-009` declares `expected_missing_fields: []`; the run
reported four.

### D20. Non-numeric OCR corruption passes unguarded
3 duplicated-character hits in 60 cases.  **observed.**

```
分折 (分析)   行生品 (衍生品)   檀长 (擅长)   投投资   CFA—级 (CFA一级)
```

The numeric guard covers digits and periods only; there is no equivalent for
corrupted glyphs in ordinary text.

### D21. Two-column reading order interleaves columns
2 cases, both `input_profile=scanned_two_column_png`.  **inferred.**

Output order follows an A1, B1, A2, B2 pattern — first halves of two distinct
sentences, then their second halves — which is the signature of reading across
a column gutter rather than down each column.  All P1 fragmentation in this
inventory originates in these two cases; `query_only` and `plain_text` inputs
are clean.

## P3 — Rendering

### D22. Name placeholder rendered as the document title
3 of 40 documents.  **confirmed.**  When no name is extracted the document is
titled `候选人`.

### D23. Empty separator paragraphs and mixed font sizes
2–3 of 40 documents.  **confirmed.**

```
'  |      '            education line with a bare pipe and no content
'  |  BSc ·     '      pipe and interpunct with empty neighbours
```

The second paragraph carries three runs at 9.5pt, 10.5pt and 11.5pt.  Template
separators are emitted even when their slots are empty.

## Disproved hypotheses

Recorded so they are not re-investigated.

- **CJK `eastAsia` font missing on `List Bullet` runs.**  Checked every
  CJK-bearing run across 40 documents: **0 occurrences**.  Font portability is
  genuinely fixed.
- **`score: 0.0` / `score_breakdown: {}` is a V3 regression.**  V2 returns the
  same on all 60 cases.  It is a mismatch between `README.md` and actual
  behaviour, plausibly intentional under evaluation mode, and not attributable
  to V3.

## Two standing observations

**Internal and external factuality metrics disagree.**  The `quality_report`
embedded in the API response gives `HV2-S2-012` precision `0.6667` and
`HV2-S2-010` `0.7333`, while the external audit reports micro precision `0.996`
over the same run.  The definitions differ, but the number the platform and the
user see is the internal one.

**Latency leaves no room for another model call.**  `HV2-S1-009` totals
`384.858s`, of which `v3_pipeline_s` is `367.067s` — about 95s of headroom
against the 480s deadline.  Every P0/P1 remedy above is deterministic
post-processing in the millisecond range; none of them fits as an additional
LLM pass.

## Suggested order

Ranked by expected perceived-quality gain per unit of change, not by severity.

1. **D1 + D2** — include `additional_sections` in the missing-field check and
   separate "not provided" from "not produced".  Removes the most visible
   self-contradiction in the product.
2. **D4** — move one line in `summary_compiler.py`.
3. **D13 + D14** — route `补充经历` content by type; drop boilerplate outright.
4. **D15** — filter metadata lines before they become facts.
5. **Sentence-integrity guard** — one gate covering D5, D6, D8 and D9: reject
   bullets that begin with punctuation, carry unbalanced brackets, or fail to
   form a complete sentence; merge with the adjacent same-record span, and
   fall back to a record-local source sentence when merging fails.

D10 (cross-language summary), D16 (planner selection) and D18 (JD evidence) are
architectural and should not be bundled with the above.

## Provenance

Compiled 2026-08-17 by reading stored artifacts and rendered documents.  No
pipeline run, model call or code change was performed for this document.
Counts over the 60-case set were produced by pattern scanning; matches were
reviewed by hand and false positives from legitimate short fields
(`远程`, `7个月`), CamelCase brand names (`TechInnovate`, `MedFramework`) and
incidental character repetition (`项目目标`) were excluded from the figures
reported here.

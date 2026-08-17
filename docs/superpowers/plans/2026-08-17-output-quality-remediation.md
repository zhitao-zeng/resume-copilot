# 输出体感缺陷修复计划（R27）

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复 `docs/quality-defects-2026-08-17.md` 中 D1–D9、D13–D15 共 12 项确定性缺陷，消除「文档里写了却报告用户未提供」的自相矛盾，并让 bullet 恢复为完整句子。不涉及 D10/D16/D18 等架构级问题。

**背景：** R24 本地 Darvin 四项分量全面优于 V2（完整度 0.8416>0.7881，表达 0.3726>0.3443，回复 0.8825>0.7991），但盲评判官以 4:0 偏好 V2。缺陷清单定位了差异来源：产出在**逐字层面正确**（micro precision 0.996）却在**句子层面破碎**。本计划修的全部是这类「校验通过但不能读」的问题。

**架构：** 全部为确定性后处理，不新增任何 LLM 调用。改动集中在 `core/v3/pipeline.py`、`core/v3/summary_compiler.py`、`core/v3/resume_adapter.py`，新增 1 个模块 `core/v3/text_integrity.py` 和 1 个测试文件 `tests/test_v3_text_integrity.py`。

**技术栈：** Python 3.13, pydantic v2, pytest。运行环境 `.venv/bin/python`。

**硬约束：**
- **不得新增 LLM 调用。** 实测 `HV2-S1-009` 总耗时 384.858s，其中 `v3_pipeline_s` 367.067s，距 480s 上限仅约 95s 余量。
- **不得把 `additional_sections` 内容直接提升为 `summary`。** 那会绕过 summary 校验器，正是本项目要防的事。任务 1 只负责**如实报告**，不负责搬运。
- **不得为通过某个 holdout case 而写针对性规则。** 见任务 0。
- V2 路径（`RESUME_PIPELINE_VERSION=v2`）必须零改动。

---

### 任务 0：Holdout 卫生（必须最先执行）

`validation_sets/public_resume_holdout/README.md` 的 holdout policy 规定：

> Once a case is inspected and used to implement a fix, move it to the normal regression suite and replace it from `shadow_v3` before the next release.

缺陷清单是通过**逐例阅读**下列 holdout case 得出的，它们已被污染，不能再作为盲验证集：

```
HV2-S1-009  HV2-S1-012  HV2-S2-008  HV2-S2-010  HV2-S2-012  HV2-S3-009
```

**文件：**
- 修改：`validation_sets/public_resume_holdout/split_manifest.json`
- 修改：`tests/fixtures/`（新增回归夹具）

- [ ] **步骤 1：** 将上述 6 个 case 的输入与期望迁入常规回归套件（`tests/fixtures/`），作为本计划各任务的验证夹具。
- [ ] **步骤 2：** 从 `shadow_v3`（24 例储备）中补入 6 例，保持 `holdout_v2` 仍为 60 例、四场景各 15 例。
- [ ] **步骤 3：** 确认补入后 resume root 与 JD root 在两个 split 间仍然不相交（`verify.py` 会检查）。
- [ ] **步骤 4：** 运行 `.venv/bin/python validation_sets/public_resume_holdout/verify.py`，确认 `verification_report.json` 通过。

**验收：** `holdout_v2` 仍为 60 例；被检查过的 6 例已不在 holdout 内；`verify.py` 通过。

---

### 任务 1：`_missing_fields` 感知 `additional_sections`，并区分「未提供」与「未产出」（D1 + D2）

**这是本计划优先级最高的一项。** 6/6 case 命中，直接违反架构文档不变量：*"written facts cannot simultaneously be reported as missing"*。

**现状：** `core/v3/pipeline.py:134` 的 `_missing_fields()` 只检查顶层 key：

```python
("summary", "个人总结", bool(str(resume_data.get("summary") or "").strip())),
("education", "教育经历", bool(resume_data.get("education"))),
```

被 `core/v3/resume_adapter.py:263-268` 路由进 `additional_sections["补充经历"]` / `["教育补充信息"]` 的内容对它完全隐形。实测 `HV2-S1-012` 的 `教育补充信息` 含 5 条资质，同时报告「教育经历：个人材料中未提供该信息」。

同时 reason 字符串写死为「个人材料中未提供该信息，未进行推断」，但谓词只判断**我们是否产出**，从未判断**用户是否提供**。`HV2-S2-010` 的 query 明确含 `Summary\nData Analyst with 0 years of experience in Healthcare.`，真实状态是 `summary_status: dropped_unverifiable`。

**文件：**
- 修改：`core/v3/pipeline.py:134`（`_missing_fields` 签名与实现）
- 修改：`core/v3/pipeline.py:258`（调用处传入降级信息）

- [ ] **步骤 1：新增 `additional_sections` 文本聚合助手**

```python
def _additional_blob(resume_data: dict[str, Any]) -> str:
    """Flatten additional_sections so misrouted content stays visible."""
    sections = resume_data.get("additional_sections")
    if not isinstance(sections, dict):
        return ""
    parts: list[str] = []
    for values in sections.values():
        if isinstance(values, list):
            parts.extend(str(value) for value in values if value)
    return "\n".join(parts)
```

- [ ] **步骤 2：扩展 `_missing_fields` 签名，接收降级状态**

```python
def _missing_fields(
    resume_data: dict[str, Any],
    *,
    degraded: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
```

`degraded` 形如 `{"summary": "dropped_unverifiable"}`，由调用方从 `summary_result.report` 传入。

- [ ] **步骤 3：引入第二种缺失来源 `not_rendered`**

判定顺序改为三态，而非二态：

| 情形 | `source` | reason |
| --- | --- | --- |
| 字段已写入正文 | 不报告 | — |
| 材料中有相关内容，但未通过校验／被路由到补充信息 | `not_rendered` | `材料中存在相关内容，但未能通过校验写入正文，已保留在补充信息中，请人工确认` |
| 材料中确实没有 | `not_provided` | `个人材料中未提供该信息，未进行推断`（保持不变） |

- `summary`：`degraded.get("summary")` 非空 → `not_rendered`；否则若 `_additional_blob` 非空 → `not_rendered`；否则 `not_provided`。
- `education`：`additional_sections` 含 `教育补充信息` → `not_rendered`。

- [ ] **步骤 4：更新调用处** `core/v3/pipeline.py:258`

```python
missing = _missing_fields(
    resume_data,
    degraded={"summary": summary_result.report.status}
    if summary_result.report.status not in {"generated", ""}
    else {},
)
```

注意 `_missing_fields` 当前在 `summary_result` 之前调用，需将其**移到 `summary_result` 计算之后**（`compile_summary` 调用之后、`build_reply` 之前）。`build_reply` 已经接收 `missing_fields=missing`，顺序调整后仍然成立。

- [ ] **步骤 5：新增测试** `tests/test_v3_summary_reply.py`
  - 构造 `resume_data` 含 `additional_sections["教育补充信息"]` 且顶层 `education` 为空 → 断言 `source == "not_rendered"`，且 reason 不含「未提供」。
  - 构造 `summary` 为空且 `degraded={"summary": "dropped_unverifiable"}` → 断言 `source == "not_rendered"`。
  - 构造真正空白输入 → 断言仍为 `not_provided`（防回归）。

**验收：** 6 个回归夹具中，任何写入 `additional_sections` 的内容都不再以 `not_provided` 上报；纯空输入路径行为不变。

---

### 任务 2：修复 summary 长度修复必然清空（D4）

**现状：** `core/v3/summary_compiler.py:236` 的 `total_chars` 在拒绝判定**之前**累加：

```python
        total_chars += _compact_len(text)     # ← 被拒句子的字数也被计入
        if sentence_violations:
            violations.extend(sentence_violations)
            continue
        verified.append({"text": text, "fact_ids": fact_ids})
```

被拒句子不进 `verified` 却永久占用预算，而修复循环只能从 `verified` 弹出：

```python
    while verified and total_chars > MAX_COMPACT_CHARS:
        removed = verified.pop()
        total_chars -= _compact_len(removed["text"])
    if not verified:
        violations.append("summary_empty_after_length_repair")
```

结果是一旦超标必然把合法句子全部弹空。实测 `summary_exceeds_100_chars` 与 `summary_empty_after_length_repair` 各 3 次，**100% 耦合**。

**文件：**
- 修改：`core/v3/summary_compiler.py:236`

- [ ] **步骤 1：将累加移到 `verified.append` 之前一行**

```python
        if sentence_violations:
            violations.extend(sentence_violations)
            continue
        total_chars += _compact_len(text)
        verified.append({"text": text, "fact_ids": fact_ids})
```

- [ ] **步骤 2：新增测试** `tests/test_v3_summary_reply.py`
  - 构造 3 个候选句：其中 2 句触发 `escape_cjk` 被拒（合计 >100 字），1 句合法且 <100 字。
  - 断言结果**保留该合法句**，且 `summary_empty_after_length_repair` **不出现**在 violations 中。
  - 补一个真正超长用例（全部合法但合计 >100 字）→ 断言仍按尾部丢弃、非空。

**验收：** `summary_empty_after_length_repair` 不再与 `summary_exceeds_100_chars` 同频出现。

---

### 任务 3：`补充经历` 按类型分流并丢弃套话（D13 + D14）

**现状：** `core/v3/resume_adapter.py:257-268` 把 `additional` / `other` 两个 section 原样倒进 `补充经历`。实测同一章节混装五类内容：个人总结、兴趣爱好、简历套话、证书资质、真实工作经历。

**文件：**
- 修改：`core/v3/resume_adapter.py:257-268`

- [ ] **步骤 1：定义套话模式常量**

```python
_BOILERPLATE_PATTERNS = (
    "推荐信可按需提供",
    "如需参考资料",
    "references available upon request",
    "面议",
)
```

匹配采用**去标点后的子串包含**，大小写不敏感。命中即整条丢弃，不进入任何 section。

- [ ] **步骤 2：把 `教育补充信息` 中的资质条目改投 `certifications`**

`HV2-S1-012` 的 `教育补充信息` 实为 5 条证书（`国际投资认证`、`资本市场文凭`、`已通过CFA—级考试`、`认证财务管理分析师`、`执业市场分析师`）。判定依据用**已有的 fact_type**（`credential`），不要引入行业词典或关键词表——这违反架构文档「Classification is section/structure based and does not contain a technology or industry dictionary」。

若对应 fact 的 `fact_type == "credential"`，写入 `data["certifications"]`（沿用 `flat_targets` 已有去重逻辑），不再进 `教育补充信息`。

- [ ] **步骤 3：`补充经历` 更名为 `补充信息`**

现标题「补充经历」把兴趣爱好、总结片段都描述成「经历」，是语义错误。改为中性的「补充信息」。

- [ ] **步骤 4：新增测试** `tests/test_v3_compiler.py`
  - 套话条目 → 断言不出现在任何 section。
  - `fact_type == "credential"` 的条目 → 断言进入 `certifications` 而非 `教育补充信息`。
  - 普通补充条目 → 断言进入 `补充信息`。

**禁止：** 不要把 `补充信息` 里疑似个人总结的长句自动提升为 `data["summary"]`。那会绕过 summary 校验器。任务 1 已保证它被如实报告。

---

### 任务 4：过滤输入元数据行（D15）

**现状：** query 中 `ANONYMIZED SYNTHETIC PROFILE` 段落的元数据行被当作个人事实写入简历：

```
"Years of Experience: 0"      → 渲染为一条工作经历 bullet
"Career Level: Junior"        → 补充经历
"Industry: Healthcare"        → 补充经历
```

**文件：**
- 修改：`core/v3/resume_adapter.py`（写入 `experience` / `additional` 前过滤）

- [ ] **步骤 1：识别 `Label: Value` 元数据形态**

```python
_METADATA_LABELS = (
    "years of experience",
    "career level",
    "industry",
    "professional direction",
)
```

判定：整行匹配 `^\s*(<label>)\s*[:：]\s*(.+)$`（label 大小写不敏感）。

- [ ] **步骤 2：命中后不整条丢弃，仅剥离标签**

值本身（`Junior`、`Healthcare`）是合法事实，不能丢。**只移除 `Label:` 前缀**，并阻止该条进入 `experience`——元数据不是经历。剥离后的裸值若无处可去，保留在 `补充信息`。

- [ ] **步骤 3：新增测试** `tests/test_v3_compiler.py`
  - 输入含 `Years of Experience: 0` → 断言 `experience` 中不含该字符串。
  - 断言值 `Healthcare` 未从产出中整体消失（防止过度过滤导致召回下降）。

**验收：** 元数据行不再出现在 `experience`；对应值仍可被检索到。

---

### 任务 5：句子完整性守卫（D5 + D6 + D8 + D9）

一个守卫覆盖四类缺陷。**这是对盲评三个扣分点（mid-word bullet splits / truncated summary leading fragment / unclosed parens）的直接回应。**

**现状实例：**

```
"当地社区中心志愿者：协助组织社区活动并为团队提供行"  +  "政支持。"     ← 词中劈开
"，顾问及"                                                          ← 纯碎片
"日均处理超过个预约"                                                ← 数字被吞后的残句
"副总裁，投资策略，亚洲（收入1.1亿美元>副总裁，全球客户管理（收入(万美元）"  ← 括号不配平
```

原子校验器接受它们，因为每个碎片都逐字来自源文——**没有任何环节检查一条 bullet 是否成句**。

**文件：**
- 新增：`core/v3/text_integrity.py`
- 修改：`core/v3/realizer_records.py`（每单元校验处调用）
- 新增：`tests/test_v3_text_integrity.py`

- [ ] **步骤 1：实现纯函数 `bullet_defects(text: str) -> list[str]`**

不做任何 IO、不调模型。返回缺陷标签列表，空列表表示通过。

```python
def bullet_defects(text: str) -> list[str]:
    """Return structural defects that make a bullet unreadable.

    Every check is deterministic and source-agnostic: a fragment that is
    verbatim source text is still unreadable as a standalone bullet.
    """
```

判定规则：

| 标签 | 规则 |
| --- | --- |
| `fragment_start` | 首字符属于 `，。、；：）%>-` |
| `unbalanced_bracket` | `（`/`）`、`(`/`)`、`【`/`】` 计数不等，或类型交叉（如 `（…】`） |
| `eaten_numeral` | 量词紧邻计数词而缺数字，如 `超过个`、`价值万`、`境内个` |
| `bare_fragment` | 去除标点后长度 < 6 **且** 不以终结标点结尾 |

- [ ] **步骤 2：在 realizer 每单元校验中接入**

`core/v3/realizer_records.py` 的 per-unit 硬校验已存在（失败时回落记录级源句）。将 `bullet_defects` 并入该校验：任一 claim 命中缺陷 → 该单元判定失败。

**处置顺序（不得跳过）：**
1. 尝试与**同记录相邻 span** 合并后重判（可修复 `政支持。`、`，顾问及` 这类劈开）；
2. 合并后仍有缺陷 → 回落到该记录的**源句**（现有机制）；
3. 源句仍不成句 → 丢弃该 claim。丢弃后 coverage ledger 与 reply 已有上报路径，不需另加提示。

- [ ] **步骤 3：`eaten_numeral` 特殊处置**

数字被隔离后留下残句，比不写更伤。命中 `eaten_numeral` 的 bullet **直接丢弃**，不尝试合并——数值已由 numeric guard 在 reply 的「待确认数字」中上报，信息未丢失。

- [ ] **步骤 4：编写测试** `tests/test_v3_text_integrity.py`

用缺陷清单中的真实字符串做夹具，逐条断言：

```python
("政支持。",                     ["bare_fragment"]),
("，顾问及",                     ["fragment_start", "bare_fragment"]),
("日均处理超过个预约",             ["eaten_numeral"]),
("（公司】",                     ["unbalanced_bracket"]),
("协助组织社区活动并为团队提供行政支持", []),      # 正常句必须通过
("Excel",                       []),            # 技能标签必须通过
("2022年至今",                   []),            # period 字段必须通过
```

**注意：** 守卫只作用于 **bullet 文本**，不得作用于 `role` / `company` / `period` / `skills` 字段——`远程`、`7个月`、`Excel` 都是合法短值。

- [ ] **步骤 5：合并逻辑测试** `tests/test_v3_record_local.py`
  - 构造同记录内被劈成两段的源 span → 断言合并后产出完整句、单元不降级。
  - 构造无法合并的孤立碎片 → 断言该单元回落到源句。

**验收：** 6 个回归夹具中 `fragment_start`、`unbalanced_bracket`、`eaten_numeral` 归零；`bare_fragment` 仅在无法合并时以「丢弃」形态出现，不再作为 bullet 渲染。

---

### 任务 6：内部状态码不得作为用户可见冲突（D3）

**现状：** `core/v3/pipeline.py:272`

```python
    conflicts = [
        {"field": "source_conflict", "description": conflict}
        for conflict in audit.conflicts
    ]
```

`audit.conflicts` 含 `unassigned:metric` 这类内部标记，被渲染成「存在多处不一致的数字表述，请核对确认」。该 case 的 `expected_conflicts` 为 `[]`——既是假阳性，也是内部 token 泄漏。

**文件：**
- 修改：`core/v3/pipeline.py:272`

- [ ] **步骤 1：** 建立内部标记白名单／黑名单，形如 `<code>:<detail>` 的纯内部标记不进入用户可见 `conflicts`。
- [ ] **步骤 2：** 无法映射为用户可读描述的条目一律**丢弃**而非透传。
- [ ] **步骤 3：新增测试** `tests/test_v3_summary_reply.py`：`audit.conflicts` 含 `unassigned:metric` → 断言用户可见 `conflicts` 为空。

**验收：** 用户可见 `conflicts` 不含冒号分隔的内部标记。

---

## 整体验收

- [ ] **全量单测通过且无净减少：** 基线为 `831 passed, 43 deselected`（`2026-08-17` 实测，耗时约 24s）。

```bash
.venv/bin/python -m pytest -m 'not integration' -q
```

- [ ] **6 例回归夹具复跑**，逐项核对：
  - `additional_sections` 内容不再以 `not_provided` 上报（任务 1）
  - `summary_empty_after_length_repair` 消失（任务 2）
  - 套话与元数据行不出现在正文（任务 3、4）
  - 四类结构缺陷归零（任务 5）
  - 用户可见 `conflicts` 无内部 token（任务 6）

- [ ] **延迟未劣化：** 单例 `total_s` 不高于修复前基线（`HV2-S1-009` 为 `384.858s`）。全部改动为确定性后处理，预期增量在毫秒级；若出现秒级增长说明实现有误。

- [ ] **V2 路径零改动：** 确认 `RESUME_PIPELINE_VERSION=v2` 的产出逐字节不变。

- [ ] **不得自行提交 Darvin 平台。** 采用门槛为 Darvin 完整均分 ≥ 66.48、零编造否决、零生成失败、全部请求 < 480s。本计划只修体感缺陷，**不构成提交依据**，是否提交由项目负责人决定。

## 不在本计划范围

以下为架构级问题，已在 `docs/quality-defects-2026-08-17.md` 记录，**不要在本计划中顺手处理**：

- **D10** 英文源无法产出 summary（逐字校验与跨语言输出根本冲突，影响 20% case）
- **D16** 规划器丢弃求职方向／职级／行业／年限
- **D18** JD 匹配证据退化（`support_rate` 0.1）
- **D21** 两栏扫描件阅读序跨栏交错（上游 OCR／版面）
- **D12** 分号堆叠压制 STAR（涉及表达分权重 20 的最大子项，需独立设计与 A/B）

## 参考

- 缺陷清单与完整证据：`docs/quality-defects-2026-08-17.md`
- 架构不变量：`docs/architecture/resume-evidence-compiler-v3.md`
- Holdout 政策：`validation_sets/public_resume_holdout/README.md`

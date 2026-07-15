# Resume Copilot V2 MVP 规格

## 目标

把 15+ 层压成 5 层，先跑通 Case 1/5/39。

## 核心原则

1. 证据必须能确定性回指原文（`quote in block.text`）
2. Verifier 裁决事实，Composer 只声明生成方式
3. Renderer 只读一份 `CanonicalResume`，不多份数据
4. Verifier 失败时宁可少输出，不上未经审核的 Draft

## 五层架构

```
Document Adapter
        ↓
SourceBundle

        ↓
ResumeComposer (LLM 1)
输出 DraftResume + Evidence

        ↓
ResumeVerifier (LLM 2)
输出 CanonicalResume

        ↓
Basic Validator (确定性)
格式 / 空记录 / 去重

        ↓
Renderer (只读 CanonicalResume)
```

---

## 第一层：SourceBundle

```python
class SourceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    source_type: Literal["resume", "query", "jd"]
    text: str
    section_hint: str | None = None

class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[SourceBlock]
```

不做：bbox/page/start/end/table_row/col/parent_block_id/section enum

---

## 第二层：ResumeComposer (LLM 1)

### Evidence (简化)

```python
class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    quote: str

class DraftField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str | None = None
    mode: Literal["direct", "normalized", "derived", "rewritten", "none"]
    evidence: list[EvidenceRef] = []
```

Composer 输出后做确定性检查：

```python
def evidence_exists(ref, blocks):
    block = next((b for b in blocks if b.block_id == ref.block_id), None)
    return block is not None and ref.quote in block.text
```

不存在 quote 的 evidence 标记 invalid，Verifier 不依赖。

### DraftResume

```python
class DraftResume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: MetaDraft
    education: list[EducationDraft]
    experience: list[ExperienceDraft]
    projects: list[ProjectDraft]
    skills: SkillsDraft
    summary: DraftField
```

record_id 由代码赋值，不由 LLM 输出。

---

## 第三层：ResumeVerifier (LLM 2)

输入 SourceBundle + DraftResume。
**直接输出 CanonicalResume**，不输出 report。

```python
class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    action: Literal["clear", "remove", "replace"]
    reason: str

class VerifiedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume: CanonicalResume
    changes: list[Change]
```

Verifier 失败时→**不返回 DraftResume**。改为：

```python
def conservative_fallback(source):
    return CanonicalResume(
        meta={"name": "", "target_role": ""},
        education=[], experiences=[], projects=[],
        skills=[], summary="",
    )
```

宁可空也不输出编造内容。

---

## 第四层：Basic Validator (确定性)

只做：

- Pydantic `extra="forbid"` 校验
- 空记录删除（全部字段为空）
- 项目去重（name）
- 手机号格式
- 邮箱格式

不做：

- 学校/实验室/职位规则
- 语义修复

---

## 第五层：Renderer

只消费 `VerifiedResult.resume`。
不再有多份副本、嵌套 projects、apply_fabrication_report。

---

## 删除或冻结的模块

| 模块 | 处理 |
|------|------|
| `fact_ledger.py` | 冻结，Step 9 删除 |
| `semantic_guard.py` | 冻结，Step 9 删除 |
| `_apply_fabrication_report` | 立即删除 |
| 多 projects 副本 | 立即删除 |
| `resume_parsing.py` parse 函数 | 冻结 |
| `resume_optimization.py` bullet 管道 | 冻结 |
| `resume_product_logic.py` 实体规则 | 冻结 |

保留：`resume_io.py`、`resume_renderer.py`、`server_runtime.py`、`llm_gateway.py`、`prompts.py`、`resume_scoring.py`

---

## 暂不做 (V2.1+)

- 精确字符 offset
- Source Policy 完整枚举
- SupportStatus 四状态
- FieldProvenance
- 日期对象
- 复杂技能分类
- 完整 Shadow 指标体系

---

## 实施步骤

### Step 1: 新 Schema
`SourceBlock` / `SourceBundle` / `DraftField` / `DraftResume` / `VerifiedResult` / `CanonicalResume`

### Step 2: ResumeComposer
Composer Prompt → DraftResume → evidence_exists 检查

### Step 3: ResumeVerifier
Verifier Prompt → CanonicalResume → conservative_fallback

### Step 4: Basic Validator + Renderer 对接
### Step 5: Shadow Run (v1 + v2 并行)
### Step 6: 三 Case 验收 → 切换

---

## 验收标准

| Case | 必须通过 |
|------|----------|
| Case 1 | 姓名/电话/邮箱正确。无虚构教育。项目不重复。 |
| Case 5 | OCR 姓名/电话保留。无虚构公司/教育。项目不重复。 |
| Case 39 | 学校/专业/学历正确。无虚构公司/职位/日期/技术栈/数字。user_stage=student。 |

# Resume Copilot 架构重构规格

## 动机

当前系统有 25 个核心模块、16,732 行、15+ 处理层。严重问题集中在同一个模式上：

1. LLM 生成了一个字段
2. 字符串规则尝试验证
3. 验证层发现了错误
4. 报告层写了日志
5. 执行层需要猜该删什么

每一层都在修正上一层，层层累积导致以下典型故障：

- FactLedger substring 校验通过，但实体方向完全错误（"全国大学"被当作学校）
- Fabrication Report 检测到了编造，但最终 resume_data 和 DOCX 仍保留
- 同一条项目存了两份，Validator 和 Renderer 各读一个副本
- 修复一个字段清空留下半条幽灵记录

根本原因不是模型能力不足——是同一份数据在不同层之间产生了多个世界观。

---

## 核心设计原则

```
① 证据必须能确定性回指原始 SourceBlock
② 事实支持状态由 Verifier 决定
③ Renderer 永远只消费一个 CanonicalResume
④ Verifier 失败时宁可少输出，不允许未经审核的 Draft 直接上线
```

---

## 最终架构

```
Document Adapter
        │
        ▼
SourceBundle
包含：
block_id
来源
结构
顺序
        │
        ▼
ResumeComposer (LLM Call 1)
输出：
DraftResume
TransformationMode
EvidenceRef (必须引用真实 block_id + offset)
        │
        ▼
Evidence Integrity Check (确定性)
只检查：
  - block_id 存在
  - start/end 合法
  - quote 与原文一致
  - 来源权限 (source policy) 是否合法
        │
        ▼
ResumeVerifier (LLM Call 2)
输出：
CanonicalResume (干净字段，无 GroundedValue)
Provenance (单独保存)
Changes
        │
        ▼
Structural Validator (确定性)
只做：
  - JSON Schema 校验
  - 空记录清理
  - 去重
  - 手机/邮箱格式
不做语义修复
        │
        ▼
Renderer
只读取 CanonicalResume
```

逻辑上只有：提取 → 生成 → 审核 → 确定性结构检查 → 渲染

---

## 第一层：Document Adapter (确定性)

现有 `resume_io.py` 的提取逻辑可以保留。

只做：
- 恢复阅读顺序（已实现）
- 保留 bbox / 段落/表格顺序
- 保留章节结构
- 输出为结构化 SourceBundle

**不做：**
- `_extract_named_entities`
- 学校后缀规则
- 实验室/研究所枚举
- 职位正则猜测

### SourceBlock

```python
class SourceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    source_type: Literal["resume", "query", "jd"]

    text: str
    page: int | None = None
    order: int

    bbox: tuple[int, int, int, int] | None = None

    section_hint: Literal[
        "header", "summary", "education", "experience",
        "project", "skill", "honor", "campus", "unknown",
    ] | None = None

    block_type: Literal[
        "paragraph", "table", "heading", "bullet",
    ]

    parent_block_id: str | None = None
    table_row: int | None = None
    table_col: int | None = None
```

### SourceBundle

```python
class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[SourceBlock]
    user_query: str
    jd_text: str
```

---

## 第二层：ResumeComposer (LLM Call 1)

### 职责

理解原始材料，生成结构化 DraftResume，为每个事实字段附加 evidence。
**Composer 不裁决最终真伪，只声明生成方式。** 也不分配 record_id（由代码确定性赋值）。

### 标签

```python
class TransformationMode(str, Enum):
    DIRECT = "direct"          # 原文逐字提取
    NORMALIZED = "normalized"  # 格式标准化（日期、手机号）
    DERIVED = "derived"        # 合理推断（"OCR识别" → "OCR识别项目"）
    REWRITTEN = "rewritten"    # 语义改写（职责描述优化）
    NONE = "none"              # 无直接来源
```

### EvidenceRef

```python
class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str              # 引用 SourceBlock.block_id
    start: int                 # 在 block.text 中的起始偏移
    end: int                   # 在 block.text 中的结束偏移
    quote: str                 # block.text[start:end] 必须等于这个值
```

### GroundedValue (仅 Draft 阶段使用)

```python
T = TypeVar("T")

class GroundedValue(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    transformation: TransformationMode
    evidence: list[EvidenceRef] = Field(default_factory=list)
```

### Draft 实体

```python
class ExperienceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience_type: Literal[
        "work", "internship", "research", "campus", "volunteer",
    ]
    organization: GroundedValue[str] = GroundedValue(transformation="none")
    role: GroundedValue[str] = GroundedValue(transformation="none")
    period: GroundedValue[str] = GroundedValue(transformation="none")
    bullets: list[GroundedValue[str]] = Field(default_factory=list)
    record_evidence: list[EvidenceRef] = Field(default_factory=list)

class EducationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    school: GroundedValue[str] = GroundedValue(transformation="none")
    degree: GroundedValue[str] = GroundedValue(transformation="none")
    major: GroundedValue[str] = GroundedValue(transformation="none")
    period: GroundedValue[str] = GroundedValue(transformation="none")

class ProjectDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: GroundedValue[str] = GroundedValue(transformation="none")
    organization: GroundedValue[str] = GroundedValue(transformation="none")
    role: GroundedValue[str] = GroundedValue(transformation="none")
    period: GroundedValue[str] = GroundedValue(transformation="none")
    tech_stack: list[GroundedValue[str]] = Field(default_factory=list)
```

**注意：** `record_id` 不由 LLM 输出，由代码在 Pydantic 校验后确定性赋值。

### DraftResume (Composer 输出)

```python
class DraftResume(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: MetaDraft
    education: list[EducationDraft] = Field(default_factory=list)
    experience: list[ExperienceDraft] = Field(default_factory=list)
    projects: list[ProjectDraft] = Field(default_factory=list)
    skills: SkillsDraft
    summary: GroundedValue[str] = GroundedValue(transformation="none")
```

---

## 第三层：Evidence Integrity Check (确定性)

Composer 和 Verifier 之间的薄安全检查，不是新的业务世界观。

```python
def resolve_evidence(
    evidence: EvidenceRef,
    source_blocks: dict[str, SourceBlock],
) -> bool:
    block = source_blocks.get(evidence.block_id)
    if block is None:
        return False
    if not (0 <= evidence.start <= evidence.end <= len(block.text)):
        return False
    return block.text[evidence.start:evidence.end] == evidence.quote
```

这一步可以拦截 Composer "顺便编一条 evidence" 的问题。校验失败的 evidence 标记为 invalid，Verifier 不依赖它做审核依据。

### Source Policy (确定性)

不同字段允许使用不同来源。JD 不能支持候选人事实。

```python
FACT_SOURCE_POLICY: dict[str, set[str]] = {
    # candidate identity — resume or query only
    "meta.name": {"resume", "query"},
    "meta.phone": {"resume", "query"},
    "meta.email": {"resume", "query"},
    "education.school": {"resume", "query"},
    "education.degree": {"resume", "query"},
    "education.major": {"resume", "query"},
    "experience.organization": {"resume", "query"},
    "experience.role": {"resume", "query"},
    "project.name": {"resume", "query"},
    "project.tech_stack": {"resume", "query"},
    # intent — query and jd are valid sources
    "meta.target_role": {"query", "resume", "jd"},
    # expression — can draw from all
    "summary": {"resume", "query", "jd"},
}
```

---

## 第四层：ResumeVerifier (LLM Call 2)

### 职责

输入 SourceBundle + DraftResume，逐字段审核证据链。
**直接返回修正后的 CanonicalResume**，不输出 report。

### SupportStatus (Verifier 产出)

Verifier 给每个字段判定状态：

```python
class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
```

Verifier **不修改** Composer 的 `transformation`（如不将 `derived` 改为 `direct`）。两个维度各自保留：

```json
{
  "value": "OCR识别项目",
  "transformation": "derived",
  "support_status": "supported",
  "evidence": [{"block_id": "resume_12", "start": 5, "end": 12, "quote": "OCR识别"}]
}
```

### 审核规则

**允许：**
- 保留有 direct/normalized/supported_derived 证据的字段
- 根据用户 Query 设置 target_role
- 项目标题合理标准化（"OCR识别" → "OCR识别项目", deried → supported）

**禁止：**
- 新增无原文证据的学校、职位、公司、时间、技术栈、数字结果
- 将奖项/荣誉片段识别为教育经历
- 将通用描述识别为正式组织名称
- 证据来源违反 Source Policy（如 JD 不能支持学校、公司）

### Record Anchor 规则 (确定性)

依赖 Verifier 的 `support_status`，不是 Composer 的 `transformation`：

**Education：** 至少一个 field 的 support_status=supported → 保留记录，清空 unsupported 字段。否则 → 删除整条记录。

**Experience：** 至少 organization 或 role 的 support_status=supported，或有 record_evidence 明确关联 → 保留。否则 → 删除整条记录。

**Project：** 至少 name 的 support_status=supported → 保留。否则 → 删除整条记录。

### 输出

Verifier 输出分三段：

```python
class CanonicalResume(BaseModel):
    """干净字段，不带 GroundedValue。Renderer 和 API 直接消费此结构。"""
    model_config = ConfigDict(extra="forbid")

    meta: MetaOutput
    education: list[EducationOutput] = Field(default_factory=list)
    experience: list[ExperienceOutput] = Field(default_factory=list)
    projects: list[ProjectOutput] = Field(default_factory=list)
    skills: SkillsOutput
    summary: str

class FieldProvenance(BaseModel):
    record_id: str
    field: str
    value: str | None
    transformation: TransformationMode
    support_status: SupportStatus
    evidence: list[EvidenceRef]

class VerificationChange(BaseModel):
    record_id: str
    field: str | None = None          # None 表示整条记录
    action: Literal["clear", "remove_record", "keep"]
    reason_code: str
    reason: str

class VerifiedResumeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume: CanonicalResume
    provenance: list[FieldProvenance]
    changes: list[VerificationChange]
```

Renderer 只读 `bundle.resume`。Provenance 和 Changes 供日志、调试、reply 消费。

---

## 第五层：Structural Validator (确定性，无 LLM)

只允许 structural repair，不允许 semantic repair。

| 校验 | 实现 |
|------|------|
| JSON Schema | Pydantic extra="forbid" + type checks |
| 空记录清理 | 记录中所有字段为 None/空，则删除 |
| 项目去重 | name + organization 去重 |
| 手机号格式 | `PHONE_PATTERN` |
| 邮箱格式 | `EMAIL_PATTERN` |

**不做：**
- 学校后缀验证
- 实验室/研究所枚举
- 职称正则猜测
- 实体方向判断
- 任何语义层面的修复

---

## 错误处理

| 场景 | 处理 |
|------|------|
| Composer structured output 失败 | 同一次请求重试一次 |
| Verifier structured output 失败 | 使用 Conservative Fallback（只保留 direct evidence 明确的字段，删除所有 none/derived/rewritten 字段和新记录） |
| 两次 LLM 都失败 | 返回结构化错误信息 |
| Evidence Integrity 发现无效引用 | 标记为 invalid，Verifier 不依赖 |

Conservative Fallback 原则：宁可少输出，也不允许未经审核的 Draft 直接上线。

---

## 日期处理

period 保持字符串。不做强制日期解析。

只做温和标准化：

```text
2023年9月 → 2023.09
至今 → 至今
```

不强制 mm-yyyy，不自动补月份，不构造 DateRange 对象。强制日期格式放到后续版本。

---

## 实施步骤

### Step 0: 冻结旧链路
保持 `pipeline_v1` 不动。加 feature flag：
```python
RESUME_PIPELINE_VERSION = "v1" | "v2" | "shadow"
```
不一边重构一边删除旧逻辑。

### Step 1: SourceBundle + Schema
- SourceBlock + SourceBundle (含 block_id, section_hint, 表格结构)
- GroundedValue, EvidenceRef, TransformationMode
- Evidence Integrity Check
- Source Policy
- Record ID 代码赋值

### Step 2: ResumeComposer
- Composer System Prompt
- 输出 DraftResume (Pydantic, extra="forbid")
- Structured Output 失败重试逻辑

### Step 3: Evidence Integrity
- 确定性检查：block_id 存在、start/end 合法、quote 一致、source_type 合规
- 失败 evidence 标记 invalid

### Step 4: ResumeVerifier
- Verifier System Prompt
- 输出 VerifiedResumeBundle
- Conservative Fallback

### Step 5: Structural Validator
- 空记录清理、项目去重、手机/邮箱格式

### Step 6: Shadow Run
- 对同一请求同时运行 v1 和 v2
- 用户继续看到 v1
- 后台比较：事实保留率、hallucination 数、关键字段召回、项目重复、幽灵记录、运行时间、Token 消耗

### Step 7: 三 Case 验收
然后扩展到完整 badcase 集

### Step 8: 切换 Renderer
V2 通过后用户切换到 v2 输出

### Step 9: 删除旧模块 (有条件)
最后才删（每步通过测试）：
- `fact_ledger.py`
- `semantic_guard.py`
- `_apply_fabrication_report`
- 旧 optimization 管道
- normalize 中的实体规则

---

## 可以删除或大幅弱化的模块

| 模块 | 当前行数 | 处理方式 |
|------|----------|----------|
| `fact_ledger.py` | 389 | Step 9 删除。provenance 移到 FieldProvenance |
| `semantic_guard.py` | 421 | Step 9 删除。职责给了 Verifier |
| `resume_validator.py` (部分) | 907 | 只保留 check_required_fields + 确定性校验 |
| `resume_product_logic.py` (部分) | 546 | Step 9 删除 normalize 实体规则 |
| `resume_parsing.py` (部分) | 1195 | Composer 替代 structured_resume_from_text |
| `resume_optimization.py` (部分) | 1439 | Step 9 删除 bullet 管道 |
| `resume_scoring.py` | 294 | 保留 |
| `_apply_fabrication_report` | — | Step 9 删除 |
| 多 projects 副本 | — | Step 5 删除。唯一 canonical projects 列表 |

保留：`resume_io.py` (提取)、`resume_renderer.py`、`server_runtime.py`、`llm_gateway.py`、`prompts.py`

---

## 验收标准

| Case | 通过标准 |
|------|----------|
| Case 1 (DOCX+JD) | 姓名/电话/邮箱/target_role 正确。原 CV 无教育 → 不虚构。项目不重复。技能分类合理。 |
| Case 5 (图片+JD) | OCR 正确提取姓名/电话/邮箱。无虚构教育/公司。项目不重复。 |
| Case 39 (纯 Query) | 学校/专业/学历正确。无虚构公司/职位。user_stage=student。科研经历不强制要求工作经历。 |

# V2 MVP 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 15+ 层 pipeline 重构为 5 层（SourceBundle → Composer → Verifier → Validator → Renderer），v1 和 v2 通过 feature flag 并行运行。

**架构：** 两次 LLM 调用 + 确定性校验。Composer 输出带 evidence 的 DraftResume；Verifier 审核后直接输出 CanonicalResume（最终唯一事实来源）；Renderer 只消费 CanonicalResume。

**技术栈：** Python 3.11+，Pydantic v2，FastAPI，结构化输出（StructuredResumeLLMOutput 模式）

---

## 文件结构

### 新建
| 文件 | 职责 |
|------|------|
| `core/v2_schemas.py` | V2 所有 Pydantic 模型（SourceBlock, SourceBundle, DraftField, DraftResume, CanonicalResume 等） |
| `core/source_adapter.py` | 从现有的 resume_io 提取结果构建 SourceBundle |
| `core/resume_composer.py` | LLM Call 1：输入 SourceBundle，输出 DraftResume + Evidence |
| `core/resume_verifier.py` | LLM Call 2：输入 SourceBundle + DraftResume，输出 CanonicalResume |
| `core/v2_pipeline.py` | 编排 5 层 pipeline，feature flag 控制 |
| `tests/test_v2_schemas.py` | Schema 校验测试 |
| `tests/test_v2_pipeline.py` | 集成测试（mock LLM） |

### 修改
| 文件 | 修改内容 |
|------|----------|
| `core/resume_copilot_service.py` | 根据 RESUME_PIPELINE_VERSION flag 路由 v1/v2 |
| `core/resume_copilot_pipeline.py` | 冻结 v1 逻辑（只加 flag，不改结构） |
| `core/resume_renderer.py` | 确保能消费 CanonicalResume；删除嵌套 projects |

### 删除（Step 8）
| 文件 | 说明 |
|------|------|
| `core/semantic_guard.py` | V2.1 删除 |
| `core/fact_ledger.py` | V2.1 删除 |
| `core/_apply_fabrication_report` 逻辑 | 立即删除 |

---

### 任务 1：V2 Schema

**文件：**
- 创建：`core/v2_schemas.py`
- 测试：`tests/test_v2_schemas.py`

- [ ] **步骤 1：编写 SourceBlock + SourceBundle 测试**

```python
"""tests/test_v2_schemas.py"""
import pytest
from pydantic import ValidationError

def test_source_block_requires_block_id():
    from v2_schemas import SourceBlock
    SourceBlock(block_id="b1", source_type="resume", text="陈媛媛 Abbey")
    # must not raise

def test_source_block_rejects_extra_field():
    from v2_schemas import SourceBlock
    with pytest.raises(ValidationError):
        SourceBlock(block_id="b1", source_type="resume", text="test", unknown_field="x")

def test_source_block_rejects_invalid_source_type():
    from v2_schemas import SourceBlock
    with pytest.raises(ValidationError):
        SourceBlock(block_id="b1", source_type="weibo", text="test")

def test_source_bundle_accepts_blocks():
    from v2_schemas import SourceBundle, SourceBlock
    b = SourceBundle(blocks=[
        SourceBlock(block_id="b1", source_type="resume", text="hello"),
    ])
    assert len(b.blocks) == 1

def test_draft_field_defaults():
    from v2_schemas import DraftField
    f = DraftField()
    assert f.value is None
    assert f.mode == "none"
    assert f.evidence == []

def test_draft_field_rejects_bad_mode():
    from v2_schemas import DraftField
    with pytest.raises(ValidationError):
        DraftField(mode="invalid")

def test_evidence_ref_rejects_extra():
    from v2_schemas import EvidenceRef
    with pytest.raises(ValidationError):
        EvidenceRef(block_id="b1", quote="hello", extra=True)

def test_verified_result_has_resume_and_changes():
    from v2_schemas import VerifiedResult, CanonicalResume, Change
    r = VerifiedResult(
        resume=CanonicalResume(meta={"name": ""}, education=[], experiences=[], projects=[], skills=[], summary=""),
        changes=[Change(path="edu[0]", action="remove", reason="不真实")],
    )
    assert len(r.changes) == 1
```

- [ ] **步骤 2：运行测试验证失败**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
PYTHONPATH=core python3 -m pytest tests/test_v2_schemas.py -v --tb=short
```
预期：FAIL（ModuleNotFoundError: No module named 'v2_schemas'）

- [ ] **步骤 3：创建 Schema 实现**

```python
"""core/v2_schemas.py"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional


class SourceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    source_type: Literal["resume", "query", "jd"]
    text: str
    section_hint: Optional[str] = None


class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[SourceBlock]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    quote: str


class DraftField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Optional[str] = None
    mode: Literal["direct", "normalized", "derived", "rewritten", "none"] = "none"
    evidence: list[EvidenceRef] = Field(default_factory=list)


class MetaDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: DraftField = Field(default_factory=DraftField)
    phone: DraftField = Field(default_factory=DraftField)
    email: DraftField = Field(default_factory=DraftField)
    target_role: DraftField = Field(default_factory=DraftField)


class EducationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    school: DraftField = Field(default_factory=DraftField)
    degree: DraftField = Field(default_factory=DraftField)
    major: DraftField = Field(default_factory=DraftField)
    period: DraftField = Field(default_factory=DraftField)


class ExperienceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    organization: DraftField = Field(default_factory=DraftField)
    role: DraftField = Field(default_factory=DraftField)
    period: DraftField = Field(default_factory=DraftField)
    bullets: list[DraftField] = Field(default_factory=list)


class ProjectDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: DraftField = Field(default_factory=DraftField)
    organization: DraftField = Field(default_factory=DraftField)
    role: DraftField = Field(default_factory=DraftField)
    period: DraftField = Field(default_factory=DraftField)


class SkillsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    languages: list[DraftField] = Field(default_factory=list)
    frameworks: list[DraftField] = Field(default_factory=list)
    tools: list[DraftField] = Field(default_factory=list)
    domains: list[DraftField] = Field(default_factory=list)


class DraftResume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: MetaDraft = Field(default_factory=MetaDraft)
    education: list[EducationDraft] = Field(default_factory=list)
    experience: list[ExperienceDraft] = Field(default_factory=list)
    projects: list[ProjectDraft] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: DraftField = Field(default_factory=DraftField)


# ---- Canonical (clean, no DraftField) ----

class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    phone: str = ""
    email: str = ""
    target_role: str = ""
    work_experience: str = ""


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")
    school: str = ""
    degree: str = ""
    major: str = ""
    period: str = ""


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")
    organization: str = ""
    role: str = ""
    period: str = ""
    bullets: list[str] = Field(default_factory=list)


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    organization: str = ""
    role: str = ""
    period: str = ""


class CanonicalResume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    action: Literal["clear", "remove", "replace"]
    reason: str


class VerifiedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume: CanonicalResume
    changes: list[Change] = Field(default_factory=list)
```

- [ ] **步骤 4：运行测试验证通过**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_schemas.py -v --tb=short
```
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add core/v2_schemas.py tests/test_v2_schemas.py
git commit -m "feat(v2): add schema models (SourceBlock, DraftResume, CanonicalResume)"
```

---

### 任务 2：SourceAdapter

**文件：**
- 创建：`core/source_adapter.py`
- 测试：`tests/test_v2_schemas.py`（追加）

- [ ] **步骤 1：编写测试**

```python
def test_build_source_bundle_from_cv_text():
    from source_adapter import build_source_bundle
    bundle = build_source_bundle(
        cv_text="陈媛媛Abbey 188-8888-8888\n工作经历\n超级公司",
        query_text="帮我的简历优化",
        jd_text="",
    )
    assert len(bundle.blocks) >= 2
    resume_blocks = [b for b in bundle.blocks if b.source_type == "resume"]
    assert any("陈媛媛" in b.text for b in resume_blocks)
    query_blocks = [b for b in bundle.blocks if b.source_type == "query"]
    assert len(query_blocks) == 1

def test_source_block_id_unique():
    from source_adapter import build_source_bundle
    bundle = build_source_bundle(cv_text="line1\nline2", query_text="q", jd_text="")
    ids = [b.block_id for b in bundle.blocks]
    assert len(ids) == len(set(ids))

def test_build_source_bundle_empty_cv():
    from source_adapter import build_source_bundle
    bundle = build_source_bundle(cv_text="", query_text="query only", jd_text="")
    assert len(bundle.blocks) >= 1
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_schemas.py::test_build_source_bundle_from_cv_text -v --tb=short
```
预期：FAIL

- [ ] **步骤 3：实现 SourceAdapter**

```python
"""core/source_adapter.py"""
from __future__ import annotations
import re
from v2_schemas import SourceBlock, SourceBundle


def _split_into_blocks(text: str, source_type: str, section_hint: str | None = None) -> list[SourceBlock]:
    """Split text into SourceBlocks by newline."""
    blocks: list[SourceBlock] = []
    for i, line in enumerate(text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        blocks.append(SourceBlock(
            block_id=f"{source_type}_{i}",
            source_type=source_type,  # type: ignore
            text=line,
            section_hint=section_hint,
        ))
    return blocks


def build_source_bundle(
    cv_text: str,
    query_text: str,
    jd_text: str,
) -> SourceBundle:
    blocks: list[SourceBlock] = []

    # CV/resume text blocks
    if cv_text.strip():
        blocks.extend(_split_into_blocks(cv_text, "resume"))

    # Query as single block
    if query_text.strip():
        blocks.append(SourceBlock(
            block_id="query_0",
            source_type="query",
            text=query_text.strip(),
        ))

    # JD text
    if jd_text.strip():
        blocks.append(SourceBlock(
            block_id="jd_0",
            source_type="jd",
            text=jd_text.strip(),
        ))

    return SourceBundle(blocks=blocks)
```

- [ ] **步骤 4：运行测试验证通过**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_schemas.py::test_build_source_bundle_from_cv_text -v --tb=short
```
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add core/source_adapter.py
git commit -m "feat(v2): add SourceAdapter to build SourceBundle from text"
```

---

### 任务 3：Evidence 完整性检查

**文件：**
- 创建：添加到核心函数（放在 `core/resume_composer.py` 中）
- 测试：`tests/test_v2_schemas.py`（追加）

- [ ] **步骤 1：编写测试**

```python
def test_evidence_exists_valid():
    from v2_schemas import SourceBlock, EvidenceRef
    from resume_composer import evidence_exists
    blocks = [SourceBlock(block_id="b1", source_type="resume", text="陈媛媛 Abbey")]
    ref = EvidenceRef(block_id="b1", quote="陈媛媛")
    assert evidence_exists(ref, blocks)

def test_evidence_exists_invalid_block_id():
    from v2_schemas import EvidenceRef
    from resume_composer import evidence_exists
    ref = EvidenceRef(block_id="b_nonexist", quote="text")
    assert not evidence_exists(ref, [])

def test_evidence_exists_quote_missing():
    from v2_schemas import SourceBlock, EvidenceRef
    from resume_composer import evidence_exists
    blocks = [SourceBlock(block_id="b1", source_type="resume", text="陈媛媛 Abbey")]
    ref = EvidenceRef(block_id="b1", quote="北京大学")
    assert not evidence_exists(ref, blocks)
```

- [ ] **步骤 2：在 `resume_composer.py` 中实现 `evidence_exists`**

```python
"""core/resume_composer.py"""
from __future__ import annotations

from v2_schemas import SourceBlock, EvidenceRef


def evidence_exists(ref: EvidenceRef, blocks: list[SourceBlock]) -> bool:
    """Check that the evidence quote actually exists in the referenced block.
    This is a deterministic check — NOT another LLM call."""
    block = next((b for b in blocks if b.block_id == ref.block_id), None)
    if block is None:
        return False
    return ref.quote in block.text
```

- [ ] **步骤 3：运行测试验证通过**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_schemas.py::test_evidence_exists_valid -v --tb=short
```

- [ ] **步骤 4：Commit**

```bash
git add core/resume_composer.py
git commit -m "feat(v2): add evidence_exists deterministic check"
```

---

### 任务 4：ResumeComposer (LLM Call 1)

**文件：**
- 修改：`core/resume_composer.py`
- 修改：`core/prompts.py`（追加 Composer prompt）
- 测试：`tests/test_v2_pipeline.py`

- [ ] **步骤 1：编写 Composer Prompt**

追加到 `core/prompts.py`：

```python
# ── V2 Resume Composer ──

RESUME_COMPOSER_SYSTEM_PROMPT = """你是一名简历结构解析器。
请将提供的简历文本精确解析为结构化 JSON，不遗漏任何章节。

关键规则：
1. 每个字段必须附带 evidence，evidence 必须引用原始文本中的 quote
2. 字段来源分级：
   - direct: 文本中存在完全一致的原文
   - normalized: 格式标准化（日期、电话）
   - derived: 合理推断（"OCR识别" → "OCR识别项目"）
   - rewritten: 语义改写（重新组织职责描述）
   - none: 无直接来源，系统推断
3. target_role 可以从 Query 或 JD 提取
4. 禁止编造学校、公司、职位、时间、数字成果
5. 禁止将荣誉奖项中的片段识别为学校名称
6. 输出必须是严格 JSON，不要额外解释"""
```

- [ ] **步骤 2：实现 Composer LLM 调用**

在 `core/resume_composer.py` 中：

```python
from __future__ import annotations

import json
import logging

from server_runtime import call_llm_typed, llm_enabled, sanitize_user_text
from prompts import RESUME_COMPOSER_SYSTEM_PROMPT
from v2_schemas import (
    SourceBundle, SourceBlock, EvidenceRef, DraftResume,
    MetaDraft, EducationDraft, ExperienceDraft, ProjectDraft,
    SkillsDraft, DraftField,
)

logger = logging.getLogger(__name__)


def evidence_exists(ref: EvidenceRef, blocks: list[SourceBlock]) -> bool:
    block = next((b for b in blocks if b.block_id == ref.block_id), None)
    if block is None:
        return False
    return ref.quote in block.text


def _strip_invalid_evidence(draft: DraftResume, blocks: list[SourceBlock]) -> None:
    """Remove evidence refs that don't resolve to real text."""
    def _clean(field: DraftField) -> None:
        field.evidence = [e for e in field.evidence if evidence_exists(e, blocks)]

    for edu in draft.education:
        for f in (edu.school, edu.degree, edu.major, edu.period):
            _clean(f)
    for exp in draft.experience:
        for f in (exp.organization, exp.role, exp.period):
            _clean(f)
        for b in exp.bullets:
            _clean(b)
    for proj in draft.projects:
        for f in (proj.name, proj.organization, proj.role, proj.period):
            _clean(f)
    _clean(draft.summary)
    for f in (draft.meta.name, draft.meta.phone, draft.meta.email, draft.meta.target_role):
        _clean(f)


def compose_resume(source: SourceBundle) -> DraftResume:
    """Call LLM to produce DraftResume from source material."""
    if not llm_enabled():
        return DraftResume()

    # Build source text for LLM input
    parts = []
    for block in source.blocks:
        parts.append(f"[{block.block_id}] {block.text}")
    source_text = "\n".join(parts)

    prompt = (
        "请将以下简历文本解析为结构化 JSON。每个字段必须附带证据引用。\n\n"
        "【原始材料】\n"
        f"{source_text}"
    )

    try:
        parsed = call_llm_typed(
            DraftResume,
            RESUME_COMPOSER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("ResumeComposer LLM call failed: %s", exc)
        return DraftResume()

    if not isinstance(parsed, dict) or not parsed:
        return DraftResume()

    draft = DraftResume(**parsed)

    # Strip invalid evidence
    _strip_invalid_evidence(draft, source.blocks)

    return draft
```

- [ ] **步骤 3：编写测试**

```python
"""tests/test_v2_pipeline.py"""
import pytest

def test_composer_returns_draft_on_failure():
    from resume_composer import compose_resume
    from v2_schemas import SourceBundle, SourceBlock
    source = SourceBundle(blocks=[
        SourceBlock(block_id="cv_0", source_type="resume", text="陈媛媛 产品经理"),
    ])
    # Without LLM, should return empty draft
    draft = compose_resume(source)
    assert draft.meta.name.value is None

def test_evidence_exists():
    from resume_composer import evidence_exists
    from v2_schemas import SourceBlock, EvidenceRef
    blocks = [SourceBlock(block_id="b1", source_type="resume", text="北京大学硕士")]
    assert evidence_exists(EvidenceRef(block_id="b1", quote="北京大学"), blocks)
    assert not evidence_exists(EvidenceRef(block_id="b1", quote="清华大学"), blocks)
    assert not evidence_exists(EvidenceRef(block_id="b_none", quote="test"), blocks)
```

- [ ] **步骤 4：运行测试验证通过**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_pipeline.py -v --tb=short
```

- [ ] **步骤 5：Commit**

```bash
git add core/resume_composer.py core/prompts.py tests/test_v2_pipeline.py
git commit -m "feat(v2): ResumeComposer — LLM call 1 with evidence checking"
```

---

### 任务 5：ResumeVerifier (LLM Call 2) + Conservative Fallback

**文件：**
- 创建部分：`verifier` 函数添加到新文件或 composer 文件
- 修改：`core/prompts.py`（追加 Verifier prompt）
- 测试：追加到 `tests/test_v2_pipeline.py`

- [ ] **步骤 1：编写 Verifier Prompt**

追加到 `core/prompts.py`：

```python
# ── V2 Resume Verifier ──

RESUME_VERIFIER_SYSTEM_PROMPT = """你是一名简历事实审核专家。
请审核 DraftResume 中的每个字段是否被原始证据支持。

审核规则：
1. 允许：
   - 保留 evidence 引用有效的字段
   - format 标准化日期/电话
   - 项目标题合理标准化（"OCR识别" → "OCR识别项目"）
2. 禁止：
   - 新增无原文证据的学校、公司、职位、时间、技术栈、数字成果
   - 将荣誉奖项中的片段识别为教育经历（如"全国大学"来自"全国大学生创新创业大赛"）
   - 将通用描述识别为正式组织名称
   - 保留 evidence 为空的字段值
3. 判断标准：
   - mode=direct/normalized：通常是可信的
   - mode=derived：需要原文有相关内容支撑
   - mode=none/rewritten 且无直接证据：应清空
4. 整条记录如果没有任何可信字段（如整条 education 的 school/degree/major/period 全部无证据），应删除整条记录

输出：直接输出修正后的 JSON，不要输出审核报告。"""
```

- [ ] **步骤 2：实现 Verifier**

```python
# 在 core/resume_verifier.py 中

from __future__ import annotations

import json
import logging

from server_runtime import call_llm_typed, llm_enabled
from prompts import RESUME_VERIFIER_SYSTEM_PROMPT
from v2_schemas import (
    SourceBundle, DraftResume, VerifiedResult, CanonicalResume,
    Change, Meta, Education, Experience, Project,
)

logger = logging.getLogger(__name__)


def conservative_fallback() -> VerifiedResult:
    """When Verifier fails, return empty safe result — never return unverified Draft."""
    return VerifiedResult(
        resume=CanonicalResume(
            meta=Meta(),
            education=[],
            experiences=[],
            projects=[],
            summary="",
        ),
        changes=[Change(path="*", action="remove", reason="Verifier failed, emitted empty fallback")],
    )


def verify_resume(source: SourceBundle, draft: DraftResume) -> VerifiedResult:
    """Call LLM to verify and produce CanonicalResume."""
    if not llm_enabled():
        # No LLM available, build conservative result
        return conservative_fallback()

    # Build prompt with source + draft
    source_parts = [f"[{b.block_id}] {b.text}" for b in source.blocks]
    draft_json = draft.model_dump_json(exclude_none=True)

    prompt = (
        "请审核以下 DraftResume，输出修正后的最终简历。\n\n"
        "【原始材料】\n"
        f"{chr(10).join(source_parts)}\n\n"
        "【DraftResume】\n"
        f"{draft_json}"
    )

    try:
        parsed = call_llm_typed(
            VerifiedResult,
            RESUME_VERIFIER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("ResumeVerifier LLM call failed: %s", exc)
        return conservative_fallback()

    if not isinstance(parsed, dict) or not parsed:
        return conservative_fallback()

    try:
        result = VerifiedResult(**parsed)
        return result
    except Exception as exc:
        logger.warning("ResumeVerifier output validation failed: %s", exc)
        return conservative_fallback()
```

- [ ] **步骤 3：编写测试**

```python
def test_conservative_fallback_empty():
    from resume_verifier import conservative_fallback
    result = conservative_fallback()
    assert result.resume.meta.name == ""
    assert result.resume.education == []
    assert len(result.changes) == 1

def test_verifier_returns_fallback_on_no_llm():
    from resume_verifier import verify_resume
    from v2_schemas import SourceBundle, SourceBlock, DraftResume
    source = SourceBundle(blocks=[])
    draft = DraftResume()
    result = verify_resume(source, draft)
    assert result.resume.meta.name == ""
```

- [ ] **步骤 4：运行测试**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_pipeline.py -v --tb=short
```

- [ ] **步骤 5：Commit**

```bash
git add core/resume_verifier.py core/prompts.py
git commit -m "feat(v2): ResumeVerifier — LLM call 2 with conservative fallback"
```

---

### 任务 6：Basic Validator

**文件：**
- 创建：`core/v2_validator.py`
- 测试：追加到 `tests/test_v2_pipeline.py`

- [ ] **步骤 1：编写测试**

```python
def test_validator_removes_empty_education():
    from v2_validator import validate_resume
    from v2_schemas import CanonicalResume, Meta, Education
    resume = CanonicalResume(
        meta=Meta(name="张三"),
        education=[
            Education(school="", degree="", major="", period=""),
            Education(school="北京大学", degree="", major="", period=""),
        ],
    )
    result = validate_resume(resume)
    assert len(result.education) == 1
    assert result.education[0].school == "北京大学"

def test_validator_deduplicates_projects():
    from v2_validator import validate_resume
    from v2_schemas import CanonicalResume, Project
    resume = CanonicalResume(
        projects=[
            Project(name="智能家居", organization="", role="", period=""),
            Project(name="智能家居", organization="", role="", period=""),
            Project(name="OCR识别", organization="", role="", period=""),
        ],
    )
    result = validate_resume(resume)
    assert len(result.projects) == 2
```

- [ ] **步骤 2：实现 Validator**

```python
"""core/v2_validator.py"""
from __future__ import annotations
import re
from v2_schemas import CanonicalResume, Education, Project


def _is_empty_edu(edu: Education) -> bool:
    return not any([edu.school, edu.degree, edu.major])


def validate_resume(resume: CanonicalResume) -> CanonicalResume:
    # Remove empty education records
    resume.education = [e for e in resume.education if not _is_empty_edu(e)]

    # Deduplicate projects by name
    seen: set[str] = set()
    deduped: list[Project] = []
    for p in resume.projects:
        key = p.name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)
    resume.projects = deduped

    # Phone format (basic)
    if resume.meta.phone:
        phones = re.findall(r"1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}", resume.meta.phone)
        if phones:
            resume.meta.phone = phones[0].replace(" ", "").replace("-", "")

    return resume
```

- [ ] **步骤 3：运行测试**

```bash
PYTHONPATH=core python3 -m pytest tests/test_v2_pipeline.py -v --tb=short
```

- [ ] **步骤 4：Commit**

```bash
git add core/v2_validator.py
git commit -m "feat(v2): Basic Validator — empty records, dedup, phone format"
```

---

### 任务 7：V2 Pipeline + Feature Flag

**文件：**
- 创建：`core/v2_pipeline.py`
- 修改：`core/resume_copilot_service.py`

- [ ] **步骤 1：实现 v2_pipeline.py**

```python
"""core/v2_pipeline.py"""
from __future__ import annotations

import logging
import os
from v2_schemas import VerifiedResult, CanonicalResume
from source_adapter import build_source_bundle
from resume_composer import compose_resume
from resume_verifier import verify_resume
from v2_validator import validate_resume

logger = logging.getLogger(__name__)


def run_v2_pipeline(
    cv_text: str,
    query_text: str,
    jd_text: str,
) -> VerifiedResult:
    """Run the V2 5-layer pipeline. Returns VerifiedResult or fallback."""
    # Layer 1: SourceBundle
    source = build_source_bundle(cv_text, query_text, jd_text)
    logger.info("V2 | SourceBundle: %d blocks", len(source.blocks))

    # Layer 2: Composer
    draft = compose_resume(source)
    logger.info("V2 | DraftResume: %d edu, %d exp, %d proj",
                len(draft.education), len(draft.experience), len(draft.projects))

    # Layer 3: Verifier
    result = verify_resume(source, draft)
    logger.info("V2 | VerifiedResult: %d education, %d experiences, %d changes",
                len(result.resume.education), len(result.resume.experiences), len(result.changes))

    # Layer 4: Validator
    result.resume = validate_resume(result.resume)

    return result
```

- [ ] **步骤 2：在 `resume_copilot_service.py` 中添加 feature flag**

找到 `resume_copilot_service` 路由函数（约 760 行附近）：

```python
_RESUME_PIPELINE_VERSION = os.environ.get("RESUME_PIPELINE_VERSION", "v1").strip()

def _get_pipeline_version() -> str:
    return _RESUME_PIPELINE_VERSION
```

在 `resume_copilot_service` 函数中，在 `stage_ingest` 和 `stage_classify` 之后添加分支：

```python
# At the top of resume_copilot_service.py:
import os
_RESUME_PIPELINE_VERSION = os.environ.get("RESUME_PIPELINE_VERSION", "v1").strip()
```

在 route 函数中（大约 ctx = await stage_classify 之后）：

```python
    if _RESUME_PIPELINE_VERSION == "v2":
        # V2 pipeline
        try:
            from v2_pipeline import run_v2_pipeline
            v2_result = run_v2_pipeline(
                cv_text=ctx.cv_text,
                query_text=ctx.query_text,
                jd_text=ctx.jd_text,
            )
            # Map back to pipeline context (temporary bridge)
            ctx.resume_data = v2_result.resume.model_dump()
            ctx.fabrication_report = None
            ctx.missing_fields = []
        except Exception as exc:
            logger.error("V2 pipeline failed, falling back to V1: %s", exc)
            # Fall through to V1
            if ctx.has_cv:
                ctx = await rewrite_path(ctx)
            else:
                ctx = await generate_path(ctx)
    else:
        # V1 pipeline
        if ctx.has_cv:
            ctx = await rewrite_path(ctx)
        else:
            ctx = await generate_path(ctx)
```

- [ ] **步骤 3：Commit**

```bash
git add core/v2_pipeline.py
git commit -m "feat(v2): pipeline orchestration + feature flag in service"
```

---

### 任务 8：Shadow Run 支持

**文件：**
- 修改：`core/resume_copilot_service.py`

- [ ] **步骤 1：在服务中添加 shadow 模式**

在 `resume_copilot_service` 中：

```python
    if _RESUME_PIPELINE_VERSION in ("v2", "shadow"):
        try:
            from v2_pipeline import run_v2_pipeline
            v2_result = run_v2_pipeline(
                cv_text=ctx.cv_text,
                query_text=ctx.query_text,
                jd_text=ctx.jd_text,
            )
            if _RESUME_PIPELINE_VERSION == "shadow":
                # Run V1 as primary, V2 for comparison
                logger.info("SHADOW | V2 produced %d edu, %d exp",
                            len(v2_result.resume.education),
                            len(v2_result.resume.experiences))
            else:
                # V2 is primary
                ctx.resume_data = v2_result.resume.model_dump()
                ctx.fabrication_report = None
                ctx.missing_fields = []
        except Exception as exc:
            logger.error("V2 pipeline failed: %s", exc)
            if _RESUME_PIPELINE_VERSION == "v2":
                # Fall through to V1
                ...
    else:
        # V1
        ...
```

- [ ] **步骤 2：Commit**

```bash
git commit -m "feat(v2): add shadow run mode for v1/v2 comparison"
```

---

### 任务 9：删除 `_apply_fabrication_report` + 清理嵌套 projects

**文件：**
- 修改：`core/resume_copilot_pipeline.py`

- [ ] **步骤 1：删除 `_apply_fabrication_report`**

找到并移除 `_apply_fabrication_report` 函数和 `final_fact_guard` 中对它的调用。`final_fact_guard` 恢复为：

```python
def final_fact_guard(source_truth_text, resume_data, has_cv=True, *, max_iterations=1, ledger=None):
    if not source_truth_text.strip():
        return resume_data, FabricationReport(fabrication_found=False, details=[])
    fab = check_fabrication_heuristic(source_truth_text, resume_data)
    return resume_data, fab
```

- [ ] **步骤 2：删除嵌套 projects**

在 `rewrite_path` 中 `normalize` 之后的 `_dedup_projects` 调用和函数本身都删除（v2 不用这个逻辑了，v1 冻结不动）。

- [ ] **步骤 3：Commit**

```bash
git add core/resume_copilot_pipeline.py
git commit -m "refactor: remove _apply_fabrication_report (v2 replaces this)"
```

---

### 任务 10：Renderer 对接 CanonicalResume

**文件：**
- 修改：`core/resume_renderer.py`

- [ ] **步骤 1：确保 Renderer 能消费 CanonicalResume**

当前 Renderer 的函数签名（如 `export_resume_files`）接收 `resume_data: dict`。CanonicalResume 的 `.model_dump()` 输出应与现有格式兼容。

主要差异：
- `experiences`（复数）vs `experience`（单数）
- 没有嵌套 `experience[*].projects`
- 技能使用 `SkillsDraft` 格式

在 `v2_pipeline.py` 中，输出前做一次格式兼容：

```python
def _canonical_to_v1_format(canonical: CanonicalResume) -> dict:
    data = canonical.model_dump()
    # Rename experiences -> experience for renderer compat
    data["experience"] = data.pop("experiences", [])
    # Flatten skills
    skills = data.get("skills", {})
    if isinstance(skills, dict):
        for key in ("languages", "frameworks", "tools", "domains"):
            if isinstance(skills.get(key), list):
                skills[key] = [s.get("value", "") if isinstance(s, dict) else s for s in skills[key]]
    return data
```

在 `run_v2_pipeline` 最后：

```python
    result.resume_dict = _canonical_to_v1_format(result.resume)
    return result
```

- [ ] **步骤 2：Commit**

```bash
git add core/v2_pipeline.py
git commit -m "feat(v2): add format bridge for renderer compatibility"
```

---

### 任务 11：端到端集成验证

**文件：** 无修改，纯测试

- [ ] **步骤 1：启动容器并运行 3 个 badcase**

```bash
docker run -d --rm --name test-v2 \
  -e MODELHUB_BASE_URL="http://172.17.0.1:8003/v1" \
  -e RESUME_PIPELINE_VERSION=v2 \
  -e PORT=8000 \
  -e OUTPUT_DIR=/root/app/output \
  -e DEBUG_OUTPUT_DIR=/root/app/debug_out \
  -p 8008:8000 \
  -v /tmp/badcase-verify:/root/app/badcase-0715:ro \
  --entrypoint python3 \
  harbor-contest.4pd.io/zengzhitao/resume-copilot:fix-v6 \
  main.py
```

- [ ] **步骤 2：跑 Case 1/5/39**

使用之前同样的 case 脚本验证。

- [ ] **步骤 3：检查结果**

```text
Case 1: 姓名/电话/邮箱保留。无虚构教育。
Case 5: OCR 姓名/电话保留。无虚构公司/教育。
Case 39: 学校/专业正确。无虚构公司/职位。
```

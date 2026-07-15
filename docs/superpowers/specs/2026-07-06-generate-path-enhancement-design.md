# generate_path 增强 + 模板水印过滤设计

## 背景

评估 12 个样本发现 4 类问题，其中 P0/P1 问题需要在本轮修复：

- **类别 A（P0）**：generate_path 输出太弱 — 无 CV 场景下 LLM 过于保守，用户 query 中有明确信息但不敢写入 resume 字段
- **类别 B/C（P1）**：LLM 输出中存在知页/WonderCV 模板水印和 Demo 占位数据
- **类别 D（P2）**：JD URL 解析失败（nowcoder 等 SPA 站）— 暂不修复，需额外引入 Chromium

## 改动 1: generate_path prompt 放宽（generate_resume_with_llm_from_profile）

### 文件
`resume_copilot_service.py` → `generate_resume_with_llm_from_profile()`

### 改动内容
system_prompt 的硬约束第 3 条从"不得猜测"改为"从描述中提取"：

```python
# 旧
"3) 缺失的姓名、电话、邮箱、学校、学历、时间、岗位、成果、技能必须留空或保留原文弱表达，不得猜测。"

# 新
"3) 没有原始简历，完全依赖用户的文字描述。请从描述中提取所有可用的经历、技能、项目信息填入对应字段。
    只留空不能从描述中推断的内容（姓名、电话、邮箱等个人联系方式），其余字段尽量填充。"
```

同时第 6 条末尾加：
```python
"6) ...（保留旧内容）..."
"7) 输出必须使用中文。"
"8) 不得使用知页、WonderCV 等模板网站的示例占位数据。"
```

### 风险
- 用户描述过于模糊时可能偶发编造，但 FactLedger 已兜底
- 无 CV 场景用户本就期待"AI 帮我写出来"，适度放宽是合理的

## 改动 2: 三层的 fallback 链（generate_path）

### 文件
`resume_copilot_pipeline.py` → `generate_path()`

### 改动内容
在 LLM profile 生成后，增加输出长度/内容密度判断，过短时回退到 `structured_resume_from_text`（旧版本就有的第二层 LLM 解析）：

```python
def _profile_output_too_short(resume_data: dict) -> bool:
    """Check if profile output is too sparse — less than 3 bullet points."""
    total_bullets = sum(
        len(r.get("bullets", []) or [])
        for r in resume_data.get("experience", [])
    )
    text_len = len(json.dumps(resume_data, ensure_ascii=False))
    return total_bullets < 3 or text_len < 500

# generate_path 中
resume_data = generate_resume_with_llm_from_profile(...)
if not resume_data or _profile_output_too_short(resume_data):
    # 第二层：LLM 把 query 当简历内容解析
    resume_data = structured_resume_from_text(ctx.generation_text)
if not resume_data:
    # 第三层：规则解析（兜底）
    resume_data = heuristic_resume_from_text(...)
```

## 改动 3: 模板水印过滤

### 文件
`resume_copilot_pipeline.py` → 新增 `_clean_template_watermarks()`

### 黑名单正则

```python
_KNOWN_TEMPLATE_PATTERNS = {
    "email": re.compile(
        r"(abbey@wondercv\.com|job@weiapp\.com|job@wonder\.com|"
        r"hr@\w+\.com|career@\w+\.com|recruit@\w+\.com)"
    ),
    "phone": re.compile(
        r"(13800000000|188-?8888-?8888|15900000000|"
        r"13600000000|010-?8888-?8888)"
    ),
    "company": re.compile(
        r"(超级公司|知页|WonderCV)",
    ),
}
```

### 逻辑
1. 在 `stage_render` 中，渲染 `ctx.resume_data` 前调用 `_clean_template_watermarks()`
2. 全字段递归扫描（meta/experience/education/skills/projects 等）
3. 匹配到黑名单模式的字段值 → 置空
4. 产生一条 `template_warning` 追加到 `ctx.ocr_warnings`

### 额外的 prompt 约束
同时在 `generate_resume_with_llm_from_profile` 和 LLM optimize prompt 中加入：「不得使用知页、WonderCV 等模板网站的示例占位数据」

## 验证方法

1. 用 badcase-0702.zip 中 12 个样本重新生成
2. 检查类别 A 的 6 个样本（31/41/51/91/101/111）输出长度是否 > 500 chars
3. 检查类别 B/C 的 4 个样本（11/21/71/81）不再出现知页水印和 demo 数据
4. 回归：上一轮修复的 4 个 badcase 不退化

## 不改的部分

- **JD URL 解析**（nowcoder 等 SPA 站）：需 puppeteer/Chromium（~500MB 镜像体积），留作后续单独优化
- **OCR CLS 模型**、**query 泄漏**、**行业分类**、**bullet 去重**、**技能分类**：上一轮已修复，本轮不涉及

# Query 泄漏修复 + generate_path 阈值调优 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。

**目标：** 修复 2 个倒退——query 泄漏到 resume 字段（样本71）和 generate_path 空输出（样本41）

**架构：** 1 个文件 3 处改动：(1) 在 `resume_copilot_pipeline.py` 新增 `_clean_query_leakage()` LLM 字段清理函数；(2) 在 `stage_render` 中调用；(3) 在 `generate_path` 中加调试日志排查 `_profile_output_too_short` 未触发问题

**技术栈：** Python, vLLM (Qwen3)

---

## 文件修改清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `resume_copilot_pipeline.py:110-130` | 新增 | `_clean_query_leakage()` 函数 |
| `resume_copilot_pipeline.py:920-928` | 修改 | `stage_render` 中水印清洗后调用 query 清理 |
| `resume_copilot_pipeline.py:806-810` | 修改 | `generate_path` 中 profile 生成后加调试日志 |

---

### 任务 1：新增 `_clean_query_leakage()` 函数

**文件：** `resume_copilot_pipeline.py`（约第 110 行，在水印函数之后）

- [ ] **步骤 1：新增函数**

在 `_clean_template_watermarks` 函数之后（约第 105 行），添加：

```python
def _clean_query_leakage(resume_data: dict, query_text: str) -> dict:
    """Use LLM to check and clean query instruction text leaked into resume fields.

    When the pipeline mixes query_text with cv_text, the LLM parser may write
    user instructions (e.g. "帮我改改我的简历") into resume fields like company
    name, role, or description. This function:
    1. Serializes resume_data to JSON
    2. Sends it to LLM with the original query_text
    3. LLM returns a cleaned JSON with query-text fields blanked
    4. Falls back to original data on failure
    """
    if not resume_data or not query_text or not llm_enabled():
        return resume_data
    try:
        resume_json = json.dumps(resume_data, ensure_ascii=False, default=str)
        system_prompt = (
            "你是一名简历质检员。检查简历 JSON 中的每一个字符串字段，判断其是否被用户的"
            "提问/指令文本污染，而非真实的简历内容。\n\n"
            "常见泄漏模式：\n"
            "1. 公司名/学校名出现用户提问的句子（例如 "帮我改改"、"生成一份"）\n"
            "2. 岗位名出现 query 中的指令性文字\n"
            "3. 项目描述或工作职责中出现 query 原文片段\n"
            "4. summary 或技能中出现用户的问题文本\n\n"
            "规则：\n"
            "- 明显是用户指令/提问的字段值 → 清空为 ""\n"
            "- 如果字段只有占位词（"未明确"/"待补充"/"目标投递"等）→ 清空\n"
            "- 正常的简历内容原样保留，不确定时也保留\n"
            "- 输出完整的清洗后简历 JSON，结构不能改变"
        )
        prompt = (
            f"用户原始查询（可能有泄漏到简历中）：\n{query_text}\n\n"
            f"当前简历 JSON：\n{resume_json}\n\n"
            "请输出清洗后的简历 JSON（仅修改被污染的字段，其他不变）："
        )
        cleaned = call_llm_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=4096,
        )
        if cleaned:
            # Parse the result — it might be wrapped in ```json ... ``` or plain JSON
            cleaned_clean = cleaned.strip()
            if "```json" in cleaned_clean:
                cleaned_clean = cleaned_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_clean:
                cleaned_clean = cleaned_clean.split("```")[1].split("```")[0].strip()
            parsed = json.loads(cleaned_clean)
            if isinstance(parsed, dict):
                logger.info("Query leakage cleaning applied to resume_data")
                return parsed
    except Exception as exc:
        logger.warning("_clean_query_leakage failed: %s", exc)
    return resume_data
```

- [ ] **步骤 2：验证修改**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
grep -n "_clean_query_leakage" resume_copilot_pipeline.py
```
预期输出：显示函数定义行。

- [ ] **步骤 3：Commit**

```bash
git add resume_copilot_pipeline.py
git commit -m "fix: 新增 _clean_query_leakage LLM字段清理函数"
```

---

### 任务 2：在 `stage_render` 中调用 query 清理

**文件：** `resume_copilot_pipeline.py:920-928`

- [ ] **步骤 1：插入调用代码**

在 `stage_render` 中，水印清洗之后、`reply_text = _build_llm_reply(...)` 之前插入：

```python
    # ── Clean template watermarks before rendering ──
    _watermark_warnings = _clean_template_watermarks(ctx.resume_data)
    if _watermark_warnings:
        for _w in _watermark_warnings:
            ctx.ocr_warnings.append({"source": "template_watermark", "message": _w})
        logger.info("Cleaned %d template watermark(s)", len(_watermark_warnings))

    # ── Clean query leakage from resume data fields ──
    if ctx.query_text and llm_enabled():
        _cleaned = _clean_query_leakage(ctx.resume_data, ctx.query_text)
        if _cleaned and _cleaned is not ctx.resume_data:
            ctx.resume_data = _cleaned

    reply_text = _build_llm_reply(
```

- [ ] **步骤 2：验证修改**

```bash
grep -n "query_leakage\|query_text and llm_enabled" /mnt/disk1/zengzhitao/resume-copilot-server-acceptance/resume_copilot_pipeline.py
```
预期输出：显示 `stage_render` 中的调用行。

- [ ] **步骤 3：Commit**

```bash
git add resume_copilot_pipeline.py
git commit -m "fix: 在stage_render中添加query泄漏LLM清洁调用"
```

---

### 任务 3：generate_path 调试日志 + 阈值排查

**文件：** `resume_copilot_pipeline.py:806-810`

- [ ] **步骤 1：在 profile 生成后添加调试日志**

```python
    if llm_enabled():
        from resume_copilot_service import generate_resume_with_llm_from_profile
        resume_data = generate_resume_with_llm_from_profile(
            query_text=ctx.query_text, jd_text=ctx.jd_text,
            scenario=ctx.scenario, industry=ctx.industry,
            target_role=ctx.target_role, user_stage=ctx.user_stage,
        )
    if resume_data:
        try:
            _dbg = len(json.dumps(resume_data, ensure_ascii=False))
            _blt = _count_resume_bullets(resume_data)
            logger.info("generate_path profile: %d chars, %d bullets, keys=%s",
                        _dbg, _blt, list(resume_data.keys()))
        except Exception as exc:
            logger.info("generate_path profile debug log failed: %s", exc)
```

- [ ] **步骤 2：验证修改**

```bash
grep -n "generate_path profile:" resume_copilot_pipeline.py
```
预期输出：显示日志行。

- [ ] **步骤 3：Commit**

```bash
git add resume_copilot_pipeline.py
git commit -m "fix: generate_path添加profile输出调试日志"
```

---

### 任务 4：构建 Docker 镜像并测试 9 个样本

- [ ] **步骤 1：构建镜像**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
COMMIT_HASH=$(git rev-parse --short HEAD)
docker build -t resume-copilot-test:$COMMIT_HASH .
```

- [ ] **步骤 2：启动测试容器**

```bash
docker stop resume-copilot-test 2>/dev/null; docker rm resume-copilot-test 2>/dev/null
docker run -d --name resume-copilot-test \
  --gpus '"device=1"' \
  -p 5100:80 \
  -v /mnt/disk1/zengzhitao/data/Qwen3-14B-GPTQ-Int4:/model \
  resume-copilot-test:$COMMIT_HASH
```

等待就绪后，运行验证脚本。

- [ ] **步骤 3：运行完整验证**

```bash
cd /mnt/disk1/zengzhitao/tmp/badcase-0702-extracted && python3 test_badcases.py 2>&1
```

- [ ] **步骤 4：对比结果**

比较 9 个样本的新旧版本（从首次构建 2274127 到本次构建）的字符数和内容质量，重点关注：
- 样本 71：query 泄漏是否清除
- 样本 41：generate_path 输出是否恢复（>200 chars）
- 样本 31/51：generate_path 是否有改善
- 样本 1/11/21：是否没有新的退化

- [ ] **步骤 5：生成对比报告**

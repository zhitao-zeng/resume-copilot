# Query 泄漏修复 + generate_path 阈值调优 设计

## 背景

上一轮修复（prompt 放宽 + 3层fallback + 水印过滤）在 9 个完整测试样本中发现 2 个倒退：

- **样本 71**：rewrite_path 输出从 907 chars 跌倒 128 chars，query 指令文本被写入 experience 的公司字段（"简历有点太水了"作为公司名）
- **样本 41**：generate_path 输出从 289 chars 跌倒 88 chars，3 层 fallback 未正确触发
- 水印清除后 rewrite_path 输出变短是预期行为，不修复

## 改动 1: LLM 字段清理 — 修复 query 泄漏

### 文件
`resume_copilot_pipeline.py` — 新增 `_clean_query_leakage()` + 在 `stage_render` 中调用

### 做法
在 `stage_render` 中、水印清洗 (`_clean_template_watermarks`) **之后**、reply 生成 **之前** 调用。

**新增函数：**
```python
def _clean_query_leakage(resume_data: dict, query_text: str) -> dict:
    """Use LLM to check if any resume_data field contains query instruction text
    (not actual resume content) and clear those fields.
    
    Returns cleaned resume_data dict.
    """
    prompt = (
        "你正在检查一份简历数据的字段是否被用户的指令文字污染。\n\n"
        "用户原始查询：\n{query}\n\n"
        "当前简历 JSON：\n{resume_json}\n\n"
        "请检查：是否有任何字段的值明显来自用户查询的指令文本（而非真实简历内容）？\n"
        "常见泄漏模式：\n"
        "1. 公司名/学校名/岗位名中出现了用户提问的句子片段\n"
        "2. 项目描述中出现 query 的指令性语言（"帮我改改"、"生成一份"等）\n"
        "3. summary 或技能中出现 query 文本\n\n"
        "输出规则：\n"
        "- 只清空明显是 query 指令的字段，正常的简历内容原样保留\n"
        "- 如果字段只有"未明确"/"待补充"/"目标投递"等占位词，也清空\n"
        "- 不确定时留空字段，不要保留可疑内容\n"
        "- 输出完整的清理后的简历 JSON，结构不变"
    )
    # 调用 LLM...
```

**调用位置**（`stage_render` 中）：
```python
    # ── Clean template watermarks before rendering ──
    _watermark_warnings = _clean_template_watermarks(ctx.resume_data)
    ...
    # ── Clean query leakage from resume data fields ──
    if ctx.query_text and llm_enabled():
        try:
            _cleaned = _clean_query_leakage(ctx.resume_data, ctx.query_text)
            if _cleaned:
                ctx.resume_data = _cleaned
        except Exception as exc:
            logger.warning("Query leakage cleanup failed: %s", exc)
```

### 风险
- 额外一次 LLM 调用（+5-10s/请求）
- 可能误伤正常的字段内容
- 缓解：prompt 偏保守，不确定时保留

## 改动 2: _profile_output_too_short 阈值调优

### 问题
样本 41（88 chars）未触发 fallback，说明判断逻辑有问题。

### 排查方法
在 `generate_path` 中 profile 生成后加日志：
```python
resume_data = generate_resume_with_llm_from_profile(...)
if resume_data:
    _dbg = len(json.dumps(resume_data, ensure_ascii=False)) if isinstance(resume_data, dict) else -1
    _blt = _count_resume_bullets(resume_data)
    logger.info("generate_path profile: %d chars, %d bullets, keys=%s",
                _dbg, _blt, list(resume_data.keys()))
```

根据日志定位具体 bug 后修复。可能的问题：
- A: `json.dumps` 抛异常 → 函数返回 True → 但外层 try 捕获有问题
- B: `_count_resume_bullets` 统计口径不同，88 chars 的 dict 恰好有 ≥3 bullets
- C: LLM profile 返回了非 dict（如 None/空 list）→ `_profile_output_too_short(not_a_dict)` 可能报错

## 不改的部分

- **rewrite_path 输出变短**：水印清除后，伪数据被清空是真正常行为，不修
- **JD URL 解析**：nowcoder 等 SPA 站点需额外引入 Chromium，不修
- **generate_path 内容扩展**：当前 focus 是 fix regression，后续再考虑 richness

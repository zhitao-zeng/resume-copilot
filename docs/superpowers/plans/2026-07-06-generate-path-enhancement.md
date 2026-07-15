# generate_path 增强 + 模板水印过滤 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复 generate_path（无 CV 场景）输出太弱和 LLM 模板水印泄露两个 P0/P1 问题

**架构：** 3 处修改：(1) 放宽 `generate_resume_with_llm_from_profile` 的 prompt，让 LLM 敢于从用户描述中提取事实；(2) 在 `generate_path` 中恢复旧版本的 3 层 fallback 链；(3) 增加模板水印后处理过滤 + prompt 约束

**技术栈：** Python, re, vLLM (Qwen3)

---

## 文件修改清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `resume_copilot_service.py:698-708` | 修改 | 放宽 system_prompt 第3条 + 增加水印约束 |
| `resume_copilot_pipeline.py:739-757` | 修改 | 3 层 fallback 链 |
| `resume_copilot_pipeline.py:55-70` | 修改 | 新增常量 + 导入 |
| `resume_copilot_pipeline.py:835-860` | 修改 | 在 stage_render 中调用水印过滤 |

---

### 任务 1：修改 generate_path prompt

**文件：** `resume_copilot_service.py:698-708`

- [ ] **步骤 1：替换 system_prompt 文本**

将第 3 条硬约束从"不得猜测"改写为"从描述中提取"，增加第 8 条禁止模板水印：

**旧：**
```python
    system_prompt = (
        "你是一位多行业简历生成专家，负责把用户提供的个人事实整理为可投递简历 JSON。\n"
        "硬约束：\n"
        "1) 只能把【用户事实】写成候选人已有经历、学校、岗位、技能、证书、时间、数字结果。\n"
        "2) 【目标JD】只能用于确定表达方向、关键词侧重和个人总结，不得写成候选人已具备事实。\n"
        "3) 缺失的姓名、电话、邮箱、学校、学历、时间、岗位、成果、技能必须留空或保留原文弱表达，不得猜测。\n"
        "4) 原文没有数字结果时，不得编造百分比、金额、人数、病例数、学生数、客户数。\n"
        "5) 个人总结必须不超过100字，且要匹配目标行业。\n"
        "6) 输出必须是结构化简历 JSON，不要输出解释或 markdown。\n"
        "7) 输出语言必须与用户输入保持一致：用户用中文，则全部字段用中文输出。"
    )
```

**新：**
```python
    system_prompt = (
        "你是一位多行业简历生成专家，负责把用户提供的个人事实整理为可投递简历 JSON。\n"
        "硬约束：\n"
        "1) 只能把【用户事实】写成候选人已有经历、学校、岗位、技能、证书、时间、数字结果。\n"
        "2) 【目标JD】只能用于确定表达方向、关键词侧重和个人总结，不得写成候选人已具备事实。\n"
        "3) 没有原始简历，完全依赖用户的文字描述。请从描述中提取所有可用的经历、技能、项目信息填入对应字段。"
        "只留空不能从描述中推断的内容（姓名、电话、邮箱等个人联系方式），其余字段尽量填充。\n"
        "4) 原文没有数字结果时，不得编造百分比、金额、人数、病例数、学生数、客户数。\n"
        "5) 个人总结必须不超过100字，且要匹配目标行业。\n"
        "6) 输出必须是结构化简历 JSON，不要输出解释或 markdown。\n"
        "7) 输出语言必须与用户输入保持一致：用户用中文，则全部字段用中文输出。\n"
        "8) 不得使用知页、WonderCV 等模板网站的示例占位数据，必须使用描述中真实的用户信息。"
    )
```

- [ ] **步骤 2：验证修改**

```bash
grep -n "不得猜测\|完全依赖用户的文字描述" /mnt/disk1/zengzhitao/resume-copilot-server-acceptance/resume_copilot_service.py
```
预期输出：第 703 行显示新内容，不再有"不得猜测"字样。

- [ ] **步骤 3：Commit**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
git add resume_copilot_service.py
git commit -m "fix: 放宽generate_path prompt约束，允许从用户描述中提取事实"
```

---

### 任务 2：恢复 3 层 fallback 链

**文件：** `resume_copilot_pipeline.py:739-757`

- [ ] **步骤 1：在 `generate_path` 中添加 `_profile_output_too_short` 和 3 层 fallback**

在 `generate_path` 函数顶部（`resume_copilot_pipeline.py` 约第 739 行，`def generate_path` 后），添加 `_profile_output_too_short` 辅助函数：

```python
def _profile_output_too_short(resume_data: dict) -> bool:
    """Check if LLM profile output is too sparse to use.

    Returns True when the output has fewer than 3 total bullet points
    or the serialized JSON is shorter than 500 chars — both indicate
    the LLM was too conservative and didn't extract enough from the query.
    """
    total_bullets = sum(
        len(r.get("bullets", []) or [])
        for r in resume_data.get("experience", []) if isinstance(r, dict)
    )
    total_bullets += sum(
        len(p.get("bullets", []) or [])
        for p in resume_data.get("projects", []) if isinstance(p, dict)
    )
    text_len = len(json.dumps(resume_data, ensure_ascii=False))
    return total_bullets < 3 or text_len < 500
```

确认 `json` 已在 pipeline.py 顶部 import（第 11 行有 `import re` — 确认 `json` 是否已导入）：

```bash
head -20 /mnt/disk1/zengzhitao/resume-copilot-server-acceptance/resume_copilot_pipeline.py | grep json
```

如果 `json` 未导入，在 `import re` 后加一行 `import json`。

然后修改 `generate_path` 中的 fallback 逻辑（约第 754 行）——用 3 层替代当前的 2 层：

**旧：**
```python
    if not resume_data:
        resume_data = product_logic.heuristic_resume_from_text(
            ctx.generation_text, ctx.industry, ctx.target_role,
        )
```

**新：**
```python
    if not resume_data or _profile_output_too_short(resume_data):
        # 第2层: LLM把query文本当原始简历内容解析
        try:
            resume_data = structured_resume_from_text(ctx.generation_text)
        except Exception as exc:
            logger.warning("structured_resume_from_text fallback failed: %s", exc)
            resume_data = {}
    if not resume_data:
        # 第3层: 规则解析（兜底）
        resume_data = product_logic.heuristic_resume_from_text(
            ctx.generation_text, ctx.industry, ctx.target_role,
        )
```

- [ ] **步骤 2：验证修改**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
grep -n "structured_resume_from_text\|heuristic_resume_from_text\|_profile_output_too_short" resume_copilot_pipeline.py | head -10
```

预期输出：`structured_resume_from_text` 在 import 行 + `generate_path` 中均有出现。
`_profile_output_too_short` 出现在 `generate_path` 函数前或函数内。

- [ ] **步骤 3：Commit**

```bash
git add resume_copilot_pipeline.py
git commit -m "fix: generate_path恢复3层fallback链 - LLM profile→LLM parse→规则解析"
```

---

### 任务 3：添加模板水印过滤

**文件：** `resume_copilot_pipeline.py`

- [ ] **步骤 1：在文件顶部（约第 65 行，imports 结束处）添加常量 + 辅助函数**

```python
# ── Template watermark patterns ──
_TEMPLATE_WATERMARK_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(
        r"(abbey@wondercv\.com|job@weiapp\.com|job@wonder\.com|"
        r"hr@\w+\.com|career@\w+\.com|recruit@\w+\.com)",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(13800000000|188-?8888-?8888|15900000000|"
        r"13600000000|010-?8888-?8888)",
    ),
    "company": re.compile(r"(超级公司|知页|WonderCV)"),
}

_name_and_contact_pattern = re.compile(r"知页\S+男|知页\S+女")


def _clean_template_watermarks(resume_data: dict) -> list[str]:
    """Scan all fields in resume_data for known watermark/template placeholders
    and clear them. Returns a list of warning messages."""
    warnings: list[str] = []

    def _scan_value(value: Any, path: str = "") -> Any:
        """Recursively scan a value for watermark patterns."""
        if isinstance(value, str):
            for name, pattern in _TEMPLATE_WATERMARK_PATTERNS.items():
                if pattern.search(value):
                    warnings.append(f"已清除占位数据（{name}）位于 {path}")
                    return ""
            if _name_and_contact_pattern.search(value):
                warnings.append(f"已清除占位数据（模板水印）位于 {path}")
                return ""
            return value
        if isinstance(value, dict):
            return {k: _scan_value(v, f"{path}.{k}") for k, v in value.items()}
        if isinstance(value, list):
            return [_scan_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value

    _scan_value(resume_data)
    return warnings
```

- [ ] **步骤 2：在 `stage_render` 中调用水印过滤**

在 `stage_render` 函数中，`missing_dict = _dicts(ctx.missing_fields)` 之后（第 838 行后）、`reply_text = _build_llm_reply(...)` 之前插入：

```python
    # ── Clean template watermarks before rendering ──
    _watermark_warnings = _clean_template_watermarks(ctx.resume_data)
    if _watermark_warnings:
        for _w in _watermark_warnings:
            ctx.ocr_warnings.append({"source": "template_watermark", "message": _w})
        logger.info("Cleaned %d template watermark(s)", len(_watermark_warnings))
```

- [ ] **步骤 3：验证修改**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
grep -n "_clean_template_watermarks\|_TEMPLATE_WATERMARK" resume_copilot_pipeline.py
```

预期输出：显示常量定义、`_clean_template_watermarks` 函数，以及在 `stage_render` 中的调用。

```bash
python3 -c "
import re
ctx = {
    'meta': {'email': 'abbey@wondercv.com', 'phone': '13800000000', 'name': '张三'},
    'experience': [{'company': '超级公司', 'role': '产品经理'}],
}
_TEMPLATE_WATERMARK_PATTERNS = {
    'email': re.compile(r'(abbey@wondercv\.com|job@weiapp\.com|job@wonder\.com|hr@\w+\.com|career@\w+\.com|recruit@\w+\.com)', re.IGNORECASE),
    'phone': re.compile(r'(13800000000|188-?8888-?8888|15900000000|13600000000|010-?8888-?8888)'),
    'company': re.compile(r'(超级公司|知页|WonderCV)'),
}
_name_and_contact_pattern = re.compile(r'知页\S+男|知页\S+女')
def _scan_value(value, path='', warnings=None):
    if warnings is None: warnings = []
    if isinstance(value, str):
        for name, pattern in _TEMPLATE_WATERMARK_PATTERNS.items():
            if pattern.search(value):
                warnings.append(f'  清除: {path}')
                return '', warnings
        if _name_and_contact_pattern.search(value):
            warnings.append(f'  清除: {path}')
            return '', warnings
        return value, warnings
    if isinstance(value, dict):
        for k, v in value.items():
            value[k], _ = _scan_value(v, f'{path}.{k}', warnings)
        return value, warnings
    if isinstance(value, list):
        for i, item in enumerate(value):
            value[i], _ = _scan_value(item, f'{path}[{i}]', warnings)
        return value, warnings
    return value, warnings
w = []
result, w = _scan_value(ctx, 'root', w)
print('Warnings:', w)
print('Result:', result)
assert result['meta']['email'] == '', 'email should be cleared'
assert result['meta']['phone'] == '', 'phone should be cleared'
assert result['experience'][0]['company'] == '', 'company should be cleared'
print('All assertions passed')
"
```

预期输出：3 条 warning 消息 + "All assertions passed"。

- [ ] **步骤 4：Commit**

```bash
git add resume_copilot_pipeline.py
git commit -m "fix: 添加模板水印后处理过滤（知页/WonderCV/demo数据）"
```

---

### 任务 4：验证 12 个 badcase

- [ ] **步骤 1：运行 Docker 测试脚本（如果已有容器）或启动测试容器**

检查当前是否有可用的测试容器：

```bash
docker ps -a | grep resume-copilot
```

如果已有旧容器在跑，停掉并用最新镜像启动测试容器（GPU 3，端口 5100）：

```bash
docker stop resume-copilot-test 2>/dev/null; docker rm resume-copilot-test 2>/dev/null
docker run -d --name resume-copilot-test \
  --gpus '"device=3"' \
  -p 5100:1025 \
  harbor-contest.4pd.io/zengzhitao/resume-copilot:latest
```

等待容器就绪（约 60 秒）：

```bash
sleep 60 && curl -s http://localhost:5100/ready
```

- [ ] **步骤 2：提取 12 个样本的 manifest.json 或构造测试输入**

```bash
cd /mnt/disk1/zengzhitao/tmp/badcase-0702

# 针对无 CV 样本（31/41/51/91/101/111）构造测试请求
# 对每个样本发送 POST 请求检查输出
```

可以通过已有的 `test_badcase.sh` 或手动调用 3 个代表性样本：

**样本 31（无 CV/JD，纯 query）：**
```bash
curl -s -X POST http://localhost:5100/process \
  -F "query=帮我生成一份产品经理简历。我现在在一家创业公司做产品负责人，带过3个人的小团队，主要负责从0到1搭建企业内部管理系统。之前还做过两年运营，对用户增长和数据分析比较熟悉。学历是本科，计算机相关专业。" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('score:', d.get('score')); print('reply:', d.get('reply_text','')[:200])"
```

**样本 111（无 CV，有 JD）：**
```bash
curl -s -X POST http://localhost:5100/process \
  -F "query=根据我目标岗位的jd生成一般你觉得有竞争力的简历，尽量可以覆盖jd提到的点。" \
  -F "target_jd=一、岗位职责 1、负责工程部日常资料管理，包含施工图纸、变更签证、工程联系单、报审报验资料的整理、归档、扫描、台账登记。 2、跟进施工现场进度，协助工程师日常巡查、质量检查、安全巡检，做好每日施工日志、现场问题记录。 3、对接施工单位、监理单位，完成资料对接、流程报送、消息传达、整改事项跟进闭环。 4、负责工程会议筹备、会议记录撰写、整改问题汇总、进度节点统计，输出周报、月报、进度台账等。 5、协助完成工程签证、工程量核对、现场拍照留底、材料进场验收辅助工作。 6、负责部门办公用品、工程物资台账、文件报审、流程审批跟进，完成领导交办的工程类辅助工作。" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('score:', d.get('score')); print('reply:', d.get('reply_text','')[:200])"
```

**样本 21（有 CV 的知页水印检测）：**
```bash
curl -s -X POST http://localhost:5100/process \
  -F "query=结合岗位职责和任职要求，帮我优化简历并提升通过率。" \
  -F "cv=@cv/21_cv.docx" \
  -F "target_jd=主要职责 战略法律咨询与风险管理：..." \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('score:', d.get('score')); print('warnings:', [w for w in d.get('ocr_warnings',[]) if 'watermark' in w.get('source','') or '占位' in w.get('message','')])"
```

- [ ] **步骤 3：下载生成的 DOCX 并检查质量**

```python
# 对 generate_path 样本检查输出长度
# 预期：之前 58-289 chars → 现在至少 500+ chars
```

- [ ] **步骤 4：确认知页水印已被清除**

检查样本 11/21/71/81 的输出：
- 邮箱、电话字段不再出现 `job@weiapp.com` / `13800000000`
- 公司字段不再出现 `知页` / `超级公司`
- ctx.ocr_warnings 中包含 `template_watermark` 来源的提示

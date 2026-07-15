# Resume Copilot Acceptance Testset

这套测试集用于接入 LLM 后的端到端效果验收。它不是无 LLM heuristic 的基准集，最终结论以真实 `/resume-copilot` 输出的 DOCX 和 `reply_text` 为准。

## 内容

- `cases.jsonl`：48 条端到端 case
- `cases.csv`：便于人工浏览的摘要表
- `manifest.json`：场景、行业、用户阶段、文件格式覆盖统计
- `qa_report.json`：fixture 文件结构校验和 DOCX 渲染 QA 状态
- `files/cv/`：原始简历 fixture，覆盖 DOCX/PDF/PNG
- `files/jd/`：目标 JD fixture，覆盖 TXT/DOCX/PDF/PNG
- `files/templates/`：模板 fixture，覆盖 DOCX/PDF/PNG
- `run_api_testset.py`：批量调用 `/resume-copilot` 的脚本

## 覆盖范围

- 场景 1：有原始简历 + 有目标 JD
- 场景 2：无原始简历 + 无 JD + 有个人信息
- 场景 3：有原始简历 + 目标岗位说明
- 场景 4：无原始简历 + 有 JD + 有个人信息

行业覆盖：

- 产研
- 运营
- 医生
- 老师
- 销售/售前
- 金融
- 设计
- 教育
- 其他

## Case 字段

- `id`：case 编号
- `scenario`：预期业务场景
- `industry`：预期行业
- `user_stage`：预期用户阶段
- `query`：用户输入文本
- `cv_path`：原始简历文件路径，可为空
- `target_jd`：目标 JD 文本，可为空
- `target_jd_file_path`：目标 JD 文件路径，可为空
- `cv_template_path`：简历模板路径，可为空
- `expected_missing_fields`：预期应提示补充的字段
- `expected_conflicts`：预期应提示确认的冲突
- `forbidden_fabrication`：不得出现在最终简历里的编造内容
- `expected_output`：交付要求，默认 DOCX、最多 3 页、LLM 后目标分 90+

## 运行方式

先启动服务：

```bash
python main.py
```

再跑测试集：

```bash
python acceptance_testset/run_api_testset.py --base-url http://127.0.0.1:8001
```

只跑前 3 条 smoke：

```bash
python acceptance_testset/run_api_testset.py --base-url http://127.0.0.1:8001 --limit 3
```

输出：

- 控制台打印汇总指标
- `results.json` 保存每条 case 的响应、耗时、分数、DOCX 路径和回复文本
- 每条 row 的 `validation_failures` 会记录场景/行业不匹配、预期缺失项未提示、预期冲突未提示、禁止编造词出现、DOCX 缺失、回复缺失等问题

## 验收口径

优先看：

- `usable_rate`：单条分数 90+ 的比例
- `average_score`
- `failure_rate`
- `validation_failure_count`
- `average_elapsed_s`，单请求应小于 8 分钟

人工复核重点：

- 是否编造公司、学校、岗位、时间、学历、数字成果
- 缺失姓名、电话、邮箱、学历、教育时间、技能时是否在 `reply_text` 里提示
- 时间重叠或冲突时是否提示确认
- DOCX 是否可编辑、结构清晰、最多 3 页
- 个人总结是否不超过 100 字
- 工作/项目经历是否按 STAR 和行业表达策略改写

## QA 状态

已完成：

- 48 条 case 的引用路径校验
- 57 个 fixture 文件结构性打开校验
- DOCX/PDF/PNG/TXT 格式覆盖检查

未完成：

- DOCX 视觉渲染 QA。本地 LibreOffice 渲染器缺少 `liblcms2.2.dylib`，无法生成页面 PNG。fixture 文件本身可被 `python-docx` 打开，后续在镜像或具备完整 LibreOffice 依赖的环境中可重新渲染复核。

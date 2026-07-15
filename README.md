# Resume Copilot Server

验收版主流程面向多行业、多用户阶段的简历生成与优化。主交付格式为可编辑 DOCX，旧接口仍保留兼容，但推荐统一使用 `/resume-copilot`。

## 主接口

- `POST /resume-copilot`
- Form fields:
  - `query`: 用户说明文本，可包含个人信息、求职目标、修改诉求
  - `cv`: 原始简历文件，支持 PDF、DOCX、PNG、JPG 等图片
  - `cv_template`: 简历模板文件，DOCX 优先；PDF/图片模板仅抽取版式偏好并回退标准模板
  - `target_jd`: 目标 JD 文本或链接
  - `target_jd_file`: 目标 JD 文件，支持 PDF、DOCX、图片
  - `target_jd_url` / `jd_url`: 目标 JD 链接
  - `jd_text`: 兼容字段，等同 JD 文本

返回核心字段：

```json
{
  "files": { "docx": "outputs/resume_xxx.docx" },
  "reply_text": "已按场景完成一版可编辑 DOCX...",
  "score": 86.0,
  "missing_fields": [],
  "conflicts": [],
  "scenario": "scenario1",
  "industry": "finance",
  "user_stage": "experienced",
  "perf": { "total_s": 12.3 },
  "score_breakdown": {
    "readability": 8,
    "completeness": 26,
    "expression": 42,
    "response": 10,
    "fabrication": false
  },
  "ocr_warnings": [],
  "user_report": {}
}
```

## 场景覆盖

- 场景 1：有原始简历 + 有目标 JD，优化为更匹配目标岗位的简历
- 场景 2：无原始简历 + 无 JD + 有个人信息，生成标准版简历
- 场景 3：有原始简历 + 目标岗位说明，按目标岗位优化简历表达
- 场景 4：无原始简历 + 有 JD + 有个人信息，生成适配目标 JD 的简历

## 能力范围

- 统一解析 `query`、`cv`、`cv_template`、`target_jd`
- 图片简历、图片 JD、图片模板走本地 OCR
- 多行业策略：产研、运营、医生、老师、销售/售前、金融、设计、教育、其他
- 用户阶段策略：在校生优先校园/项目/实习，职场人优先工作经历和成果
- 事实保真：不新增公司、学校、岗位、时间、学历、数字结果
- 缺失字段和冲突字段进入 `reply_text` 与 `user_report`
- 评分规则对齐验收口径：编造总分 0，可阅读性 10，完整度 30，表达 50，回复 10

## 兼容接口

- `POST /audit-and-optimize`
- `POST /audit-and-optimize/upload`
- `POST /generate`
- `GET /health`
- `GET /models`

旧接口保留原有请求结构，内部评分、校验、渲染能力已补充验收字段。

`/generate` 已作为兼容入口转到 `/resume-copilot` 主编排；JD 不再拼入用户事实文本，只作为目标方向参与生成。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

默认地址：`http://0.0.0.0:8001`

关键运行配置：

- `LLM_TIMEOUT_SECONDS`：单次 LLM 调用超时，默认 `180`
- `REQUEST_TIMEOUT_SECONDS`：单请求总预算，默认 `480`
- `ENABLE_LLM_JSON_REPAIR`：LLM JSON 失败后进行一次纠错重试，默认开启
- `ENABLE_LLM_FAILURE_DUMP`：是否落盘 LLM 失败原文，默认关闭；开启后会做邮箱和手机号脱敏

图片 OCR 需要系统安装 Tesseract。Docker 镜像已安装：

- `tesseract-ocr`
- `tesseract-ocr-chi-sim`
- `tesseract-ocr-eng`

## Docker

```bash
docker build -t resume-copilot-server:acceptance .
docker run --rm -p 8001:8001 resume-copilot-server:acceptance
```

## 测试与评测

单元测试：

```bash
python -m unittest discover -s tests
```

离线评测：

```bash
python eval_runner.py
```

`eval_cases.jsonl` 当前包含 40 条 smoke case，覆盖四类场景、产研/运营/医生/老师/金融等行业、在校生和职场人。离线评测输出 `mode=heuristic_baseline_no_llm`，不代表最终简历效果；它只验证解析、字段完整度、冲突检测、评分规则与报表输出。

接入 LLM 后的端到端验收集位于 `acceptance_testset/`，包含 48 条 case 和 DOCX/PDF/PNG/TXT fixture 文件，可用 `python acceptance_testset/run_api_testset.py --base-url http://127.0.0.1:8001` 批量调用 `/resume-copilot`。
该 runner 会自动检查预期缺失字段、预期冲突、禁止编造词、DOCX 返回和 `reply_text`。

报表指标：

- 简历可用率
- 平均分
- 可阅读性、完整度、表达、回复平均分
- 失败率和平均生成时间

## 代码结构

- `main.py`：FastAPI 入口与兼容接口
- `resume_copilot_service.py`：验收版统一编排
- `resume_product_logic.py`：无外部依赖的纯产品逻辑，用于测试和离线评测
- `resume_io.py`：PDF、DOCX、图片 OCR 文本抽取
- `resume_validator.py`：必填字段、时间冲突、排序和防编造校验
- `resume_scoring.py`：验收评分
- `resume_renderer.py`：DOCX/PDF 渲染
- `eval_runner.py`：固定评测集离线跑分

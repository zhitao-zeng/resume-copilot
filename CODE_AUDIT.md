# Resume Copilot Server — 代码审计报告

> 审计日期：2026-06-12 | 项目路径：`resume-copilot-server-acceptance/`

---

## 一、代码使用层级总览

```
入口点
├── main.py (port 8001)            ← 主入口
│   ├── POST /resume-copilot       → resume_copilot_service  ██████████ 核心流程
│   ├── POST /generate             → 兼容转到 /resume-copilot ████ 兼容链路
│   ├── POST /audit-and-optimize   → resume_service           ████ 旧链路
│   ├── POST /score                → score_resume             ██ 独立评分子链路
│   └── port 80 only               → resume_generate_api      ██ Flask异步模式
├── cli.py                         → resume_service           █ CLI快捷
└── eval_runner.py                 → 纯product_logic+scoring  ██ 离线评测
```

## 二、每个文件的使用状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `resume_copilot_service.py` | 🟢 **核心活跃** | 主接口 `/resume-copilot` 的完整编排 |
| `main.py` | 🟢 **核心活跃** | FastAPI Web 服务器入口 |
| `server_runtime.py` | 🟢 **核心活跃** | 全部符号均被使用（内部引用或跨模块导入） |
| `resume_scoring.py` | 🟢 **核心活跃** | 评分逻辑，被主流程和独立端点调用 |
| `resume_validator.py` | 🟢 **核心活跃** | 必填校验、编造检查、时间冲突 |
| `resume_product_logic.py` | 🟢 **核心活跃** | 纯函数产品逻辑（行业/阶段分类等） |
| `resume_classifier.py` | 🟢 **核心活跃** | LLM 分类入口 |
| `resume_parsing.py` | 🟢 **核心活跃** | 文本→结构化简历 |
| `resume_renderer.py` | 🟢 **核心活跃** | DOCX 导出 |
| `resume_io.py` | 🟢 **核心活跃** | 文件提取（PDF/DOCX/OCR） |
| `llm_gateway.py` | 🟢 **核心活跃** | LLM 调用网关 |
| `json_parsing.py` | 🟢 **核心活跃** | JSON 解析与修复 |
| `audit_logic.py` | 🟢 **核心活跃** | 审计逻辑（主流程和兼容链路都用） |
| `resume_optimization.py` | 🟢 **核心活跃** | 简历优化改写 |
| `resume_common.py` | 🟢 **核心活跃** | 通用辅助函数 |
| `prompts.py` | 🟢 **核心活跃** | 所有 LLM 提示词 |
| `schemas.py` | 🟢 **核心活跃** | Pydantic 数据模型 |
| `http_compat.py` | 🟢 **核心活跃** | FastAPI 兼容层 |
| `drafts.py` | 🟢 **核心活跃** | 草稿持久化 |
| `request_utils.py` | 🟢 **核心活跃** | 请求解析（兼容链路使用） |
| `resume_service.py` | 🟡 **兼容链路活跃** | 旧 `/audit-and-optimize` 端点 |
| `resume_generator.py` | 🟡 **兼容链路活跃** | 旧 `generate_resume_from_profile` 已不被调用，但 `_build_generation_direction` 仍被主流程使用 |
| `resume_generate_api.py` | 🟡 **特定模式** | 仅 `port=80` 时激活的 Flask 异步模式 |
| `cli.py` | 🟡 **特定模式** | CLI 快捷入口，3 个命令 |
| `eval_runner.py` | 🟡 **特定模式** | 离线评测，独立运行 |
| `vllm_openai_proxy.py` | 🟡 **特定模式** | 仅 Docker `entrypoint.sh` + `Dockerfile.vllm-openai` 使用 |

## 三、确认的死代码

### 3.1 从未被调用的函数（Dead Functions）

| 位置 | 函数 | 说明 |
|------|------|------|
| `resume_generator.py` | `generate_resume_from_profile()` | 旧的生成函数，已被 `resume_copilot_service.generate_resume_with_llm_from_profile()` 替代 |
| `llm_gateway.py` | `call_json()` | 通用 JSON 调用接口从未被使用（所有调用都走 `call_typed`） |

### 3.2 未使用的 Import

| 文件 | 未使用的 Import |
|------|----------------|
| `resume_generator.py` | `render_docx`, `GenerateResponse`, `UserStage` |
| `resume_optimization.py` | `DETAIL_HINT_WORDS`, `RESPONSIBILITY_WORDS` |
| `resume_scoring.py` | `FabricationDetail` |
| `main.py` | `Path` |
| `resume_copilot_service.py` | `run_single_optimize_with_audit_pass` |
| `resume_validator.py` | `BaseModel` |
| `resume_validator.py` | `logger` |
| `resume_generate_api.py` | `sanitize_user_text`, `MAX_FILE_SIZE` |
| `resume_io.py` | `ElementTree` |
| `resume_service.py` | `score_resume` |

---

## 四、已修复的流程问题

| # | 问题 | 修复 | 状态 |
|---|------|------|------|
| 1 | `resume_copilot_service.py` 与 `resume_product_logic.py` 间 ~400行完全重复代码 | 删除死代码，统一使用 product_logic | ✅ |
| 2 | 编造清洗在必填校验之后执行 | 将 `final_fact_guard` 移到 `check_required_fields` 之前 | ✅ |
| 3 | 审核可能被调用两次 | 使用 `_has_audit` 标记变量 | ✅ |
| 4 | `score_resume` 内部重复编造检查 | 简化 `fabrication_report` 参数处理逻辑 | ✅ |

## 五、审计发现的问题

| 级别 | 数量 | 主要影响 |
|------|------|----------|
| 🔴 高 | 1 | `LLMGateway.call_text()` 绕过 `_call_raw()` 中的错误处理层 |
| 🔴 中高 | 1 | `_build_user_report` 函数名在 `audit_logic.py` 和 `resume_copilot_service.py` 中冲突（同名但签名/输出完全不同） |
| 🟡 中 | 4 | 错误处理、代码重复、能力不一致 |
| 🟢 低 | 2 | 封装边界（跨模块调用私有函数） |

---

## 六、验证结果

- ✅ 17/17 单元测试全部通过
- ✅ 40/40 离线评测用例正常运行，0 错误
- ✅ Python 语法检查和模块导入验证通过
- ✅ 服务启动正常（uvicorn on port 8001）

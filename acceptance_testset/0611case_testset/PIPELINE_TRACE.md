# S1-FIN-JD-001 完整流水线追踪

Case: 金融专业学生，有 DOCX 简历 + JD，投金融分析师岗位。

---

## Step 1: 文件提取（DOCX → Text）

耗时: 0.024s | cv_financial.docx (49545 bytes)

```
工作/实习经历
南方基金管理有限公司 | 实习分析师
2017.06-2017.12
搜集研究数据及信息，构建财务模型撰写公司点评，并完成2篇行业深度报告
负责团队日报推送，更新上市公司公告，点评军工行业新闻，加深对行业了解
所在团队荣获2017年最佳分析师团队第二名

安永华明会计师事务所深圳分所 | 实习审计师
2016.11-2017.02
参与中石油国际石油勘探开发有限公司的审计项目，团队在9天内高效完成7个事业部的审计工作
协助编制3家子公司的工作底稿及报表，独立完成4册底稿的抽凭检查
核对报表附注，并整理装订案卷近35册

教育经历
香港中文大学（深圳） | 本科 · 金融统计 | 2014.09-2018.06
GPA排名：3.54/4.0（学院前10%）
连续三年获校级奖学金（每年学院1%学生获此殊荣）
核心课程：会计、财务报表分析、线性回归分析、计量经济学、统计推断等

荣誉与奖项
2017年中国工商银行校园商战大赛一等奖
2016年香港中文大学（深圳）The Voice 第三名
2015年"万人之上"中国大学生金融挑战赛全国总决赛第三名
2014年经管学院案例比赛第一名

专业技能：MS Office、会声会影、Goldwave、SPSS、Stata、VBA、SQL、RapidMiner、MAMP
英语：CET-6（625分）
推荐人：张小明 安永华明会计师事务所深圳分所合伙人 | 徐扬生 香港中文大学（深圳）校长
```

---

## Step 2: JD 文本

```
行业研究员 / 金融分析师
岗位职责：
- 负责行业研究、公司调研，撰写深度研究报告和公司点评
- 构建财务模型，进行估值分析和投资逻辑梳理
任职要求：
- 金融、统计、经济等相关专业本科及以上学历
- 具备数据分析能力，熟练掌握 SQL、Excel、Python、R 等工具
- 有券商、基金、投行人实习经历者优先
```

---

## Step 3: OCR 质量检查

```json
{"acceptable": true, "score": 100, "zh_ratio": 0.68, "noise_ratio": 0.011}
```

✅ 高质量 DOCX 文本，无需降级

---

## Step 4: source_truth_text（编造检测基准）

2341 chars = query + cv_text 拼接。编造检测时所有 resume_data 字段必须能在此文本中找到对应。

---

## Step 5: LLM OCR 文本清理

耗时: **24.4s** | 原始 2341 → 清理后 12354 chars

```
Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** OCR Text Repair Expert.
    *   **Task:** Clean up noisy OCR text into a coherent resume text.
    ...
    *   **Input:** A messy OCR text containing resume content with repetitions, broken lines, and some noise.

2.  **Analyze the Input Text:**
    There are repeated blocks of text. For example, the internship details (Southern Fund, EY) appear multiple times.
    There is a header "项目经历" followed by "2014.09-2018.06 香港中文大学（深圳）金融统计本科..." which is actually Education.
    There is a header "推荐人..." which seems to belong to the contact info.
    "数独拉丁舞丁若虚" — "丁若虚" might be a name or OCR error, but rule 4 says preserve all information.
```

⚠️ **问题**: Qwen3.5-9B 又输出思考链了！OCR 清理的 system prompt 要求"只输出文本"，但模型还是在前面加了 Thinking Process。

---

## Step 6: LLM 分类（行业/阶段/岗位）

耗时: **2.0s** | confidence=0.9 | used_llm=True

| 分类 | 值 | 证据 |
|------|-----|------|
| industry | **finance** | `金融统计`(cv), `南方基金管理有限公司`(cv), `金融分析师`(query) |
| user_stage | **experienced** | `2017年`(cv), `香港中文大学本科`(cv) |
| target_role | **金融分析师** | `金融分析师`(query), `行业研究员/金融分析师`(jd) |

---

## Step 7: LLM 结构化简历解析

耗时: **15.7s**

```json
{
  "meta": {"name":"", "email":"abcdefg@cn.ey.com", "phone":"136-0000-0000", "education_level":"本科", "work_experience":"1年", "target_role":"金融分析师"},
  "summary": "...",
  "experience": [
    {"company":"南方基金管理有限公司", "role":"实习分析师", "period":"2017.06-2017.12"},
    {"company":"安永华明会计师事务所深圳分所", "role":"实习审计师", "period":"2016.11-2017.02"}
  ],
  "education": [
    {"school":"香港中文大学", "period":"2014.09-2018.06"},
    {"school":"徐扬生香港中文大学", "period":"2016.11-2017.02"}  ← ⚠️ 导师名被当成学校
  ],
  "projects": [
    {"name":"香港中文大学（深圳）金融统计本科", "period":"2014.09-2018.06"},
    {"name":"巴厘岛义工支教志愿服务", "period":"2016.05-2016.07"},
    {"name":"职业发展协会", "period":"2015.05-2016.05"},
    {"name":"科研项目"}  ← ⚠️ LLM 编造的空壳项目
  ],
  "skills": {"languages":["CET-6（625分）"], "tools":["MS Office","SPSS","Stata","VBA","SQL"...], "domains":["金融","审计","数据分析"]},
  "publications": [
    {"title":"5. Preserve section headers (e.g., "教育经历", "项目经历") on their own line."},  ← ⚠️ LLM思考泄露!
    {"title":"If I include "我想投..." in the output, it might look weird as a resume."},
    {"title":"The first line "我想投..." is likely a mistake in the prompt construction."},
    {"title":"4. **Structure:** Keep headers on their own line."}
  ]
}
```

⚠️ **发现 3 个问题**:
1. `徐扬生香港中文大学` 被当成独立教育经历（实为推荐人信息）
2. `科研项目` 是 LLM 编造的空壳项目名称
3. `publications` 里混入了 LLM 的内部思考文本

---

## Step 8: 标准化（normalize_resume_data_for_product）

- 日期归一化: `2017.06-2017.12` → `06-2017 - 12-2017`
- Experience 按时间倒序排列
- Summary 重写为规范版本
- user_stage fix: work_years=1 → `experienced`

标准化后 experience:
```
南方基金管理有限公司 | 实习分析师 | 06-2017 - 12-2017
  → 搜集研究数据及信息，构建财务模型撰写公司点评，完成行业深度报告...
  → 所在团队荣获 2017 年最佳分析师团队第二名。
安永华明会计师事务所深圳分所 | 实习审计师 | 11-2016 - 02-2017
  → 参与中石油审计项目，协助编制工作底稿及报表...
```

---

## Step 9: LLM 审核（audit_resume_core）

耗时: **9.7s** | overall_score=2.1

| 维度 | 得分 | 问题 |
|------|------|------|
| technical_depth | 2.5/10 | "构建模型"未说明 DCF/LBO 类型 |
| quantification | 3.0/10 | "第二名"无基线/团队规模 |
| responsibility_clarity | 3.5/10 | "日报推送"模糊 |
| authenticity | 2.0/10 | 条目重复 + OCR 乱码 |

关键审核意见:
- 🔴 未说明具体财务模型类型（DCF、LBO）和估值方法
- 🔴 "团队获奖第二名"缺少个人贡献占比
- 🟡 简历中有 OCR 识别乱码影响专业度
- 🟡 GPA 排名未与金融分析能力直接关联

---

## Step 10: LLM 优化（optimize_resume_core）

耗时: **15.9s** | 产出 5 处 changes

| 位置 | 优化内容 |
|------|----------|
| 摘要 | 加入 `GPA 3.54/4.0（前10%），具备 DCF 三表预测及估值分析能力` |
| publications[0-3] | 清除 4 条 LLM 思考泄露 → 清空 |

⚠️ 优化只改了摘要和清除了 publication 垃圾，**没有改 experience 的子弹内容**。因为 LLM 审核报告过于苛刻（overall_score=2.1），Qwen3.5-9B 可能认为需要"深度改写"但受限于事实保真约束。

---

## Step 11: 编造检测与清洗

### 原始检测
found=True, details=2:
- `科研项目`: 该项目名称未出现在用户原始输入中 ✅ 正确检测
- `CET-6（625分）`: 技能项未出现在原始输入中 ⚠️ 实际在CV里有

### final_fact_guard 清洗
found=False, details=0 | 耗时 0.004s

`科研项目` 被移除。`CET-6（625分）` 在 source_truth 中有 `CET-6（625分）` → 第二次检查通过。

---

## Step 12: 必填校验 + 冲突检测

### 缺失字段（5个）
| field | label | reason |
|-------|-------|--------|
| meta.name | 姓名 | 必填 |
| education[0].degree | 学位 | edu第1段无学位 |
| education[0].major | 专业名称 | edu第1段无专业 |
| education[1].degree | 学位 | edu第2段无学位 |
| education[1].major | 专业名称 | edu第2段无专业 |

### 冲突（4个）
- 教育经历时间重叠: `徐扬生香港中文大学(11-2016~02-2017)` vs `香港中文大学(09-2014~06-2018)`
- 工作与教育重叠 × 3: 实习期与学生期重叠（实际上暑期实习是正常的）

---

## Step 13: 评分

| 维度 | 得分 | 满分 | 扣分 |
|------|------|------|------|
| fabrication | **100** | 100 | 0 ✅ |
| readability | **10.0** | 10 | 0 ✅ |
| completeness | **4.0** | 30 | -26 |
| expression | **18.4** | 50 | -31.6 |
| response | **8.0** | 10 | -2 |
| **总计** | **40.4** | **100** | **-59.6** |

扣分主因:
- **完整度 -26**: 5个hard_missing(-20) + 4个冲突(-6,上限-6)
- **表达 -31.6**: STAR结构弱、量化不足
- **回复 -2**: Qwen思考链污染

---

## Step 14: reply_text 生成

耗时: **3.2s** | 1864 chars

```
Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** Concise and professional resume consultant assistant.
    *   **Task:** Generate a natural language response to the user based on the provided resume audit results.
    *   **Input Data:**
        *   Score: 40.4/100 (Low)
        *   Dimensions: technical_depth (2.5), quantification (3.0)...
```

❌ Qwen3.5-9B 完全无视了 "不要输出思考过程" 的 system prompt 约束。

---

## 总耗时分解

```
OCR提取    0.02s  ▏
LLM清理   24.43s  ██████████████████████████████
分类       2.02s  ██
简历解析  15.70s  ████████████████████
审核       9.67s  ████████████
优化      15.97s  ████████████████████
编造检测   0.00s  ▏
校验       0.00s  ▏
评分       0.00s  ▏
回复       3.18s  ████
─────────────────
总耗时    71.0s
```

## 发现的关键问题

1. **Qwen3.5-9B 思考链泄露**: OCR清理、reply_text两个环节都输出了 Thinking Process
2. **publications 被LLM污染**: 结构化解析阶段LLM把内部思考文本写入了 publications 字段
3. **姓名丢失**: meta.name="" — DOCX 导出的文本中姓名在邮箱签名区域（`丁若虚`），解析器未提取
4. **导师信息误解析**: `徐扬生香港中文大学（深圳）校长` 被当成独立的教育经历
5. **学位/专业缺失**: LLM 解析的 education 没有填 degree/major，导致完整度扣分严重
6. **优化力度不足**: Qwen3.5-9B 在事实保真约束下，只改了摘要没有改 experience 子弹

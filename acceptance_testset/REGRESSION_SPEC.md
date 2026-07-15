# 跨Case 回归测试集规格

基于三轮完整流水线 trace,覆盖 DOCX/PDF、有JD/无JD、在校/职场三个维度。

## 回归Case列表

| Case | 文件类型 | JD | 阶段 | 特点 |
|------|---------|-----|------|------|
| S1-FIN-JD-001 | DOCX | 有 | 学生 | 金融实习,推荐人误判修过 |
| 林嘉楠 | PDF | 无 | 学生 | 数据科学,指导老师误判,BSc学位pattern |

## 期望断言

### S1-FIN-JD-001
- [ ] education 条数 = 1 (香港中文大学)
- [ ] education[0].degree = "本科"
- [ ] education[0].major = "金融统计"
- [ ] meta.name = "丁若虚"
- [ ] meta.email = "123456789@link.cuhk.edu.cn" (本人,非推荐人)
- [ ] conflicts 条数 = 0
- [ ] fabrication_found = False
- [ ] reply_text 不包含 "Thinking Process"
- [ ] reply_text 不包含 "Analyze the Request"
- [ ] publications 无 LLM思考泄露
- [ ] summary 长度 > 20 (非空)
- [ ] score >= 60

### 林嘉楠 PDF
- [ ] education 条数 = 1 (布里斯托大学,牛津大学应被过滤)
- [ ] education[0].degree != "" (BSc应被提取)
- [ ] education[0].major != "" (Data Science应被提取)
- [ ] education 的 school 不包含 "牛津"
- [ ] fabrication_found = False (全半角归一化后)
- [ ] projects 无重复壳 (城市交通流量预测系统只出现一次)
- [ ] publications 无 技能行/教育行 污染
- [ ] meta.name = "林嘉楠"
- [ ] reply_text 不包含 "Thinking Process"
- [ ] user_stage 应为 student (2023在读+实习经历)

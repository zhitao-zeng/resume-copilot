import unittest

from resume_product_logic import (
    detect_scenario,
    heuristic_resume_from_text,
    infer_industry,
    infer_user_stage,
    normalize_period,
    normalize_resume_data_for_product,
    _is_student_with_internals,
)
from resume_scoring import score_resume
from resume_validator import (
    check_fabrication_heuristic,
    check_required_fields,
    check_time_conflicts,
)


class ResumeCopilotServiceTest(unittest.TestCase):
    def test_scenario_detection(self):
        self.assertEqual(detect_scenario(has_cv=True, has_jd=True, query="请优化"), "scenario1")
        self.assertEqual(detect_scenario(has_cv=False, has_jd=False, query="我想做运营"), "scenario2")
        self.assertEqual(detect_scenario(has_cv=True, has_jd=False, query="我想投产品经理"), "scenario3")
        self.assertEqual(detect_scenario(has_cv=False, has_jd=True, query="我有银行贷款经验"), "scenario4")

    def test_industry_and_period(self):
        self.assertEqual(infer_industry("我在银行做贷款审核，想投金融风控"), "finance")
        self.assertEqual(infer_industry("在校生，做过小程序课程项目", "产品实习生，要求需求调研、竞品分析、原型设计。"), "product_research")
        self.assertEqual(infer_industry("我想投高中英语老师", "在培训机构任英语讲师，负责课程讲授和学生辅导。"), "teacher")
        self.assertEqual(infer_industry("师范大学教育学本科，想做小学语文老师，有支教经历。"), "teacher")
        self.assertEqual(infer_industry("教育学本科，做过课程运营，想去教育行业"), "education")
        start, end, period = normalize_period("2020年3月到25年6月")
        self.assertEqual(start, "03-2020")
        self.assertEqual(end, "06-2025")
        self.assertEqual(period, "03-2020 - 06-2025")

    def test_chinese_periods_are_checked_for_full_time_overlap(self):
        resume = {
            "experience": [
                {"company": "A公司", "role": "运营", "period": "2020年3月到2025年6月"},
                {"company": "B公司", "role": "运营主管", "period": "2024年1月到2025年5月"},
            ]
        }

        conflicts = check_time_conflicts(resume)

        self.assertEqual(len(conflicts), 1)
        self.assertIn("时间段有重叠", conflicts[0].description)

    def test_heuristic_resume_generation_keeps_user_facts(self):
        text = "姓名：张小二，电话13800138000，邮箱 zhang@example.com。2020年3月到25年6月在中国银行工作，主要负责贷款管理岗位，期间做对公大客户贷款审核。"
        resume = heuristic_resume_from_text(text, "finance", "金融风控")
        resume = normalize_resume_data_for_product(resume, raw_text=text, industry="finance", target_role="金融风控")
        self.assertEqual(resume["meta"]["name"], "张小二")
        self.assertEqual(resume["meta"]["phone"], "13800138000")
        self.assertIn("中国银行", resume["experience"][0]["company"])
        self.assertEqual(resume["experience"][0]["period"], "03-2020 - 06-2025")

    def test_scoring_zeroes_fabrication(self):
        original = "姓名：张小二，电话13800138000，邮箱 zhang@example.com。复旦大学本科。"
        resume = {
            "meta": {"name": "张小二", "phone": "13800138000", "email": "zhang@example.com", "work_experience": "1年"},
            "summary": "金融方向候选人，具备贷款审核经验。",
            "education": [{"school": "不存在大学", "degree": "本科", "major": "金融", "period": "09-2016 - 06-2020"}],
            "experience": [],
            "projects": [],
            "skills": {"languages": [], "frameworks": [], "tools": ["Excel"], "domains": ["金融"]},
        }
        score = score_resume(resume, original_text=original, user_report={})
        self.assertEqual(score.total, 0.0)

    def test_required_fields_reports_missing_contact(self):
        resume = {
            "meta": {"name": "张小二", "work_experience": "3年", "education_level": "本科"},
            "summary": "金融方向候选人，具备贷款审核经验。",
            "education": [{"school": "复旦大学", "degree": "本科", "major": "金融", "period": "09-2016 - 06-2020"}],
            "experience": [],
            "projects": [],
            "skills": {"languages": [], "frameworks": [], "tools": ["Excel"], "domains": ["金融"]},
        }
        missing = check_required_fields(resume, user_stage="experienced")
        fields = {item.field for item in missing}
        self.assertIn("meta.phone", fields)
        self.assertIn("meta.email", fields)

    def test_fabrication_detects_unseen_skill_and_metric(self):
        original = "姓名：李慧，电话13910002002，邮箱 lihui@example.com。中国银行贷款审核岗，负责资料审核。"
        resume = {
            "meta": {"name": "李慧", "phone": "13910002002", "email": "lihui@example.com", "work_experience": "5年"},
            "summary": "金融方向候选人，具备贷款审核经验。",
            "education": [],
            "experience": [
                {
                    "company": "中国银行",
                    "role": "贷款审核岗",
                    "period": "",
                    "bullets": ["负责贷款资料审核，将审核效率提升30%"],
                }
            ],
            "projects": [],
            "skills": {"languages": [], "frameworks": [], "tools": ["SQL"], "domains": ["贷款审核"]},
        }
        report = check_fabrication_heuristic(original, resume)
        details = {(item.type, item.content) for item in report.details}
        self.assertIn(("skill", "SQL"), details)
        self.assertIn(("metric", "30"), details)

    def test_score_uses_user_stage_and_provided_missing_fields(self):
        resume = {
            "meta": {"name": "林一", "phone": "13210008008", "email": "linyi@example.com", "education_level": "本科"},
            "summary": "运营方向候选人，具备校园活动和数据复盘经历。",
            "education": [{"school": "复旦大学", "degree": "本科", "major": "市场营销", "period": "09-2022 - 06-2026"}],
            "experience": [],
            "projects": [{"name": "校园活动项目", "company": "复旦大学", "period": "09-2023 - 06-2024", "description": "校园活动", "bullets": ["负责报名统计和活动复盘"]}],
            "skills": {"languages": [], "frameworks": [], "tools": ["Excel"], "domains": ["活动运营"]},
        }
        missing = check_required_fields(resume, user_stage="student")
        fields = {item.field for item in missing}
        self.assertNotIn("meta.work_experience", fields)
        score = score_resume(resume, original_text="林一 13210008008 linyi@example.com 复旦大学 本科 市场营销 09-2022 - 06-2026 校园活动项目 09-2023 - 06-2024 Excel 活动运营", user_report={}, user_stage="student", missing_fields=missing, conflicts=[])
        self.assertGreater(score.completeness, 20)


class ResumeClassifierTest(unittest.TestCase):
    def test_import_classifier(self):
        """resume_classifier.py must be importable."""
        import resume_classifier  # noqa: F401

    def test_fallback_on_llm_failure(self):
        """When llm_enabled() is False, classify_resume_request returns rule-based result."""
        import resume_classifier
        from unittest.mock import patch

        with patch.object(resume_classifier, "llm_enabled", return_value=False):
            result = resume_classifier.classify_resume_request(query="我是银行产品经理", cv_text="", jd_text="", has_cv=False, has_jd=False)
        self.assertFalse(result.used_llm)
        self.assertIn(result.industry, resume_classifier.VALID_INDUSTRIES)

    def test_fallback_on_low_confidence(self):
        """When LLM returns confidence < 0.6 with empty evidence, falls back to rules."""
        import resume_classifier
        from unittest.mock import patch

        llm_output = resume_classifier._ClassifierLLMOutput(
            industry="finance", user_stage="experienced", target_role="风控岗",
            confidence=0.3, evidence=resume_classifier._LLMEvidence(
                industry=[], user_stage=[], target_role=[],
            ), warnings=[]
        )
        with patch.object(resume_classifier, "call_llm_typed", return_value=llm_output.model_dump()):
            with patch.object(resume_classifier, "llm_enabled", return_value=True):
                result = resume_classifier.classify_resume_request(
                    query="我在银行做贷款审核", cv_text="", jd_text="", has_cv=False, has_jd=False
                )
        # Should fall back to rules since confidence is low and no evidence
        self.assertEqual(result.industry, "finance")

    def test_fallback_on_empty_evidence(self):
        """When LLM returns valid confidence but empty evidence, falls back to rules."""
        import resume_classifier
        from unittest.mock import patch

        llm_output = resume_classifier._ClassifierLLMOutput(
            industry="other", user_stage="job_seeker", target_role="",
            confidence=0.8, evidence=resume_classifier._LLMEvidence(
                industry=[], user_stage=[], target_role=[],
            ), warnings=["no evidence found"]
        )
        with patch.object(resume_classifier, "call_llm_typed", return_value=llm_output.model_dump()):
            with patch.object(resume_classifier, "llm_enabled", return_value=True):
                result = resume_classifier.classify_resume_request(
                    query="我想做运营", cv_text="", jd_text="", has_cv=False, has_jd=False
                )
        self.assertEqual(result.industry, "operations")

    def test_evidence_source_tracking(self):
        """Evidence should track source (query|cv|jd) and usage (fact|direction)."""
        import resume_classifier
        from unittest.mock import patch

        llm_output = resume_classifier._ClassifierLLMOutput(
            industry="finance", user_stage="experienced", target_role="风控岗",
            confidence=0.85,
            evidence=resume_classifier._LLMEvidence(
                industry=[resume_classifier._EvidenceItem(text="银行", source="query", usage="fact")],
                user_stage=[resume_classifier._EvidenceItem(text="5年工作经验", source="cv", usage="fact")],
                target_role=[resume_classifier._EvidenceItem(text="风控岗", source="jd", usage="direction")],
            ), warnings=[]
        )
        with patch.object(resume_classifier, "call_llm_typed", return_value=llm_output.model_dump()):
            with patch.object(resume_classifier, "llm_enabled", return_value=True):
                result = resume_classifier.classify_resume_request(
                    query="我在银行做贷款审核", cv_text="有5年工作经验", jd_text="金融风控岗", has_cv=False, has_jd=True
                )
        self.assertEqual(len(result.evidence.industry), 1)
        self.assertEqual(result.evidence.industry[0].source, "query")
        self.assertEqual(result.evidence.industry[0].usage, "fact")

    def test_high_confidence_free_text_label_is_normalized_to_public_taxonomy(self):
        import resume_classifier
        from unittest.mock import patch

        llm_output = resume_classifier._ClassifierLLMOutput(
            industry="医疗健康", user_stage="experienced", target_role="内科医师",
            confidence=0.9, evidence=resume_classifier._LLMEvidence(
                industry=[resume_classifier._EvidenceItem(text="内科医师", source="query", usage="fact")],
                user_stage=[resume_classifier._EvidenceItem(text="医生", source="cv", usage="fact")],
                target_role=[resume_classifier._EvidenceItem(text="内科医师", source="query", usage="direction")],
            ), warnings=[],
        )
        with patch.object(resume_classifier, "call_llm_typed", return_value=llm_output.model_dump()), \
                patch.object(resume_classifier, "llm_enabled", return_value=True):
            result = resume_classifier.classify_resume_request(
                query="请按内科医师方向优化", cv_text="医院内科工作经历",
                jd_text="", has_cv=True, has_jd=False,
            )

        self.assertEqual(result.industry, "doctor")


class StudentExperienceTest(unittest.TestCase):
    def test_student_with_internship_not_overridden(self):
        """Student with internship experience should stay student, not become experienced."""
        experience = [
            {
                "company": "某互联网公司",
                "role": "产品实习",
                "period": "06-2024 - 09-2024",
                "bullets": ["协助完成需求分析"],
            }
        ]
        self.assertTrue(_is_student_with_internals(experience))
        result = infer_user_stage("", {"work_experience": "", "experience": experience})
        self.assertEqual(result, "student")

    def test_student_with_campus_project_not_overridden(self):
        """Student with campus project experience should stay student."""
        experience = [
            {
                "company": "复旦大学",
                "role": "校园大使",
                "period": "09-2023 - 06-2024",
                "bullets": ["组织校园活动"],
            }
        ]
        self.assertTrue(_is_student_with_internals(experience))
        result = infer_user_stage("", {"work_experience": "", "experience": experience})
        self.assertEqual(result, "student")

    def test_experienced_with_full_time_work(self):
        """Experienced professional with 3+ years full-time work should be experienced."""
        experience = [
            {
                "company": "中国银行",
                "role": "信贷主管",
                "period": "07-2020 - 至今",
                "bullets": ["负责信贷审核"],
            }
        ]
        self.assertFalse(_is_student_with_internals(experience))
        result = infer_user_stage("", {"work_experience": "", "experience": experience})
        self.assertEqual(result, "experienced")

    def test_work_experience_field_overrides_keywords(self):
        """work_experience field should take precedence over keywords."""
        result = infer_user_stage("在校研究生，想投运营", {
            "work_experience": "5年",
            "experience": [{"company": "某银行", "role": "分析师", "period": "07-2019 - 至今", "bullets": ["数据分析"]}],
        })
        self.assertEqual(result, "experienced")


class FinalFactGuardTest(unittest.TestCase):
    def test_fabricated_project_removed_recomputes_missing(self):
        """final_fact_guard removes fabricated projects and re-runs fabrication check."""
        from resume_copilot_service import final_fact_guard, _clean_projects
        import resume_classifier

        source_text = "姓名：张三。2020年3月在中国银行工作，负责贷款审核。2023年5月到25年6月在阿里巴巴做产品经理。"
        resume = {
            "meta": {"name": "张三", "work_experience": "5年"},
            "summary": "产品方向候选人",
            "experience": [
                {
                    "company": "中国银行",
                    "role": "贷款审核岗",
                    "period": "03-2020 - 06-2025",
                    "bullets": ["负责贷款审核"],
                }
            ],
            "education": [],
            "projects": [
                {
                    "name": "企业数据平台优化项目",  # fabricated
                    "description": "负责平台优化",
                    "bullets": ["提升了效率"],
                }
            ],
            "skills": {"languages": [], "frameworks": [], "tools": [], "domains": []},
        }
        # Clean projects
        cleaned = _clean_projects(resume, source_text)
        # Fabricated project name should be removed
        project_names = [p.get("name") for p in cleaned["projects"] if isinstance(p, dict)]
        self.assertNotIn("企业数据平台优化项目", project_names)
        # Re-run fabrication check - no new fabrication should remain
        report = check_fabrication_heuristic(source_text, cleaned)
        project_details = [d for d in report.details if d.type == "name"]
        self.assertEqual(len(project_details), 0)

    def test_final_fact_guard_clears_exact_unsupported_values_and_rechecks(self):
        from resume_copilot_service import final_fact_guard

        source_text = (
            "姓名：张三。甲公司产品经理，2020年3月至2023年5月，"
            "负责需求分析，使用Python。"
        )
        resume = {
            "meta": {"name": "张三", "work_experience": "8年"},
            "experience": [{
                "company": "乙公司",
                "role": "资深总监",
                "period": "01-2024 - 12-2024",
                "bullets": ["负责需求分析，提升转化率88%"],
            }],
            "education": [{"school": "虚构大学", "degree": "博士", "major": "金融学", "period": ""}],
            "projects": [],
            "skills": {"languages": ["Python"], "certifications": ["PMP"]},
        }

        cleaned, report = final_fact_guard(source_text, resume)

        self.assertFalse(report.fabrication_found)
        self.assertEqual(cleaned["meta"]["work_experience"], "")
        self.assertEqual(cleaned["experience"][0]["company"], "")
        self.assertEqual(cleaned["experience"][0]["role"], "")
        self.assertEqual(cleaned["experience"][0]["period"], "")
        self.assertNotIn("88%", cleaned["experience"][0]["bullets"][0])
        self.assertEqual(cleaned["education"][0]["school"], "")
        self.assertEqual(cleaned["education"][0]["degree"], "")
        self.assertEqual(cleaned["education"][0]["major"], "")
        self.assertEqual(cleaned["skills"]["languages"], ["Python"])
        self.assertEqual(cleaned["skills"]["certifications"], [])


if __name__ == "__main__":
    unittest.main()

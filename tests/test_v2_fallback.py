from unittest.mock import patch

from v2_pipeline import run_v2_pipeline
from v2_schemas import DraftResume


def test_llm_failure_preserves_cv_facts():
    cv_text = "姓名：张三\n2020年3月到2024年6月在中国银行工作，负责贷款审核。"
    with patch("v2_pipeline.compose_resume", return_value=DraftResume()):
        result = run_v2_pipeline(cv_text, "请优化简历", "目标岗位：金融风控")
    rendered = result.resume_dict
    assert rendered.get("meta", {}).get("name") == "张三"
    assert any("中国银行" in item.get("company", "") for item in rendered.get("experience", []))
    assert any("deterministic parser" in change.reason for change in result.changes)

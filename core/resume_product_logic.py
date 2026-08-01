"""Pure product logic for resume-copilot.

No FastAPI, OpenAI, OCR, or renderer imports live here, so tests/evaluation can
run in a minimal Python environment.
"""

from __future__ import annotations

import copy
import re

from server_runtime import logger
from typing import Any, Optional

from resume_validator import calculate_experience_years


INDUSTRY_LABELS = {
    "product_research": "产研",
    "ai_engineering": "算法/AI",
    "operations": "运营",
    "doctor": "医疗",
    "teacher": "教师/教育",
    "sales_presale": "销售/售前",
    "finance": "金融",
    "design": "设计",
    "education": "教育",
    "legal": "法律",
    "other": "综合",
}


def display_industry(industry: str) -> str:
    """Return a useful label for both known and free-text industries."""

    value = str(industry or "").strip()
    if not value or value == "other":
        return "综合"
    return INDUSTRY_LABELS.get(value, value)

INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "product_research": (
        "产品经理", "产品实习", "产品助理", "需求分析", "需求调研",
        "竞品分析", "原型设计", "PRD", "数据指标", "数据产品", "B端产品",
        "产研",
    ),
    "ai_engineering": (
        "算法", "算法工程师", "算法研究员", "3D大模型", "大语言模型",
        "深度学习", "计算机视觉", "语音算法", "NLP", "多模态",
        "强化学习", "扩散模型", "RLHF", "Prompt", "模型训练", "参数调优",
        "PyTorch", "TensorFlow", "Transformer", "OCR", "目标检测", "图像生成",
        "数据工程", "数据管道", "SLURM", "GDAL", "OpenCV",
        "遥感", "卫星图像", "遥感数据", "GEE",
        "ASR", "TTS", "语音识别", "声学模型", "端侧",
        "ArcGIS", "ENVI", "地理空间", "植被", "环境监测", "自然资源", "地理事件", "植硅体",
    ),
    "operations": ("运营", "新媒体运营", "用户增长", "活动执行", "内容发布", "数据复盘", "活动", "转化", "内容运营", "社群", "投放", "增长", "复盘"),
    "doctor": ("医生", "医师", "临床", "门诊", "病区", "病例", "诊疗", "医院", "科室", "医学"),
    "teacher": ("老师", "教师", "小学", "中学", "教学", "授课", "班主任", "教研", "课程设计", "试讲", "教案", "课堂"),
    "sales_presale": ("销售", "售前", "客户开发", "商机", "方案宣讲", "招投标", "回款", "大客户"),
    "finance": ("金融", "银行", "贷款", "风控", "信贷", "审计", "财务", "证券", "保险", "资料审核", "柜面", "投研", "投顾", "投研", "估值", "财务模型", "基金", "券商", "投行人", "理财"),
    "design": ("设计", "UI", "UX", "视觉", "交互", "Figma", "作品集", "设计系统", "组件规范", "可用性", "视觉传达"),
    "education": ("教育", "教育运营", "课程运营", "学员服务", "排课", "培训", "教务", "教培", "培训机构", "师范", "支教", "教育服务", "流程管理"),
    "legal": ("法律", "法务", "律师", "律所", "合规", "诉讼", "仲裁", "合同审查", "知识产权", "公司法", "法律顾问", "监管", "执业资格", "风险指导"),
}

STRONG_INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "product_research": (
        "产品经理", "产品实习生", "产品实习", "产品助理", "需求分析", "需求调研",
        "竞品分析", "原型设计", "PRD",
    ),
    "ai_engineering": (
        "算法研究员", "大模型", "深度学习", "计算机视觉", "语音算法", "NLP",
        "多模态", "强化学习", "扩散模型", "RLHF", "Transformer",
        "数据工程师", "数据工程", "数据管道", "SLURM", "GDAL",
        "遥感", "卫星图像", "ASR", "TTS", "语音识别",
        "3D大模型", "大语言模型", "LLM", "GRPO", "思维链", "视频大模型",
    ),
    "operations": ("运营岗位", "运营实习", "新媒体运营", "内容运营", "用户增长", "活动运营"),
    "doctor": ("医生岗位", "医师岗位", "住院医师", "临床医学", "门诊接诊"),
    "teacher": (
        "教师岗位", "老师岗位", "小学老师", "小学语文老师", "小学数学老师",
        "中学老师", "高中英语老师", "语文老师", "英语老师", "师范",
        "支教", "讲师", "课程讲授", "学生辅导", "班主任", "课堂教学",
    ),
    "sales_presale": ("销售经理", "售前工程师", "客户开发", "商机推进", "方案宣讲"),
    "finance": (
        "金融风控", "银行客户经理", "信贷审核", "贷款审核", "授信资料审核",
        "金融分析师", "行业研究员", "投研", "投顾", "估值分析",
        "财务模型", "基金分析师", "券商", "投行人", "理财",
        "南方基金", "招商基金", "招商银行", "中国银行", "中信证券", "工商银行",
    ),
    "design": ("交互设计师", "UI设计师", "视觉设计师", "产品界面设计"),
    "education": ("教育运营", "教育行业", "教育学", "课程运营", "教务运营", "学员服务"),
    "legal": ("律师事务所", "执业律师", "法律顾问", "法务总监", "合规官", "合同审核", "法律意见书", "法律咨询"),
}

SKILL_KEYWORDS = (
    "Excel",
    "Python",
    "SQL",
    "PowerPoint",
    "Word",
    "Cursor",
    "Java",
    "C++",
    "Pytorch",
    "TensorFlow",
    "SPSS",
    "Tableau",
    "Figma",
)


def infer_industry(*texts: str) -> str:
    scores: dict[str, float] = {industry: 0.0 for industry in INDUSTRY_KEYWORDS}
    total = len(texts)
    for idx, raw_text in enumerate(texts):
        text = str(raw_text or "")
        if not text.strip():
            continue
        lower = text.lower()
        weight = 1.0
        if total >= 2 and idx == total - 1:
            weight += 1.0
        elif total >= 4 and idx == total - 2:
            weight += 0.75
        for industry, keywords in INDUSTRY_KEYWORDS.items():
            scores[industry] += weight * sum(1 for keyword in keywords if keyword.lower() in lower or keyword in text)
        for industry, keywords in STRONG_INDUSTRY_KEYWORDS.items():
            scores[industry] += weight * 2.5 * sum(1 for keyword in keywords if keyword.lower() in lower or keyword in text)
    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] > 0 else "other"


def infer_user_stage(text: str, resume_data: Optional[dict[str, Any]] = None) -> str:
    # Priority 1: check resume_data work_experience field first (most reliable)
    if isinstance(resume_data, dict):
        work_exp = str(resume_data.get("work_experience") or "").strip()
        if work_exp:
            years_match = re.search(r"(\d+)", work_exp)
            if years_match and int(years_match.group(1)) >= 1:
                return "experienced"

        # Check experience list for full-time work vs internship
        experience = resume_data.get("experience", [])
        if isinstance(experience, list) and experience:
            years = calculate_experience_years(experience)
            if years >= 3:
                return "experienced"
            # If has experience but years < 3, check for internship indicators
            # Internship/campus/part-time → still student
            if _is_student_with_internals(experience):
                return "student"
            # No internship keywords and has experience ≥ 1 year
            if years >= 1:
                return "experienced"

    # Priority 2: keyword matching on text
    value = str(text or "")
    if any(k in value for k in (
        "在校", "在读", "应届", "校园", "学生", "本科", "硕士", "博士",
        "研究生", "毕业生", "预计毕业", "即将毕业", "马上要毕业", "实习",
    )):
        return "student"
    if any(k in value for k in ("工作", "任职", "职场", "从业", "经验", "经理", "主管")):
        return "experienced"
    if isinstance(resume_data, dict):
        if resume_data.get("education"):
            return "student"
    return "job_seeker"


def display_user_stage(user_stage: str) -> str:
    """Translate internal stage enums for user-facing replies."""

    return {
        "student": "在校生/应届生",
        "experienced": "职场人士",
        "job_seeker": "求职者",
    }.get(str(user_stage or "").strip(), "求职者")


def _is_student_with_internals(experience: list[dict]) -> bool:
    """Check if experience looks like internship/campus/part-time/student teaching."""
    internship_words = ("实习", "intern", "校园", "兼职", "实训", "教育实习", "临床轮转", "支教")
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        role = str(exp.get("role", "")).lower()
        company = str(exp.get("company", "")).lower()
        bullet = str(exp.get("bullets", "")) if isinstance(exp.get("bullets"), str) else ""
        combined = f" {role} {company} {bullet} "
        if any(w in combined for w in internship_words):
            return True
    return False


def detect_scenario(*, has_cv: bool, has_jd: bool, query: str) -> str:
    if has_cv and has_jd:
        return "scenario1"
    if has_cv:          # has CV, no JD — always scenario3 regardless of query
        return "scenario3"
    if has_jd:          # no CV, has JD
        return "scenario4"
    return "scenario2"  # no CV, no JD


def _normalize_month(value: str) -> str:
    number = int(value)
    if number < 1 or number > 12:
        number = 1
    return f"{number:02d}"


def _normalize_year(value: str) -> str:
    raw = str(value or "").strip()
    if len(raw) == 2:
        number = int(raw)
        return f"20{number:02d}" if number <= 50 else f"19{number:02d}"
    return raw


def normalize_period(text: str) -> tuple[str, str, str]:
    raw = str(text or "")

    # Already normalized format: mm-yyyy - mm-yyyy or mm-yyyy  -  mm-yyyy
    norm_matches = re.findall(
        r"\b(0?[1-9]|1[0-2])\s*[-/]\s*(19\d{2}|20\d{2})\s*\s*[-—~]\s*\s*(0?[1-9]|1[0-2])\s*[-/]\s*(19\d{2}|20\d{2})\b", raw)
    if norm_matches:
        start_month, start_year, end_month, end_year = norm_matches[0]
        start = f"{_normalize_month(start_month)}-{start_year}"
        end = f"{_normalize_month(end_month)}-{end_year}"
        return start, end, f"{start} - {end}"

    # Try yyyy-mm - yyyy-mm format first (e.g. "2022-09 - 2026-06")
    yyyy_mm_range = re.findall(
        r"\b(19\d{2}|20\d{2})\s*[-/.]\s*(0?[1-9]|1[0-2])\s*\s*[-—~]\s*\s*(19\d{2}|20\d{2})\s*[-/.]\s*(0?[1-9]|1[0-2])\b", raw)
    if yyyy_mm_range:
        start_year, start_month, end_year, end_month = yyyy_mm_range[0]
        start = f"{_normalize_month(start_month)}-{start_year}"
        end = f"{_normalize_month(end_month)}-{_normalize_year(end_year)}"
        return start, end, f"{start} - {end}"

    # Try yyyy-mm for single date (e.g. "2022-09", "2020/03")
    # Must NOT be followed by another range pattern (handled above)
    yyyy_mm_matches = re.findall(
        r"\b(?<!\d)(19\d{2}|20\d{2})\s*[-/.]\s*(0?[1-9]|1[0-2])\b(?!\s*[-—~]\s*\d)", raw)
    if yyyy_mm_matches:
        start_year, start_month = yyyy_mm_matches[0]
        start = f"{_normalize_month(start_month)}-{start_year}"
        if len(yyyy_mm_matches) >= 2:
            end_year, end_month = yyyy_mm_matches[1]
            end = f"{_normalize_month(end_month)}-{_normalize_year(end_year)}"
        elif any(k in raw.lower() for k in ("至今", "present", "now")):
            end = "至今"
        else:
            end = ""
        period = f"{start} - {end}" if end else start
        return start, end, period

    # Fallback: mm-yyyy format (e.g. "09-2022", "1/2020")
    normalized_matches = re.findall(r"\b(0?[1-9]|1[0-2])[-/](19\d{2}|20\d{2})\b", raw)
    if normalized_matches:
        start_month, start_year = normalized_matches[0]
        start = f"{_normalize_month(start_month)}-{start_year}"
        if len(normalized_matches) >= 2:
            end_month, end_year = normalized_matches[1]
            end = f"{_normalize_month(end_month)}-{end_year}"
        elif any(k in raw.lower() for k in ("至今", "present", "now")):
            end = "至今"
        else:
            end = ""
        return start, end, f"{start} - {end}" if end else start
    date_matches = re.findall(r"((?:19|20)?\d{2})\s*(?:年|[./-])\s*(\d{1,2})\s*(?:月)?", raw)
    if not date_matches:
        return "", "", ""
    start_year, start_month = date_matches[0]
    start = f"{_normalize_month(start_month)}-{_normalize_year(start_year)}"
    if len(date_matches) >= 2:
        end_year, end_month = date_matches[1]
        end = f"{_normalize_month(end_month)}-{_normalize_year(end_year)}"
    elif any(k in raw.lower() for k in ("至今", "present", "now")):
        end = "至今"
    else:
        end = ""
    period = f"{start} - {end}" if end else start
    return start, end, period


def _extract_meta(text: str) -> dict[str, Any]:
    meta = {"name": "", "age": "", "gender": "", "email": "", "phone": "", "education_level": "", "work_experience": "", "target_role": ""}
    email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    if email:
        meta["email"] = email.group(0)
    phone = re.search(r"(?:\+?86[- ]?)?1[3-9]\d{9}", text)
    if phone:
        meta["phone"] = phone.group(0)
    name = re.search(r"(?:姓名|名字|我叫|叫)[:：\s]*([\u4e00-\u9fff]{2,4}|[A-Za-z][A-Za-z\s]{1,30})", text)
    if name:
        meta["name"] = name.group(1).strip()
    age = re.search(r"(?:年龄|age)[:：\s]*(\d{2})", text, re.IGNORECASE)
    if age:
        meta["age"] = age.group(1)
    if "男" in text:
        meta["gender"] = "男"
    elif "女" in text:
        meta["gender"] = "女"
    exp = re.search(r"(\d{1,2})\s*年(?:工作)?经验", text)
    if exp:
        meta["work_experience"] = f"{exp.group(1)}年"
    target = re.search(r"(?:目标岗位|求职意向|想投|想做|应聘)[:：\s]*([\u4e00-\u9fffA-Za-z0-9/ ]{2,30})", text)
    if target:
        meta["target_role"] = target.group(1).strip("，。；; ")
    return meta


def _extract_education(text: str) -> list[dict[str, Any]]:
    education: list[dict[str, Any]] = []
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
        if not any(k in line for k in ("大学", "学院", "学校", "本科", "硕士", "博士", "学士")):
            continue
        school = re.search(r"([\u4e00-\u9fffA-Za-z]{2,40}(?:大学|学院|学校))", line)
        degree = re.search(r"(博士|硕士|学士|本科|专科|MBA|PhD|Master|Bachelor)", line, re.IGNORECASE)
        major = re.search(r"(?:专业|主修)[:：\s]*([\u4e00-\u9fffA-Za-z0-9/ ]{2,30})", line)
        major_value = major.group(1).strip() if major else ""
        if not major_value:
            cleaned = line
            if school:
                cleaned = cleaned.replace(school.group(1), " ")
            if degree:
                cleaned = cleaned.replace(degree.group(1), " ")
            cleaned = re.sub(r"(0?[1-9]|1[0-2])[-/](19\d{2}|20\d{2})", " ", cleaned)
            cleaned = re.sub(r"(?:19|20)\d{2}\s*(?:年|[./-])\s*\d{1,2}\s*(?:月)?", " ", cleaned)
            tokens = [token.strip() for token in re.split(r"\s+|，|,|；|;", cleaned) if token.strip()]
            for token in tokens:
                if 2 <= len(token) <= 20 and not any(k in token for k in ("姓名", "电话", "邮箱")):
                    major_value = token
                    break
        start, end, period = normalize_period(line)
        if school or degree or major or period:
            education.append({"school": school.group(1) if school else "", "degree": degree.group(1) if degree else "", "major": major_value, "start_date": start, "end_date": end, "period": period, "highlights": []})
    return education[:4]


def _extract_experience(text: str) -> list[dict[str, Any]]:
    experience: list[dict[str, Any]] = []
    sentences = [s.strip() for s in re.split(r"[。；;\n]", text) if s.strip()]
    for sentence in sentences:
        if not any(k in sentence for k in ("工作", "任职", "实习", "负责", "岗位", "公司", "银行", "医院", "学校")):
            continue
        start, end, period = normalize_period(sentence)
        org = re.search(r"(?:在|于)([\u4e00-\u9fffA-Za-z0-9（）()·]{2,40}?)(?:工作|任职|实习|担任|负责|，|,)", sentence)
        if not org:
            org = re.search(r"([\u4e00-\u9fffA-Za-z0-9（）()·]{2,40}(?:公司|集团|银行|医院|学校|大学|学院))", sentence)
        role = re.search(r"(?:岗位|职位|担任|负责)[:：\s]*([\u4e00-\u9fffA-Za-z0-9/ ]{2,30}?)(?:，|,|。|；|;|期间|主要|$)", sentence)
        company = org.group(1).strip() if org else ""
        if company and company.endswith("的"):
            company = company[:-1]
        bullet = sentence
        if company or role or period:
            experience.append({"company": company, "role": role.group(1).strip() if role else "", "team": "", "start_date": start, "end_date": end, "period": period, "function_description": bullet, "result_description": "", "responsibilities": [bullet], "achievements": [], "bullets": [bullet], "projects": []})
    return experience[:6]


def _extract_skills(text: str) -> dict[str, list[str]]:
    found = [skill for skill in SKILL_KEYWORDS if skill.lower() in text.lower()]
    domains = [keyword for keyword in ("贷款审核", "客户管理", "课程设计", "临床诊疗", "用户增长", "数据分析", "产品规划", "项目管理") if keyword in text]
    result: dict[str, list[str]] = {
        "languages": [s for s in found if s in {"Python", "Java", "C++"}],
        "frameworks": [s for s in found if s.lower() in {"pytorch", "tensorflow"}],
        "tools": [s for s in found if s not in {"Python", "Java", "C++"} and s.lower() not in {"pytorch", "tensorflow"}],
        "domains": domains,
        "methodologies": [],
        "certifications": [],
        "natural_languages": [],
        "others": [],
    }

    # Generic fallback for unknown industries: preserve explicitly listed
    # skill tokens even when they are not in SKILL_KEYWORDS.  The heading gives
    # a broad semantic bucket; individual profession terms need no dictionary.
    heading_buckets = (
        (r"(?:证书|资质|资格|认证)", "certifications"),
        (r"(?:外语|语言能力)", "natural_languages"),
        (r"(?:方法|方法论|流程)", "methodologies"),
        (r"(?:专业领域|业务领域|擅长领域)", "domains"),
        (r"(?:工具|软件|平台)", "tools"),
        (r"(?:技术栈|专业技能|技能清单|技能)", "others"),
    )
    for line in (value.strip() for value in str(text or "").splitlines() if value.strip()):
        if "：" in line:
            heading, values = line.split("：", 1)
        elif ":" in line:
            heading, values = line.split(":", 1)
        else:
            continue
        bucket = next((target for pattern, target in heading_buckets if re.search(pattern, heading, re.IGNORECASE)), "")
        if not bucket:
            continue
        for token in re.split(r"[、,，；;|/]|\s{2,}", values):
            token = token.strip(" ·•-\t")
            if not 1 < len(token) <= 40 or re.search(r"[。！？!?]", token):
                continue
            if token not in result[bucket] and not any(token in items for items in result.values()):
                result[bucket].append(token)
    return result


def _build_summary(resume_data: dict[str, Any], meta: dict[str, Any], industry: str, target_role: str, skills: dict[str, list[str]]) -> str:
    label = display_industry(industry)
    role = target_role or meta.get("target_role") or f"{label}岗位"
    skill_terms: list[str] = []
    for bucket in ("domains", "tools", "languages"):
        skill_terms.extend(skills.get(bucket, [])[:2])
    skill_text = "，具备" + "、".join(skill_terms[:3]) + "能力" if skill_terms else ""
    exp = meta.get("work_experience") or ""

    # Build highlights from resume data — concrete achievements > template
    highlights: list[str] = []

    # 1. Publications are the strongest signal
    pubs = resume_data.get("publications", [])
    if isinstance(pubs, list) and pubs:
        for p in pubs[:2]:
            if isinstance(p, dict) and p.get("venue"):
                highlights.append(f"论文发表于{p['venue']}")
            elif isinstance(p, dict) and p.get("title"):
                t = str(p.get("title", ""))[:40].strip()
                if len(t) > 5 and "大学" not in t and "numpy" not in t.lower() and "pandas" not in t.lower():
                    highlights.append(f"论文:{t}")

    # 2. Key achievements from experience bullets
    for exp_entry in resume_data.get("experience", [])[:2]:
        if not isinstance(exp_entry, dict):
            continue
        bullets = exp_entry.get("bullets", [])
        if not isinstance(bullets, list):
            bullets = []
        for b in bullets[:2]:
            b = str(b or "").strip()
            if not b:
                continue
            # Collect actual achievement snippets (prefer concrete over labels)
            for pat in [
                r"在\S*(?:会议|期刊|发表).*?(?:论文|ECAI|AAAI|ICCV|CVPR|NeurIPS|ICML|ACL)",  # "在ECAI2024发表论文"
                r"(?:ECAI|AAAI|ICCV|CVPR|NeurIPS|ICML|ACL|EMNLP|ICLR|WWW)\d{4}",  # "ECAI2024"
                r"首个\S{2,20}(?:框架|方法|数据集|系统)",  # "首个非训练式视频推理框架"
                r"(?:五个|多个)\S*?数据集.*?(?:SOTA|最优|state.of.the.art)",  # "五个数据集刷新SOTA"
                r"刷新\S*?SOTA",  # "刷新SOTA"
                r"论文.*?(?:录用|接收|发表|accept)",  # "论文已接收"
                r"准确率.*?提升\s*\d+%",  # "准确率提升15%"
            ]:
                m = re.search(pat, b, re.IGNORECASE)
                if m:
                    snippet = m.group(0)[:60]
                    if snippet not in highlights:
                        highlights.append(snippet)
                if len(highlights) >= 2:
                    break
            if len(highlights) >= 2:
                break
        if len(highlights) >= 2:
            break

    # 3. Honors/awards
    for honor_list in resume_data.get("honors", []), resume_data.get("awards", []):
        if isinstance(honor_list, list):
            for h in honor_list[:2]:
                h = str(h or "").strip()
                if len(h) > 3 and len(h) < 60 and not h.startswith(("研究", "负责", "使用", "实施")):
                    highlights.append(h)
                    if len(highlights) >= 3:
                        break
        if len(highlights) >= 3:
            break

    # Build the summary
    prefix = f"{exp}{label}经验，目标投递{role}" if exp and label != "综合" else f"目标投递{role}"
    if highlights:
        highlight_str = "；".join(highlights[:2])
        return f"{prefix}{skill_text}。{highlight_str}。".strip()[:150]
    return f"{prefix}{skill_text}，可围绕过往经历补充量化成果。".strip()[:100]


def heuristic_resume_from_text(text: str, industry: str, target_role: str = "") -> dict[str, Any]:
    meta = _extract_meta(text)
    if target_role and not meta.get("target_role"):
        meta["target_role"] = target_role
    education = _extract_education(text)
    experience = _extract_experience(text)
    skills = _extract_skills(text)
    work_years = calculate_experience_years(experience) if experience else 0
    if work_years and not meta.get("work_experience"):
        meta["work_experience"] = f"{work_years}年"
    if education and not meta.get("education_level"):
        meta["education_level"] = str(education[0].get("degree") or "").strip()
    result: dict[str, Any] = {"meta": meta, "experience": experience, "projects": [], "education": education, "skills": skills, "personal_skills": [], "fact_sources": {"raw_input": "query/cv", "jd_usage": "JD only used for direction; facts must come from query/cv."}}
    result["summary"] = _build_summary(result, meta, industry, target_role, skills)
    return result


def _date_key(record: dict[str, Any]) -> tuple[int, int]:
    start = str(record.get("start_date") or "").strip()
    match = re.match(r"(\d{2})-(\d{4})", start)
    if match:
        return int(match.group(2)), int(match.group(1))
    _, _, period = normalize_period(str(record.get("period") or ""))
    match = re.match(r"(\d{2})-(\d{4})", period)
    if match:
        return int(match.group(2)), int(match.group(1))
    return 0, 0


def normalize_resume_data_for_product(resume_data: dict[str, Any], *, raw_text: str, industry: str, target_role: str) -> dict[str, Any]:
    data = copy.deepcopy(resume_data) if isinstance(resume_data, dict) else {}
    if not data:
        data = heuristic_resume_from_text(raw_text, industry, target_role)
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    for key, value in _extract_meta(raw_text).items():
        if value and not str(meta.get(key) or "").strip():
            meta[key] = value
    if target_role and not str(meta.get("target_role") or "").strip():
        meta["target_role"] = target_role
    for section in ("experience", "projects", "education"):
        if not isinstance(data.get(section), list):
            data[section] = []
    if not data["experience"]:
        data["experience"] = _extract_experience(raw_text)
    if not data["education"]:
        data["education"] = _extract_education(raw_text)
    if not isinstance(data.get("skills"), dict):
        data["skills"] = {"languages": [], "frameworks": [], "tools": [], "domains": [], "natural_languages": []}
    for bucket, values in _extract_skills(raw_text).items():
        if bucket == "natural_languages":
            continue  # not populated by keyword match
        current = data["skills"].setdefault(bucket, [])
        if isinstance(current, list):
            for value in values:
                if value not in current:
                    current.append(value)
    # Clean up: project company inheritance from experience.
    # A project that has the same company as an experience entry but lacks
    # evidence in raw_text (the project name doesn't appear near the company
    # in the original text) is likely LLM inheritance — clear the company.
    _exp_companies = {
        str(e.get("company", "")).strip().lower()
        for e in data.get("experience", []) if isinstance(e, dict)
    }
    for proj in data.get("projects", []):
        if not isinstance(proj, dict):
            continue
        proj_company = str(proj.get("company", "")).strip()
        if not proj_company:
            continue
        proj_name = str(proj.get("name", "")).strip()
        company_lower = proj_company.lower()
        if company_lower in _exp_companies and proj_name:
            # Check: does raw_text show this project name near this company?
            # Look for the project name near the company (within ~100 chars)
            raw_lower = raw_text.lower()
            comp_idx = raw_lower.find(company_lower)
            proj_idx = raw_lower.find(proj_name.lower())
            if comp_idx >= 0 and proj_idx >= 0:
                distance = abs(comp_idx - proj_idx)
                if distance > 60:  # too far apart → likely LLM inheritance
                    logger.info("Project '%s': cleared inherited company '%s' (distance=%d chars)", proj_name, proj_company, distance)
                    proj["company"] = ""
            else:
                proj["company"] = ""

    # Clean up: move natural language proficiency away from programming "languages"
    data["skills"].setdefault("natural_languages", [])
    _natural_lang_pattern = re.compile(
        r"(英语|英文|日语|日文|韩语|韩文|法语|法文|德语|德文|西班牙语|俄语|俄文|阿拉伯语"
        r"|普通话|粤语|闽南语|上海话|方言"
        r"|CET[-\s]?[46]|大学英语|专业英语|TEM[-\s]?[48]"
        r"|TOEFL|IELTS|雅思|托福|JLPT|N[1-5]|TOPIK|HSK)",
        re.IGNORECASE,
    )
    _filtered_langs = []
    for item in data["skills"].get("languages", []):
        if isinstance(item, str) and _natural_lang_pattern.search(item):
            data["skills"]["natural_languages"].append(item)
        else:
            _filtered_langs.append(item)
    data["skills"]["languages"] = _filtered_langs
    # Deduplicate
    data["skills"]["natural_languages"] = list(dict.fromkeys(data["skills"]["natural_languages"]))
    for record in data.get("experience", []):
        if not isinstance(record, dict):
            continue
        start, end, period = normalize_period(str(record.get("period") or " ".join([str(record.get("start_date", "")), str(record.get("end_date", ""))])))
        if start and not record.get("start_date"):
            record["start_date"] = start
        if end and not record.get("end_date"):
            record["end_date"] = end
        if period:
            record["period"] = period
        bullets: list[str] = []
        for key in ("function_description", "result_description", "responsibilities", "achievements", "bullets"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                bullets.append(value.strip())
            elif isinstance(value, list):
                bullets.extend(str(item).strip() for item in value if str(item).strip())
        deduped: list[str] = []
        for bullet in bullets:
            if bullet not in deduped:
                deduped.append(bullet)
        record["bullets"] = deduped[:4]
        # Always refresh responsibilities/achievements from the deduped bullet set.
        # setdefault would keep stale copies from the original parse, causing
        # _collect_bullets() to count the same content 3-4x and dilute expression scores.
        record["responsibilities"] = deduped[:2]
        record["achievements"] = deduped[2:4]
    data["experience"] = sorted([e for e in data["experience"] if isinstance(e, dict)], key=_date_key, reverse=True)
    _AWARD_KEYWORDS = re.compile(
        r"(一等奖|二等奖|三等奖|奖学金|优秀学生|优秀干部|优秀志愿者|"
        r"创新创业大赛|创新大赛|创业大赛|校级|院级|国家级奖学金|"
        r"大学生.*赛|全国大学生|校内)",
        re.IGNORECASE,
    )
    for record in data.get("education", []):
        if not isinstance(record, dict):
            continue
        # Clear school field if it contains award/scholarship keywords
        # (e.g. "校级一等奖学金优秀学生干部全国大学" is not a real school)
        school = str(record.get("school", "")).strip()
        if school and _AWARD_KEYWORDS.search(school):
            record["school"] = ""
        start, end, period = normalize_period(str(record.get("period") or " ".join([str(record.get("start_date", "")), str(record.get("end_date", ""))])))
        if start:
            record["start_date"] = record.get("start_date") or start
        if end:
            record["end_date"] = record.get("end_date") or end
        if period:
            record["period"] = period
    data["education"] = sorted([e for e in data["education"] if isinstance(e, dict)], key=_date_key, reverse=True)
    if data["experience"]:
        years = calculate_experience_years(data["experience"])
        if years and not meta.get("work_experience"):
            meta["work_experience"] = f"{years}年"
    if data["education"] and not meta.get("education_level"):
        meta["education_level"] = str(data["education"][0].get("degree") or "").strip()
    if not str(data.get("summary") or "").strip():
        data["summary"] = _build_summary(data, meta, industry, target_role, data["skills"])
    elif len(str(data.get("summary") or "")) > 120:
        data["summary"] = str(data["summary"]).strip()[:120]
        data["summary"] = str(data["summary"]).strip()
    data.setdefault("fact_sources", {"raw_input": "query/cv", "jd_usage": "JD only used for direction; facts must come from query/cv."})
    return data

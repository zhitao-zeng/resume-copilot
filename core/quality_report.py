"""Deterministic, score-free quality reporting for generated resumes.

The report separates three questions that must not be conflated:

* Which candidate source items were not represented in the final resume?
* Which generated claims were removed because they had no candidate evidence?
* Which job requirements have direct, related, or no evidence in the current
  candidate material?

No function in this module calls an LLM.  Job requirements are extracted from
the supplied JD at runtime, so the primary path does not depend on an
ever-growing industry or skill dictionary.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from atomic_fact_audit import audit_atomic_facts
from evidence_binding import measure_source_coverage, source_fact_units
from source_adapter import candidate_blocks
from v2_schemas import CanonicalResume, EvidenceBinding, SourceBundle


_MAX_REQUIREMENTS = 10
_MAX_UNREPRESENTED_ITEMS = 12
_MAX_REMOVED_ITEMS = 12
_MAX_CLAIM_GAPS = 4
_MAX_FOLLOW_UP_QUESTIONS = 4

_JD_HEADINGS = {
    "岗位职责", "工作职责", "职位职责", "职责描述", "职位描述", "工作内容",
    "任职要求", "岗位要求", "职位要求", "任职资格", "岗位资格", "职位资格",
    "优先条件", "加分项", "我们希望你", "你需要具备",
    "responsibilities", "responsibility", "requirements", "requirement",
    "qualifications", "qualification", "nice to have", "preferred qualifications",
}
_JD_STOP_HEADINGS = {
    "公司介绍", "企业介绍", "团队介绍", "关于我们", "薪资福利", "薪酬福利",
    "福利待遇", "工作地点", "办公地点", "投递方式", "company profile",
    "about us", "benefits", "compensation", "location", "how to apply",
}
_JD_NOISE = (
    "公司介绍", "企业介绍", "关于我们", "薪资", "薪酬", "福利", "五险一金",
    "工作地点", "办公地点", "工作地址", "办公地址", "联系地址", "上班地点", "校区",
    "招聘人数", "立即申请", "投递简历", "职位类别",
    "职位详情", "招聘官网", "联系我们", "company profile", "benefits",
    "salary", "location", "apply now", "about us",
)
_JD_PREDICATES = (
    "负责", "参与", "主导", "推动", "协助", "支持", "完成", "输出", "设计",
    "开发", "构建", "管理", "分析", "规划", "制定", "实施", "执行", "维护",
    "要求", "必须", "需要", "应当", "应具备", "应负责", "具备", "熟悉", "掌握", "能够", "持有",
    "优先", "经验", "学历", "专业", "responsible", "develop", "design",
    "build", "manage", "analyze", "must", "required", "proficient", "familiar",
    "experience", "degree", "qualification", "preferred",
)
_REQUIREMENT_SHELL_WORDS = (
    "候选人应", "候选人需", "我们希望你", "你需要具备", "岗位要求", "任职要求",
    "任职资格", "职位要求", "工作职责", "岗位职责", "职位职责", "主要负责",
    "负责", "要求", "必须", "需要", "应当", "应具备", "应负责", "具备", "熟悉", "掌握",
    "能够", "可以", "相关", "工作", "岗位", "能力", "经验", "优先考虑",
    "优先", "以上", "responsible for", "responsibilities include",
    "requirements include", "required", "must have", "must", "should",
    "proficient in", "familiar with", "experience with", "preferred",
)
_MATCH_STOP_WORDS = {
    "负责", "参与", "协助", "支持", "完成", "进行", "开展", "推动", "具备",
    "熟悉", "掌握", "能够", "要求", "必须", "需要", "相关", "工作", "岗位",
    "能力", "经验", "优先", "以上", "以及", "并且", "可以", "良好", "较强",
    "responsible", "required", "must", "should", "experience", "ability",
    "skills", "skill", "with", "and", "the", "for", "have", "plus",
}
_STRONG_OWNERSHIP = ("主导", "牵头", "独立负责", "独立完成", "全权负责", "owner", "led")
_MEDIUM_OWNERSHIP = ("负责", "管理", "组织", "推动", "设计", "开发", "implemented")
_WEAK_OWNERSHIP = ("参与", "协助", "支持", "配合", "接触", "了解", "assisted", "supported")
_NEGATIVE_FACT = re.compile(
    r"(?:没有(?:相关|对应|这方面|此类|任何)?(?:经验|经历|技能|资质|证书|背景)|"
    r"无(?:相关|对应|该领域|工作|项目)?(?:经验|经历|技能|资质|证书|背景)|"
    r"未曾|未参与|不具备|不会|未使用|没有做过|"
    r"no\s+(?:relevant\s+)?experience|never\s+used)",
    re.IGNORECASE,
)
_ACTION_SIGNAL = re.compile(
    r"(?:负责|主导|牵头|参与|协助|支持|组织|推动|设计|开发|构建|实现|制定|"
    r"实施|执行|维护|运营|教学|诊疗|审核|销售|研究|撰写|协调|分析|排查|"
    r"built|designed|implemented|developed|managed|analyzed|delivered|led)",
    re.IGNORECASE,
)
_METHOD_SIGNAL = re.compile(
    r"(?:通过|采用|使用|基于|借助|运用|利用|结合|按照|围绕|依托|针对|"
    r"using|via|based\s+on|with\s+the\s+use\s+of)",
    re.IGNORECASE,
)
_OUTPUT_SIGNAL = re.compile(
    r"(?:输出|交付|上线|发布|形成|完成|实现|达成|获得|发表|获奖|覆盖|解决|"
    r"提升|降低|增长|减少|缩短|节省|产出|delivered|launched|published|"
    r"achieved|improved|reduced|increased|resolved)",
    re.IGNORECASE,
)
_QUANTIFIED_SIGNAL = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:%|人|次|项|个|条|篇|例|台|套|万|万元|"
    r"元|年|个月|月|天|小时|分钟|ms|s|qps|tps|fps|mb|gb)?",
    re.IGNORECASE,
)
_CONTEXT_ONLY_BULLET = re.compile(
    r"^(?:行业|领域|部门|地点|工作地点|客户类型|业务范围|服务对象|"
    r"技术环境|项目背景|专业方向|研究方向|产品|焦点)\s*[:：]",
    re.IGNORECASE,
)


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[^a-z0-9+.#/_\-\u4e00-\u9fff]+", "", value)


def _safe_excerpt(text: str, limit: int = 160) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    cutoff = limit - 1
    for marker in ("。", "；", ";", "，", ",", " "):
        position = value.rfind(marker, max(1, limit // 2), cutoff)
        if position > 0:
            cutoff = position + 1
            break
    return value[:cutoff].rstrip() + "…"


def _preferred_jd_text(jd_text: str) -> str:
    """Prefer explicit pasted JD text over fetched page boilerplate."""

    value = str(jd_text or "")
    if "【链接页面正文（仅作补充）】" in value:
        value = value.split("【链接页面正文（仅作补充）】", 1)[0]
    return value


def _clean_requirement(value: str) -> str:
    text = re.sub(r"^\s*(?:[-*•●▪◦]|\d+[.、)）]|[（(]?[一二三四五六七八九十]+[)）、.])\s*", "", value)
    text = re.sub(r"^【[^】]+】\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ：:；;。")


def _requirement_core(value: str) -> str:
    core = unicodedata.normalize("NFKC", str(value or "")).casefold()
    # Remove sentence shells only where they act as a prefix.  Global Chinese
    # replacement corrupts real terms such as "需求优先级" -> "需求级".
    prefix_pattern = "|".join(
        re.escape(word.casefold())
        for word in sorted(_REQUIREMENT_SHELL_WORDS, key=len, reverse=True)
    )
    previous = None
    while core != previous:
        previous = core
        core = re.sub(
            rf"^\s*(?:{prefix_pattern})\s*[:：,，;；-]*\s*",
            "",
            core,
            count=1,
            flags=re.IGNORECASE,
        )
    core = re.sub(r"[\s:：,，。；;()（）【】\[\]]+", "", core)
    return core


def _looks_like_title_only(value: str) -> bool:
    text = value.strip().casefold()
    has_obligation = bool(re.search(
        r"(?:负责|参与|主导|推动|协助|支持|要求|必须|需要|具备|熟悉|掌握|能够|持有|"
        r"responsible|required|must|proficient|familiar)",
        text,
        re.IGNORECASE,
    ))
    # Exported files often prepend a short document title such as
    # "design JD". Because "design" is also a responsibility predicate, the
    # generic extractor used to report that title as an unmet requirement.
    document_title = bool(re.fullmatch(
        r"(?:[\u4e00-\u9fffa-z0-9+.#/_\- ]{1,40}\s+)?"
        r"(?:jd|job\s+description|职位描述|岗位描述|岗位说明书?)",
        text,
        re.IGNORECASE,
    ))
    if document_title:
        return not has_obligation

    title_shape = bool(re.fullmatch(
        r"[\u4e00-\u9fffa-z0-9+.#/_\- ]{2,28}(?:岗位|职位|工程师|经理|专员|分析师|"
        r"设计师|顾问|医生|医师|教师|老师|研究员|intern|engineer|manager|analyst)",
        text,
        re.IGNORECASE,
    ))
    if not title_shape:
        return False
    # Domain words such as "开发/设计/分析" often form part of a role title.
    # Require an explicit obligation/action shell before treating a title-shaped
    # line as a genuine requirement sentence.
    return not has_obligation


def extract_jd_requirements(jd_text: str, *, limit: int = _MAX_REQUIREMENTS) -> list[str]:
    """Extract ordered requirement statements without an industry dictionary."""

    value = _preferred_jd_text(jd_text)
    if not value.strip():
        return []

    # Preserve list lines first, but also support compact one-line JDs.
    raw_segments: list[str] = []
    for line in re.split(r"[\r\n]+", value):
        line = line.strip()
        if not line:
            continue
        raw_segments.extend(part for part in re.split(r"(?<=[。；;])", line) if part.strip())

    requirements: list[str] = []
    seen: set[str] = set()
    in_relevant_section = False
    for raw in raw_segments:
        text = _clean_requirement(raw)
        if not text:
            continue
        heading_key = re.sub(r"[\s:：]+", " ", text).strip().casefold()
        if heading_key in _JD_HEADINGS:
            in_relevant_section = True
            continue
        if any(
            heading_key == stop or heading_key.startswith(stop + " ")
            for stop in _JD_STOP_HEADINGS
        ):
            in_relevant_section = False
            continue
        # Location and campus rows often follow the requirement section with
        # no new heading.  They describe where the job is based, not evidence
        # the candidate should claim on a resume.
        if re.match(
            r"^(?:校区(?:[一二三四五六七八九十\d]+)?|工作地点|办公地点|工作地址|"
            r"办公地址|联系地址|上班地点|location)\s*[:：]",
            text,
            re.IGNORECASE,
        ):
            continue
        # A short machine label before the first real JD section (for example
        # "design" or "product_pm") is a document title, not a candidate
        # requirement. Inside an explicit requirements section, a bare skill
        # such as "Python" remains eligible.
        if (
            not in_relevant_section
            and re.fullmatch(r"[a-z][a-z0-9_-]{1,40}", heading_key)
        ):
            continue
        if any(noise.casefold() in text.casefold() for noise in _JD_NOISE):
            # A mixed line may still contain a valid requirement after a
            # heading, but pure page/navigation boilerplate is discarded.
            if not any(predicate.casefold() in text.casefold() for predicate in _JD_PREDICATES):
                continue
        if text.lower().startswith(("http://", "https://", "www.")):
            continue
        if len(text) < 3 or len(text) > 220 or _looks_like_title_only(text):
            continue
        has_predicate = any(predicate.casefold() in text.casefold() for predicate in _JD_PREDICATES)
        has_list_shape = bool(re.search(r"[、,/]|\band\b|\bor\b", text, re.IGNORECASE))
        if not in_relevant_section and not has_predicate and not has_list_shape:
            continue
        core = _normalize(_requirement_core(text))
        if len(core) < 2 or core in seen:
            continue
        seen.add(core)
        requirements.append(text)
        if len(requirements) >= 100:
            break
    # Reserve the bounded output for explicit hard requirements before broad
    # responsibilities or nice-to-have statements, then restore document
    # order so the report remains easy to compare with the original JD.
    rank = {"required": 0, "responsibility": 1, "preferred": 2}
    selected_indexes = sorted(
        range(len(requirements)),
        key=lambda index: (rank[_priority(requirements[index])], index),
    )[:max(0, limit)]
    return [requirements[index] for index in sorted(selected_indexes)]


def _priority(requirement: str) -> str:
    value = requirement.casefold()
    if any(word in value for word in (
        "优先条件", "优先考虑", "者优先", "加分项", "加分",
        "preferred", "nice to have", "plus",
    )):
        return "preferred"
    if any(word in value for word in (
        "必须", "要求", "至少", "学历", "经验", "熟悉", "掌握", "具备", "能够", "持有",
        "must", "required", "degree", "proficient", "familiar",
    )):
        return "required"
    return "responsibility"


def _kind(requirement: str) -> str:
    value = requirement.casefold()
    if re.search(r"\d+(?:\.\d+)?\s*年(?:以上)?", value):
        return "tenure"
    if any(word in value for word in ("学历", "本科", "硕士", "博士", "大专", "degree", "bachelor", "master", "phd")):
        return "qualification"
    if any(word in value for word in ("证书", "资格证", "执业", "认证", "certificate", "license")):
        return "qualification"
    if any(word in value for word in ("负责", "主导", "参与", "管理", "设计", "开发", "构建", "输出", "推动", "responsible", "build", "design", "manage")):
        return "responsibility"
    if any(word in value for word in ("熟悉", "掌握", "使用", "技能", "proficient", "familiar", "skill")):
        return "skill"
    return "other"


def _ownership_level(text: str) -> int:
    value = str(text or "").casefold()
    if any(token.casefold() in value for token in _STRONG_OWNERSHIP):
        return 4
    if any(token.casefold() in value for token in _MEDIUM_OWNERSHIP):
        return 3
    if "参与" in value:
        return 2
    if any(token.casefold() in value for token in _WEAK_OWNERSHIP):
        return 1
    return 0


def _features(text: str) -> tuple[set[str], set[str], set[str]]:
    """Return dynamic phrases, Latin terms and Chinese bi/tri-grams."""

    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    latin = {
        token for token in re.findall(r"[a-z][a-z0-9+.#/_\-]{1,31}", value)
        if token not in _MATCH_STOP_WORDS
    }
    phrase_source = value
    for word in sorted(_MATCH_STOP_WORDS, key=len, reverse=True):
        escaped = re.escape(word)
        if re.fullmatch(r"[a-z ]+", word, re.IGNORECASE):
            phrase_source = re.sub(
                rf"(?<![a-z]){escaped}(?![a-z])", " ", phrase_source,
                flags=re.IGNORECASE,
            )
        else:
            # Chinese action shells are ignored only at a segment boundary;
            # a substring inside a domain term remains meaningful.
            phrase_source = re.sub(
                rf"(^|[\s,，。；;、/|()（）【】\[\]]){escaped}",
                r"\1 ",
                phrase_source,
                flags=re.IGNORECASE,
            )
    raw_phrases = re.split(r"[\s,，。；;、/|()（）【】\[\]]+|(?:以及|并且|and|or)", phrase_source)
    phrases = {
        _normalize(phrase)
        for phrase in raw_phrases
        if 2 <= len(_normalize(phrase)) <= 32
    }
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", phrase_source))
    grams: set[str] = set()
    for size in (2, 3):
        grams.update(chinese[index:index + size] for index in range(max(0, len(chinese) - size + 1)))
    grams.difference_update({"相关", "工作", "岗位", "能力", "经验", "要求", "负责", "参与"})
    return phrases, latin, grams


def _canonical_path_values(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten canonical data using the exact path format used by bindings.

    A lookup table is safer than parsing paths: a long-tail
    ``additional_sections`` title may itself contain ``.`` or ``[``.
    """

    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_canonical_path_values(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_canonical_path_values(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and prefix:
        result[prefix] = value
    return result


def _record_root(path: str) -> str:
    match = re.match(r"(experience|research|activities|projects)\[(\d+)]", path)
    return match.group(0) if match else path.split(".", 1)[0]


def _trusted_claims(
    resume: CanonicalResume,
    source: SourceBundle,
    bindings: Iterable[EvidenceBinding],
) -> list[dict[str, str]]:
    data = resume.model_dump()
    path_values = _canonical_path_values(data)
    block_by_id = {block.block_id: block for block in source.blocks}
    trusted_block_ids = {block.block_id for block in candidate_blocks(source)}
    claims: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    excluded_paths = {"meta.target_role", "meta.job_intention", "summary", "framework"}
    for binding in bindings:
        path = str(_value(binding, "path", ""))
        block_id = str(_value(binding, "block_id", ""))
        if not path or path in seen_paths or path in excluded_paths or path.startswith("summary"):
            continue
        if block_id not in trusted_block_ids:
            continue
        block = block_by_id.get(block_id)
        if block is None or block.source_type == "jd":
            continue
        value = path_values.get(path)
        if not isinstance(value, str) or not value.strip():
            continue
        # A negative statement mentions the term but is evidence of absence,
        # never evidence that the candidate has the capability.
        negative = bool(_NEGATIVE_FACT.search(block.text) or _NEGATIVE_FACT.search(value))
        claims.append({
            "path": path,
            "text": value.strip(),
            "source_text": block.text.strip(),
            "block_id": block_id,
            "root": _record_root(path),
            "negative": "1" if negative else "0",
        })
        seen_paths.add(path)
    return claims


def _best_claim_match(requirement: str, claims: list[dict[str, str]]) -> dict[str, Any]:
    req_phrases, req_latin, req_grams = _features(_requirement_core(requirement))
    req_core = _normalize(_requirement_core(requirement))
    req_ownership = _ownership_level(requirement)
    candidates: list[dict[str, Any]] = []
    for claim in claims:
        claim_blob = f"{claim['text']} {claim['source_text']}"
        claim_normalized = _normalize(claim_blob)
        claim_phrases, claim_latin, claim_grams = _features(claim_blob)
        direct_phrases = {
            phrase for phrase in req_phrases
            if len(phrase) >= 3 and phrase in claim_normalized
        }
        latin_matches = req_latin & claim_latin
        gram_matches = req_grams & claim_grams
        gram_coverage = len(gram_matches) / max(1, len(req_grams))
        phrase_coverage = len(direct_phrases) / max(1, len(req_phrases))
        latin_coverage = len(latin_matches) / max(1, len(req_latin)) if req_latin else 0.0
        direct_core = bool(len(req_core) >= 3 and req_core in claim_normalized)
        rank = max(
            1.0 if direct_core else 0.0,
            gram_coverage,
            phrase_coverage,
            latin_coverage,
        )
        # Prefer behavior evidence over a bare skill or title when ranks tie.
        path_strength = 2 if ".bullets[" in claim["path"] else 1
        candidates.append({
            "claim": claim,
            "rank": rank,
            "path_strength": path_strength,
            "direct_core": direct_core,
            "direct_phrases": direct_phrases,
            "latin_matches": latin_matches,
            "gram_coverage": gram_coverage,
            "req_phrases": req_phrases,
            "claim_ownership": _ownership_level(claim_blob),
            "negative": claim["negative"] == "1",
        })
    if not candidates:
        return {}
    best = max(candidates, key=lambda item: (item["rank"], item["path_strength"]))
    best["required_ownership"] = req_ownership
    return best


def _degree_status(requirement: str, claims: list[dict[str, str]]) -> str | None:
    ranks = {"大专": 1, "专科": 1, "本科": 2, "学士": 2, "硕士": 3, "研究生": 3, "博士": 4}
    requirement_rank = max((rank for name, rank in ranks.items() if name in requirement), default=0)
    if not requirement_rank:
        return None
    candidate_rank = max(
        (
            rank
            for claim in claims
            if re.fullmatch(r"education\[\d+]\.degree", claim["path"])
            and claim["negative"] != "1"
            for name, rank in ranks.items()
            if name in claim["text"]
        ),
        default=0,
    )
    if candidate_rank >= requirement_rank:
        return "supported"
    return "partial" if candidate_rank else "missing"


def _tenure_status(
    requirement: str,
    claims: list[dict[str, str]],
    best: dict[str, Any],
) -> str | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*年(?:以上)?", requirement)
    if not match:
        return None
    if best.get("negative"):
        return "missing"
    required = float(match.group(1))
    explicit_years = [
        float(value)
        for claim in claims
        if claim["path"] == "meta.work_experience" and claim["negative"] != "1"
        for value in re.findall(r"(\d+(?:\.\d+)?)\s*年", claim["text"])
    ]
    if explicit_years and max(explicit_years) >= required:
        # Total tenure supports only a generic work-experience threshold.  It
        # does not prove the same number of years in a named domain.
        domain = re.sub(r"\d+(?:\.\d+)?\s*年(?:以上)?", "", requirement)
        domain = re.sub(r"(?:相关|工作|从业|经验|要求|至少|具备|以上)", "", domain)
        return "supported" if len(_normalize(domain)) < 2 else "partial"
    if explicit_years or (best and not best.get("negative") and best.get("rank", 0.0) >= 0.20):
        return "partial"
    return "missing"


def _missing_aspects(requirement: str, best: dict[str, Any]) -> list[str]:
    if not best:
        return [_safe_excerpt(_requirement_core(requirement), 48)]
    req_phrases = best.get("req_phrases") or set()
    claim_blob = _normalize(
        f"{best['claim']['text']} {best['claim']['source_text']}"
    )
    missing = [phrase for phrase in sorted(req_phrases, key=len, reverse=True) if phrase not in claim_blob]
    req_level = int(best.get("required_ownership", 0))
    claim_level = int(best.get("claim_ownership", 0))
    if req_level >= 3 and claim_level < req_level:
        missing.insert(0, "责任边界")
    return [_safe_excerpt(item, 40) for item in missing[:3] if item]


_ASPECT_ACTION_START = (
    "负责", "参与", "主导", "推动", "协助", "支持", "完成", "输出", "交付", "搭建",
    "设计", "开发", "构建", "管理", "分析", "规划", "制定", "实施", "执行", "维护",
    "使用", "熟悉", "掌握", "具备", "能够", "持有", "build", "design", "develop",
    "manage", "analyze", "deliver", "use", "using",
)


def _requirement_aspects(requirement: str) -> list[list[str]]:
    """Split explicit AND facets while keeping OR alternatives together."""

    value = _clean_requirement(requirement)
    action_pattern = "|".join(re.escape(item) for item in _ASPECT_ACTION_START)
    value = re.sub(
        rf"(?:并且|同时|且|并)(?=(?:{action_pattern}))",
        "；",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+and\s+", "；", value, flags=re.IGNORECASE)
    # “以及” is an unambiguous conjunction. Bare 和/与 are split only when an
    # earlier comma/list delimiter has established an enumeration (A、B和C).
    # This keeps exact gap reporting for JD lists without corrupting lexical
    # words used by unrelated industries, such as 和平、中和、参与、与会.
    value = re.sub(
        r"(?<=[\u4e00-\u9fffA-Za-z0-9])以及(?=[\u4e00-\u9fffA-Za-z0-9])",
        "；",
        value,
    )

    def split_enumerated_conjunction(match: re.Match[str]) -> str:
        prefix = value[:match.start()]
        if not re.search(r"[，,；;、]", prefix):
            return match.group(0)
        conjunction = match.group(0)
        previous = value[match.start() - 1:match.start()]
        following = value[match.end():match.end() + 1]
        if conjunction == "和" and (
            previous in "共调中饱温亲柔缓总"
            or following in "平谐解声睦蔼善气服"
        ):
            return conjunction
        if conjunction == "与" and (previous in "参赠授" or following == "会"):
            return conjunction
        return "；"

    value = re.sub(
        r"(?<=[\u4e00-\u9fffA-Za-z0-9])(?:和|与)(?=[\u4e00-\u9fffA-Za-z0-9])",
        split_enumerated_conjunction,
        value,
    )
    parts = [item.strip() for item in re.split(r"[，,；;、]+", value) if item.strip()]
    groups: list[list[str]] = []
    for part in parts:
        alternatives = [
            item.strip()
            for item in re.split(r"(?:\s+or\s+|或者|或)", part, flags=re.IGNORECASE)
            if item.strip()
        ]
        if alternatives:
            groups.append(alternatives)
    return groups or [[value]]


def _single_aspect_status(
    aspect: str,
    claims: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    best = _best_claim_match(aspect, claims)
    if best.get("negative"):
        return "missing", best

    status = "missing"
    if best:
        direct_or_strong = (
            best.get("direct_core")
            or (
                best.get("rank", 0.0) >= 0.72
                and best.get("path_strength", 0) >= 2
            )
        )
        related = (
            best.get("rank", 0.0) >= 0.20
            or bool(best.get("direct_phrases"))
            or bool(best.get("latin_matches"))
        )
        if direct_or_strong:
            status = "supported"
        elif related:
            status = "partial"

    degree_status = _degree_status(aspect, claims)
    if degree_status is not None:
        status = degree_status
    tenure_status = _tenure_status(aspect, claims, best)
    if tenure_status is not None:
        status = tenure_status

    if best and status == "supported":
        req_level = int(best.get("required_ownership", 0))
        claim_level = int(best.get("claim_ownership", 0))
        if req_level >= 3 and claim_level < req_level:
            status = "partial"
        if _kind(aspect) == "responsibility" and best.get("path_strength", 0) < 2:
            status = "partial"
    return status, best


def assess_jd_requirements(
    jd_text: str,
    target_role: str,
    resume: CanonicalResume,
    evidence_bindings: Iterable[EvidenceBinding],
    source: SourceBundle,
    *,
    jd_supplied: bool | None = None,
    jd_unavailable: bool = False,
) -> dict[str, Any]:
    """Return evidence states for dynamically extracted JD requirements."""

    jd_available = bool(str(jd_text or "").strip())
    supplied = jd_available if jd_supplied is None else bool(jd_supplied)
    unavailable = bool(supplied and jd_unavailable and not jd_available)
    if unavailable:
        recommendation = (
            "已收到JD链接，但当前无法读取页面内容。请直接粘贴岗位职责和任职要求；"
            "在此之前不会推测岗位要求，也不会把链接内容写成候选人经历。"
        )
        return {
            "has_job_description": True,
            "job_description_available": False,
            "source_status": "unavailable",
            "target_role": str(target_role or "").strip(),
            "requirements": [],
            "supported_requirement_count": 0,
            "partial_requirement_count": 0,
            "missing_requirement_count": 0,
            "recommendations": [recommendation],
            "follow_up_questions": ["请粘贴目标岗位的岗位职责和任职要求，以便逐项核对匹配情况。"],
        }

    bindings = list(evidence_bindings)
    claims = _trusted_claims(resume, source, bindings)
    items: list[dict[str, Any]] = []
    recommendation_entries: list[tuple[str, str]] = []
    question_entries: list[tuple[str, str]] = []
    for requirement in extract_jd_requirements(jd_text):
        group_results: list[dict[str, Any]] = []
        for alternatives in _requirement_aspects(requirement):
            alternative_results = []
            for aspect in alternatives:
                aspect_status, best = _single_aspect_status(aspect, claims)
                alternative_results.append({
                    "aspect": aspect,
                    "status": aspect_status,
                    "best": best,
                })
            status_rank = {"missing": 0, "partial": 1, "supported": 2}
            group_results.append(max(
                alternative_results,
                key=lambda item: (
                    status_rank[item["status"]],
                    item["best"].get("rank", 0.0),
                ),
            ))

        statuses = [item["status"] for item in group_results]
        if statuses and all(value == "supported" for value in statuses):
            status = "supported"
        elif any(value in {"supported", "partial"} for value in statuses):
            status = "partial"
        else:
            status = "missing"

        missing_aspects: list[str] = []
        for result in group_results:
            if result["status"] == "supported":
                continue
            details = _missing_aspects(result["aspect"], result["best"])
            missing_aspects.extend(details or [_safe_excerpt(_requirement_core(result["aspect"]), 48)])
        missing_aspects = list(dict.fromkeys(item for item in missing_aspects if item))[:3]
        evidence: list[dict[str, str]] = []
        evidence_paths: set[str] = set()
        for result in group_results:
            best = result["best"]
            if result["status"] == "missing" or not best or best.get("negative"):
                continue
            claim = best["claim"]
            if claim["path"] in evidence_paths:
                continue
            evidence_paths.add(claim["path"])
            evidence.append({
                "canonical_field_path": claim["path"],
                "excerpt": _safe_excerpt(claim["text"], 120),
            })

        if status == "supported":
            recommendation = (
                f"JD重点“{_safe_excerpt(requirement, 72)}”已有直接事实依据，"
                "建议将对应经历前置，并保留具体行动和结果。"
            )
            question = ""
        elif status == "partial":
            gap_text = "、".join(missing_aspects) or "直接事实与可核验结果"
            recommendation = (
                f"JD重点“{_safe_excerpt(requirement, 72)}”已有部分相关证据，"
                f"但仍缺少“{gap_text}”；请只补充真实发生的个人动作、交付物或结果。"
            )
            question = (
                f"对于“{_safe_excerpt(requirement, 64)}”，当前已有部分证据，"
                f"还需确认“{gap_text}”。若真实发生，请补充对应的个人行动和可核验结果。"
            )
        else:
            gap_text = "、".join(missing_aspects) or "相关场景、个人行动与可核验结果"
            recommendation = (
                f"JD重点“{_safe_excerpt(requirement, 72)}”在当前材料中未找到直接证据；"
                f"需优先确认“{gap_text}”。若确有相关经历请补充，未参与则不要写入简历。"
            )
            question = (
                f"你是否有与“{_safe_excerpt(requirement, 64)}”相关的真实经历？"
                f"如有，请围绕“{gap_text}”补充真实场景、个人行动、交付物和结果。"
            )

        requirement_id = "req_" + hashlib.sha1(
            _normalize(requirement).encode("utf-8")
        ).hexdigest()[:10]
        items.append({
            "requirement_id": requirement_id,
            "requirement": requirement,
            "kind": _kind(requirement),
            "priority": _priority(requirement),
            "status": status,
            "evidence": evidence,
            "missing_aspects": missing_aspects,
            "recommendation": recommendation,
        })
        recommendation_entries.append((status, recommendation))
        if question:
            question_entries.append((status, question))

    gap_rank = {"missing": 0, "partial": 1, "supported": 2}
    recommendations = [
        value for _, value in sorted(recommendation_entries, key=lambda item: gap_rank[item[0]])
    ]
    questions = [
        value for _, value in sorted(question_entries, key=lambda item: gap_rank[item[0]])
    ]
    return {
        "has_job_description": supplied,
        "job_description_available": jd_available,
        "source_status": "available" if jd_available else "not_provided",
        "target_role": str(target_role or "").strip(),
        "requirements": items,
        "supported_requirement_count": sum(item["status"] == "supported" for item in items),
        "partial_requirement_count": sum(item["status"] == "partial" for item in items),
        "missing_requirement_count": sum(item["status"] == "missing" for item in items),
        "recommendations": recommendations[:3],
        "follow_up_questions": questions[:4],
    }


def _normalize_bindings(values: Iterable[Any]) -> list[EvidenceBinding]:
    result: list[EvidenceBinding] = []
    for value in values:
        if isinstance(value, EvidenceBinding):
            result.append(value)
            continue
        try:
            result.append(EvidenceBinding.model_validate(value))
        except Exception:
            continue
    return result


def _normalize_dict_items(values: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            result.append(dict(value))
        elif hasattr(value, "model_dump"):
            result.append(value.model_dump())
        elif hasattr(value, "__dict__"):
            result.append(dict(value.__dict__))
    return result


def _friendly_path(path: str) -> str:
    labels = {
        "meta.name": "姓名", "meta.phone": "联系电话", "meta.email": "邮箱",
        "meta.work_experience": "工作年限", "education": "教育经历",
        "experience": "工作/实习经历", "research": "科研经历",
        "activities": "校园/社会活动", "projects": "项目经历",
        "skills.items": "专业技能", "awards": "荣誉奖项",
        "publications": "论文成果", "patents": "专利成果",
        "certifications": "证书/资质", "training": "培训经历",
        "teaching": "教学经历", "additional_sections": "其它经历",
    }
    if path in labels:
        return labels[path]
    for prefix, label in labels.items():
        if path.startswith(prefix + "[") or path.startswith(prefix + "."):
            indexes = [int(value) + 1 for value in re.findall(r"\[(\d+)]", path)]
            suffix = ""
            if indexes:
                suffix = f"第{indexes[0]}项"
                if len(indexes) > 1:
                    suffix += f"第{indexes[1]}条"
            return label + suffix
    return path


def _claim_record_label(resume: CanonicalResume, path: str) -> str:
    """Resolve a bullet path to the exact record the user should amend."""

    match = re.match(r"(experience|projects|research|activities)\[(\d+)]", path)
    if not match:
        return ""
    section, raw_index = match.groups()
    records = getattr(resume, section)
    index = int(raw_index)
    if index >= len(records):
        return ""
    record = records[index]
    if section == "experience":
        values = (record.organization, record.role)
    elif section == "projects":
        values = (record.name, record.role)
    elif section == "research":
        values = (record.institution, record.topic)
    else:
        values = (record.organization, record.role)
    return "｜".join(dict.fromkeys(
        str(value or "").strip() for value in values if str(value or "").strip()
    ))


def _claim_improvement_items(
    resume: CanonicalResume,
    source: SourceBundle,
    bindings: list[EvidenceBinding],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for claim in _trusted_claims(resume, source, bindings):
        if ".bullets[" not in claim["path"]:
            continue
        text = claim["text"]
        # Context labels preserve useful source facts, but they are not
        # accomplishment claims and should not trigger a request for an
        # action/method/result that never existed in the source.
        if _CONTEXT_ONLY_BULLET.match(text.strip()):
            continue
        missing: list[str] = []
        if not _ACTION_SIGNAL.search(text):
            missing.append("个人行动")
        if not _METHOD_SIGNAL.search(text):
            missing.append("方法或过程")
        if not (_OUTPUT_SIGNAL.search(text) or _QUANTIFIED_SIGNAL.search(text)):
            missing.append("交付物或结果")
        if not missing:
            continue
        record_label = _claim_record_label(resume, claim["path"])
        prompt_by_dimension = {
            "个人行动": "你本人具体负责哪一步",
            "方法或过程": "具体通过哪些步骤、工具或协作方式完成",
            "交付物或结果": "最终产出了什么、如何验收，或产生了什么可核验影响",
        }
        exact_questions = "；".join(prompt_by_dimension[value] for value in missing)
        location = f"{record_label}中的" if record_label else ""
        question = (
            f"{location}“{_safe_excerpt(text, 54)}”仍缺{'、'.join(missing)}。"
            f"请补充：{exact_questions}；只填写真实发生且能够核验的信息。"
        )
        items.append({
            "canonical_field_path": claim["path"],
            "record_label": record_label,
            "excerpt": _safe_excerpt(text, 120),
            "missing_dimensions": missing,
            "question": question,
        })
        if len(items) >= _MAX_CLAIM_GAPS:
            break
    return items


def build_quality_report(
    *,
    source: SourceBundle,
    resume: CanonicalResume | dict[str, Any],
    evidence_bindings: Iterable[Any],
    changes: Iterable[Any] = (),
    missing_fields: Iterable[Any] = (),
    jd_text: str = "",
    jd_supplied: bool | None = None,
    jd_unavailable: bool = False,
    target_role: str = "",
    framework_mode: bool = False,
) -> dict[str, Any]:
    """Build a bounded, deterministic QualityReport with no synthetic score."""

    canonical = resume if isinstance(resume, CanonicalResume) else CanonicalResume.model_validate(resume)
    bindings = _normalize_bindings(evidence_bindings)
    normalized_changes = _normalize_dict_items(changes)
    normalized_missing = _normalize_dict_items(missing_fields)

    units = source_fact_units(source)
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    _, missing_unit_ids = measure_source_coverage(
        source,
        bindings,
        allow_distributed=True,
    )
    missing_units = [unit_by_id[unit_id] for unit_id in missing_unit_ids if unit_id in unit_by_id]
    source_item_count = len(units)

    if framework_mode and source_item_count == 0:
        preservation_status = "not_applicable"
    elif missing_units:
        preservation_status = "unrepresented_items_detected"
    else:
        preservation_status = "no_unrepresented_items_detected"

    unrepresented_items = [
        {
            "block_id": unit["block_id"],
            "unit_id": unit["unit_id"],
            "source_type": unit["source_type"],
            "section_hint": unit["section_hint"] or None,
            "record_id": unit["record_id"] or None,
            "dimensions": unit["dimensions"],
            "excerpt": _safe_excerpt(unit["text"]),
        }
        for unit in missing_units[:_MAX_UNREPRESENTED_ITEMS]
    ]

    final_bound_paths = {binding.path for binding in bindings}
    unsupported_by_path: dict[str, dict[str, Any]] = {}
    for item in normalized_changes:
        path = str(item.get("path", "")).strip()
        reason = str(item.get("reason", "")).casefold()
        if (
            str(item.get("action", "")) not in {"remove", "clear"}
            or not (
                "evidence" in reason
                or "unsupported" in reason
                or "ground" in reason
            )
            or not path
            or path == "*"
            # Draft summaries are deliberately discarded and rebuilt from
            # grounded final fields. Reporting each intermediate sentence as
            # a removed fabrication produced alarming counts such as 10-15
            # even when no final resume claim was unsupported.
            or path.startswith("summary[")
            # A path recovered later with a valid final binding was corrected,
            # not omitted from the delivered resume.
            or path in final_bound_paths
        ):
            continue
        unsupported_by_path.setdefault(path, item)
    unsupported = list(unsupported_by_path.values())
    removed_items = [
        {
            "canonical_field_path": str(item.get("path", "")),
            "field_label": _friendly_path(str(item.get("path", ""))),
            "message": "缺少候选人事实依据，未写入最终简历。",
        }
        for item in unsupported[:_MAX_REMOVED_ITEMS]
    ]

    job_alignment = assess_jd_requirements(
        jd_text,
        target_role,
        canonical,
        bindings,
        source,
        jd_supplied=jd_supplied,
        jd_unavailable=jd_unavailable,
    )
    claim_gaps = _claim_improvement_items(canonical, source, bindings)
    atomic_audit = audit_atomic_facts(
        source=source,
        resume=canonical,
        evidence_bindings=bindings,
    )
    follow_ups = list(job_alignment.get("follow_up_questions", []))
    follow_ups.extend(item["question"] for item in claim_gaps)
    follow_ups = list(dict.fromkeys(follow_ups))[:_MAX_FOLLOW_UP_QUESTIONS]

    return {
        "schema_version": "1.1",
        "document_mode": "framework" if framework_mode else "resume",
        "source_preservation": {
            "status": preservation_status,
            "source_item_count": source_item_count,
            "represented_source_item_count": max(0, source_item_count - len(missing_units)),
            "unrepresented_item_count": len(missing_units),
            "unrepresented_items": unrepresented_items,
            "truncated": len(missing_units) > len(unrepresented_items),
        },
        "fact_grounding": {
            "evidence_bound_item_count": len({binding.path for binding in bindings}),
            "unsupported_item_count": len(unsupported),
            "unsupported_items_removed": removed_items,
            "truncated": len(unsupported) > len(removed_items),
        },
        "atomic_factuality": atomic_audit["atomic_factuality"],
        "ownership_integrity": atomic_audit["ownership_integrity"],
        "structural_invariants": atomic_audit["structural_invariants"],
        "missing_information": normalized_missing,
        "claim_improvement_opportunities": claim_gaps,
        "job_alignment": job_alignment,
        "follow_up_questions": follow_ups,
    }

import html as html_lib
import copy
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from resume_common import _normalize_project_name

from docx import Document as DocxDocument
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from weasyprint import HTML as WeasyHTML
except ImportError:
    WeasyHTML = None

SUPPORTED_TEMPLATES = {"classic", "modern", "minimal"}
DEFAULT_DOC_FONT = os.getenv("RESUME_DOC_FONT", "Noto Sans CJK SC")
DEFAULT_DOC_MONO_FONT = os.getenv("RESUME_DOC_MONO_FONT", "DejaVu Sans Mono")
PDF_ACCENT_COLORS = {
    "classic": (0x0F, 0x34, 0x60),
    "modern": (0x00, 0x5A, 0x64),
    "minimal": (0x33, 0x33, 0x33),
}
PDF_BODY_COLOR = (0x22, 0x22, 0x22)
PDF_MUTED_COLOR = (0x66, 0x66, 0x66)
PDF_MARGIN_MM = os.getenv("RESUME_PDF_MARGIN_MM", "10mm")
logger = logging.getLogger(__name__)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", value.strip())
    return cleaned.strip("_") or "resume"


def _normalize_template(template: str) -> str:
    tpl = (template or "classic").strip()
    if not tpl:
        return "classic"
    # If it looks like a file path, check if file exists and return as-is
    if "/" in tpl or "\\" in tpl:
        if Path(tpl).is_file():
            return tpl
        return "classic"
    if tpl not in SUPPORTED_TEMPLATES:
        return "classic"
    return tpl


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (round(color[0] / 255, 4), round(color[1] / 255, 4), round(color[2] / 255, 4))


def _html_escape(value: Any) -> str:
    return html_lib.escape(str(value or ""), quote=True)


def _has_publication_content(pub: Any) -> bool:
    if isinstance(pub, dict):
        return any(str(pub.get(key, "")).strip() for key in ("venue", "title", "authors"))
    if isinstance(pub, str):
        return bool(pub.strip())
    return False


def _publication_line(pub: Any) -> str:
    if isinstance(pub, dict):
        venue = str(pub.get("venue", "")).strip()
        title = str(pub.get("title", "")).strip()
        authors = str(pub.get("authors", "")).strip()
        return " - ".join(part for part in (venue, title, authors) if part)
    if isinstance(pub, str):
        return pub.strip()
    return ""


def _normalize_text_items(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _collect_experience_bullets(exp: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    for key in (
        "function_description",
        "result_description",
        "bullets",
        "responsibilities",
        "achievements",
        "highlights",
        "details",
        "description",
    ):
        merged.extend(_normalize_text_items(exp.get(key)))
    # Keep order while deduping.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in merged:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:4]


def _collect_text_entries(payload: Any, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    merged: list[str] = []
    for key in keys:
        merged.extend(_normalize_text_items(payload.get(key)))
    seen: set[str] = set()
    deduped: list[str] = []
    for item in merged:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _collect_project_bullets(project: dict[str, Any]) -> list[str]:
    return _collect_text_entries(
        project,
        ("description", "function_description", "result_description", "bullets", "achievements", "highlights"),
    )


def _compress_resume_data_for_docx(resume_data: dict[str, Any]) -> dict[str, Any]:
    """Compress dense content to fit <= 3 pages while preserving all entries."""
    data = copy.deepcopy(resume_data) if isinstance(resume_data, dict) else {}
    # Summary: max 80 chars (fits 2-3 lines in DOCX)
    if isinstance(data.get("summary"), str):
        s = data["summary"].strip()
        if len(s) > 80:
            s = s[:77].rstrip("，,；; ") + "..."
        data["summary"] = s

    def _cap_texts(values: Any, limit: int) -> list[str]:
        items = _normalize_text_items(values)
        result: list[str] = []
        for item in items:
            text = re.sub(r"\s+", " ", item).strip()
            # More aggressive cap: 100 chars per bullet to fit 3 pages
            if len(text) > 100:
                text = text[:97].rstrip("，,；; ") + "..."
            if text and text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return result

    # Experience: cap to 1-2 bullets max, merge from all sources
    for exp in data.get("experience", []) if isinstance(data.get("experience"), list) else []:
        if not isinstance(exp, dict):
            continue
        merged = _collect_experience_bullets(exp)
        exp["bullets"] = _cap_texts(merged, 2)
        exp["responsibilities"] = _cap_texts(exp.get("responsibilities"), 1)
        exp["achievements"] = _cap_texts(exp.get("achievements"), 1)
        for proj in exp.get("projects", []) if isinstance(exp.get("projects"), list) else []:
            if isinstance(proj, dict):
                proj["bullets"] = _cap_texts(_collect_project_bullets(proj), 3)

    # Projects: cap to 3 bullets each
    for proj in data.get("projects", []) if isinstance(data.get("projects"), list) else []:
        if isinstance(proj, dict):
            proj["bullets"] = _cap_texts(_collect_project_bullets(proj), 3)

    skills = data.get("skills")
    if isinstance(skills, dict):
        for key, values in list(skills.items()):
            if isinstance(values, list):
                skills[key] = [str(item).strip() for item in values if str(item).strip()][:8]

    # Publications/honors: cap to 4 entries to save space
    for key in ("publications", "honors", "awards", "certifications", "personal_skills"):
        value = data.get(key)
        if isinstance(value, list):
            data[key] = [item for item in value if str(item).strip()][:4]
    return data


def _collect_profile_items(meta: Any) -> list[tuple[str, str]]:
    if not isinstance(meta, dict):
        return []
    mapping = [
        ("年龄", "age"),
        ("性别", "gender"),
        ("工作经验", "work_experience"),
        ("学历", "education_level"),
        ("政治面貌", "political_status"),
        ("期望城市", "expected_city"),
        ("目标岗位", "target_role"),
        ("求职意向", "job_intention"),
    ]
    rows: list[tuple[str, str]] = []
    for label, key in mapping:
        value = str(meta.get(key, "")).strip()
        if value:
            rows.append((label, value))
    return rows


def _resolve_avatar_path(meta: Any) -> Optional[Path]:
    if not isinstance(meta, dict):
        return None
    for key in ("avatar_path", "avatar", "photo_path", "portrait_path"):
        raw = str(meta.get(key, "") or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path.is_file():
            return path
    return None


def _avatar_image_uri(meta: Any) -> str:
    avatar_path = _resolve_avatar_path(meta)
    if avatar_path is None:
        return ""
    try:
        return avatar_path.resolve().as_uri()
    except Exception:
        return ""


def _append_docx_avatar(doc: DocxDocument, meta: Any, template: str) -> None:
    avatar_path = _resolve_avatar_path(meta)
    if avatar_path is None:
        return
    width_map = {
        "minimal": Inches(0.66),
        "classic": Inches(0.76),
        "modern": Inches(0.72),
    }
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = para.add_run()
    try:
        run.add_picture(str(avatar_path), width=width_map.get(template, Inches(1.0)))
        para.space_after = Pt(4)
    except Exception as exc:
        logger.warning("Failed to insert avatar into DOCX | path=%s error=%s", avatar_path, exc)


def _append_docx_header_block(
    doc: DocxDocument,
    *,
    meta: Any,
    template: str,
    name: str,
    summary: str = "",
) -> None:
    avatar_path = _resolve_avatar_path(meta)
    has_avatar = avatar_path is not None

    # --- Avatar (centered above name when present) ---
    if avatar_path is not None:
        try:
            p_av = doc.add_paragraph()
            p_av.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_av = p_av.add_run()
            run_av.add_picture(str(avatar_path), width=Inches(0.8))
            p_av.space_after = Pt(4)
        except Exception as exc:
            logger.warning("Failed to insert header avatar into DOCX | path=%s error=%s", avatar_path, exc)

    # --- Name ---
    name_align = WD_ALIGN_PARAGRAPH.LEFT
    name_size = Pt(22)
    name_color = RGBColor(0x0F, 0x34, 0x60)

    if template == "classic":
        name_align = WD_ALIGN_PARAGRAPH.CENTER
        name_size = Pt(22)
        name_color = RGBColor(0x0F, 0x34, 0x60)
    elif template == "modern":
        name_size = Pt(22)
        name_color = RGBColor(0x00, 0x5A, 0x64)

    p_name = doc.add_paragraph()
    p_name.alignment = name_align
    run = p_name.add_run(name)
    run.bold = True
    run.font.size = name_size
    run.font.name = DEFAULT_DOC_FONT
    run.font.color.rgb = name_color
    p_name.space_after = Pt(2)

    # --- Contact line (horizontal) ---
    contacts = [
        v for k in ("email", "phone", "wechat", "github", "linkedin", "website")
        if isinstance(meta, dict) and (v := meta.get(k))
    ]
    if contacts:
        contact_sep = "  |  "
        p_contact = doc.add_paragraph(contact_sep.join(contacts))
        p_contact.alignment = name_align
        p_contact.paragraph_format.space_after = Pt(2)
        for item in p_contact.runs:
            item.font.size = Pt(9.5)
            item.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # --- Profile tags (age/gender/edu/exp in one compact line) ---
    if isinstance(meta, dict):
        profile_bits = []
        for label, key in [
            ("性别", "gender"), ("年龄", "age"),
            ("学历", "education_level"), ("工作经验", "work_experience"),
        ]:
            val = str(meta.get(key, "")).strip()
            if val:
                profile_bits.append(f"{label}: {val}")
        if profile_bits:
            p_profile = doc.add_paragraph("  |  ".join(profile_bits))
            p_profile.alignment = name_align
            p_profile.paragraph_format.space_after = Pt(4)
            for item in p_profile.runs:
                item.font.size = Pt(9.5)
                item.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # --- Target role / intention (if set) ---
    if isinstance(meta, dict):
        target = str(meta.get("target_role", "") or meta.get("job_intention", "")).strip()
        if target:
            p_target = doc.add_paragraph(f"求职意向: {target}")
            p_target.alignment = name_align
            p_target.paragraph_format.space_after = Pt(6)
            for item in p_target.runs:
                item.font.size = Pt(10)
                item.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # --- Summary (if present) ---
    if isinstance(summary, str) and summary.strip():
        p_summary = doc.add_paragraph(summary.strip())
        p_summary.alignment = name_align
        p_summary.paragraph_format.space_after = Pt(8 if template != "classic" else 6)
        for item in p_summary.runs:
            item.font.size = Pt(10)
            item.font.color.rgb = RGBColor(0x44, 0x44, 0x44) if template == "classic" else RGBColor(0x33, 0x33, 0x33)
            if template == "classic":
                item.italic = True


def _collect_projects_from_experience(experience: Any) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    if not isinstance(experience, list):
        return projects

    for exp in experience:
        if not isinstance(exp, dict):
            continue

        company = str(exp.get("company", "")).strip()
        role = str(exp.get("role", "")).strip()
        period = str(exp.get("period", "")).strip()
        affiliation = " | ".join(part for part in [company, role, period] if part)

        exp_projects = exp.get("projects", [])
        if not isinstance(exp_projects, list):
            continue

        for proj in exp_projects:
            if not isinstance(proj, dict):
                continue
            name = str(proj.get("name", "项目")).strip() or "项目"
            bullets = _normalize_text_items(proj.get("bullets"))
            tech_stack = _normalize_text_items(proj.get("tech_stack"))
            for field_name in ("description", "function_description", "result_description"):
                value = str(proj.get(field_name, "")).strip()
                if value and value not in bullets:
                    bullets = [value] + bullets
            projects.append(
                {
                    "name": name,
                    "affiliation": affiliation,
                    "bullets": bullets[:4],
                    "tech_stack": tech_stack,
                }
            )

    return projects


def _collect_projects_from_top_level(resume_data: Any) -> list[dict[str, Any]]:
    if not isinstance(resume_data, dict):
        return []

    projects: list[dict[str, Any]] = []
    candidate_keys = ("projects", "project_experience", "project_experiences")
    for key in candidate_keys:
        records = resume_data.get(key, [])
        if not isinstance(records, list):
            continue
        for proj in records:
            if not isinstance(proj, dict):
                continue
            name = str(proj.get("name", "项目")).strip() or "项目"
            period = str(proj.get("period", "")).strip()
            company = str(proj.get("company", "")).strip()
            role = str(proj.get("role", "")).strip()
            affiliation = " | ".join(part for part in (company, role, period) if part)

            bullets = _normalize_text_items(proj.get("bullets"))
            for field_name in ("description", "function_description", "result_description"):
                value = str(proj.get(field_name, "")).strip()
                if value and value not in bullets:
                    bullets = [value] + bullets
            tech_stack = _normalize_text_items(proj.get("tech_stack"))

            projects.append(
                {
                    "name": name,
                    "affiliation": affiliation,
                    "bullets": bullets[:4],
                    "tech_stack": tech_stack,
                }
            )
    return projects


def _project_signature(project: dict[str, Any]) -> tuple[str, str, str]:
    name = _normalize_project_name(project.get("name", ""))
    affiliation = str(project.get("affiliation", "")).strip().lower()
    bullets = project.get("bullets", [])
    if isinstance(bullets, list):
        parts = [str(item).strip() for item in bullets[:2] if str(item).strip()]
    else:
        parts = []
    bullet_sig = " ".join(parts).lower()
    bullet_sig = re.sub(
        r"((19|20)\d{2}[./-]\d{1,2}\s*(?:[-–—至到~]\s*((19|20)\d{2}[./-]\d{1,2}|至今|present))?)",
        " ",
        bullet_sig,
        flags=re.IGNORECASE,
    )
    bullet_sig = re.sub(r"[-–—|:：()\[\]{}【】]+", " ", bullet_sig)
    bullet_sig = re.sub(r"\s+", " ", bullet_sig).strip()
    return name, affiliation, bullet_sig


def _is_duplicate_project(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name, left_aff, left_sig = _project_signature(left)
    right_name, right_aff, right_sig = _project_signature(right)

    if left_name and right_name and left_name == right_name:
        if not left_sig or not right_sig or left_sig == right_sig:
            return True
        if left_sig and right_sig and (left_sig in right_sig or right_sig in left_sig):
            return True
        if left_aff and right_aff and left_aff == right_aff:
            return True

    if left_sig and right_sig and left_sig == right_sig:
        if not left_name or not right_name:
            return True
        if left_name in right_name or right_name in left_name:
            return True

    return False


def _collect_all_projects(resume_data: Any, experience: Any) -> list[dict[str, Any]]:
    merged = _collect_projects_from_experience(experience) + _collect_projects_from_top_level(resume_data)
    deduped: list[dict[str, Any]] = []
    for proj in merged:
        if not isinstance(proj, dict):
            continue
        if any(_is_duplicate_project(proj, existing) for existing in deduped):
            continue
        deduped.append(proj)
    return deduped


def _collect_all_experience(resume_data: Any) -> list[dict[str, Any]]:
    if not isinstance(resume_data, dict):
        return []

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    candidate_keys = (
        "experience",
        "work_experience",
        "work_experiences",
        "internships",
        "internship_experience",
        "internship",
    )

    for key in candidate_keys:
        records = resume_data.get(key, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            company = str(record.get("company", "")).strip()
            role = str(record.get("role", "")).strip()
            period = str(record.get("period", "")).strip()
            dedupe_key = (company, role, period)
            if company or role or period:
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
            else:
                # All empty fields — still include, but no dedup key
                pass
            merged.append(record)

    return merged


def _build_resume_html(resume_data: dict[str, Any], template: str = "classic") -> str:
    tpl = _normalize_template(template)
    theme = {
        "classic": {
            "accent": "#0f3460",
            "title": "#10243f",
            "text": "#222222",
            "muted": "#5f6368",
            "divider": "#d8dde6",
            "bg": "#ffffff",
        },
        "modern": {
            "accent": "#005a64",
            "title": "#08373d",
            "text": "#1f2a2e",
            "muted": "#58666b",
            "divider": "#d1dde0",
            "bg": "#ffffff",
        },
        "minimal": {
            "accent": "#333333",
            "title": "#202020",
            "text": "#2a2a2a",
            "muted": "#616161",
            "divider": "#dddddd",
            "bg": "#ffffff",
        },
    }[tpl]

    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    summary = resume_data.get("summary", "") if isinstance(resume_data, dict) else ""
    experience = _collect_all_experience(resume_data)
    project_items = _collect_all_projects(resume_data, experience)
    education = resume_data.get("education", []) if isinstance(resume_data, dict) else []
    skills = resume_data.get("skills", {}) if isinstance(resume_data, dict) else {}
    publications = resume_data.get("publications", []) if isinstance(resume_data, dict) else []

    name = _html_escape(meta.get("name", "Candidate")) if isinstance(meta, dict) else "Candidate"
    contacts: list[str] = []
    if isinstance(meta, dict):
        for key in ("email", "phone", "wechat", "github", "linkedin", "website"):
            value = str(meta.get(key) or "").strip()
            if value:
                contacts.append(_html_escape(value))
    contacts_html = " | ".join(contacts)
    avatar_uri = _avatar_image_uri(meta)
    profile_items = _collect_profile_items(meta)

    summary_html = ""
    if isinstance(summary, str) and summary.strip():
        summary_html = f'<p class="summary">{_html_escape(summary.strip())}</p>'

    exp_blocks: list[str] = []
    if isinstance(experience, list):
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            company = _html_escape(exp.get("company", ""))
            role = _html_escape(exp.get("role", ""))
            period = _html_escape(exp.get("period", ""))
            head_parts = [part for part in [company, role, period] if part]
            if not head_parts:
                continue
            exp_bullets = _collect_experience_bullets(exp)
            exp_bullets_html = ""
            if exp_bullets:
                exp_bullets_html = (
                    '<ul class="bullets">'
                    + "".join(f"<li>{_html_escape(item)}</li>" for item in exp_bullets)
                    + "</ul>"
                )
            exp_blocks.append(
                f"""
                <article class="entry">
                  <h3>{" | ".join(head_parts)}</h3>
                  {exp_bullets_html}
                </article>
                """
            )

    project_blocks: list[str] = []
    for proj in project_items:
        proj_name = _html_escape(proj.get("name", "Project"))
        affiliation = _html_escape(proj.get("affiliation", ""))
        bullets_html = ""
        bullets = proj.get("bullets", [])
        if isinstance(bullets, list) and bullets:
            bullet_items = "".join(f"<li>{_html_escape(item)}</li>" for item in bullets if str(item).strip())
            if bullet_items:
                bullets_html = f'<ul class="bullets">{bullet_items}</ul>'
        tech_html = ""
        tech = proj.get("tech_stack", [])
        if isinstance(tech, list) and tech:
            clean_tech = [_html_escape(item) for item in tech if str(item).strip()]
            if clean_tech:
                tech_html = f'<p class="tech"><span>Tech</span>{", ".join(clean_tech)}</p>'

        project_blocks.append(
            f"""
            <article class="entry">
              <h4>{proj_name}</h4>
              {'<p class="project-meta">' + affiliation + '</p>' if affiliation else ''}
              {bullets_html}
              {tech_html}
            </article>
            """
        )

    edu_blocks: list[str] = []
    if isinstance(education, list):
        for edu in education:
            if not isinstance(edu, dict):
                continue
            school = _html_escape(edu.get("school", ""))
            degree = _html_escape(edu.get("degree", ""))
            major = _html_escape(edu.get("major", ""))
            period = _html_escape(edu.get("period", ""))
            head = " | ".join(part for part in [school, f"{degree} {major}".strip(), period] if part)
            if not head:
                continue
            highlights_html = ""
            highlights = edu.get("highlights", [])
            if isinstance(highlights, list) and highlights:
                rows = "".join(f"<li>{_html_escape(item)}</li>" for item in highlights if str(item).strip())
                if rows:
                    highlights_html = f'<ul class="bullets">{rows}</ul>'
            edu_blocks.append(f'<article class="entry"><h3>{head}</h3>{highlights_html}</article>')

    skill_blocks: list[str] = []
    if isinstance(skills, dict):
        labels = {
            "languages": "编程语言",
            "frameworks": "框架工具",
            "tools": "开发工具",
            "domains": "业务领域",
            "natural_languages": "语言能力",
        }
        for key, label in labels.items():
            items = skills.get(key, [])
            if isinstance(items, list) and items:
                tags = "".join(
                    f'<span class="tag">{_html_escape(item)}</span>' for item in items if str(item).strip()
                )
                if tags:
                    skill_blocks.append(
                        f"""
                        <div class="skill-row">
                          <div class="skill-label">{label}</div>
                          <div class="skill-tags">{tags}</div>
                        </div>
                        """
                    )

    pub_blocks: list[str] = []
    if isinstance(publications, list):
        for pub in publications:
            if not _has_publication_content(pub):
                continue
            line = _publication_line(pub)
            if line:
                pub_blocks.append(f"<li>{_html_escape(line)}</li>")

    honor_items = _collect_text_entries(
        resume_data if isinstance(resume_data, dict) else {},
        ("honors", "awards", "certifications"),
    )
    personal_skill_items = _collect_text_entries(
        resume_data if isinstance(resume_data, dict) else {},
        ("personal_skills",),
    )

    section_summary = ""
    if summary_html:
        section_summary = f'<section><h2>个人简介</h2>{summary_html}</section>'
    section_profile = ""
    if profile_items:
        profile_rows = "".join(
            f'<li><span class="meta-key">{_html_escape(k)}：</span>{_html_escape(v)}</li>'
            for k, v in profile_items
        )
        section_profile = f'<section><h2>求职信息</h2><ul class="bullets">{profile_rows}</ul></section>'
    empty_experience = '<p class="empty">暂无经历信息</p>'
    empty_projects = '<p class="empty">暂无项目信息</p>'
    empty_education = '<p class="empty">暂无教育信息</p>'
    section_exp = f'<section><h2>工作/实习经历</h2>{"".join(exp_blocks) or empty_experience}</section>'
    section_projects = f'<section><h2>项目经历</h2>{"".join(project_blocks) or empty_projects}</section>'
    section_edu = f'<section><h2>教育经历</h2>{"".join(edu_blocks) or empty_education}</section>'
    section_skills = ""
    if skill_blocks:
        section_skills = f'<section><h2>专业技能</h2>{"".join(skill_blocks)}</section>'
    section_pubs = ""
    if pub_blocks:
        section_pubs = f'<section><h2>论文成果</h2><ul class="bullets">{"".join(pub_blocks)}</ul></section>'
    section_honors = ""
    if honor_items:
        rows = "".join(f"<li>{_html_escape(item)}</li>" for item in honor_items)
        section_honors = f'<section><h2>荣誉与奖项</h2><ul class="bullets">{rows}</ul></section>'
    section_personal_skills = ""
    if personal_skill_items:
        rows = "".join(f"<li>{_html_escape(item)}</li>" for item in personal_skill_items)
        section_personal_skills = f'<section><h2>个人技能</h2><ul class="bullets">{rows}</ul></section>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    @page {{
      size: A4;
      margin: {PDF_MARGIN_MM};
    }}
    html {{
      background: #ffffff !important;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: #ffffff !important;
      font-family: "Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Microsoft YaHei", "PingFang SC", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: {theme["text"]};
      font-size: 12px;
      line-height: 1.55;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .resume-print {{
      width: 100%;
      max-width: none;
      margin: 0;
      background: #ffffff !important;
      padding: 0;
    }}
    .header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 2px solid {theme["divider"]};
      margin-bottom: 12px;
    }}
    .header-main {{
      flex: 1 1 auto;
      min-width: 0;
    }}
    .name {{
      margin: 0;
      color: {theme["title"]};
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: 0.2px;
      font-weight: 700;
    }}
    .contacts {{
      margin-top: 8px;
      color: {theme["muted"]};
      font-size: 12px;
      overflow-wrap: anywhere;
      word-wrap: break-word;
    }}
    .avatar-wrap {{
      flex: 0 0 auto;
      width: 86px;
      text-align: right;
    }}
    .avatar {{
      width: 82px;
      height: 106px;
      object-fit: cover;
      border-radius: 3px;
      border: 1px solid {theme["divider"]};
      background: #fff;
    }}
    section {{
      margin-top: 10px;
      page-break-inside: auto;
      break-inside: auto;
    }}
    h2 {{
      margin: 0 0 8px;
      color: {theme["accent"]};
      font-size: 15px;
      border-bottom: 1px solid {theme["divider"]};
      padding-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    h3 {{
      margin: 0 0 2px;
      color: {theme["title"]};
      font-size: 13px;
      font-weight: 600;
    }}
    h4 {{
      margin: 6px 0 2px;
      color: {theme["text"]};
      font-size: 12.5px;
      font-weight: 600;
    }}
    .summary {{
      margin: 0;
      color: {theme["muted"]};
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .entry {{
      margin-bottom: 7px;
      page-break-inside: auto;
      break-inside: auto;
    }}
    .bullets {{
      margin: 3px 0 0 16px;
      padding: 0;
    }}
    .bullets li {{
      margin: 2px 0;
      white-space: pre-wrap;
    }}
    .tech {{
      margin: 4px 0 0;
      color: {theme["muted"]};
      font-size: 11px;
    }}
    .project-meta {{
      margin: 0 0 2px;
      color: {theme["muted"]};
      font-size: 11px;
    }}
    .project-list {{
      margin-top: 3px;
    }}
    .project-item {{
      margin-top: 4px;
      padding-left: 10px;
      border-left: 2px solid {theme["divider"]};
      break-inside: auto;
      page-break-inside: auto;
    }}
    .tech span {{
      color: {theme["accent"]};
      font-weight: 600;
      margin-right: 6px;
    }}
    .skill-row {{
      display: block;
      margin-bottom: 5px;
    }}
    .skill-label {{
      display: inline-block;
      width: auto;
      color: {theme["accent"]};
      font-weight: 600;
      font-size: 11.5px;
      margin-right: 6px;
    }}
    .skill-tags {{
      display: inline;
    }}
    .tag {{
      background: transparent;
      border: 0;
      border-radius: 0;
      padding: 0;
      font-size: 11px;
      margin-right: 4px;
    }}
    .empty {{
      margin: 0;
      color: {theme["muted"]};
      font-style: italic;
    }}
    .meta-key {{
      color: {theme["accent"]};
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <main class="resume-print template-{tpl}">
    <header class="header">
      <div class="header-main">
        <h1 class="name">{name}</h1>
        {'<div class="contacts">' + contacts_html + '</div>' if contacts_html else ''}
      </div>
      {'<div class="avatar-wrap"><img class="avatar" src="' + _html_escape(avatar_uri) + '" alt="avatar" /></div>' if avatar_uri else ''}
    </header>
    {section_summary}
    {section_profile}
    {section_exp}
    {section_projects}
    {section_edu}
    {section_skills}
    {section_pubs}
    {section_honors}
    {section_personal_skills}
  </main>
</body>
</html>"""


def _render_pdf_via_weasyprint(resume_data: dict[str, Any], output_path: Path, template: str = "classic") -> bool:
    if WeasyHTML is None:
        return False

    html_content = _build_resume_html(resume_data, template=template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        WeasyHTML(string=html_content, base_url=str(output_path.parent)).write_pdf(str(output_path))
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        import logging
        logging.getLogger("resume_renderer").error("WeasyPrint PDF render failed: %s", exc, exc_info=True)
        return False


def _pdf_has_middle_blank_pages(pdf_path: Path) -> bool:
    """Detect blank pages introduced by unstable pagination in HTML->PDF engines."""
    if fitz is None or not pdf_path.exists():
        return False
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return False

    try:
        if doc.page_count < 3:
            return False
        for page_index in range(1, doc.page_count - 1):
            page = doc.load_page(page_index)
            text = (page.get_text("text") or "").strip()
            drawings = page.get_drawings() or []
            images = page.get_images(full=True) or []
            if len(text) < 12 and not drawings and not images:
                return True
        return False
    finally:
        doc.close()


def _fitz_font_for_text(text: str) -> str:
    if _contains_cjk(text):
        # CJK fallback font name used by PyMuPDF on many platforms.
        return "china-s"
    return "helv"


def _measure_fitz_text(text: str, fontname: str, fontsize: float) -> float:
    if not text:
        return 0.0
    try:
        return float(fitz.get_text_length(text, fontname=fontname, fontsize=fontsize))
    except Exception:
        # Approximate width fallback: CJK ~1.0em, Latin ~0.55em
        width = 0.0
        for ch in text:
            width += fontsize * (1.0 if _contains_cjk(ch) else 0.55)
        return width


def _wrap_fitz_text(text: str, max_width: float, fontname: str, fontsize: float) -> list[str]:
    content = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    wrapped: list[str] = []
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            wrapped.append("")
            continue
        buf = ""
        for ch in line:
            trial = f"{buf}{ch}"
            if buf and _measure_fitz_text(trial, fontname, fontsize) > max_width:
                wrapped.append(buf.rstrip())
                buf = ch.lstrip() if ch == " " else ch
            else:
                buf = trial
        if buf:
            wrapped.append(buf.rstrip())
    return wrapped or [""]


def _add_section_heading(doc: DocxDocument, title: str, template: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(title)

    if template == "minimal":
        run.bold = True
        run.font.size = Pt(12)
        run.font.name = DEFAULT_DOC_FONT
        run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        p.space_before = Pt(10)
        p.space_after = Pt(4)
        return
    if template == "modern":
        run.bold = True
        run.font.size = Pt(11)
        run.font.name = DEFAULT_DOC_FONT
        run.font.color.rgb = RGBColor(0x00, 0x5A, 0x64)
        p.space_before = Pt(12)
        p.space_after = Pt(5)
        return

    run.bold = True
    run.font.size = Pt(13)
    run.font.name = DEFAULT_DOC_FONT
    run.font.color.rgb = RGBColor(0x0F, 0x34, 0x60)
    p.space_before = Pt(12)
    p.space_after = Pt(4)


def _append_docx_profile_section(doc: DocxDocument, meta: Any, template: str) -> None:
    items = _collect_profile_items(meta)
    if not items:
        return
    _add_section_heading(doc, "求职信息", template)
    for key, value in items:
        p = doc.add_paragraph(f"{key}: {value}")
        for run in p.runs:
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def _append_docx_text_section(doc: DocxDocument, title: str, items: list[str], template: str) -> None:
    clean = [str(x).strip() for x in items if str(x).strip()]
    if not clean:
        return
    _add_section_heading(doc, title, template)
    for item in clean:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def _render_docx_minimal(doc: DocxDocument, resume_data: dict[str, Any]) -> None:
    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    name = meta.get("name", "候选人") if isinstance(meta, dict) else "候选人"
    summary = resume_data.get("summary")
    _append_docx_header_block(
        doc,
        meta=meta,
        template="minimal",
        name=str(name),
        summary=summary if isinstance(summary, str) else "",
    )
    experience = _collect_all_experience(resume_data)
    project_items = _collect_all_projects(resume_data, experience)
    if isinstance(experience, list) and experience:
        _add_section_heading(doc, "工作/实习经历", "minimal")
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            head = " | ".join([x for x in [exp.get("company", ""), exp.get("role", ""), exp.get("period", "")] if x])
            if head:
                doc.add_paragraph(head)
            for bullet in _collect_experience_bullets(exp):
                pb = doc.add_paragraph(style="List Bullet")
                pb.add_run(str(bullet))

    if project_items:
        _add_section_heading(doc, "项目经历", "minimal")
        for proj in project_items:
            doc.add_paragraph(f"- {proj.get('name', '项目')}")
            affiliation = str(proj.get("affiliation", "")).strip()
            if affiliation:
                p_aff = doc.add_paragraph(affiliation)
                for run in p_aff.runs:
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                for bullet in bullets:
                    pb = doc.add_paragraph(style="List Bullet")
                    pb.add_run(str(bullet))
            tech = proj.get("tech_stack", [])
            if isinstance(tech, list) and tech:
                doc.add_paragraph("Tech: " + " · ".join(str(x) for x in tech))

    _append_docx_text_section(
        doc,
        "荣誉与奖项",
        _collect_text_entries(resume_data, ("honors", "awards", "certifications")),
        "minimal",
    )
    _append_docx_text_section(
        doc,
        "个人技能",
        _collect_text_entries(resume_data, ("personal_skills",)),
        "minimal",
    )


def _render_docx_classic(doc: DocxDocument, resume_data: dict[str, Any]) -> None:
    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    experience = _collect_all_experience(resume_data)
    project_items = _collect_all_projects(resume_data, experience)
    education = resume_data.get("education", []) if isinstance(resume_data, dict) else []
    skills = resume_data.get("skills", {}) if isinstance(resume_data, dict) else {}
    summary = resume_data.get("summary", "") if isinstance(resume_data, dict) else ""
    publications = resume_data.get("publications", []) if isinstance(resume_data, dict) else []

    name = meta.get("name", "候选人") if isinstance(meta, dict) else "候选人"
    _append_docx_header_block(
        doc,
        meta=meta,
        template="classic",
        name=str(name),
        summary=summary if isinstance(summary, str) else "",
    )
    if isinstance(experience, list) and experience:
        _add_section_heading(doc, "工作/实习经历", "classic")
        for exp in experience:
            if not isinstance(exp, dict):
                continue

            p_company = doc.add_paragraph()
            run = p_company.add_run(exp.get("company", ""))
            run.bold = True
            run.font.size = Pt(11.5)
            run.font.color.rgb = RGBColor(0x0F, 0x34, 0x60)

            run = p_company.add_run(f"  |  {exp.get('role', '')}")
            run.font.size = Pt(10.5)

            run = p_company.add_run(f"    {exp.get('period', '')}")
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            run.italic = True
            p_company.space_before = Pt(6)
            p_company.space_after = Pt(2)

            exp_bullets = _collect_experience_bullets(exp)
            for bullet in exp_bullets:
                p_bullet = doc.add_paragraph(style="List Bullet")
                run = p_bullet.add_run(str(bullet))
                run.font.size = Pt(10)
                p_bullet.space_after = Pt(1)

    if project_items:
        _add_section_heading(doc, "项目经历", "classic")
        for proj in project_items:
            p_proj = doc.add_paragraph()
            run = p_proj.add_run(f"▸ {proj.get('name', '')}")
            run.bold = True
            run.font.size = Pt(10.5)
            p_proj.space_before = Pt(4)
            p_proj.space_after = Pt(1)

            affiliation = str(proj.get("affiliation", "")).strip()
            if affiliation:
                p_aff = doc.add_paragraph(affiliation)
                for aff_run in p_aff.runs:
                    aff_run.font.size = Pt(9.2)
                    aff_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
                    aff_run.italic = True
                p_aff.space_after = Pt(1)

            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                for bullet in bullets:
                    p_bullet = doc.add_paragraph(style="List Bullet")
                    run = p_bullet.add_run(str(bullet))
                    run.font.size = Pt(10)
                    p_bullet.space_after = Pt(1)

            tech = proj.get("tech_stack", [])
            if isinstance(tech, list) and tech:
                p_tech = doc.add_paragraph()
                run = p_tech.add_run("Tech: ")
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
                run = p_tech.add_run("  ·  ".join(str(x) for x in tech))
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
                run.italic = True
                p_tech.space_after = Pt(4)

    if isinstance(education, list) and education:
        _add_section_heading(doc, "教育经历", "classic")
        for edu in education:
            if not isinstance(edu, dict):
                continue

            p_edu = doc.add_paragraph()
            run = p_edu.add_run(edu.get("school", ""))
            run.bold = True
            run.font.size = Pt(11.5)
            run.font.color.rgb = RGBColor(0x0F, 0x34, 0x60)

            degree_major = f"  |  {edu.get('degree', '')} · {edu.get('major', '')}"
            run = p_edu.add_run(degree_major)
            run.font.size = Pt(10.5)

            run = p_edu.add_run(f"    {edu.get('period', '')}")
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            run.italic = True
            p_edu.space_after = Pt(2)

            highlights = edu.get("highlights", [])
            if isinstance(highlights, list) and highlights:
                p_hl = doc.add_paragraph()
                run = p_hl.add_run("  ·  ".join(str(x) for x in highlights))
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
                p_hl.paragraph_format.left_indent = Inches(0.25)
                p_hl.space_after = Pt(4)

    if isinstance(skills, dict) and any(skills.values()):
        _add_section_heading(doc, "专业技能", "classic")
        label_map = {
            "languages": "编程语言",
            "frameworks": "框架工具",
            "tools": "基础设施",
            "domains": "专业领域",
            "natural_languages": "语言能力",
        }

        for key, label in label_map.items():
            items = skills.get(key, [])
            if isinstance(items, list) and items:
                p_skill = doc.add_paragraph()
                run = p_skill.add_run(f"{label}：")
                run.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0x0F, 0x34, 0x60)
                run = p_skill.add_run("  ·  ".join(str(x) for x in items))
                run.font.size = Pt(10)
                p_skill.space_after = Pt(1)
                p_skill.paragraph_format.left_indent = Inches(0.15)

    valid_publications = [pub for pub in publications if _has_publication_content(pub)] if isinstance(publications, list) else []
    if valid_publications:
        _add_section_heading(doc, "论文成果", "classic")
        for pub in valid_publications:
            if not isinstance(pub, dict):
                p_pub = doc.add_paragraph(_publication_line(pub))
                p_pub.space_after = Pt(2)
                p_pub.paragraph_format.left_indent = Inches(0.15)
                continue
            p_pub = doc.add_paragraph()
            venue = pub.get("venue", "")
            title = pub.get("title", "")
            run = p_pub.add_run(f"[{venue}]  ")
            run.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x0F, 0x34, 0x60)
            run = p_pub.add_run(str(title))
            run.font.size = Pt(9.5)
            run.italic = True
            p_pub.space_after = Pt(2)
            p_pub.paragraph_format.left_indent = Inches(0.15)
    _append_docx_text_section(
        doc,
        "荣誉与奖项",
        _collect_text_entries(resume_data, ("honors", "awards", "certifications")),
        "classic",
    )
    _append_docx_text_section(
        doc,
        "个人技能",
        _collect_text_entries(resume_data, ("personal_skills",)),
        "classic",
    )


def _render_docx_modern(doc: DocxDocument, resume_data: dict[str, Any]) -> None:
    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    experience = _collect_all_experience(resume_data)
    project_items = _collect_all_projects(resume_data, experience)
    education = resume_data.get("education", []) if isinstance(resume_data, dict) else []
    skills = resume_data.get("skills", {}) if isinstance(resume_data, dict) else {}
    summary = resume_data.get("summary", "") if isinstance(resume_data, dict) else ""
    publications = resume_data.get("publications", []) if isinstance(resume_data, dict) else []

    name = meta.get("name", "候选人") if isinstance(meta, dict) else "候选人"
    _append_docx_header_block(
        doc,
        meta=meta,
        template="modern",
        name=str(name),
        summary=summary if isinstance(summary, str) else "",
    )
    if isinstance(experience, list) and experience:
        _add_section_heading(doc, "工作/实习经历", "modern")
        for exp in experience:
            if not isinstance(exp, dict):
                continue

            p_head = doc.add_paragraph()
            run = p_head.add_run(exp.get("company", ""))
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(0x00, 0x5A, 0x64)
            role = exp.get("role", "")
            period = exp.get("period", "")
            if role:
                rr = p_head.add_run(f"  |  {role}")
                rr.font.size = Pt(10.5)
            if period:
                pr = p_head.add_run(f"  ·  {period}")
                pr.font.size = Pt(9.5)
                pr.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
            p_head.space_before = Pt(4)
            p_head.space_after = Pt(2)

            exp_bullets = _collect_experience_bullets(exp)
            for bullet in exp_bullets:
                p_bullet = doc.add_paragraph(f"• {bullet}")
                p_bullet.paragraph_format.left_indent = Inches(0.2)
                p_bullet.space_after = Pt(1)
                for br in p_bullet.runs:
                    br.font.size = Pt(10)

    if project_items:
        _add_section_heading(doc, "项目经历", "modern")
        for proj in project_items:
            p_proj = doc.add_paragraph()
            r = p_proj.add_run(f"- {proj.get('name', '')}")
            r.bold = True
            r.font.size = Pt(10.5)
            r.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
            p_proj.space_before = Pt(2)
            p_proj.space_after = Pt(1)

            affiliation = str(proj.get("affiliation", "")).strip()
            if affiliation:
                p_aff = doc.add_paragraph(affiliation)
                p_aff.paragraph_format.left_indent = Inches(0.2)
                p_aff.space_after = Pt(1)
                for ar in p_aff.runs:
                    ar.font.size = Pt(9)
                    ar.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                for bullet in bullets:
                    p_bullet = doc.add_paragraph(f"• {bullet}")
                    p_bullet.paragraph_format.left_indent = Inches(0.2)
                    p_bullet.space_after = Pt(1)
                    for br in p_bullet.runs:
                        br.font.size = Pt(10)

            tech = proj.get("tech_stack", [])
            if isinstance(tech, list) and tech:
                p_tech = doc.add_paragraph("TECH  " + " / ".join(str(x) for x in tech))
                p_tech.paragraph_format.left_indent = Inches(0.2)
                p_tech.space_after = Pt(3)
                for tr in p_tech.runs:
                    tr.font.size = Pt(9)
                    tr.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    if isinstance(education, list) and education:
        _add_section_heading(doc, "教育经历", "modern")
        for edu in education:
            if not isinstance(edu, dict):
                continue
            p_edu = doc.add_paragraph()
            run = p_edu.add_run(edu.get("school", ""))
            run.bold = True
            run.font.size = Pt(10.8)
            run.font.color.rgb = RGBColor(0x00, 0x5A, 0x64)
            dm = f"{edu.get('degree', '')} {edu.get('major', '')}".strip()
            if dm:
                rr = p_edu.add_run(f"  |  {dm}")
                rr.font.size = Pt(10)
            if edu.get("period", ""):
                pr = p_edu.add_run(f"  ·  {edu.get('period', '')}")
                pr.font.size = Pt(9.5)
                pr.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
            p_edu.space_after = Pt(2)

            highlights = edu.get("highlights", [])
            if isinstance(highlights, list) and highlights:
                p_hl = doc.add_paragraph(" / ".join(str(x) for x in highlights))
                p_hl.paragraph_format.left_indent = Inches(0.2)
                p_hl.space_after = Pt(3)
                for hr in p_hl.runs:
                    hr.font.size = Pt(9.5)
                    hr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    if isinstance(skills, dict) and any(skills.values()):
        _add_section_heading(doc, "专业技能", "modern")
        for key, label in {
            "languages": "编程语言",
            "frameworks": "框架工具",
            "tools": "开发工具",
            "domains": "业务领域",
            "natural_languages": "语言能力",
        }.items():
            items = skills.get(key, [])
            if isinstance(items, list) and items:
                p_skill = doc.add_paragraph()
                run = p_skill.add_run(f"{label}: ")
                run.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x00, 0x5A, 0x64)
                rr = p_skill.add_run(" · ".join(str(x) for x in items))
                rr.font.size = Pt(9.8)
                p_skill.space_after = Pt(1)

    valid_publications = [pub for pub in publications if _has_publication_content(pub)] if isinstance(publications, list) else []
    if valid_publications:
        _add_section_heading(doc, "论文成果", "modern")
        for pub in valid_publications:
            if not isinstance(pub, dict):
                p_pub = doc.add_paragraph(_publication_line(pub))
                p_pub.space_after = Pt(2)
                continue
            p_pub = doc.add_paragraph()
            venue = pub.get("venue", "")
            title = pub.get("title", "")
            run = p_pub.add_run(f"{venue}  ")
            run.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x00, 0x5A, 0x64)
            rr = p_pub.add_run(str(title))
            rr.italic = True
            rr.font.size = Pt(9.5)
            p_pub.space_after = Pt(2)
    _append_docx_text_section(
        doc,
        "荣誉与奖项",
        _collect_text_entries(resume_data, ("honors", "awards", "certifications")),
        "modern",
    )
    _append_docx_text_section(
        doc,
        "个人技能",
        _collect_text_entries(resume_data, ("personal_skills",)),
        "modern",
    )


def render_docx(resume_data: dict[str, Any], output_path: Path, template: str = "classic") -> None:
    tpl = _normalize_template(template)
    resume_data = _compress_resume_data_for_docx(resume_data)

    # Custom template: load as source doc and merge resume data
    if "/" in tpl or "\\" in tpl:
        template_path = Path(tpl)
        if template_path.is_file() and template_path.suffix.lower() == ".docx":
            doc = DocxDocument(str(template_path))
            # Only treat as an injection template if it has actual content
            # (tables or placeholders). Style-guide documents with only static
            # text would produce empty output since nothing can be injected.
            _has_tables = len(doc.tables) > 0
            _has_placeholders = False
            for p in doc.paragraphs:
                if re.search(r"[{【].*[}】]|name|phone|email|experience|education|skills", p.text, re.IGNORECASE):
                    _has_placeholders = True
                    break
            if _has_tables or _has_placeholders:
                _apply_resume_data_to_template(doc, resume_data)
                doc.save(str(output_path))
                return
            # Style-guide template: fall through to built-in renderer
            logger.info(
                "Template %s has no tables/placeholders; using built-in renderer instead",
                template_path.name,
            )

    # Built-in templates
    doc = DocxDocument()

    normal_style = doc.styles["Normal"]
    normal_style.font.name = DEFAULT_DOC_FONT
    normal_style.font.size = Pt(10.5)
    normal_style.paragraph_format.space_after = Pt(2)
    normal_style.paragraph_format.line_spacing = 1.2

    section = doc.sections[0]
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    if tpl == "minimal":
        _render_docx_minimal(doc, resume_data)
    elif tpl == "modern":
        _render_docx_modern(doc, resume_data)
    else:
        _render_docx_classic(doc, resume_data)

    doc.save(str(output_path))

    # Estimate page count for logging. The compression in
    # _compress_resume_data_for_docx keeps content tight to fit <= 3 pages.
    # We no longer add a note paragraph (that would make it worse).
    doc_reloaded = DocxDocument(str(output_path))
    estimated = _estimate_docx_pages(doc_reloaded)
    if estimated > 3.0:
        logger.warning("Generated DOCX estimated at %.1f pages (may affect visual score) | path=%s", estimated, output_path)


def _apply_resume_data_to_template(doc: "DocxDocument", resume_data: dict[str, Any]) -> None:
    """Apply resume data to a user-provided template by replacing placeholder runs."""
    _replace_template_placeholders(doc, resume_data)
    # Collect resume sections as a flat list of (heading, items)
    sections = _build_renderable_sections(resume_data)

    for section_heading, items in sections:
        for para in doc.paragraphs:
            full_text = para.text.strip()
            if _template_section_matches(full_text, section_heading):
                # Clear existing content under this heading
                parent = para._element
                parent.getparent().remove(parent)
                # Add new content under this heading
                section_para = doc.add_paragraph()
                run = section_para.add_run(section_heading)
                run.bold = True
                run.font.size = Pt(12)
                section_para.paragraph_format.space_before = Pt(8)
                section_para.paragraph_format.space_after = Pt(4)
                for item in items:
                    item_para = doc.add_paragraph()
                    if isinstance(item, tuple) and len(item) == 2:
                        sub_heading, sub_items = item
                        sub_run = item_para.add_run(sub_heading)
                        sub_run.bold = True
                        sub_run.font.size = Pt(11)
                        for sub_item in sub_items:
                            bullet = doc.add_paragraph(style="List Bullet")
                            bullet.text = sub_item
                            bullet.paragraph_format.space_after = Pt(2)
                            for run in bullet.runs:
                                run.font.size = Pt(10)
                    else:
                        item_para.text = item
                        item_para.paragraph_format.space_after = Pt(2)
                        for run in item_para.runs:
                            run.font.size = Pt(10)

                # Remove any paragraphs after this section (could be stale content)
                break


def _template_section_matches(text: str, section_heading: str) -> bool:
    normalized = re.sub(r"[\s：:]+", "", str(text or ""))
    target = re.sub(r"[\s：:]+", "", str(section_heading or ""))
    aliases = {
        "基本信息": {"个人信息", "基础信息", "联系方式"},
        "个人总结": {"个人简介", "自我评价", "职业总结"},
        "教育背景": {"教育经历", "学历背景"},
        "专业技能": {"技能", "技能清单", "个人技能"},
    }
    candidates = {target, *aliases.get(target, set())}
    return any(candidate and (normalized == candidate or candidate in normalized) for candidate in candidates)


def _replace_template_placeholders(doc: "DocxDocument", resume_data: dict[str, Any]) -> None:
    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    values = {
        "name": meta.get("name", ""),
        "phone": meta.get("phone", ""),
        "email": meta.get("email", ""),
        "age": meta.get("age", ""),
        "gender": meta.get("gender", ""),
        "work_experience": meta.get("work_experience", ""),
        "education_level": meta.get("education_level", ""),
        "target_role": meta.get("target_role", ""),
        "summary": resume_data.get("summary", "") if isinstance(resume_data, dict) else "",
    }

    def _replace_text(text: str) -> str:
        result = text
        for key, value in values.items():
            for token in (f"{{{{{key}}}}}", f"${{{key}}}", f"[[{key}]]"):
                result = result.replace(token, str(value or ""))
        return result

    def _replace_para(para: Any) -> None:
        original = para.text
        replaced = _replace_text(original)
        if replaced == original:
            return
        for run in para.runs:
            run.text = ""
        if para.runs:
            para.runs[0].text = replaced
        else:
            para.add_run(replaced)

    for para in doc.paragraphs:
        _replace_para(para)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    _replace_para(para)


def _build_renderable_sections(resume_data: dict[str, Any]) -> list[tuple[str, list]]:
    """Build sections list for template rendering."""
    meta = resume_data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    sections = []

    # Basic info section
    basic_items = []
    for key, label in [("name", "姓名"), ("phone", "电话"), ("email", "邮箱"),
                        ("wechat", "微信"), ("github", "GitHub"), ("linkedin", "LinkedIn"),
                        ("political_status", "政治面貌")]:
        val = str(meta.get(key, "")).strip()
        if val:
            basic_items.append(val)
    if meta.get("expected_city") or meta.get("target_role"):
        info_parts = []
        if meta.get("expected_city"):
            info_parts.append(f"期望城市: {meta['expected_city']}")
        if meta.get("target_role"):
            info_parts.append(f"目标岗位: {meta['target_role']}")
        if info_parts:
            basic_items.append(" | ".join(info_parts))
    if basic_items:
        sections.append(("基本信息", basic_items))

    # Summary
    summary = str(resume_data.get("summary", "")).strip()
    if summary:
        sections.append(("个人总结", [summary]))

    # Education
    education = resume_data.get("education", [])
    if isinstance(education, list) and education:
        edu_items = []
        for edu in education:
            if not isinstance(edu, dict):
                continue
            parts = []
            if edu.get("school"):
                parts.append(f"{edu['school']}")
            if edu.get("degree") and edu.get("major"):
                parts.append(f"{edu['degree']} {edu['major']}")
            elif edu.get("degree"):
                parts.append(edu["degree"])
            elif edu.get("major"):
                parts.append(edu["major"])
            if edu.get("period"):
                parts.append(edu["period"])
            if parts:
                edu_items.append(" | ".join(parts))
            highlights = edu.get("highlights", [])
            if isinstance(highlights, list):
                edu_items.extend([h for h in highlights if h])
        sections.append(("教育背景", edu_items))

    # Experience
    experience = resume_data.get("experience", [])
    if isinstance(experience, list) and experience:
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            exp_heading = f"{exp.get('company', '')} - {exp.get('role', '')}"
            exp_parts = []
            if exp.get("period"):
                exp_parts.append(exp["period"])
            if exp.get("team"):
                exp_parts.append(exp["team"])
            if exp_parts:
                exp_heading += f" ({' | '.join(exp_parts)})"
            items = []
            bullets = exp.get("bullets", [])
            if isinstance(bullets, list):
                items.extend([b for b in bullets if b])
            resp = exp.get("responsibilities", [])
            if isinstance(resp, list):
                items.extend([r for r in resp if r])
            ach = exp.get("achievements", [])
            if isinstance(ach, list):
                items.extend([a for a in ach if a])
            sections.append((exp_heading, items))

    # Projects
    projects = resume_data.get("projects", [])
    if isinstance(projects, list) and projects:
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            heading = proj.get("name", "")
            if not heading:
                continue
            parts = []
            if proj.get("period"):
                parts.append(proj["period"])
            if proj.get("description"):
                parts.append(proj["description"])
            if parts:
                heading += f" ({' | '.join(parts)})"
            items = []
            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                items.extend([b for b in bullets if b])
            tech = proj.get("tech_stack", [])
            if isinstance(tech, list) and tech:
                items.append("技术栈: " + " · ".join(str(t) for t in tech))
            sections.append((heading, items))

    # Skills
    skills = resume_data.get("skills", {})
    if isinstance(skills, dict) and any(skills.values()):
        skill_items = []
        for key, label in [("languages", "编程语言"), ("frameworks", "框架"),
                           ("tools", "工具"), ("domains", "领域"),
                           ("natural_languages", "语言能力")]:
            items = skills.get(key, [])
            if isinstance(items, list) and items:
                skill_items.append(f"{label}: " + " · ".join(str(x) for x in items))
        if skill_items:
            sections.append(("专业技能", skill_items))

    # Honors
    honors = []
    for key in ("honors", "awards", "certifications"):
        items = resume_data.get(key, [])
        if isinstance(items, list):
            honors.extend([h for h in items if h])
    if honors:
        sections.append(("荣誉与奖项", honors))

    # Personal skills
    personal_skills = resume_data.get("personal_skills", [])
    if isinstance(personal_skills, list):
        personal_skills = [s for s in personal_skills if s]
    if personal_skills:
        sections.append(("个人技能", personal_skills))

    return sections


def _estimate_docx_pages(doc: "DocxDocument") -> float:
    """Estimate the number of pages a DOCX document would have."""
    # Count total characters and paragraphs
    total_chars = 0
    total_paragraphs = len(doc.paragraphs)
    for para in doc.paragraphs:
        total_chars += len(para.text)
        # Tables contribute too
        if para._element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'):
            total_chars += 500  # Approximate for tables

    # Rough estimate: ~3000 chars per page for 10.5pt font, 1.2 line spacing
    # With CJK characters, ~2500 chars per page
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", doc.element.xml))
    chars_per_page = 2500 if has_cjk else 3500
    return max(1.0, total_chars / chars_per_page)


def _truncate_to_pages(doc: "DocxDocument", max_pages: float = 3.0) -> None:
    """Truncate paragraphs to fit within max_pages."""
    # Estimate current pages
    estimated = _estimate_docx_pages(doc)
    if estimated <= max_pages:
        return

    # Calculate how many paragraphs to keep
    total_paragraphs = len(doc.paragraphs)
    keep_ratio = max_pages / estimated
    keep_count = max(1, int(total_paragraphs * keep_ratio))

    # Remove paragraphs from the end
    for _ in range(total_paragraphs - keep_count):
        p = doc.paragraphs[-1]._element
        p.getparent().remove(p)

    # Add a note
    note_para = doc.add_paragraph()
    run = note_para.add_run("（注：为控制在3页以内，部分内容已简化。详见完整版简历。）")
    run.font.size = Pt(9)
    run.italic = True


def _convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> bool:
    libreoffice = shutil.which("libreoffice")
    if not libreoffice:
        return False

    with tempfile.TemporaryDirectory() as temp_out:
        result = subprocess.run(
            [libreoffice, "--headless", "--convert-to", "pdf", "--outdir", temp_out, str(docx_path)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            return False

        generated_pdf = Path(temp_out) / f"{docx_path.stem}.pdf"
        if not generated_pdf.exists():
            return False

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_pdf, pdf_path)
        return True


def _pdf_insert_wrapped_lines(
    *,
    pdf: Any,
    page: Any,
    text: str,
    x: float,
    y: float,
    max_width: float,
    fontsize: float,
    color: tuple[int, int, int],
    line_spacing: float = 1.35,
) -> tuple[Any, float]:
    lines = _wrap_fitz_text(text, max_width=max_width, fontname=_fitz_font_for_text(text), fontsize=fontsize)
    step = max(11.0, fontsize * line_spacing)

    for line in lines:
        if y + step > 800:
            page = pdf.new_page(width=595, height=842)
            y = 42

        draw_text = line if line else " "
        fontname = _fitz_font_for_text(draw_text)
        draw_kwargs = {
            "fontsize": fontsize,
            "color": _rgb01(color),
        }
        try:
            page.insert_text((x, y), draw_text, fontname=fontname, **draw_kwargs)
        except Exception:
            page.insert_text((x, y), draw_text, **draw_kwargs)
        y += step

    return page, y


def _fallback_render_pdf(resume_data: dict[str, Any], output_path: Path, template: str = "classic") -> None:
    if fitz is None:
        raise RuntimeError("PDF export needs LibreOffice or pymupdf")

    tpl = _normalize_template(template)
    accent = PDF_ACCENT_COLORS.get(tpl, PDF_ACCENT_COLORS["classic"])

    left = 42.0
    right = 42.0
    content_width = 595 - left - right

    lines: list[tuple[str, str]] = []
    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    name = meta.get("name", "候选人") if isinstance(meta, dict) else "候选人"
    lines.append(("title", str(name)))

    contacts = [v for k in ("email", "phone", "github", "linkedin", "website") if isinstance(meta, dict) and (v := meta.get(k))]
    if contacts:
        lines.append(("meta", "  |  ".join(str(v) for v in contacts)))

    summary = resume_data.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(("section", "个人简介"))
        lines.append(("body", summary.strip()))

    profile_items = _collect_profile_items(meta)
    if profile_items:
        lines.append(("section", "求职信息"))
        for key, value in profile_items:
            lines.append(("body", f"{key}: {value}"))

    experience = _collect_all_experience(resume_data)
    project_items = _collect_all_projects(resume_data, experience)
    if isinstance(experience, list):
        lines.append(("section", "工作/实习经历"))
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            head = " | ".join([x for x in [exp.get("company", ""), exp.get("role", ""), exp.get("period", "")] if x])
            if head:
                lines.append(("item_head", head))
            for bullet in _collect_experience_bullets(exp):
                lines.append(("bullet", f"• {bullet}"))

    if project_items:
        lines.append(("section", "项目经历"))
        for proj in project_items:
            lines.append(("project", f"- {proj.get('name', '')}"))
            affiliation = str(proj.get("affiliation", "")).strip()
            if affiliation:
                lines.append(("body", affiliation))
            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                for bullet in bullets:
                    lines.append(("bullet", f"• {bullet}"))
            tech = proj.get("tech_stack", [])
            if isinstance(tech, list) and tech:
                lines.append(("body", "Tech: " + " · ".join(str(x) for x in tech)))

    education = resume_data.get("education", []) if isinstance(resume_data, dict) else []
    if isinstance(education, list) and education:
        lines.append(("section", "教育经历"))
        for edu in education:
            if not isinstance(edu, dict):
                continue
            edu_head = " | ".join([x for x in [edu.get("school", ""), f"{edu.get('degree', '')} {edu.get('major', '')}".strip(), edu.get("period", "")] if x])
            if edu_head:
                lines.append(("item_head", edu_head))
            highlights = edu.get("highlights", [])
            if isinstance(highlights, list):
                for item in highlights:
                    text = str(item).strip()
                    if text:
                        lines.append(("bullet", f"• {text}"))

    skills = resume_data.get("skills", {}) if isinstance(resume_data, dict) else {}
    if isinstance(skills, dict) and any(skills.values()):
        lines.append(("section", "专业技能"))
        for key, label in {
            "languages": "编程语言",
            "frameworks": "框架工具",
            "tools": "开发工具",
            "domains": "业务领域",
            "natural_languages": "语言能力",
        }.items():
            items = skills.get(key, [])
            if isinstance(items, list) and items:
                lines.append(("body", f"{label}: " + " · ".join(str(x) for x in items)))

    honor_items = _collect_text_entries(resume_data, ("honors", "awards", "certifications"))
    if honor_items:
        lines.append(("section", "荣誉与奖项"))
        for item in honor_items:
            lines.append(("bullet", f"• {item}"))

    personal_skill_items = _collect_text_entries(resume_data, ("personal_skills",))
    if personal_skill_items:
        lines.append(("section", "个人技能"))
        for item in personal_skill_items:
            lines.append(("bullet", f"• {item}"))

    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)

    avatar_reserved = 0.0
    avatar_path = _resolve_avatar_path(meta)
    if avatar_path is not None:
        try:
            avatar_w = 66.0
            avatar_h = 86.0
            x1 = 595 - right
            x0 = x1 - avatar_w
            y0 = 42.0
            y1 = y0 + avatar_h
            page.insert_image(fitz.Rect(x0, y0, x1, y1), filename=str(avatar_path), keep_proportion=True)
            avatar_reserved = avatar_w + 12.0
        except Exception as exc:
            logger.warning("Fallback PDF failed to draw avatar | path=%s error=%s", avatar_path, exc)

    content_width = max(220.0, 595 - left - right - avatar_reserved)

    y = 42.0
    for kind, text in lines:
        if kind == "title":
            page, y = _pdf_insert_wrapped_lines(
                pdf=pdf,
                page=page,
                text=text,
                x=left,
                y=y,
                max_width=content_width,
                fontsize=20,
                color=accent,
                line_spacing=1.2,
            )
            y += 4
            continue
        if kind == "meta":
            page, y = _pdf_insert_wrapped_lines(
                pdf=pdf,
                page=page,
                text=text,
                x=left,
                y=y,
                max_width=content_width,
                fontsize=9.5,
                color=PDF_MUTED_COLOR,
                line_spacing=1.25,
            )
            y += 4
            continue
        if kind == "section":
            page, y = _pdf_insert_wrapped_lines(
                pdf=pdf,
                page=page,
                text=text,
                x=left,
                y=y,
                max_width=content_width,
                fontsize=11.5,
                color=accent,
                line_spacing=1.2,
            )
            if y + 3 > 800:
                page = pdf.new_page(width=595, height=842)
                y = 42
            page.draw_line((left, y), (left + content_width, y), color=_rgb01(accent), width=0.8)
            y += 7
            continue
        if kind == "item_head":
            page, y = _pdf_insert_wrapped_lines(
                pdf=pdf,
                page=page,
                text=text,
                x=left,
                y=y,
                max_width=content_width,
                fontsize=10.2,
                color=PDF_BODY_COLOR,
                line_spacing=1.25,
            )
            y += 1
            continue
        if kind == "project":
            page, y = _pdf_insert_wrapped_lines(
                pdf=pdf,
                page=page,
                text=text,
                x=left + 6,
                y=y,
                max_width=content_width - 6,
                fontsize=10.0,
                color=PDF_BODY_COLOR,
                line_spacing=1.25,
            )
            continue
        if kind == "bullet":
            page, y = _pdf_insert_wrapped_lines(
                pdf=pdf,
                page=page,
                text=text,
                x=left + 18,
                y=y,
                max_width=content_width - 18,
                fontsize=9.7,
                color=PDF_BODY_COLOR,
                line_spacing=1.3,
            )
            continue
        page, y = _pdf_insert_wrapped_lines(
            pdf=pdf,
            page=page,
            text=text,
            x=left,
            y=y,
            max_width=content_width,
            fontsize=9.8,
            color=PDF_BODY_COLOR,
        )

    pdf.save(str(output_path))
    pdf.close()


def render_pdf(
    resume_data: dict[str, Any],
    output_path: Path,
    template: str = "classic",
    docx_source: Optional[Path] = None,
) -> None:
    tpl = _normalize_template(template)
    _ = docx_source
    if _render_pdf_via_weasyprint(resume_data, output_path, template=tpl):
        if _pdf_has_middle_blank_pages(output_path):
            _fallback_render_pdf(resume_data, output_path, template=tpl)
        return

    if fitz is not None:
        _fallback_render_pdf(resume_data, output_path, template=tpl)
        return

    raise RuntimeError(
        "WeasyPrint HTML-to-PDF failed and pymupdf is unavailable. "
        "Please install Linux system libs: cairo/pango/gdk-pixbuf or install pymupdf."
    )


def export_resume_files(
    resume_data: dict[str, Any],
    output_dir: Path,
    output_format: str = "both",
    template: str = "classic",
    file_prefix: Optional[str] = None,
) -> dict[str, Optional[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = (output_format or "both").lower()
    if fmt not in {"docx", "pdf", "both"}:
        raise ValueError("output_format must be docx/pdf/both")

    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    name = meta.get("name", "候选人") if isinstance(meta, dict) else "候选人"
    prefix = _safe_filename(file_prefix or f"简历_{name}_优化版")

    docx_path: Optional[Path] = None
    pdf_path: Optional[Path] = None

    tpl = _normalize_template(template)

    if fmt in {"docx", "both"}:
        docx_path = output_dir / f"{prefix}.docx"
        render_docx(resume_data, docx_path, template=tpl)

    if fmt in {"pdf", "both"}:
        pdf_path = output_dir / f"{prefix}.pdf"

        temp_docx_for_pdf: Optional[Path] = None
        source_docx = docx_path
        if source_docx is None:
            temp_docx_for_pdf = output_dir / f"{prefix}.__pdf_tmp__.docx"
            render_docx(resume_data, temp_docx_for_pdf, template=tpl)
            source_docx = temp_docx_for_pdf

        try:
            render_pdf(resume_data, pdf_path, template=tpl, docx_source=source_docx)
        finally:
            if temp_docx_for_pdf and temp_docx_for_pdf.exists():
                temp_docx_for_pdf.unlink(missing_ok=True)

    return {
        "docx": str(docx_path) if docx_path else None,
        "pdf": str(pdf_path) if pdf_path else None,
    }

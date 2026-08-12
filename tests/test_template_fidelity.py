from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch
import zipfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

import resume_copilot_pipeline
from resume_generate_api import _MockUpload
from resume_renderer import render_docx
from template_layout import extract_template_style_profile


FIXTURES = Path(__file__).parent.parent / "acceptance_testset" / "files" / "templates"


def _payload() -> dict:
    return {
        "meta": {
            "name": "李明",
            "phone": "13800000000",
            "email": "liming@example.com",
            "target_role": "高级产品经理",
        },
        "summary": "具备五年B端产品经验，负责需求分析、产品规划与跨团队交付。",
        "experience": [{
            "company": "星河科技有限公司",
            "role": "产品经理",
            "period": "2021.03-2025.06",
            "bullets": [
                "访谈业务用户并梳理核心流程，输出需求文档与产品方案。",
                "协同研发、测试推进版本上线，并基于反馈持续迭代。",
            ],
        }],
        "projects": [{
            "name": "企业数据平台",
            "bullets": ["负责指标体系与看板规划，支持运营复盘。"],
            "tech_stack": ["Axure", "SQL"],
        }],
        "education": [{
            "school": "江南大学",
            "degree": "本科",
            "major": "工业工程",
            "period": "2017.09-2021.06",
        }],
        "skills": {"tools": ["Axure", "SQL", "Excel"], "domains": ["B端产品", "数据分析"]},
    }


def _render(payload: dict, output: Path, template: Path) -> Document:
    with patch("resume_renderer.inspect_docx_layout", return_value={"available": False, "issues": []}):
        render_docx(payload, output, template=str(template))
    return Document(output)


def _document_xml(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read("word/document.xml")


def _paragraph_shading(paragraph) -> str:
    shading = paragraph._element.find("./w:pPr/w:shd", {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"})
    return shading.get(qn("w:fill"), "") if shading is not None else ""


def _paragraph_bottom_border(paragraph) -> str:
    border = paragraph._element.find("./w:pPr/w:pBdr/w:bottom", {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"})
    return border.get(qn("w:val"), "") if border is not None else ""


def test_compact_and_modern_docx_templates_produce_distinct_editable_layouts(tmp_path):
    compact_path = tmp_path / "compact.docx"
    modern_path = tmp_path / "modern.docx"
    compact = _render(_payload(), compact_path, FIXTURES / "template_compact.docx")
    modern = _render(_payload(), modern_path, FIXTURES / "template_modern.docx")

    assert _document_xml(compact_path) != _document_xml(modern_path)
    for expected in ("李明", "星河科技有限公司", "企业数据平台", "江南大学", "Axure"):
        assert expected in "\n".join(paragraph.text for paragraph in compact.paragraphs)
        assert expected in "\n".join(paragraph.text for paragraph in modern.paragraphs)

    modern_name = next(paragraph for paragraph in modern.paragraphs if paragraph.text == "李明")
    modern_heading = next(paragraph for paragraph in modern.paragraphs if paragraph.text == "工作/实习经历")
    assert modern_name.alignment == WD_ALIGN_PARAGRAPH.CENTER
    assert _paragraph_bottom_border(modern_heading) == "single"


def test_image_template_drives_accent_alignment_and_heading_bar(tmp_path):
    template = FIXTURES / "template_blue.png"
    profile = extract_template_style_profile(str(template))
    assert profile.source_kind == "image"
    assert profile.name_alignment == "left"
    assert profile.heading_decoration == "bar"
    assert profile.accent_hex != "0F3460"

    output = tmp_path / "blue.docx"
    rendered = _render(_payload(), output, template)
    name = next(paragraph for paragraph in rendered.paragraphs if paragraph.text == "李明")
    heading = next(paragraph for paragraph in rendered.paragraphs if paragraph.text == "工作/实习经历")
    assert name.alignment == WD_ALIGN_PARAGRAPH.LEFT
    assert _paragraph_shading(heading) not in {"", "FFFFFF", "auto"}


def test_pdf_template_is_retained_as_a_minimal_style_source(tmp_path):
    template = FIXTURES / "template_minimal.pdf"
    profile = extract_template_style_profile(str(template))
    assert profile.source_kind == "pdf"
    assert profile.preset == "minimal"
    assert profile.compact is True

    output = tmp_path / "minimal.docx"
    rendered = _render(_payload(), output, template)
    text = "\n".join(paragraph.text for paragraph in rendered.paragraphs)
    assert "李明" in text and "星河科技有限公司" in text and "企业数据平台" in text


def test_framework_mode_uses_uploaded_docx_style(tmp_path):
    payload = {
        "meta": {"target_role": "病理科技师"},
        "framework": {
            "mode": "empty_profile",
            "target_role": "病理科技师",
            "notice": "以下内容均为待填写结构，不代表候选人已有事实。",
            "sections": [
                {"title": "基本信息"},
                {"title": "个人总结"},
                {"title": "教育经历"},
                {"title": "工作/实习经历"},
                {"title": "项目经历"},
                {"title": "专业技能"},
            ],
        },
    }
    output = tmp_path / "framework.docx"
    rendered = _render(payload, output, FIXTURES / "template_modern.docx")
    name = next(paragraph for paragraph in rendered.paragraphs if paragraph.text == "个人简历框架")
    heading = next(paragraph for paragraph in rendered.paragraphs if paragraph.text == "基本信息")
    text = "\n".join(paragraph.text for paragraph in rendered.paragraphs)
    assert name.alignment == WD_ALIGN_PARAGRAPH.CENTER
    assert _paragraph_bottom_border(heading) == "single"
    assert "现代简历模板" not in text and "版式要求" not in text


def test_injected_section_content_clones_run_level_formatting(tmp_path):
    template = tmp_path / "section-template.docx"
    source = Document()
    heading = source.add_paragraph("工作经历")
    heading.runs[0].font.color.rgb = RGBColor(0x1E, 0x40, 0x78)
    sample = source.add_paragraph()
    sample_run = sample.add_run("示例工作内容")
    sample_run.font.name = "Arial"
    sample_run.font.size = Pt(8.5)
    sample_run.font.italic = True
    sample_run.font.color.rgb = RGBColor(0x77, 0x22, 0x22)
    source.save(template)

    output = tmp_path / "injected.docx"
    rendered = _render(_payload(), output, template)
    paragraph = next(
        paragraph for paragraph in rendered.paragraphs
        if paragraph.text == "访谈业务用户并梳理核心流程，输出需求文档与产品方案。"
    )
    run = paragraph.runs[0]
    assert run.font.name == "Arial"
    assert round(run.font.size.pt, 1) == 8.5
    assert run.font.italic is True
    assert str(run.font.color.rgb) == "772222"


def test_unanchored_table_template_does_not_leak_sample_content(tmp_path):
    template = tmp_path / "sample-table.docx"
    source = Document()
    table = source.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "示例姓名"
    table.cell(0, 1).text = "示例电话"
    source.save(template)

    output = tmp_path / "from-table.docx"
    rendered = _render(_payload(), output, template)
    text = "\n".join(paragraph.text for paragraph in rendered.paragraphs)
    assert "示例姓名" not in text and "示例电话" not in text
    assert "李明" in text and "星河科技有限公司" in text


def test_template_upload_keeps_pdf_and_image_for_layout_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(resume_copilot_pipeline, "AVATAR_DIR", tmp_path)

    async def resolve(filename: str, payload: str):
        warnings = []
        upload = _MockUpload(None, filename, content=payload)
        path = await resume_copilot_pipeline._resolve_template_path(upload, warnings)
        return Path(path), warnings

    pdf_path, pdf_warnings = asyncio.run(resolve("style.pdf", "%PDF-style"))
    image_path, image_warnings = asyncio.run(resolve("style.png", "png-style"))
    assert pdf_path.suffix == ".pdf" and pdf_path.read_bytes() == b"%PDF-style"
    assert image_path.suffix == ".png" and image_path.read_bytes() == b"png-style"
    assert pdf_warnings == [] and image_warnings == []

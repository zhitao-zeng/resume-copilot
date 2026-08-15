"""Content-only renderer boundary for V3.

Actual DOCX rendering remains an integration concern.  This boundary ensures
the renderer receives a frozen content model and cannot invent facts.
"""
from .template_renderer import render_text, render_sections

__all__ = ["render_sections", "render_text"]

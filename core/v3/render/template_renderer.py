from __future__ import annotations

from ..contracts import FrozenResume, TemplateAST


def render_sections(frozen: FrozenResume, template: TemplateAST | None = None) -> dict[str, list[str]]:
    """Return editable section content without changing frozen claims."""
    rendered: dict[str, list[str]] = {}
    for section, claims in frozen.sections.items():
        rendered[section] = [claim.text for claim in claims]
    return rendered


def render_text(frozen: FrozenResume, template: TemplateAST | None = None) -> str:
    rendered = render_sections(frozen, template)
    return "\n\n".join(f"{section}\n" + "\n".join(values) for section, values in rendered.items())


__all__ = ["render_sections", "render_text"]

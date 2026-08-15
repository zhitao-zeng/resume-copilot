"""Strict, provenance-first contracts for Resume Evidence Compiler V3.

The models in this module are intentionally domain-neutral.  A source span is
the authority for a candidate fact; layout, a JD and a template can organize
or prioritize content, but cannot create a candidate fact.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SourceKind = Literal["cv", "resume", "query", "jd", "template"]
ExtractionEngine = Literal["native_docx", "native_pdf", "text", "ocr", "ppstructure"]
FactType = Literal[
    "identity", "contact", "organization", "role", "period", "action",
    "method", "deliverable", "result", "skill", "education", "credential",
    "metric", "project", "other",
]
LayoutKind = Literal[
    "document", "page", "column", "region", "heading", "paragraph", "line",
    "span", "table", "row", "cell", "image", "footer", "header", "other",
]


class V3Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SourceAsset(V3Model):
    """One input asset and its extraction characteristics.

    ``text`` is the normalized text used for offset validation.  For binary
    files it may be empty while native/layout adapters provide spans.  It is
    never used as a candidate fact merely because it is present.
    """

    source_id: str = Field(min_length=1)
    source_type: SourceKind
    filename: str = ""
    media_type: str = ""
    text: str = ""
    native: bool = False
    page_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourcePolicy(V3Model):
    """Explicit source eligibility policy; safe defaults are restrictive."""

    candidate_fact_sources: set[SourceKind] = Field(default_factory=lambda: {"cv", "resume", "query"})
    query_intent_tokens: tuple[str, ...] = (
        "帮我", "请", "优化", "生成", "改写", "调整", "希望", "想要",
        "不要编造", "不需要编造", "按照岗位", "匹配岗位",
    )

    def source_can_support_facts(self, source_type: SourceKind) -> bool:
        if source_type in {"jd", "template"}:
            return False
        return source_type in self.candidate_fact_sources

    def query_clause_kind(self, text: str) -> Literal["fact", "intent"]:
        value = str(text or "").strip()
        if not value:
            return "intent"
        return "intent" if any(token in value for token in self.query_intent_tokens) else "fact"


class SourceSpan(V3Model):
    """Half-open, exact source range.  Offsets are Python string offsets."""

    source_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    paragraph_id: str | None = None
    node_id: str | None = None

    @model_validator(mode="after")
    def valid_range(self) -> "SourceSpan":
        if self.char_end <= self.char_start:
            raise ValueError("SourceSpan must be a non-empty half-open range")
        return self

    def quote(self, documents: dict[str, str]) -> str:
        if self.source_id not in documents:
            raise KeyError(self.source_id)
        text = documents[self.source_id]
        if self.char_end > len(text):
            raise ValueError("SourceSpan exceeds source document length")
        return text[self.char_start:self.char_end]


class Anchor(V3Model):
    """An entity anchor used by hard verification (organization, date, etc.)."""

    entity_type: Literal[
        "organization", "role", "period", "metric", "skill", "credential",
        "education", "identity", "contact", "project", "other",
    ]
    text: str = Field(min_length=1)
    span: SourceSpan


class LayoutNode(V3Model):
    """Layout-preserving node from native parser or PP-StructureV3."""

    node_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    kind: LayoutKind = "paragraph"
    text: str = ""
    page: int = Field(default=1, ge=1)
    column_id: str | None = None
    region_id: str | None = None
    parent_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    order: int = Field(default=0, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    label: str = ""
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentGraph(V3Model):
    source_id: str = Field(min_length=1)
    source_type: SourceKind
    extraction_engine: ExtractionEngine = "text"
    source_text: str = ""
    nodes: list[LayoutNode] = Field(default_factory=list)
    root_id: str = "document"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_nodes_and_spans(self) -> "DocumentGraph":
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("DocumentGraph node_id values must be unique")
        for node in self.nodes:
            quotes: list[str] = []
            for span in node.source_spans:
                if span.source_id != self.source_id:
                    raise ValueError("layout node span belongs to another source")
                if span.char_end > len(self.source_text):
                    raise ValueError("layout node span exceeds source document")
                quotes.append(self.source_text[span.char_start:span.char_end])
            if quotes and "".join(quotes) != node.text:
                raise ValueError("layout node spans must compose node.text exactly")
        return self

    def ordered_nodes(self) -> list[LayoutNode]:
        return sorted(self.nodes, key=lambda node: (node.page, node.order, node.node_id))

    def documents(self) -> dict[str, str]:
        return {self.source_id: self.source_text}


class SectionNode(V3Model):
    section_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = ""
    section_type: str = "other"
    node_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    order: int = Field(default=0, ge=0)


class RecordNode(V3Model):
    record_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    section_id: str
    node_ids: list[str] = Field(default_factory=list)
    title: str = ""
    organization: str = ""
    role: str = ""
    period: str = ""
    fact_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FactUnit(V3Model):
    fact_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: SourceKind
    fact_type: FactType = "other"
    text: str = Field(min_length=1)
    spans: list[SourceSpan] = Field(min_length=1)
    section_id: str | None = None
    record_id: str | None = None
    anchors: list[Anchor] = Field(default_factory=list)
    eligible: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    classification: Literal["fact", "intent", "instruction", "ineligible"] = "fact"

    @model_validator(mode="after")
    def source_and_span_match(self) -> "FactUnit":
        if any(span.source_id != self.source_id for span in self.spans):
            raise ValueError("fact spans must belong to fact source")
        if self.source_type in {"jd", "template"} and self.eligible:
            raise ValueError("JD/template facts cannot be eligible")
        if self.classification != "fact" and self.eligible:
            raise ValueError("only classified facts can be eligible")
        return self

    @property
    def source_spans(self) -> list[SourceSpan]:
        """Compatibility spelling used by the existing V2 evidence ledger."""
        return self.spans

    def quotes(self, documents: dict[str, str]) -> list[str]:
        return [span.quote(documents) for span in self.spans]


class FactGraph(V3Model):
    documents: dict[str, str] = Field(default_factory=dict)
    sections: list[SectionNode] = Field(default_factory=list)
    records: list[RecordNode] = Field(default_factory=list)
    facts: list[FactUnit] = Field(default_factory=list)

    @model_validator(mode="after")
    def fact_ids_unique(self) -> "FactGraph":
        section_ids = [section.section_id for section in self.sections]
        record_ids = [record.record_id for record in self.records]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("FactGraph section_id values must be unique")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("FactGraph record_id values must be unique")
        section_map = {section.section_id: section for section in self.sections}
        record_map = {record.record_id: record for record in self.records}
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("FactGraph fact_id values must be unique")
        for fact in self.facts:
            if fact.source_id not in self.documents:
                raise ValueError(f"fact source document is missing: {fact.source_id}")
            quotes = []
            document = self.documents[fact.source_id]
            for span in fact.spans:
                if span.source_id != fact.source_id or span.char_end > len(document):
                    raise ValueError(f"fact span is not inside its source document: {fact.fact_id}")
                quotes.append(document[span.char_start:span.char_end])
            if "".join(quotes) != fact.text:
                raise ValueError(f"fact text does not equal its source span composition: {fact.fact_id}")
            if fact.section_id is not None:
                section = section_map.get(fact.section_id)
                if section is None or section.source_id != fact.source_id:
                    raise ValueError(f"fact section is missing or belongs to another source: {fact.fact_id}")
            if fact.record_id is not None:
                record = record_map.get(fact.record_id)
                if record is None or record.source_id != fact.source_id or record.section_id != fact.section_id:
                    raise ValueError(f"fact record is missing or crosses source/section: {fact.fact_id}")
            for anchor in fact.anchors:
                span = anchor.span
                if span.source_id != fact.source_id or span.char_end > len(document):
                    raise ValueError(f"fact anchor is outside its source: {fact.fact_id}")
                if document[span.char_start:span.char_end] != anchor.text:
                    raise ValueError(f"fact anchor is not an exact source quote: {fact.fact_id}")
                if not any(
                    parent.char_start <= span.char_start and span.char_end <= parent.char_end
                    for parent in fact.spans
                ):
                    raise ValueError(f"fact anchor is outside the fact spans: {fact.fact_id}")
        fact_map = {fact.fact_id: fact for fact in self.facts}
        for record in self.records:
            if record.section_id not in section_map:
                raise ValueError(f"record section is missing: {record.record_id}")
            for fact_id in record.fact_ids:
                fact = fact_map.get(fact_id)
                if fact is None or fact.record_id != record.record_id:
                    raise ValueError(f"record references a missing or foreign fact: {record.record_id}")
        return self

    def eligible_facts(self) -> list[FactUnit]:
        return [fact for fact in self.facts if fact.eligible and fact.classification == "fact"]

    def fact_map(self) -> dict[str, FactUnit]:
        return {fact.fact_id: fact for fact in self.facts}

    @property
    def eligible_fact_ids(self) -> list[str]:
        return [fact.fact_id for fact in self.eligible_facts()]


class JobRequirement(V3Model):
    requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    requirement_type: Literal["skill", "responsibility", "qualification", "other"] = "other"
    priority: int = Field(default=0, ge=0)
    source_span: SourceSpan | None = None


class RequirementGraph(V3Model):
    source_id: str | None = None
    requirements: list[JobRequirement] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class NarrativeGroup(V3Model):
    group_id: str = Field(min_length=1)
    section: str
    source_id: str | None = None
    section_id: str | None = None
    record_id: str | None = None
    fact_ids: list[str] = Field(default_factory=list)
    purpose: Literal["summary", "record", "skills", "education", "contact", "framework", "other"] = "other"
    priority: int = Field(default=0, ge=0)


class CoverageLedger(V3Model):
    eligible_fact_ids: list[str] = Field(default_factory=list)
    planned_fact_ids: list[str] = Field(default_factory=list)
    written_fact_ids: list[str] = Field(default_factory=list)
    omitted_fact_ids: list[str] = Field(default_factory=list)
    omission_reasons: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def disjoint_omitted_written(self) -> "CoverageLedger":
        eligible = set(self.eligible_fact_ids)
        planned = set(self.planned_fact_ids)
        written = set(self.written_fact_ids)
        omitted = set(self.omitted_fact_ids)
        if not planned <= eligible or not written <= eligible or not omitted <= eligible:
            raise ValueError("coverage states must contain only eligible fact ids")
        if not written <= planned or not omitted <= planned:
            raise ValueError("written and omitted facts must be planned")
        if written & omitted:
            raise ValueError("a fact cannot be both written and omitted")
        if any(fact_id not in self.omission_reasons or not self.omission_reasons[fact_id].strip() for fact_id in omitted):
            raise ValueError("every omitted fact requires an omission reason")
        return self


class ResumePlan(V3Model):
    groups: list[NarrativeGroup] = Field(default_factory=list)
    ledger: CoverageLedger
    target_pages: tuple[int, int] = (1, 3)
    skeleton: bool = False
    template_mode: Literal["none", "tagged", "anchored", "style_only"] = "none"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def page_range_valid(self) -> "ResumePlan":
        if self.target_pages[0] < 1 or self.target_pages[1] < self.target_pages[0]:
            raise ValueError("target_pages must be a valid non-empty range")
        return self


class RealizedClaim(V3Model):
    claim_id: str = Field(min_length=1)
    section: str
    text: str = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    record_id: str | None = None
    anchors: list[Anchor] = Field(default_factory=list)
    atomic: bool = True
    generated: bool = False


class RealizerResponse(V3Model):
    """Response protocol for an optional LLM realization call.

    The response cannot introduce fact IDs outside the request.  Semantic and
    hard-anchor checks that need the FactGraph are implemented in
    :func:`v3.realizer.validate_realizer_response`.
    """

    request_fact_ids: list[str] = Field(default_factory=list)
    claims: list[RealizedClaim] = Field(default_factory=list)


class FrozenResume(V3Model):
    sections: dict[str, list[RealizedClaim]] = Field(default_factory=dict)
    claims: list[RealizedClaim] = Field(default_factory=list)
    skeleton: bool = False
    template_mode: Literal["none", "tagged", "anchored", "style_only"] = "none"

    @model_validator(mode="after")
    def claims_match_sections(self) -> "FrozenResume":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("FrozenResume claim_id values must be unique")
        for claim in self.claims:
            if len(claim.fact_ids) != len(set(claim.fact_ids)):
                raise ValueError(f"claim fact_ids are not unique: {claim.claim_id}")
        flattened = [claim for values in self.sections.values() for claim in values]
        if flattened != self.claims:
            raise ValueError("FrozenResume sections and claims content/order disagree")
        return self


class Audit(V3Model):
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    written_fact_ids: list[str] = Field(default_factory=list)
    missing_fact_ids: list[str] = Field(default_factory=list)
    ownership_errors: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    fact_reasons: dict[str, str] = Field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.unsupported_claim_ids and not self.ownership_errors and not self.violations


class TemplateAST(V3Model):
    mode: Literal["tagged", "anchored", "style_only"] = "style_only"
    section_anchors: dict[str, str] = Field(default_factory=dict)
    style_metadata: dict[str, Any] = Field(default_factory=dict)
    sample_text: list[str] = Field(default_factory=list)
    source_id: str | None = None


class V3Output(V3Model):
    """Stable result returned by the shadow orchestrator."""

    graph: FactGraph
    plan: ResumePlan
    frozen: FrozenResume
    audit: Audit
    reply: str


__all__ = [
    "Anchor", "Audit", "CoverageLedger", "DocumentGraph", "FactGraph", "FactUnit",
    "FrozenResume", "JobRequirement", "LayoutNode", "NarrativeGroup", "RealizedClaim", "RealizerResponse",
    "RecordNode", "RequirementGraph", "ResumePlan", "SectionNode", "SourceAsset",
    "SourcePolicy", "SourceSpan", "TemplateAST", "V3Output",
]

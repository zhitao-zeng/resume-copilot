"""Frozen model-facing schemas for Resume Evidence Compiler V3.

These contracts are intentionally generic across professions.  New industries
must be learned as mappings into the existing fact/section/field ontology,
not added as production keyword dictionaries.  Changing a field or enum is a
dataset migration and therefore requires a new ``SCHEMA_VERSION``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, model_validator

from .contracts import FactType, ResumeField, ResumeSection, V3Model


SCHEMA_VERSION = "resume_compiler_v3.4"
SchemaVersion = Literal["resume_compiler_v3.4"]
# Updated only together with an intentional schema-version migration.  The
# export tool and tests recompute the canonical bundle and reject drift.
SCHEMA_FINGERPRINT = "e5292be47ea8d84f43c1f0fc58e0835756a89c648f2552ef29f67906c48fb36f"


class SemanticAtomDecision(V3Model):
    """One exact, non-overlapping substring of a candidate source fact."""

    quote: str = Field(min_length=1)
    fact_type: FactType
    destination_section: ResumeSection
    destination_field: ResumeField


class SemanticContextSpan(V3Model):
    """Exact source text intentionally not emitted as a candidate fact."""

    quote: str = Field(min_length=1)
    reason: Literal["label", "separator", "instruction", "placeholder", "duplicate"]


class SemanticFactDecision(V3Model):
    """Semantic decision for one transport-level fact from the ledger."""

    candidate_fact_id: str = Field(min_length=1)
    classification: Literal["fact", "context", "intent", "instruction", "ambiguous"]
    record_id: str | None
    atoms: list[SemanticAtomDecision]
    context_spans: list[SemanticContextSpan]

    @model_validator(mode="after")
    def factual_decisions_have_atoms(self) -> "SemanticFactDecision":
        if self.classification == "fact" and not self.atoms:
            raise ValueError("a factual decision requires at least one exact atom")
        if self.classification != "fact" and self.atoms:
            raise ValueError("non-factual decisions cannot emit resume atoms")
        if self.classification in {"context", "intent", "instruction"} and not self.context_spans:
            raise ValueError("a non-factual decision requires exact context spans")
        return self


class SemanticCompilationResponse(V3Model):
    schema_version: SchemaVersion
    decisions: list[SemanticFactDecision] = Field(min_length=1)


class RealizerClaimDecision(V3Model):
    """Only fields the model is allowed to choose at realization time."""

    claim_id: str = Field(min_length=1)
    section: ResumeSection
    field: ResumeField
    text: str = Field(min_length=1)
    fact_ids: list[str] = Field(min_length=1)
    record_id: str | None
    group_id: str = Field(min_length=1)


class ConstrainedRealizerResponse(V3Model):
    """Flat claims are the stable realization target used by SFT/DPO data."""

    schema_version: SchemaVersion
    request_fact_ids: list[str] = Field(min_length=1)
    claims: list[RealizerClaimDecision] = Field(min_length=1)


def schema_bundle() -> dict[str, object]:
    return {
        "semantic_compile": SemanticCompilationResponse.model_json_schema(),
        "realize": ConstrainedRealizerResponse.model_json_schema(),
    }


def schema_fingerprint() -> str:
    encoded = json.dumps(
        schema_bundle(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ConstrainedRealizerResponse", "RealizerClaimDecision", "SCHEMA_VERSION", "SchemaVersion",
    "SCHEMA_FINGERPRINT", "SemanticAtomDecision", "SemanticCompilationResponse",
    "SemanticContextSpan", "SemanticFactDecision", "schema_bundle", "schema_fingerprint",
]

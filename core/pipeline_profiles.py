"""Resolved, auditable feature bundles for controlled pipeline ablations.

Profiles intentionally control only output-mutating content stages.  OCR,
request deadlines, logging, schemas, and rendering remain shared so a content
comparison cannot silently change the serving backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().casefold() in _TRUE_VALUES


@dataclass(frozen=True)
class PipelineProfile:
    name: str
    fact_compiler_mode: str
    source_structure_recovery: bool
    record_fact_recovery: bool
    attested_summary_recovery: bool
    query_narrative: bool
    cv_narrative: bool
    record_compiler_recovery: bool = False
    structure_fields_only: bool = False

    def trace_payload(self) -> dict[str, object]:
        return asdict(self)


def resolve_pipeline_profile() -> PipelineProfile:
    """Resolve one named content bundle without changing the public API.

    ``current_control`` is deliberately the default and mirrors the existing
    environment switches.  ``f507_compatible`` is an ablation bundle, not a
    claim of byte-identical reproduction; the exact f507 image stays the
    external control.  ``candidate`` enables the current grounded additions
    explicitly so its changed variables are visible in traces.
    """

    requested = os.getenv("PIPELINE_PROFILE", "current_control").strip().casefold()
    aliases = {
        "current": "current_control",
        "f507": "f507_compatible",
        "reference": "f507_compatible",
    }
    name = aliases.get(requested, requested)
    if name not in {
        "current_control",
        "f507_compatible",
        "ledger_shadow",
        "local_repair",
        "fact_compiler",
        "candidate",
        "quality_v2",
    }:
        name = "current_control"

    configured_compiler = os.getenv("FACT_COMPILER_MODE", "legacy").strip().casefold()
    if configured_compiler not in {"legacy", "shadow", "on"}:
        configured_compiler = "legacy"

    if name == "f507_compatible":
        return PipelineProfile(
            name=name,
            fact_compiler_mode="legacy",
            source_structure_recovery=False,
            record_fact_recovery=False,
            attested_summary_recovery=False,
            query_narrative=False,
            cv_narrative=False,
            record_compiler_recovery=False,
        )
    if name == "ledger_shadow":
        return PipelineProfile(
            name=name,
            fact_compiler_mode="shadow",
            source_structure_recovery=False,
            record_fact_recovery=False,
            attested_summary_recovery=False,
            query_narrative=False,
            cv_narrative=False,
            record_compiler_recovery=False,
        )
    if name == "local_repair":
        return PipelineProfile(
            name=name,
            fact_compiler_mode="legacy",
            source_structure_recovery=True,
            record_fact_recovery=True,
            attested_summary_recovery=True,
            query_narrative=False,
            cv_narrative=False,
            record_compiler_recovery=False,
        )
    if name == "fact_compiler":
        return PipelineProfile(
            name=name,
            fact_compiler_mode="on",
            source_structure_recovery=True,
            record_fact_recovery=True,
            attested_summary_recovery=True,
            query_narrative=False,
            cv_narrative=False,
            record_compiler_recovery=False,
        )
    if name == "candidate":
        return PipelineProfile(
            name=name,
            fact_compiler_mode="on",
            source_structure_recovery=True,
            record_fact_recovery=True,
            attested_summary_recovery=True,
            query_narrative=True,
            cv_narrative=True,
            record_compiler_recovery=False,
        )
    if name == "quality_v2":
        return PipelineProfile(
            name=name,
            # Keep the normal Composer path.  The compiler may recover only
            # uniquely owned record-body facts before the wording optimizer;
            # it cannot merge its deterministic scaffold or own the output.
            fact_compiler_mode="legacy",
            source_structure_recovery=True,
            # The legacy recovery path can infer ownership from body bullets.
            # Quality-v2 uses only the compiler's identity-bound transaction.
            record_fact_recovery=False,
            attested_summary_recovery=False,
            query_narrative=False,
            cv_narrative=False,
            record_compiler_recovery=True,
            structure_fields_only=True,
        )
    return PipelineProfile(
        name=name,
        fact_compiler_mode=configured_compiler,
        source_structure_recovery=True,
        record_fact_recovery=True,
        attested_summary_recovery=True,
        query_narrative=_env_enabled("LLM_NARRATIVE_QUERY_FASTPATH"),
        cv_narrative=_env_enabled("LLM_NARRATIVE_CV_FASTPATH"),
        record_compiler_recovery=False,
    )

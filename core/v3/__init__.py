"""Resume Evidence Compiler V3.

V3 is an isolated, shadow-only evidence compiler.  It is deliberately not
imported by the legacy V2 request path.  The public surface is small enough to
use from experiments while the individual modules remain replaceable.
"""

from .contracts import (
    Anchor,
    Audit,
    CoverageLedger,
    DocumentGraph,
    FactGraph,
    FactUnit,
    FrozenResume,
    JobRequirement,
    LayoutNode,
    NarrativeGroup,
    RealizedClaim,
    RealizerResponse,
    RecordNode,
    RequirementGraph,
    ResumePlan,
    SectionNode,
    SourceAsset,
    SourcePolicy,
    SourceSpan,
    TemplateAST,
    V3Output,
)
from .orchestrator import V3Result, run_v3

__all__ = [
    "Anchor", "Audit", "CoverageLedger", "DocumentGraph", "FactUnit",
    "FactGraph", "FrozenResume", "JobRequirement", "LayoutNode", "NarrativeGroup",
    "RealizedClaim", "RealizerResponse", "RecordNode", "RequirementGraph", "ResumePlan",
    "SectionNode", "SourceAsset", "SourcePolicy", "SourceSpan", "TemplateAST",
    "V3Output", "V3Result", "run_v3",
]

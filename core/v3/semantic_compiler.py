"""Semantic compilation facade for V3."""
from .fact_graph import build_fact_graph, build_document_structure, facts_from_graph, section_type, split_query_clauses

__all__ = ["build_document_structure", "build_fact_graph", "facts_from_graph", "section_type", "split_query_clauses"]

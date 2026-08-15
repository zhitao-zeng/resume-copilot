"""ResumeComposer: LLM Call 1 — extract structured JSON from source material.

V2 Layer 2.  Outputs plain JSON (no evidence wrapping).
Verifier judges factuality.

Source separation: JD (Target Context) is NEVER a candidate fact source.
Only Candidate Evidence (resume + query) can produce experience/education/projects/skills.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

from llm_gateway import (
    ContextBudgetError,
    LLMDeadlineExceeded,
    LLMGateway,
    estimate_chat_tokens,
)
from prompts import RESUME_COMPOSER_SYSTEM_PROMPT, GEN_COMPOSER_SYSTEM_PROMPT
from server_runtime import (
    call_llm_typed,
    llm_enabled,
    remaining_request_seconds,
    reset_request_deadline,
    set_request_deadline,
)
from source_adapter import build_source_bundle, candidate_blocks
from v2_schemas import SourceBlock, SourceBundle, DraftResume, CanonicalResume
from diagnostic_trace import trace_event

logger = logging.getLogger(__name__)

_COMPOSER_MAX_TOKENS = 4096
_COMPOSER_SAFETY_TOKENS = 256
_MIN_TYPED_OUTPUT_TOKENS = 2048
_COMPOSER_MIN_FACT_TOKENS = 1024
_COMPOSER_CONTEXT_CHARS = 1800
_MAX_COMPOSER_FACT_CHARS = 60_000
_MAX_COMPOSER_CHUNKS = 16
_DEFAULT_COMPOSER_MAX_FACT_BLOCKS = 36
_DEFAULT_COMPOSER_MIN_REMAINING_SECONDS = 10
_DEFAULT_COMPOSER_CALL_TIMEOUT_SECONDS = 120
_MAX_COMPOSER_CONCURRENCY = 2


@dataclass(frozen=True)
class ComposeOutcome:
    """Partial Composer result with explicit recovery metadata."""

    draft: DraftResume
    failed_chunks: list[SourceBundle] = field(default_factory=list)
    total_chunks: int = 0
    completed_chunks: int = 0


class _ComposerChunkTimeout(RuntimeError):
    """One Composer chunk exhausted its local budget, not the whole request."""


def _safe_positive_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _llm_context_window() -> int:
    """Return the application-side budget for the configured LLM backend."""

    return (
        _safe_positive_env_int("LLM_CONTEXT_WINDOW")
        or _safe_positive_env_int("MAX_MODEL_LEN")
        or 8192
    )


def _composer_min_remaining_seconds() -> int:
    return (
        _safe_positive_env_int("LLM_COMPOSER_MIN_REMAINING_SECONDS")
        or _DEFAULT_COMPOSER_MIN_REMAINING_SECONDS
    )


def _composer_call_timeout_seconds() -> int:
    """Bound one extraction chunk while reserving time for verification.

    The service-level LLM timeout is intentionally larger because verification
    may need a long completion.  Applying that full timeout to each Composer
    chunk lets one pathological JSON generation consume almost the entire
    480-second request budget even though the remaining chunks can be restored
    deterministically.
    """

    return (
        _safe_positive_env_int("LLM_COMPOSER_CALL_TIMEOUT_SECONDS")
        or _DEFAULT_COMPOSER_CALL_TIMEOUT_SECONDS
    )


def _composer_has_time_budget() -> bool:
    remaining = remaining_request_seconds()
    return remaining is None or remaining >= _composer_min_remaining_seconds()


def _composer_concurrency() -> int:
    """Return the bounded fan-out supported by the 40 GiB production profile."""

    configured = _safe_positive_env_int("LLM_COMPOSER_CONCURRENCY") or 2
    return max(1, min(_MAX_COMPOSER_CONCURRENCY, configured))


def _composer_output_tokens() -> int:
    """Return a configurable completion budget for dense full-CV extraction."""

    return _safe_positive_env_int("LLM_COMPOSER_MAX_TOKENS") or _COMPOSER_MAX_TOKENS


def _composer_max_fact_blocks() -> int:
    """Bound output complexity independently from input token length.

    Resume lines are short, so a request can fit the context window while its
    structured JSON still exceeds the 4096-token completion budget.  Platform
    truncations started with 45-60 factual blocks in one request; 36 preserves
    the largest observed non-truncated boundary and lets the two supported
    model sequences process the first pair in parallel.
    """

    return _safe_positive_env_int("LLM_COMPOSER_MAX_FACT_BLOCKS") or _DEFAULT_COMPOSER_MAX_FACT_BLOCKS


def _build_source_text(source: SourceBundle) -> str:
    """Build source text with strict semantic separation.

    Two sections:
      1. CANDIDATE EVIDENCE — resume text + user query.  This is the
         ONLY source for experience, education, projects, skills.
      2. TARGET CONTEXT — job description.  ONLY for target_role and
         writing style hints.  NOT a source of candidate facts.
    """
    parts = []

    # Section 1: Candidate Evidence (resume + query)
    evidence_parts = []
    resume_texts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in source.blocks if b.source_type == "resume"
    ]
    if resume_texts:
        evidence_parts.append("-- CANDIDATE RESUME --\n" + "\n".join(resume_texts))

    query_texts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in candidate_blocks(source) if b.source_type == "query"
    ]
    if query_texts:
        evidence_parts.append("-- EXPLICIT CANDIDATE FACT ADDITIONS --\n" + "\n".join(query_texts))

    if evidence_parts:
        parts.append("## CANDIDATE EVIDENCE (facts)\n" + "\n\n".join(evidence_parts))

    direction_texts = [
        f"[{b.block_id}] {b.text}"
        for b in source.blocks if b.source_type == "query" and not b.fact_eligible
    ]
    if direction_texts:
        parts.append("## USER DIRECTIONS (instructions only — NOT candidate facts)\n"
                     + "\n".join(direction_texts))

    # Section 2: Target Context (JD only)
    jd_texts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in source.blocks if b.source_type == "jd"
    ]
    if jd_texts:
        parts.append("## TARGET CONTEXT (reference only — NOT candidate facts)\n"
                     + "\n".join(jd_texts))

    return "\n\n".join(parts)


def _build_resume_prompt(source: SourceBundle) -> str:
    source_text = _build_source_text(source)
    return (
        "请从以下材料中完整提取简历信息。注意材料分为两部分：\n"
        "1) CANDIDATE EVIDENCE：候选人的简历原文和用户明确补充，这是事实来源。\n"
        "2) USER DIRECTIONS/TARGET CONTEXT：只做编辑和目标参考，不是候选人事实。\n\n"
        "【材料】\n"
        f"{source_text}"
    )


def _typed_prompt_token_estimate(
    output_model: type,
    system_prompt: str,
    user_prompt: str,
) -> int:
    messages = LLMGateway._build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=output_model.model_json_schema(),
        prefill="{",
    )
    return estimate_chat_tokens(messages)


def _composer_prompt_token_estimate(source: SourceBundle) -> int:
    return _typed_prompt_token_estimate(
        DraftResume,
        RESUME_COMPOSER_SYSTEM_PROMPT,
        _build_resume_prompt(source),
    )


def _build_generate_prompt(query_text: str, jd_text: str) -> str:
    prompt = ""
    if query_text:
        prompt += f"【用户描述】\n{query_text}\n\n"
    if jd_text:
        prompt += f"【目标岗位 JD】\n{jd_text}\n\n"
    return prompt + "请根据以上信息生成简历结构化框架。"


def _prepare_generate_request(query_text: str, jd_text: str) -> tuple[str, int]:
    """Fit a no-CV request while retaining a useful structured output budget."""

    # Never discard candidate text at a fixed character boundary.  Candidate
    # facts use the chunked extraction path in ``compose_from_query``; this
    # helper is primarily for instruction/JD-only framework generation.
    query = query_text.strip()
    jd = jd_text.strip()
    context_window = _llm_context_window()
    while True:
        prompt = _build_generate_prompt(query, jd)
        prompt_tokens = _typed_prompt_token_estimate(
            CanonicalResume,
            GEN_COMPOSER_SYSTEM_PROMPT,
            prompt,
        )
        available = context_window - prompt_tokens - _COMPOSER_SAFETY_TOKENS
        output_tokens = _composer_output_tokens()
        if available >= output_tokens:
            return prompt, output_tokens

        # Candidate facts are more important than target context. Shorten JD
        # first to recover the full output budget. Only lower the completion
        # budget when the candidate query alone still cannot fit 4096 tokens.
        if jd:
            new_length = max(0, int(len(jd) * 0.75) - 1)
            jd = jd[:new_length]
            continue
        if available >= _MIN_TYPED_OUTPUT_TOKENS:
            return prompt, available
        raise ContextBudgetError(
            f"LLM context window {context_window} is too small for no-CV generation without truncating the user input"
        )


def _copy_block_with_text(block: SourceBlock, text: str, suffix: str) -> SourceBlock:
    return block.model_copy(update={
        "block_id": f"{block.block_id}#{suffix}",
        "text": text,
    })


def _natural_prefix_length(text: str, maximum: int) -> int:
    """Prefer a nearby sentence/phrase boundary without dropping characters."""

    if maximum >= len(text):
        return len(text)
    lower_bound = max(1, int(maximum * 0.6))
    best = -1
    for marker in ("。", "！", "？", "；", ";", "，", ",", " "):
        index = text.rfind(marker, lower_bound, maximum)
        if index >= lower_bound:
            best = max(best, index + 1)
    return best if best > 0 else maximum


def _largest_fitting_prefix(
    block: SourceBlock,
    text: str,
    shared_context: list[SourceBlock],
    *,
    prompt_limit: int,
    max_chars: int,
) -> int:
    """Find the largest text prefix that fits a complete typed request."""

    low = 1
    high = min(len(text), max_chars)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = _copy_block_with_text(block, text[:middle], "probe")
        estimate = _composer_prompt_token_estimate(
            SourceBundle(blocks=[candidate, *shared_context])
        )
        if estimate <= prompt_limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return _natural_prefix_length(text, best) if best else 0


def _select_shared_context(
    context: list[SourceBlock],
    *,
    prompt_limit: int,
) -> list[SourceBlock]:
    """Keep ordered target context while reserving room for candidate facts."""

    if not context:
        return []
    empty_prompt_tokens = _composer_prompt_token_estimate(SourceBundle(blocks=[]))
    remaining_prompt_tokens = max(0, prompt_limit - empty_prompt_tokens)
    # On a small backend the fixed typed-schema overhead can already consume
    # most of the prompt budget. Split the remainder between target context and
    # facts instead of dropping the JD entirely. Larger contexts retain the
    # full one-thousand-token factual reserve.
    factual_reserve = min(
        _COMPOSER_MIN_FACT_TOKENS,
        max(128, remaining_prompt_tokens // 2),
    )
    context_prompt_limit = max(empty_prompt_tokens, prompt_limit - factual_reserve)
    selected: list[SourceBlock] = []
    used_chars = 0
    for index, block in enumerate(context):
        remaining_chars = _COMPOSER_CONTEXT_CHARS - used_chars
        if remaining_chars <= 0:
            break
        candidate_text = block.text[:remaining_chars]
        candidate = _copy_block_with_text(block, candidate_text, f"context{index}")
        estimate = _composer_prompt_token_estimate(
            SourceBundle(blocks=[*selected, candidate])
        )
        if estimate <= context_prompt_limit:
            selected.append(candidate)
            used_chars += len(candidate_text)
            if len(candidate_text) < len(block.text):
                break
            continue

        low = 1
        high = len(candidate_text)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            prefix = _copy_block_with_text(block, candidate_text[:middle], f"context{index}")
            estimate = _composer_prompt_token_estimate(
                SourceBundle(blocks=[*selected, prefix])
            )
            if estimate <= context_prompt_limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best:
            cutoff = _natural_prefix_length(candidate_text, best)
            selected.append(_copy_block_with_text(
                block,
                candidate_text[:cutoff],
                f"context{index}",
            ))
        break
    return selected


def _split_source_bundle(
    source: SourceBundle,
    max_fact_chars: int = 6000,
    max_fact_blocks: int | None = None,
) -> list[SourceBundle]:
    """Chunk evidence so each complete typed request fits the LLM context.

    The budget includes the system prompt, JSON schema, output rules, chat
    framing, candidate facts and repeated target context.  Oversized individual
    source lines are split at a nearby phrase boundary, then hard-split only
    when no natural boundary is available.
    """

    context_window = _llm_context_window()
    fact_block_limit = max(1, int(max_fact_blocks or _composer_max_fact_blocks()))
    prompt_limit = context_window - _composer_output_tokens() - _COMPOSER_SAFETY_TOKENS
    if prompt_limit <= 0:
        raise ValueError(
            f"LLM context window {context_window} is too small for Composer output budget"
        )

    factual = candidate_blocks(source)
    factual_ids = {block.block_id for block in factual}
    context = [block for block in source.blocks if block.block_id not in factual_ids]
    shared_context = _select_shared_context(context, prompt_limit=prompt_limit)

    fragments: list[SourceBlock] = []
    for block in factual:
        remaining = block.text
        part_index = 0
        while remaining:
            prefix_length = _largest_fitting_prefix(
                block,
                remaining,
                shared_context,
                prompt_limit=prompt_limit,
                max_chars=max_fact_chars,
            )
            if prefix_length <= 0:
                raise ValueError(
                    "Composer fixed prompt/context leaves no room for candidate evidence"
                )
            piece = remaining[:prefix_length]
            remaining = remaining[prefix_length:]
            fragments.append(_copy_block_with_text(block, piece, f"part{part_index}"))
            part_index += 1

    if not fragments:
        empty_bundle = SourceBundle(blocks=shared_context)
        if _composer_prompt_token_estimate(empty_bundle) > prompt_limit:
            raise ValueError("Composer prompt exceeds configured context window")
        return [empty_bundle]

    chunks: list[list[SourceBlock]] = []
    current: list[SourceBlock] = []
    current_chars = 0
    for fragment in fragments:
        fragment_size = len(fragment.text)
        candidate_facts = [*current, fragment]
        candidate_bundle = SourceBundle(blocks=[*candidate_facts, *shared_context])
        candidate_fits = (
            current_chars + fragment_size <= max_fact_chars
            and len(candidate_facts) <= fact_block_limit
            and _composer_prompt_token_estimate(candidate_bundle) <= prompt_limit
        )
        if current and not candidate_fits:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(fragment)
        current_chars += fragment_size
    if current:
        chunks.append(current)

    result = [
        SourceBundle(blocks=[*chunk, *shared_context])
        for chunk in chunks
    ]
    logger.info(
        "ResumeComposer budgeted %d factual block(s) into %d chunk(s) | context_window=%d | prompt_limit=%d | fact_block_limit=%d",
        len(factual),
        len(result),
        context_window,
        prompt_limit,
        fact_block_limit,
    )
    return result


def _bisect_source_bundle(source: SourceBundle) -> list[SourceBundle]:
    """Split a backend-rejected chunk without dropping candidate evidence."""

    factual = candidate_blocks(source)
    factual_ids = {block.block_id for block in factual}
    context = [block for block in source.blocks if block.block_id not in factual_ids]
    if len(factual) >= 2:
        total_chars = sum(len(block.text) for block in factual)
        target = total_chars / 2
        running = 0
        split_at = 1
        for index, block in enumerate(factual[:-1], start=1):
            running += len(block.text)
            split_at = index
            if running >= target:
                break
        return [
            SourceBundle(blocks=[*factual[:split_at], *context]),
            SourceBundle(blocks=[*factual[split_at:], *context]),
        ]

    if len(factual) == 1 and len(factual[0].text) > 1:
        block = factual[0]
        cutoff = _natural_prefix_length(block.text, max(1, len(block.text) // 2))
        return [
            SourceBundle(blocks=[
                _copy_block_with_text(block, block.text[:cutoff], "retry0"),
                *context,
            ]),
            SourceBundle(blocks=[
                _copy_block_with_text(block, block.text[cutoff:], "retry1"),
                *context,
            ]),
        ]

    # Candidate facts always take precedence over target context. If even a
    # one-character fact cannot fit, remove the repeated context for this last
    # fragment instead of silently discarding the fact itself.
    if factual and context:
        return [SourceBundle(blocks=factual)]
    return []


def _merge_drafts(drafts: list[DraftResume]) -> DraftResume:
    """Merge independently extracted chunks without inventing cross-chunk facts."""

    if not drafts:
        return DraftResume()
    merged = drafts[0].model_copy(deep=True)
    for draft in drafts[1:]:
        for field in ("name", "phone", "email", "target_role", "work_experience"):
            if not getattr(merged.meta, field) and getattr(draft.meta, field):
                setattr(merged.meta, field, getattr(draft.meta, field))
        for section in ("education", "experience", "research", "activities", "projects"):
            getattr(merged, section).extend(getattr(draft, section))
        merged.skills.items.extend(draft.skills.items)
        for section in ("awards", "publications", "patents", "certifications", "training", "teaching"):
            getattr(merged, section).extend(getattr(draft, section))
        for title, items in draft.additional_sections.items():
            merged.additional_sections.setdefault(title, []).extend(items)
        if not merged.summary and draft.summary:
            merged.summary = draft.summary
    return merged


def _draft_has_candidate_content(draft: DraftResume) -> bool:
    """Exclude schema-valid but semantically empty Composer responses."""

    return any((
        draft.meta.name,
        draft.meta.phone,
        draft.meta.email,
        draft.meta.work_experience,
        draft.education,
        draft.experience,
        draft.research,
        draft.activities,
        draft.projects,
        draft.skills.items,
        draft.summary,
        draft.awards,
        draft.publications,
        draft.patents,
        draft.certifications,
        draft.training,
        draft.teaching,
        draft.additional_sections,
    ))


def _compose_chunk(chunk: SourceBundle) -> DraftResume:
    """Run and validate one independent Composer request."""
    user_prompt = _build_resume_prompt(chunk)
    output_tokens = _composer_output_tokens()
    trace_event(
        "composer_request",
        source_blocks=chunk.blocks,
        system_prompt=RESUME_COMPOSER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=output_tokens,
    )
    outer_remaining = remaining_request_seconds()
    chunk_timeout = _composer_call_timeout_seconds()
    chunk_deadline_is_stricter = (
        outer_remaining is None or outer_remaining > chunk_timeout + 0.5
    )
    deadline_token = set_request_deadline(timeout_seconds=chunk_timeout)
    try:
        parsed = call_llm_typed(
            DraftResume,
            RESUME_COMPOSER_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            max_tokens=output_tokens,
        )
    except LLMDeadlineExceeded as exc:
        if chunk_deadline_is_stricter:
            raise _ComposerChunkTimeout(
                f"Composer chunk exceeded its {chunk_timeout}s local budget"
            ) from exc
        raise
    finally:
        reset_request_deadline(deadline_token)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("Composer returned an empty or invalid chunk")
    parsed_draft = DraftResume(**parsed)
    trace_event("composer_response", parsed=parsed_draft)
    if candidate_blocks(chunk) and not _draft_has_candidate_content(parsed_draft):
        raise ValueError("Composer returned no candidate content for a factual chunk")
    return parsed_draft


def compose_resume_with_outcome(source: SourceBundle) -> ComposeOutcome:
    """Extract every recoverable chunk and report incomplete source chunks."""

    if not llm_enabled():
        return ComposeOutcome(
            draft=DraftResume(),
            failed_chunks=[source] if candidate_blocks(source) else [],
        )

    factual_chars = sum(len(block.text) for block in candidate_blocks(source))
    if factual_chars > _MAX_COMPOSER_FACT_CHARS:
        logger.warning(
            "ResumeComposer skipped oversized candidate evidence (%d chars > %d); deterministic fallback will preserve source facts",
            factual_chars,
            _MAX_COMPOSER_FACT_CHARS,
        )
        return ComposeOutcome(draft=DraftResume(), failed_chunks=[source], total_chunks=1)

    chunks = _split_source_bundle(source)
    pending: list[tuple[tuple[int, ...], SourceBundle]] = [
        ((index,), chunk) for index, chunk in enumerate(chunks)
    ]
    if len(pending) > _MAX_COMPOSER_CHUNKS:
        logger.warning(
            "ResumeComposer skipped %d chunks (limit=%d); deterministic fallback will preserve source facts",
            len(pending),
            _MAX_COMPOSER_CHUNKS,
        )
        return ComposeOutcome(
            draft=DraftResume(),
            failed_chunks=list(chunks),
            total_chunks=len(chunks),
        )
    drafts: dict[tuple[int, ...], DraftResume] = {}
    failed_chunks: list[SourceBundle] = []
    worker_count = _composer_concurrency()
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="resume-composer",
    ) as executor:
        while pending:
            if len(drafts) + len(pending) > _MAX_COMPOSER_CHUNKS:
                logger.warning(
                    "ResumeComposer backend required more than %d chunks; retaining completed chunks and recovering the remainder deterministically",
                    _MAX_COMPOSER_CHUNKS,
                )
                failed_chunks.extend(chunk for _, chunk in pending)
                pending.clear()
                break

            wave = pending[:worker_count]
            del pending[:len(wave)]
            futures = []
            budget_exhausted_at: int | None = None
            for wave_index, (key, chunk) in enumerate(wave):
                if not _composer_has_time_budget():
                    budget_exhausted_at = wave_index
                    break
                futures.append((
                    key,
                    chunk,
                    executor.submit(copy_context().run, _compose_chunk, chunk),
                ))
            if budget_exhausted_at is not None:
                logger.warning(
                    "ResumeComposer stopped with source chunks remaining: request deadline budget is too low; retaining completed chunks",
                )
                failed_chunks.extend(chunk for _, chunk in wave[budget_exhausted_at:])
                failed_chunks.extend(chunk for _, chunk in pending)
                pending.clear()
            retries: list[tuple[tuple[int, ...], SourceBundle]] = []
            deadline_reached = False
            for key, chunk, future in futures:
                try:
                    drafts[key] = future.result()
                except ContextBudgetError as exc:
                    smaller_chunks = _bisect_source_bundle(chunk)
                    if not smaller_chunks:
                        logger.warning("ResumeComposer irreducible chunk failed: %s", exc)
                        failed_chunks.append(chunk)
                        continue
                    retries.extend(
                        (key + (child_index,), child)
                        for child_index, child in enumerate(smaller_chunks)
                    )
                    logger.warning(
                        "ResumeComposer backend context rejection; split chunk into %d smaller request(s): %s",
                        len(smaller_chunks),
                        exc,
                    )
                except _ComposerChunkTimeout as exc:
                    logger.warning(
                        "%s; recovering this chunk deterministically and continuing",
                        exc,
                    )
                    failed_chunks.append(chunk)
                except LLMDeadlineExceeded as exc:
                    logger.warning(
                        "ResumeComposer stopped at request deadline: %s; retaining completed chunks",
                        exc,
                    )
                    failed_chunks.append(chunk)
                    deadline_reached = True
                except Exception as exc:
                    logger.warning(
                        "ResumeComposer chunk failed: %s; recovering this chunk deterministically",
                        exc,
                    )
                    failed_chunks.append(chunk)
            if deadline_reached:
                failed_chunks.extend(chunk for _, chunk in retries)
                failed_chunks.extend(chunk for _, chunk in pending)
                pending.clear()
                retries.clear()
            if len(drafts) + len(retries) + len(pending) > _MAX_COMPOSER_CHUNKS:
                logger.warning(
                    "ResumeComposer backend required more than %d chunks; retaining completed chunks and recovering the remainder deterministically",
                    _MAX_COMPOSER_CHUNKS,
                )
                failed_chunks.extend(chunk for _, chunk in retries)
                failed_chunks.extend(chunk for _, chunk in pending)
                pending.clear()
                retries.clear()
            # Retry split children before later source chunks, matching the old
            # lossless queue behavior. Final merge still uses the logical key.
            pending[0:0] = retries

    ordered_drafts = [drafts[key] for key in sorted(drafts)]
    draft = _merge_drafts(ordered_drafts) if ordered_drafts else DraftResume()
    terminal_chunk_count = len(ordered_drafts) + len(failed_chunks)
    logger.info(
        "ResumeComposer extracted %d/%d completed chunk(s); failed=%d",
        len(ordered_drafts),
        terminal_chunk_count,
        len(failed_chunks),
    )
    return ComposeOutcome(
        draft=draft,
        failed_chunks=failed_chunks,
        total_chunks=terminal_chunk_count,
        completed_chunks=len(ordered_drafts),
    )


def compose_resume(source: SourceBundle) -> DraftResume:
    """Compatibility wrapper returning the recoverable partial draft."""

    return compose_resume_with_outcome(source).draft


def compose_from_query(query_text: str, jd_text: str) -> CanonicalResume:
    """Generate structured resume framework from query + JD (no CV).

    Used when the user has no resume file — constructs a framework from their
    written description and optional job description.
    """
    if not llm_enabled():
        return CanonicalResume()
    if not _composer_has_time_budget():
        logger.warning(
            "GenerateComposer skipped: request deadline budget is too low"
        )
        return CanonicalResume()

    # When the query contains candidate facts, use the same lossless chunking
    # and all-or-nothing extraction path as an uploaded CV.  A single generate
    # request used to hard-cut everything after 2,000 characters.
    source = build_source_bundle("", query_text, jd_text)
    if candidate_blocks(source):
        draft = compose_resume(source)
        try:
            return CanonicalResume.model_validate(draft.model_dump())
        except Exception as exc:
            logger.warning("GenerateComposer chunked output validation failed: %s", exc)
            return CanonicalResume()

    try:
        prompt, max_tokens = _prepare_generate_request(query_text, jd_text)
        trace_event(
            "generate_composer_request",
            system_prompt=GEN_COMPOSER_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=max_tokens,
        )
        parsed = call_llm_typed(
            CanonicalResume,
            GEN_COMPOSER_SYSTEM_PROMPT,
            prompt,
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("GenerateComposer LLM call failed: %s", exc)
        return CanonicalResume()

    if not isinstance(parsed, dict):
        return CanonicalResume()

    try:
        result = CanonicalResume(**parsed)
        trace_event("generate_composer_response", parsed=result)
        return result
    except Exception as exc:
        logger.warning("GenerateComposer output validation failed: %s", exc)
        return CanonicalResume()

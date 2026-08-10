"""LLM call gateway with typed/json parsing, thinking stripping, and recovery logic."""

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from json_parsing import parse_json_content

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


class ContextBudgetError(RuntimeError):
    """The backend cannot fit the prompt and a useful completion together."""


class LLMDeadlineExceeded(TimeoutError):
    """The request has no safe time left for another LLM backend call."""


_TOKENIZER_CACHE: dict[str, Any | None] = {}
_TOKENIZER_LOCK = threading.Lock()


def _load_local_tokenizer() -> Any | None:
    """Load a mounted backend tokenizer when one is available locally."""

    candidates = (
        os.getenv("LLM_TOKENIZER_PATH", ""),
        os.getenv("MODELHUB_MODEL_NAME", ""),
        os.getenv("MODEL_PATH", ""),
    )
    for raw_path in candidates:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_dir() or not (path / "tokenizer_config.json").is_file():
            continue
        key = str(path.resolve())
        with _TOKENIZER_LOCK:
            if key in _TOKENIZER_CACHE:
                return _TOKENIZER_CACHE[key]
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    key,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            except Exception:
                tokenizer = None
            _TOKENIZER_CACHE[key] = tokenizer
            return tokenizer
    return None


def estimate_text_tokens(text: str) -> int:
    """Conservatively estimate tokens without requiring a local tokenizer.

    The API container can use a remote LLM and intentionally does not depend on
    ``transformers``. Qwen tokenizes most CJK characters individually, while
    whitespace, dense punctuation and uncommon Unicode can consume more tokens
    than a simple characters-per-token ratio predicts. This fallback therefore
    counts those classes aggressively; the gateway still handles an
    authoritative backend context error as the final safety net.
    """

    ascii_word_chars = 0
    ascii_whitespace = 0
    ascii_punctuation = 0
    cjk_chars = 0
    other_chars = 0
    for char in str(text or ""):
        codepoint = ord(char)
        if codepoint < 128:
            if char.isspace():
                ascii_whitespace += 1
            elif char.isalnum() or char == "_":
                ascii_word_chars += 1
            else:
                ascii_punctuation += 1
        elif (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            cjk_chars += 1
        else:
            other_chars += 1
    return (
        ascii_word_chars
        + ascii_whitespace
        + ascii_punctuation
        + cjk_chars
        + (other_chars * 3)
    )


def _estimate_trusted_system_tokens(text: str) -> int:
    """Estimate static prompt/schema text without over-counting JSON syntax."""

    ascii_chars = 0
    cjk_chars = 0
    other_chars = 0
    for char in str(text or ""):
        codepoint = ord(char)
        if codepoint < 128:
            ascii_chars += 1
        elif (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            cjk_chars += 1
        else:
            other_chars += 1
    return math.ceil(ascii_chars / 3) + cjk_chars + (other_chars * 2)


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate a rendered chat request, including role/template overhead."""

    tokenizer = _load_local_tokenizer()
    if tokenizer is not None:
        try:
            last_is_assistant = bool(messages and messages[-1].get("role") == "assistant")
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=not last_is_assistant,
                continue_final_message=last_is_assistant,
                enable_thinking=False,
            )
            input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if input_ids is not None:
                return len(input_ids)
        except Exception:
            # Tokenizers and chat templates vary across compatible backends.
            # Fall through to the dependency-free conservative estimate.
            pass

    content_tokens = sum(
        (
            _estimate_trusted_system_tokens(message.get("content", ""))
            if message.get("role") == "system"
            else estimate_text_tokens(message.get("content", ""))
        )
        for message in messages
    )
    # Per-message role/separator tokens plus generation markers.  The separate
    # Composer safety margin covers model-specific chat-template variance.
    return content_tokens + (12 * len(messages)) + 64


def strip_thinking(content: str) -> str:
    """Strip thinking/chain-of-thought blocks from Qwen/open-source model output.

    Handles: <think> XML blocks, "Thinking Process:" prefix sections,  response markers.
    """
    if not content:
        return ""
    text = content

    # Pattern 1: XML  response markers
    end_think = re.search(r'</think>', text, re.IGNORECASE)
    if end_think:
        text = text[end_think.end():].lstrip()

    # Pattern 2: "Thinking Process:" blocks — heuristic strip from start
    if re.search(r'^(?:Thinking\s*Process\s*:)', text.strip(), re.IGNORECASE):
        lines = text.split('\n')
        result = []
        in_think = True
        real_content_markers = (
            '1.  **State the Score**', 'Here is', 'Based on', 'Your resume',
            '已按', '整体得分', '简历评分', '您好', '您的简历',
        )
        for line in lines:
            s = line.strip()
            if in_think:
                if any(s.startswith(m) for m in real_content_markers):
                    in_think = False
                    result.append(line)
                elif s and not any(
                    s.startswith(p) for p in (
                        'Thinking', '**Role:**', '**Task:**', '**Input', '**Constraints:',
                        '**Rules:**', '**Output:**', '**Analyze', '**Drafting',
                        '*   **Role:**', '*   **Task:**', '*   **Input',
                        '1.  **Analyze', '2.  **Analyze', '3.  **Analyze',
                        '4.  **Analyze', '5.  **Analyze', '6.  **Analyze',
                        'The input text', 'Let me', 'I need', 'I\'ll',
                        'Wait,', 'Looking at', 'First,', 'Next,', 'Then,',
                        'Now,', 'Finally,', 'Wait.',
                    )
                ):
                    if len(s) > 20 and not s.startswith('*') and not s.startswith('-'):
                        in_think = False
                        result.append(line)
            else:
                result.append(line)
        text = '\n'.join(result).strip()

    # Pattern 3: Clean residual markers and blank lines
    text = re.sub(r'(?m)^\s*(?:\d+\.\s*)?\*\*(?:Analyze|Drafting|Role|Task|Input|Constraints|Rules)\*\*:.*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


class LLMGateway:
    def __init__(
        self,
        *,
        client_factory: Callable[[], OpenAI],
        model_name: str,
        logger: logging.Logger,
        enable_json_repair: bool,
        dump_failure_payload: Callable[[str, str, str], None],
        call_timeout_seconds: Optional[float] = None,
        request_time_remaining: Optional[Callable[[], Optional[float]]] = None,
        deadline_reserve_seconds: float = 1.0,
        retry_min_remaining_seconds: float = 10.0,
    ) -> None:
        self._client_factory = client_factory
        self._model_name = model_name
        self._logger = logger
        self._enable_json_repair = enable_json_repair
        self._dump_failure_payload = dump_failure_payload
        self._call_timeout_seconds = (
            max(0.001, float(call_timeout_seconds))
            if call_timeout_seconds is not None
            else None
        )
        self._request_time_remaining = request_time_remaining
        self._deadline_reserve_seconds = max(0.0, float(deadline_reserve_seconds))
        self._retry_min_remaining_seconds = max(
            self._deadline_reserve_seconds,
            float(retry_min_remaining_seconds),
        )
        self._circuit_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _ensure_circuit_closed(self) -> None:
        with self._circuit_lock:
            if time.monotonic() < self._circuit_open_until:
                raise RuntimeError("LLM backend circuit is temporarily open")

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 2:
                self._circuit_open_until = time.monotonic() + 30.0

    def _remaining_seconds(self) -> Optional[float]:
        if self._request_time_remaining is None:
            return None
        remaining = self._request_time_remaining()
        if remaining is None:
            return None
        return max(0.0, float(remaining))

    def _effective_request_timeout(self, *, is_retry: bool = False) -> Optional[float]:
        """Cap a backend call by both its own timeout and the request deadline."""

        remaining = self._remaining_seconds()
        if remaining is not None:
            if is_retry and remaining < self._retry_min_remaining_seconds:
                raise LLMDeadlineExceeded(
                    f"only {remaining:.2f}s remain; skipping another LLM attempt"
                )
            usable = remaining - self._deadline_reserve_seconds
            if usable <= 0:
                raise LLMDeadlineExceeded("request deadline reached before LLM call")
            if self._call_timeout_seconds is None:
                return max(0.001, usable)
            return max(0.001, min(self._call_timeout_seconds, usable))
        return self._call_timeout_seconds

    def _can_start_repair(self) -> bool:
        remaining = self._remaining_seconds()
        return remaining is None or remaining >= self._retry_min_remaining_seconds

    @staticmethod
    def _is_timeout_exception(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        return type(exc).__name__ in {
            "APITimeoutError",
            "ConnectTimeout",
            "ReadTimeout",
            "TimeoutException",
        }

    def _is_deadline_limited_timeout(self, request_timeout: Optional[float]) -> bool:
        if request_timeout is None or self._remaining_seconds() is None:
            return False
        if self._call_timeout_seconds is None:
            return True
        return request_timeout < self._call_timeout_seconds - 0.001

    @staticmethod
    def _is_json_mode_unsupported_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        if not text:
            return False
        mentions_json_mode = "response_format" in text or "json_object" in text
        unsupported_hint = (
            "not support" in text
            or "unsupported" in text
            or "unknown field" in text
            or "invalid" in text
            or "unrecognized" in text
        )
        return mentions_json_mode and unsupported_hint

    @staticmethod
    def _context_retry_budget(
        exc: Exception,
        current_max_tokens: int,
        *,
        safety_margin: int = 128,
        minimum_max_tokens: int = 256,
    ) -> Optional[int]:
        """Return a safe completion budget parsed from a backend 400 error."""

        text = str(exc or "")
        if not text:
            return None
        context_match = re.search(
            r"maximum context length is\s*(\d+)\s*tokens?",
            text,
            re.IGNORECASE,
        )
        input_match = re.search(
            r"prompt contains(?: at least)?\s*(\d+)\s*(?:input )?tokens?",
            text,
            re.IGNORECASE,
        )
        if input_match is None:
            input_match = re.search(
                r"(\d+)\s*(?:tokens?\s+)?in (?:the )?messages?",
                text,
                re.IGNORECASE,
            )
        if input_match is None:
            input_match = re.search(
                r"messages?\s+(?:contain|resulted in)\s*(\d+)\s*tokens?",
                text,
                re.IGNORECASE,
            )
        if context_match is None or input_match is None:
            return None
        context_window = int(context_match.group(1))
        input_tokens = int(input_match.group(1))
        # vLLM may stop tokenization as soon as it can prove the request is too
        # large and report that the prompt contains *at least* N tokens.  N is
        # then only a lower bound, so a small exact subtraction can fail over
        # and over. Use a wider margin for that authoritative-but-incomplete
        # count; exact counts retain the smaller margin and more output room.
        lower_bound_only = bool(re.search(
            r"prompt contains\s+at least",
            text,
            re.IGNORECASE,
        ))
        effective_margin = max(safety_margin, 512) if lower_bound_only else safety_margin
        available = context_window - input_tokens - effective_margin
        if available < minimum_max_tokens or available >= current_max_tokens:
            return None
        return available

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return (
            "maximum context length" in text
            or "context_length_exceeded" in text
            or ("context length" in text and "token" in text)
        )

    def _call_raw(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        trace_tag: str = "generic",
        prefill: str = "",
    ) -> str:
        client = self._client_factory()
        self._ensure_circuit_closed()
        started = time.perf_counter()
        json_mode_used = True
        effective_max_tokens = max(1, int(max_tokens))
        minimum_max_tokens = min(
            effective_max_tokens,
            max(256, effective_max_tokens // 2),
        )
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        }
        try:
            response = None
            last_context_error: Optional[Exception] = None
            # Five attempts cover JSON-mode fallback plus convergence when a
            # backend reports only a lower bound for prompt tokens.
            for _attempt in range(5):
                request_kwargs = {
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": effective_max_tokens,
                    "extra_body": extra_body,
                }
                request_timeout = self._effective_request_timeout(is_retry=_attempt > 0)
                deadline_limited_timeout = self._is_deadline_limited_timeout(request_timeout)
                if request_timeout is not None:
                    request_kwargs["timeout"] = request_timeout
                if json_mode_used:
                    request_kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except Exception as exc:
                    if deadline_limited_timeout and self._is_timeout_exception(exc):
                        raise LLMDeadlineExceeded(
                            "LLM call reached this request's remaining deadline budget"
                        ) from exc
                    if json_mode_used and self._is_json_mode_unsupported_error(exc):
                        json_mode_used = False
                        self._logger.warning(
                            "JSON mode unsupported by backend, fallback to plain chat completion | model=%s | tag=%s | err=%s",
                            self._model_name,
                            trace_tag,
                            str(exc),
                        )
                        continue
                    retry_budget = self._context_retry_budget(
                        exc,
                        effective_max_tokens,
                        minimum_max_tokens=minimum_max_tokens,
                    )
                    if retry_budget is not None:
                        last_context_error = exc
                        self._logger.warning(
                            "LLM context budget adjusted by backend response | model=%s | tag=%s | requested=%d | retry=%d",
                            self._model_name,
                            trace_tag,
                            effective_max_tokens,
                            retry_budget,
                        )
                        effective_max_tokens = retry_budget
                        continue
                    if self._is_context_length_error(exc):
                        raise ContextBudgetError(str(exc)) from exc
                    raise
            if response is None:
                if last_context_error is not None:
                    raise ContextBudgetError(str(last_context_error)) from last_context_error
                raise RuntimeError("LLM recoverable retries exhausted")
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if str(finish_reason or "").lower() in {"length", "max_tokens"}:
                raise ContextBudgetError(
                    f"LLM structured completion was truncated at {effective_max_tokens} tokens"
                )
        except (ContextBudgetError, LLMDeadlineExceeded):
            # A deterministic request-size problem is not backend instability
            # Deadline exhaustion is also local to one request. Neither should
            # open the shared circuit breaker for subsequent requests.
            raise
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        elapsed = time.perf_counter() - started
        # messages[-1] is the prefill "{" when using call_typed/call_json;
        # the real user prompt is always at index 1 (messages[1]).
        user_msg = messages[1]["content"] if len(messages) > 1 else ""
        self._logger.info(
            "LLM completion done | tag=%s | model=%s | elapsed=%.2fs | temp=%.2f | max_tokens=%d | requested_max_tokens=%d | user_chars=%d | json_mode=%s",
            trace_tag,
            self._model_name,
            elapsed,
            temperature,
            effective_max_tokens,
            max_tokens,
            len(user_msg or ""),
            "on" if json_mode_used else "off",
        )
        content = response.choices[0].message.content or ""
        # Strip thinking chain blocks from model output
        content = strip_thinking(content)
        if prefill:
            stripped = content.lstrip()
            if not stripped.startswith(prefill):
                content = f"{prefill}{content}"
        return content

    @staticmethod
    def _build_messages(
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[dict[str, Any]] = None,
        prefill: str = "",
        error_feedback: Optional[str] = None,
        bad_output_excerpt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        rules = [
            "【输出规则】",
            "1) 只输出一个 JSON 对象，不要输出任何解释文字",
            "2) 不要输出 markdown 代码块标记（不要```json）",
            "3) 不要在 JSON 前后添加任何额外内容",
            '4) 字符串中的双引号必须转义为 \\"',
            "5) 不要在最后一个元素后加逗号",
        ]
        system_content = f"{system_prompt}\n\n" + "\n".join(rules)
        if output_schema:
            system_content += "\n\n【目标 JSON Schema】\n" + json.dumps(output_schema, ensure_ascii=False)

        user_content = user_prompt
        if error_feedback:
            user_content += (
                "\n\n【上次输出错误】\n"
                f"{error_feedback}\n"
                "请修正后重新输出严格 JSON。"
            )
        if bad_output_excerpt:
            user_content += "\n\n【上次错误输出片段】\n" + bad_output_excerpt[:500]

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return messages

    @staticmethod
    def _validate_typed_json(
        output_model: type[BaseModel], payload: dict[str, Any]
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if not isinstance(payload, dict) or not payload:
            return None, "empty-or-non-dict payload"
        try:
            parsed = output_model.model_validate(payload)
            return parsed.model_dump(), None
        except ValidationError as exc:
            return None, str(exc)

    def call_typed(
        self,
        output_model: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        prefill: str = "{",
    ) -> dict[str, Any]:
        schema_hint = output_model.model_json_schema()
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=schema_hint,
            prefill=prefill,
        )
        content = self._call_raw(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            trace_tag=f"{output_model.__name__}:single",
            prefill=prefill,
        )
        parsed = parse_json_content(content)
        validated, validation_error = self._validate_typed_json(output_model, parsed)
        if validated is not None:
            return validated

        last_error = validation_error or "json-parse-failed"
        self._logger.warning("Typed LLM parse/validate failed for %s: %s", output_model.__name__, last_error)
        self._dump_failure_payload(
            output_model.__name__,
            content,
            last_error,
        )
        if self._enable_json_repair and self._can_start_repair():
            retry_messages = self._build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=schema_hint,
                prefill=prefill,
                error_feedback=last_error,
                bad_output_excerpt=content,
            )
            retry_content = self._call_raw(
                retry_messages,
                temperature=min(temperature, 0.1),
                max_tokens=max_tokens,
                trace_tag=f"{output_model.__name__}:repair",
                prefill=prefill,
            )
            retry_parsed = parse_json_content(retry_content)
            retry_validated, retry_error = self._validate_typed_json(output_model, retry_parsed)
            if retry_validated is not None:
                return retry_validated
            self._logger.warning("Typed LLM repair failed for %s: %s", output_model.__name__, retry_error or "json-parse-failed")
            self._dump_failure_payload(
                f"{output_model.__name__}_repair",
                retry_content,
                retry_error or "json-parse-failed",
            )
        elif self._enable_json_repair:
            self._logger.warning(
                "Typed LLM repair skipped for %s: request deadline budget is too low",
                output_model.__name__,
            )
        return {}

    def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        """Call LLM and return raw text. No JSON mode, no schema injection."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = self._client_factory()
        self._ensure_circuit_closed()
        started = time.perf_counter()
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        effective_max_tokens = max(1, int(max_tokens))
        minimum_max_tokens = min(
            effective_max_tokens,
            max(256, effective_max_tokens // 2),
        )
        try:
            response = None
            last_context_error: Optional[Exception] = None
            for _attempt in range(4):
                # ``_effective_request_timeout`` itself raises when the inherited
                # request deadline is already exhausted.  Initialize the flag
                # before that call so the broad transport-error handler can
                # never mask the real deadline exception with UnboundLocalError.
                deadline_limited_timeout = False
                try:
                    request_kwargs = {
                        "model": self._model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": effective_max_tokens,
                        "extra_body": extra_body,
                    }
                    request_timeout = self._effective_request_timeout(is_retry=_attempt > 0)
                    deadline_limited_timeout = self._is_deadline_limited_timeout(request_timeout)
                    if request_timeout is not None:
                        request_kwargs["timeout"] = request_timeout
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except LLMDeadlineExceeded:
                    raise
                except Exception as exc:
                    if deadline_limited_timeout and self._is_timeout_exception(exc):
                        raise LLMDeadlineExceeded(
                            "LLM call reached this request's remaining deadline budget"
                        ) from exc
                    retry_budget = self._context_retry_budget(
                        exc,
                        effective_max_tokens,
                        minimum_max_tokens=minimum_max_tokens,
                    )
                    if retry_budget is not None:
                        last_context_error = exc
                        previous_max_tokens = effective_max_tokens
                        effective_max_tokens = retry_budget
                        self._logger.warning(
                            "LLM text context budget adjusted | model=%s | requested=%d | retry=%d",
                            self._model_name,
                            previous_max_tokens,
                            retry_budget,
                        )
                        continue
                    if self._is_context_length_error(exc):
                        raise ContextBudgetError(str(exc)) from exc
                    raise
            if response is None:
                if last_context_error is not None:
                    raise ContextBudgetError(str(last_context_error)) from last_context_error
                raise RuntimeError("LLM text recoverable retries exhausted")
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if str(finish_reason or "").lower() in {"length", "max_tokens"}:
                raise ContextBudgetError(
                    f"LLM text completion was truncated at {effective_max_tokens} tokens"
                )
        except (ContextBudgetError, LLMDeadlineExceeded):
            raise
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        elapsed = time.perf_counter() - started
        self._logger.info(
            "LLM text completion done | model=%s | elapsed=%.2fs | temp=%.2f | max_tokens=%d | requested_max_tokens=%d",
            self._model_name,
            elapsed,
            temperature,
            effective_max_tokens,
            max_tokens,
        )
        content = response.choices[0].message.content or ""
        return strip_thinking(content).strip()

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        prefill = "{"
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prefill=prefill,
        )
        content = self._call_raw(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            trace_tag="json_generic:single",
            prefill=prefill,
        )
        parsed = parse_json_content(content)
        if parsed:
            return parsed

        self._logger.warning("LLM JSON parse failed; returning empty object")
        self._dump_failure_payload(
            "json_generic",
            content,
            "generic-json-parse-failed",
        )
        if self._enable_json_repair and self._can_start_repair():
            retry_messages = self._build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prefill=prefill,
                error_feedback="generic-json-parse-failed",
                bad_output_excerpt=content,
            )
            retry_content = self._call_raw(
                retry_messages,
                temperature=min(temperature, 0.1),
                max_tokens=max_tokens,
                trace_tag="json_generic:repair",
                prefill=prefill,
            )
            retry_parsed = parse_json_content(retry_content)
            if retry_parsed:
                return retry_parsed
            self._dump_failure_payload(
                "json_generic_repair",
                retry_content,
                "generic-json-repair-failed",
            )
        elif self._enable_json_repair:
            self._logger.warning(
                "Generic JSON repair skipped: request deadline budget is too low"
            )
        return {}

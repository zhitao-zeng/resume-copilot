"""LLM call gateway with typed/json parsing, thinking stripping, and recovery logic."""

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from json_parsing import parse_json_content

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


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
    ) -> None:
        self._client_factory = client_factory
        self._model_name = model_name
        self._logger = logger
        self._enable_json_repair = enable_json_repair
        self._dump_failure_payload = dump_failure_payload
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
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        }
        try:
            try:
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body=extra_body,
                )
            except Exception as exc:
                if not self._is_json_mode_unsupported_error(exc):
                    raise
                json_mode_used = False
                self._logger.warning(
                    "JSON mode unsupported by backend, fallback to plain chat completion | model=%s | tag=%s | err=%s",
                    self._model_name,
                    trace_tag,
                    str(exc),
                )
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        elapsed = time.perf_counter() - started
        # messages[-1] is the prefill "{" when using call_typed/call_json;
        # the real user prompt is always at index 1 (messages[1]).
        user_msg = messages[1]["content"] if len(messages) > 1 else ""
        self._logger.info(
            "LLM completion done | tag=%s | model=%s | elapsed=%.2fs | temp=%.2f | max_tokens=%d | user_chars=%d | json_mode=%s",
            trace_tag,
            self._model_name,
            elapsed,
            temperature,
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
        if self._enable_json_repair:
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
        try:
            response = client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        elapsed = time.perf_counter() - started
        self._logger.info(
            "LLM text completion done | model=%s | elapsed=%.2fs | temp=%.2f | max_tokens=%d",
            self._model_name,
            elapsed,
            temperature,
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
        if self._enable_json_repair:
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
        return {}

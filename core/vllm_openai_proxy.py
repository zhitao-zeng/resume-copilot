"""Minimal OpenAI-compatible proxy -> vLLM /generate adapter.

Maps OpenAI chat/completions JSON -> vLLM /generate endpoint and vice versa.
Runs on port 8002 inside the same container or standalone.
"""

import json
import logging
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATE_URL = "http://localhost:8000/generate"

app = FastAPI()


def _build_messages_for_generate(user_prompt: str, system_prompt: str) -> dict[str, Any]:
    """Map OpenAI chat messages to the /generate prompt format."""
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"[SYSTEM]\n{system_prompt}\n\n")
    prompt_parts.append(user_prompt)
    return prompt_parts[0] if len(prompt_parts) == 1 else "\n".join(prompt_parts)


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]):
    model = body.get("model", "default")
    messages = body.get("messages", [])
    max_tokens = min(body.get("max_tokens", 4096), 4096)
    temperature = body.get("temperature", 0.2)

    # Extract system and user prompts
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content

    prompt = _build_messages_for_generate(user_prompt, system_prompt)

    payload = {
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(GENERATE_URL, json=payload)
            if resp.status_code != 200:
                return JSONResponse(status_code=502, content={"error": resp.text})
            data = resp.json()

        # vLLM /generate returns {"generated_texts": [...]}
        generated = data.get("generated_texts", [""])
        text = generated[0] if generated else ""

        return JSONResponse(content={
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(text) // 4,
                "total_tokens": (len(prompt) + len(text)) // 4,
            },
        })

    except Exception as exc:
        logger.error("Proxy error: %s", exc)
        return JSONResponse(status_code=502, content={"error": str(exc)})

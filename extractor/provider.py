"""Provider abstraction (Phase 0).

Every model call in the system goes through this one class. Keeping the VLM reader
and the LLM reasoner behind a single OpenAI-compatible interface is what lets us:
  - swap models via .env (Ollama local <-> OpenRouter hosted <-> vLLM),
  - measure tokens + latency uniformly for the cost/latency scorecard.

Do not call model SDKs anywhere else.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

log = logging.getLogger("extractor.provider")


class TruncatedResponse(RuntimeError):
    """The model stopped at max_tokens, so its output cannot be trusted as complete."""


class EmptyResponse(RuntimeError):
    """The provider answered, but with no completion (an upstream error in a 200)."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        # api_key may be a dummy for local Ollama; the SDK still requires a value.
        # The SDK retries 429/5xx with exponential backoff; free tiers (10 RPM) hit 429
        # routinely under batch concurrency, so the default budget of 2 is too small.
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_retries=int(os.environ.get("PROVIDER_MAX_RETRIES", "5")),
            timeout=float(os.environ.get("PROVIDER_TIMEOUT_S", "120")),
        )

    def complete(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Supported by Ollama, OpenRouter, vLLM. Falls back gracefully; we also
            # strip code fences downstream in case a backend ignores it.
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            log.warning("model=%s failed after %.2fs: %s: %s", self.model,
                        time.perf_counter() - start, type(exc).__name__, exc)
            raise
        latency = time.perf_counter() - start

        usage = getattr(resp, "usage", None)
        if not getattr(resp, "choices", None):
            # Seen live on OpenRouter free models: HTTP 200 whose body carries an upstream
            # error and no choices. Indexing it raised a baffling TypeError.
            detail = getattr(resp, "error", None) or getattr(resp, "model_extra", None) or "no detail"
            log.warning("model=%s returned no choices after %.2fs: %s", self.model, latency, detail)
            raise EmptyResponse(f"model={self.model} returned no choices (provider error: {detail})")
        choice = resp.choices[0]
        out = LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
            model=self.model,
            finish_reason=choice.finish_reason or "",
        )
        log.info("model=%s tokens_in=%d tokens_out=%d latency_s=%.2f finish=%s",
                 self.model, out.prompt_tokens, out.completion_tokens, latency, out.finish_reason)
        if out.finish_reason == "length":
            # A truncated JSON object otherwise surfaces as a baffling JSONDecodeError.
            # Thinking models (Gemini 2.5) spend reasoning tokens from the SAME budget.
            raise TruncatedResponse(
                f"model={self.model} hit max_tokens={max_tokens} "
                f"({out.completion_tokens} completion tokens); output is incomplete. "
                f"Raise the *_MAX_TOKENS setting.")
        return out

    # --- content-block helpers -------------------------------------------------

    @staticmethod
    def text_block(text: str) -> dict:
        return {"type": "text", "text": text}

    @staticmethod
    def image_block(image_path: str | Path) -> dict:
        p = Path(image_path)
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        ext = p.suffix.lstrip(".").lower() or "png"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
        return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}

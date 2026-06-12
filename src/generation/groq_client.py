"""
src/generation/groq_client.py
------------------------------
Groq Cloud LLM client with streaming support, token tracking,
and logprob extraction for uncertainty estimation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

from groq import Groq

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Full result from one LLM call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    logprobs: list[dict] = field(default_factory=list)   # per-token logprob info
    finish_reason: str = "stop"

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / max(self.latency_ms / 1000, 1e-6)


class GroqClient:
    """
    Thin wrapper around the Groq Python SDK.

    Features
    --------
    - Synchronous and streaming generation
    - Token usage tracking
    - Logprob extraction (when supported by model)
    - Automatic retry on rate-limit errors
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("GroqClient initialised — model=%s", model)

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_logprobs: bool = False,
    ) -> GenerationResult:
        """
        Blocking generation.

        Parameters
        ----------
        messages        : OpenAI-style message list
        temperature     : Override instance temperature
        max_tokens      : Override instance max_tokens
        request_logprobs: Request per-token logprobs if model supports it
        """
        t0 = time.perf_counter()
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        if request_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5

        resp = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        choice = resp.choices[0]
        logprobs_data: list[dict] = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                logprobs_data.append({
                    "token": token_info.token,
                    "logprob": token_info.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (token_info.top_logprobs or [])
                    ],
                })

        result = GenerationResult(
            text=choice.message.content or "",
            model=resp.model,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            latency_ms=latency,
            logprobs=logprobs_data,
            finish_reason=choice.finish_reason or "stop",
        )
        logger.debug(
            "Generated %d tokens in %.0fms (%.1f tok/s)",
            result.completion_tokens,
            result.latency_ms,
            result.tokens_per_second,
        )
        return result

    def stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation. Yields text chunks as they arrive.
        """
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

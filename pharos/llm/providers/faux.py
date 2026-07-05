"""FauxProvider — controllable mock LLM for tests, benchmarks, and demos.

Features:
- Configurable per-call latency
- Echo / counter / scripted response modes
- Optional error injection
- Records every call for assertions in tests

This is the only provider the runtime itself depends on. Everything
else (OpenAI, Anthropic, ...) sits in the providers/ subpackage.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pharos.llm.base import LLMProvider
from pharos.llm.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    StreamEvent,
    StreamOptions,
    TextContent,
    Usage,
)


@dataclass
class FauxCallRecord:
    """Record of one FauxProvider invocation. Kept for assertions."""

    model_id: str
    context_hash: str
    prompt: str
    response: str
    started_at: float
    ended_at: float
    usage: Usage
    options: StreamOptions | None = None


@dataclass
class FauxConfig:
    """Tuning knobs for the FauxProvider."""

    latency_seconds: float = 0.01  # default 10ms — fast for benchmarks
    response_mode: str = "echo"  # "echo" | "counter" | "scripted"
    # For scripted mode: a list of response strings consumed in order
    scripted_responses: list[str] = field(default_factory=list)
    # Optional token counts to report (input, output)
    input_tokens: int = 50
    output_tokens: int = 50
    # Optional error injection: "rate_limit" | "timeout" | "internal"
    error_to_inject: str | None = None
    # Optional pre-defined model list (defaults provided)
    models: list[Model] | None = None


class FauxProvider(LLMProvider):
    """In-process mock LLM. See module docstring."""

    name = "faux"

    def __init__(self, config: FauxConfig | None = None, **kwargs: Any) -> None:
        self.config = config or FauxConfig(**kwargs)
        self.calls: list[FauxCallRecord] = []
        self._script_index = 0

    # ---------- introspection ----------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_call(self) -> FauxCallRecord | None:
        return self.calls[-1] if self.calls else None

    # ---------- LLMProvider ----------

    async def list_models(self) -> list[Model]:
        if self.config.models is not None:
            return self.config.models
        return self._default_models()

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # Build the call record early so it survives early-break in callers.
        started_at = time.time()
        prompt = self._extract_prompt(context)
        record = FauxCallRecord(
            model_id=model.id,
            context_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            prompt=prompt,
            response="",
            started_at=started_at,
            ended_at=started_at,
            usage=Usage(),
            options=options,
        )

        try:
            # 1. simulate latency
            if self.config.latency_seconds > 0:
                await asyncio.sleep(self.config.latency_seconds)

            # 2. error injection
            if self.config.error_to_inject:
                yield StreamEvent(
                    type="error",
                    error=f"faux error: {self.config.error_to_inject}",
                )
                return

            # 3. compute response
            response_text = self._compute_response(prompt)

            # 4. usage
            usage = Usage(
                input=self.config.input_tokens,
                output=self.config.output_tokens,
            )

            # 5. emit events
            yield StreamEvent(type="start", partial=self._partial(model, usage))
            yield StreamEvent(type="text_start", content_index=0)
            yield StreamEvent(
                type="text_delta",
                content_index=0,
                delta=response_text,
            )
            yield StreamEvent(
                type="text_end",
                content_index=0,
                content=response_text,
            )

            # 6. final message
            final = AssistantMessage(
                content=[TextContent(text=response_text)],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=usage,
                stop_reason="stop",
            )
            yield StreamEvent(type="done", message=final)

            # 7. finalize record
            record.response = response_text
            record.usage = usage
            record.ended_at = time.time()
        finally:
            # Always record — even if the consumer breaks early.
            # The append is the last thing the generator does.
            self.calls.append(record)

    async def close(self) -> None:
        # No resources to release
        return None

    # ---------- helpers ----------

    @staticmethod
    def _extract_prompt(context: Context) -> str:
        """Pull the last user message text out of the context."""
        for msg in reversed(context.messages):
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        return text
        return context.system_prompt or ""

    def _compute_response(self, prompt: str) -> str:
        mode = self.config.response_mode
        if mode == "echo":
            return f"echo: {prompt}"
        if mode == "counter":
            n = self.call_count + 1
            return f"response #{n} for: {prompt[:50]}"
        if mode == "scripted":
            if not self.config.scripted_responses:
                return ""
            idx = self._script_index % len(self.config.scripted_responses)
            self._script_index += 1
            return self.config.scripted_responses[idx]
        raise ValueError(f"unknown response_mode: {mode!r}")

    @staticmethod
    def _partial(model: Model, usage: Usage) -> AssistantMessage:
        return AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=usage,
        )

    @staticmethod
    def _default_models() -> list[Model]:
        return [
            Model(
                id="faux-fast",
                name="Faux Fast",
                api="faux",
                provider="faux",
                base_url="local://faux",
                cost=ModelCost(input=0.0, output=0.0),
                context_window=128_000,
                max_tokens=4096,
            ),
            Model(
                id="faux-reasoning",
                name="Faux Reasoning",
                api="faux",
                provider="faux",
                base_url="local://faux",
                reasoning=True,
                cost=ModelCost(input=0.0, output=0.0),
                context_window=128_000,
                max_tokens=4096,
            ),
        ]


__all__ = ["FauxCallRecord", "FauxConfig", "FauxProvider"]

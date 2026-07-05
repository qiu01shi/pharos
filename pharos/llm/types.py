"""Core type definitions for the pharos LLM layer.

This module defines the unified data model used across all LLM providers
and the runtime. The shapes deliberately mirror Anthropic/OpenAI Message
streams so we can add new providers without re-deriving the model.

Key invariants:
- All token-bearing types are frozen (immutable) to enable safe sharing
  across concurrent tasks.
- Cost is computed from Usage + Model.cost; never stored on the wire.
- StreamEvent.type drives a finite state machine consumed by Director.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =====================================================================
# Usage / Cost
# =====================================================================


class Usage(BaseModel):
    """Token usage breakdown.

    cache_read/cache_write track provider-native prompt caching.
    reasoning is a subset of `output` (already counted there) — kept
    separate for cost analysis. Set to a number by providers that expose
    reasoning breakdowns; left undefined otherwise.
    """

    model_config = ConfigDict(frozen=True)

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0  # Anthropic-specific 1h retention split
    reasoning: int | None = None

    @property
    def total(self) -> int:
        return self.input + self.output


class ModelCost(BaseModel):
    """Per-million-token cost in USD."""

    model_config = ConfigDict(frozen=True)

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class CostRecord(BaseModel):
    """Concrete cost in USD for a single LLM call.

    Computed by `CostRecord.from_usage`; never sent on the wire.
    """

    model_config = ConfigDict(frozen=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0

    @classmethod
    def from_usage(cls, usage: Usage, cost: ModelCost) -> CostRecord:
        """Convert token usage to USD using a ModelCost reference."""
        return cls(
            input=usage.input * cost.input / 1_000_000,
            output=usage.output * cost.output / 1_000_000,
            cache_read=usage.cache_read * cost.cache_read / 1_000_000,
            cache_write=usage.cache_write * cost.cache_write / 1_000_000,
            total=(
                usage.input * cost.input
                + usage.output * cost.output
                + usage.cache_read * cost.cache_read
                + usage.cache_write * cost.cache_write
            )
            / 1_000_000,
        )


# =====================================================================
# Content blocks
# =====================================================================


class TextContent(BaseModel):
    """Plain text content."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ThinkingContent(BaseModel):
    """Reasoning/thinking content (Anthropic, DeepSeek, OpenAI o-series)."""

    model_config = ConfigDict(frozen=True)

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None  # provider-specific opaque reuse token


class ImageContent(BaseModel):
    """Image content (base64 inline or remote URL)."""

    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    data: str  # base64 or URL
    mime_type: str = "image/png"


class ToolCall(BaseModel):
    """Tool invocation request from assistant."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Messages
# =====================================================================


class UserMessage(BaseModel):
    """User-role message."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


class AssistantMessage(BaseModel):
    """Assistant-role message (a single LLM turn)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["assistant"] = "assistant"
    content: list[TextContent | ThinkingContent | ToolCall] = Field(
        default_factory=list
    )
    api: str = ""  # "openai-responses" | "anthropic-messages" | ...
    provider: str = ""  # "openai" | "anthropic" | "google" | ...
    model: str = ""
    response_id: str | None = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: Literal["stop", "length", "tool_use", "error", "aborted"] = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

    def text(self) -> str:
        """Concatenate all text content blocks."""
        return "".join(
            block.text for block in self.content if isinstance(block, TextContent)
        )


class ToolResultMessage(BaseModel):
    """Tool execution result, returned to the LLM."""

    model_config = ConfigDict(frozen=True)

    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    is_error: bool = False
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


Message = UserMessage | AssistantMessage | ToolResultMessage


# =====================================================================
# Tools
# =====================================================================


class Tool(BaseModel):
    """Tool definition exposed to the LLM.

    `parameters` follows JSON Schema. Providers convert to their native
    representation (e.g. Anthropic's input_schema).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Context
# =====================================================================


class Context(BaseModel):
    """Complete input for a single LLM call."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str | None = None
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)


# =====================================================================
# Streaming events
# =====================================================================


StreamEventType = Literal[
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
]


class StreamEvent(BaseModel):
    """A single event emitted by an LLM stream.

    Event sequence (per the contract):
        start → (text/thinking/toolcall)_start → *_delta* → *_end → done
    OR
        start → error
    """

    model_config = ConfigDict(frozen=True)

    type: StreamEventType
    content_index: int = 0
    delta: str | None = None
    content: str | None = None
    thinking: str | None = None
    tool_call: ToolCall | None = None
    partial: AssistantMessage | None = None
    message: AssistantMessage | None = None
    error: str | None = None


# =====================================================================
# Model catalog
# =====================================================================


class Model(BaseModel):
    """Static metadata for a model the system can route to."""

    model_config = ConfigDict(frozen=True)

    id: str  # "claude-sonnet-4-20250514" | "gpt-4o" | ...
    name: str  # human-readable
    api: str  # "openai-responses" | "anthropic-messages" | ...
    provider: str  # "openai" | "anthropic" | "google" | ...
    base_url: str
    reasoning: bool = False
    input: list[Literal["text", "image"]] = Field(
        default_factory=lambda: ["text"]  # type: ignore[arg-type]
    )
    cost: ModelCost
    context_window: int = 200_000
    max_tokens: int = 8192
    headers: dict[str, str] = Field(default_factory=dict)


# =====================================================================
# StreamOptions
# =====================================================================


class StreamOptions(BaseModel):
    """Per-call knobs (temperature, thinking, cache, etc.)."""

    model_config = ConfigDict(frozen=True)

    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    thinking_level: Literal["off", "minimal", "low", "medium", "high"] | None = None
    cache_retention: Literal["short", "long"] = "short"
    # JSON Schema (subset) the response must conform to. Providers with native
    # structured output (OpenAI) enforce it server-side; others rely on the
    # prompt nudge + LLMAgent's post-hoc validation.
    response_schema: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

"""ReplayProvider — replays recorded LLM outputs from a trace cache.

The Provider Protocol requires `stream()` returning
`AsyncIterator[StreamEvent]`. ReplayProvider's stream yields the
same events that were captured by `record_run`, in the same order,
with the same deltas and final message.

This makes `pharos replay --re-run <run_id>` byte-equal with the
original run (assuming input data hasn't changed), without any
network call.

Usage (Python):
    from pharos.llm.providers.replay import ReplayProvider
    from pharos.runtime import extract_cached_outputs

    cache = extract_cached_outputs("run-uuid-1234")
    provider = ReplayProvider(node_id="agent", cache=cache)
    # When this provider is used in a graph, it returns the
    # recorded output for that node/step.

The cache lookup is keyed by `(node_id, step_index)`. step_index
is computed from the FireContext.step_id (which the Director
generates as `<run_id>:N`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pharos.llm.base import LLMProvider
from pharos.llm.types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)


def _step_index_from_step_id(step_id: str) -> int:
    """`<run>:N` → N (or 0 on parse failure)."""
    if ":" in step_id:
        try:
            return int(step_id.rsplit(":", 1)[-1])
        except ValueError:
            pass
    return 0


class ReplayProvider(LLMProvider):
    """Replays a single node's recorded LLM output.

    `cache` is the dict returned by `runtime.extract_cached_outputs()`.
    `node_id` selects which key in the cache this provider serves.
    `step_counter` increments after every `stream()` call so a node
    that fires multiple times (SDF) gets the right cached output.
    """

    name = "replay"

    def __init__(
        self,
        node_id: str,
        cache: dict[str, dict[str, Any]],
    ) -> None:
        self._node_id = node_id
        self._cache = cache
        # Step counter starts at 1 because the Director generates
        # step_ids like `<run_id>:1`, `:2`, ... and the cache keys
        # are extracted as `node_id:step_index`.
        self._step_counter = 1
        # Tracks which cache keys we've already returned, so we
        # don't replay the same output twice on a long-running
        # graph (defensive; the step counter should handle this).
        self._consumed: set[int] = set()

    async def list_models(self) -> list[Model]:
        from pharos.llm.types import ModelCost

        return [
            Model(
                id="replay",
                name="Replay (no network)",
                api="replay",
                provider=self.name,
                base_url="",
                cost=ModelCost(input=0.0, output=0.0),
                context_window=128_000,
                max_tokens=8_192,
            )
        ]

    async def close(self) -> None:
        pass

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(model, context, options)

    async def _stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None,
    ) -> AsyncIterator[StreamEvent]:
        # Pick the right cache entry
        step_idx = self._step_counter
        self._step_counter += 1
        key = f"{self._node_id}:{step_idx}"
        cached = self._cache.get(key)

        if cached is None:
            # Fallback: try the next available key for this node.
            # This handles the case where the original run had a
            # different step_index scheme.
            candidates = [
                (k, v) for k, v in self._cache.items()
                if v["node_id"] == self._node_id
                and v["step_index"] not in self._consumed
            ]
            if candidates:
                # Sort by step_index so we use the earliest first
                candidates.sort(key=lambda kv: kv[1]["step_index"])
                _key, cached = candidates[0]
                self._consumed.add(cached["step_index"])
            else:
                # No more cached outputs: emit a structured error
                # event so the caller knows replay is exhausted.
                yield StreamEvent(
                    type="error",
                    error=(
                        f"replay: no cached output for {self._node_id} "
                        f"step {step_idx} (have keys: {list(self._cache.keys())})"
                    ),
                )
                return

        self._consumed.add(step_idx)

        # Replay each recorded event in order.
        for ev_dict in cached.get("events", []):
            event = _dict_to_stream_event(ev_dict)
            if event is not None:
                yield event


def _dict_to_stream_event(d: dict[str, Any]) -> StreamEvent | None:
    """Convert a recorded event dict back into a StreamEvent.

    StreamEvent fields (see pharos/llm/types.py):
        type, delta, content_index, tool_call, message, error

    Recorded events have a flat structure: `{"type": "...",
    "delta": "...", "content_index": N, ...}` after the
    asdict(span_event) call.
    """
    etype = d.get("type", "")
    delta = d.get("delta")
    content_index = d.get("content_index", 0)
    error = d.get("error")
    tool_call_dict = d.get("tool_call")
    message_dict = d.get("message")

    tool_call: ToolCall | None = None
    if isinstance(tool_call_dict, dict) and tool_call_dict:
        try:
            tool_call = ToolCall.model_validate(tool_call_dict)
        except Exception:
            tool_call = None

    message: AssistantMessage | None = None
    if isinstance(message_dict, dict) and message_dict:
        try:
            # Reconstruct content blocks properly
            content_blocks: list[Any] = []
            for b in message_dict.get("content", []):
                if not isinstance(b, dict):
                    continue
                btype = b.get("type", "")
                if btype == "text":
                    content_blocks.append(TextContent(text=b.get("text", "")))
                elif btype == "thinking":
                    content_blocks.append(
                        ThinkingContent(thinking=b.get("thinking", ""))
                    )
                # tool_use blocks are reconstructed via tool_call
            message = AssistantMessage(
                content=content_blocks,
                api=message_dict.get("api", "replay"),
                provider=message_dict.get("provider", "replay"),
                model=message_dict.get("model", "replay"),
                stop_reason=message_dict.get("stop_reason", "stop"),
            )
            # Attach usage if present
            u = message_dict.get("usage")
            if isinstance(u, dict):
                message = message.model_copy(
                    update={
                        "usage": Usage(
                            input=u.get("input", 0),
                            output=u.get("output", 0),
                            cache_read=u.get("cache_read", 0),
                            cache_write=u.get("cache_write", 0),
                            cache_write_1h=u.get("cache_write_1h", 0),
                            reasoning=u.get("reasoning", 0),
                        )
                    }
                )
        except Exception:
            message = None

    return StreamEvent(
        type=etype,
        delta=delta,
        content_index=content_index,
        tool_call=tool_call,
        message=message,
        error=error,
    )


__all__ = ["ReplayProvider"]
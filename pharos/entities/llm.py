"""LLMAgent — wraps a LLMProvider as a pharos Entity.

Behavior:
- `setup()` creates the LLM client (long-lived).
- `fire()` consumes one or more prompt tokens, calls the LLM,
  emits one `text` token and one `usage` token.
- Streams partial text to `draft` so downstream entities can react
  before the final response arrives.
- If `tools` are configured, the LLM may emit `tool_call` tokens
  that get echoed out of the `tool_calls` port.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any

from pharos.core.entity import Entity, entity
from pharos.core.port import InputPort, OutputPort, PortContractViolation
from pharos.core.schema import validate as _validate_schema
from pharos.core.token import TypedValue
from pharos.llm.base import LLMProvider
from pharos.llm.types import (
    Context,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    Usage,
    UserMessage,
)
from pharos.observability.trace import current_span as _current_span_fn


@dataclass
class LLMEntityConfig:
    """Configuration for an LLMAgent.

    `provider_class` is a class (not an instance) because the Director
    instantiates the Entity on first setup. We hold the *params* to
    pass at instantiation time.
    """

    provider_class: type[LLMProvider]
    provider_kwargs: dict[str, Any]
    model_id: str
    system_prompt: str = ""
    tools: list[Any] | None = None  # list[Tool] once provided
    temperature: float | None = None
    max_tokens: int | None = None
    thinking_level: str | None = None
    # If set, LLMAgent will execute tool calls and feed the
    # results back to the LLM in a loop, up to `max_tool_iterations`.
    # The registry's tools are sent to the LLM via the
    # provider's tool-spec protocol.
    tool_registry: Any = None  # pharos.entities.tools.ToolRegistry
    max_tool_iterations: int = 5
    # Structured output: when set, the LLM is steered to emit JSON matching
    # this JSON Schema (subset), the result is validated, and the parsed
    # object is emitted on the `json` port. `on_invalid` decides what happens
    # when the response doesn't parse/validate:
    #   "raise"      -> raise PortContractViolation (composes with RetryEntity)
    #   "error_port" -> emit the failure message on the `error` port instead
    output_schema: dict[str, Any] | None = None
    on_invalid: str = "raise"
    # Self-heal: on a schema-invalid response, feed the validation error back
    # into the SAME conversation and ask the model to return corrected JSON,
    # up to this many times before falling back to `on_invalid`. 0 = disabled
    # (validate once, then apply on_invalid — the historical behavior).
    max_repair_attempts: int = 0

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = []


@entity
class LLMAgent(Entity):
    """An Entity that calls an LLM.

    Ports:
        ins:
            prompt: TextPort        — user prompt
            system_override: TextPort — optional system-prompt override
        outs:
            text: TextPort         — final LLM response
            draft: TextPort        — streamed partial response (live)
            thinking: TextPort     — thinking/reasoning (if model emits any)
            tool_calls: JsonPort   — list of ToolCall (if model emits any)
            usage: JsonPort        — Usage object
            json: JsonPort         — validated structured output
                                     (only when output_schema is configured)
            error: TextPort        — validation failure message
                                     (only when on_invalid == "error_port")
    """

    ins = {
        "prompt": InputPort(name="prompt", accepted_types=["text"]),
        "system_override": InputPort(
            name="system_override", accepted_types=["text"]
        ),
    }
    outs = {
        "text": OutputPort(name="text", accepted_types=["text"]),
        "draft": OutputPort(name="draft", accepted_types=["text"]),
        "thinking": OutputPort(name="thinking", accepted_types=["text"]),
        "tool_calls": OutputPort(name="tool_calls", accepted_types=["json"]),
        "usage": OutputPort(name="usage", accepted_types=["json"]),
        # Structured output: the validated, parsed JSON object (only emitted
        # when `output_schema` is configured). Its schema is bound per-instance
        # in __init__ so the port itself enforces the shape on emit.
        "json": OutputPort(name="json", accepted_types=["json"]),
        # Validation failures land here when `on_invalid == "error_port"`.
        "error": OutputPort(name="error", accepted_types=["text"]),
    }

    def __init__(
        self,
        node_id: str,
        config: LLMEntityConfig,
    ) -> None:
        super().__init__(node_id=node_id)
        self.config = config
        # Bind the declared output schema to the `json` port so the port
        # boundary enforces the shape (defense in depth alongside fire()).
        if config.output_schema is not None:
            self.outs["json"].schema = config.output_schema
        self.provider: LLMProvider | None = None
        self.model: Model | None = None
        # Cumulative token count and cost across all fire() invocations.
        # The Director reads these after each fire to update the run's
        # total_tokens / total_cost.
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        # How many self-heal repair rounds the last fire() consumed. Exposed so
        # Agent CI can treat a rise (e.g. 0 -> 3) as a structural drift signal.
        self.repair_attempts_used: int = 0

    async def setup(self, ctx) -> None:
        # Construct the provider (this may fail if API key is missing;
        # we let the error bubble — the Director will mark the run failed).
        self.provider = self.config.provider_class(
            **self.config.provider_kwargs
        )
        models = await self.provider.list_models()
        # Find the requested model, or fall back to the first one
        for m in models:
            if m.id == self.config.model_id:
                self.model = m
                break
        if self.model is None:
            self.model = models[0]

    async def teardown(self) -> None:
        if self.provider is not None:
            await self.provider.close()
            self.provider = None

    def reset_run_state(self) -> None:
        super().reset_run_state()
        self.total_tokens = 0
        self.total_cost = 0.0
        self.repair_attempts_used = 0

    async def fire(self, ctx) -> None:
        if self.provider is None or self.model is None:
            await self.setup(ctx)
        assert self.provider is not None
        assert self.model is not None

        # Consume prompts (concatenate multiple)
        prompt_tokens = self.ins["prompt"].consume()
        if not prompt_tokens:
            return
        user_text = "".join(t.value.payload for t in prompt_tokens)

        # Optional system override
        sys_override = self.ins["system_override"].consume()
        system_prompt = self.config.system_prompt
        if sys_override:
            system_prompt = "".join(t.value.payload for t in sys_override)

        # Structured output: steer every provider (including ones without
        # native JSON-schema enforcement) by appending a schema instruction.
        if self.config.output_schema is not None:
            system_prompt = (
                f"{system_prompt}\n\n"
                "You must respond with a single JSON value that conforms to "
                "this JSON Schema. Output only the JSON — no prose, no "
                "markdown code fences:\n"
                f"{_json.dumps(self.config.output_schema)}"
            ).strip()

        # Build the tool spec. ToolRegistry.list_tools() returns
        # list[Tool] (pharos.llm.types.Tool), which is what
        # Context.tools and providers expect.
        context_tools: list[Any] = self.config.tools or []
        if self.config.tool_registry is not None:
            context_tools = self.config.tool_registry.list_tools()

        # Messages start with the user's prompt; tool results are
        # appended as we go through the tool-call loop.
        from pharos.llm.types import AssistantMessage as _AM
        from pharos.llm.types import ToolResultMessage as _TR

        messages: list[Any] = [UserMessage(content=user_text)]
        options = StreamOptions(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            thinking_level=self.config.thinking_level,  # type: ignore[arg-type]
            response_schema=self.config.output_schema,
        )

        # Accumulators across all LLM calls in this fire()
        accumulated = ""
        accumulated_thinking = ""
        all_tool_calls: list[ToolCall] = []
        total_usage = Usage()
        # Record events onto the active span (if any) for replay.
        span = _current_span_fn()
        # Tracer from the fire context lets us open a child span per tool
        # execution, so tools show up as first-class nodes in the trace
        # tree (nested under this entity's fire span) instead of being an
        # opaque black box inside one node.
        tracer = getattr(ctx, "tracer", None)

        # Tool-call loop: keep calling LLM until it stops emitting
        # tool calls, or max_tool_iterations is hit.
        granted: set[str] = getattr(ctx, "granted_permissions", set()) or set()
        for _iteration in range(self.config.max_tool_iterations + 1):
            context = Context(
                system_prompt=system_prompt,
                messages=list(messages),
                tools=context_tools,
            )
            (
                text_delta,
                think_delta,
                tool_calls_this_round,
                round_usage,
                round_error,
                round_text_final,
            ) = await self._stream_one_round(context, options, span)

            if text_delta:
                accumulated += text_delta
            if think_delta:
                accumulated_thinking += think_delta
            if round_text_final and not accumulated:
                accumulated = round_text_final
            # Accumulate usage across every round of the tool-call loop.
            # Usage is frozen, so build a new summed instance each round —
            # otherwise multi-turn tool loops would only report the LAST
            # round's tokens/cost.
            if round_usage.input or round_usage.output:
                total_usage = Usage(
                    input=total_usage.input + round_usage.input,
                    output=total_usage.output + round_usage.output,
                    cache_read=total_usage.cache_read + round_usage.cache_read,
                    cache_write=total_usage.cache_write + round_usage.cache_write,
                    cache_write_1h=(
                        total_usage.cache_write_1h + round_usage.cache_write_1h
                    ),
                )
            all_tool_calls.extend(tool_calls_this_round)

            # If we have an error or no tool calls, we're done.
            if round_error:
                self.outs["text"].emit(
                    TypedValue(type="text", payload=f"[error: {round_error}]")
                )
                return

            if not tool_calls_this_round:
                # No more tool calls → LLM produced final answer
                break

            # Execute tool calls and append results to messages.
            # Persist what the LLM said (it might be partial reasoning).
            if text_delta or round_text_final:
                messages.append(
                    _AM(
                        content=[TextContent(text=text_delta or round_text_final)],
                        api="",
                        provider="",
                        model="",
                    )
                )
            # Track assistant's tool_call declarations so the
            # provider round-trip stays consistent.
            messages.append(
                _AM(
                    content=[
                        ToolCall(
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                        for tc in tool_calls_this_round
                    ],
                    api="",
                    provider="",
                    model="",
                )
            )

            if self.config.tool_registry is None:
                # Tool calls emitted but no registry configured;
                # surface them but don't execute.
                break

            # Execute all tool calls in this round in parallel.
            tool_call_dicts = [
                {"name": tc.name, "arguments": tc.arguments, "tool_call_id": tc.id}
                for tc in tool_calls_this_round
            ]
            # Open one child span per tool call (nested under this fire's
            # span) before executing them in parallel.
            tool_spans: dict[str, Any] = {}
            if tracer is not None and span is not None:
                for tc in tool_calls_this_round:
                    tool_spans[tc.id] = tracer.start_span(
                        f"tool.execute.{tc.name}",
                        parent=span,
                        attributes={
                            "tool_name": tc.name,
                            "arguments": tc.arguments,
                            "tool_call_id": tc.id,
                        },
                    )
            results = await self.config.tool_registry.execute_batch(
                tool_call_dicts,
                granted_permissions=granted,
            )
            for tc, result in zip(tool_calls_this_round, results, strict=True):
                tsp = tool_spans.get(tc.id)
                if tsp is not None and tracer is not None:
                    out = result.output or ""
                    maxc = tsp.max_attr_chars
                    if len(out) > maxc:
                        out = out[:maxc] + f"...(truncated {len(out) - maxc} chars)"
                    tsp.set_attributes(
                        {
                            "output": out,
                            "is_error": result.is_error,
                            "error": result.error,
                        }
                    )
                    if result.is_error:
                        tsp.status = "error"
                    tracer.finish_span(tsp)
                messages.append(
                    _TR(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text=result.output)],
                        is_error=result.is_error,
                    )
                )
        # Structured output: validate and, if enabled, self-heal by feeding the
        # schema error back into the same conversation before falling back to
        # on_invalid. Runs after the tool loop so `accumulated` is the final
        # answer. total_usage keeps accruing across repair rounds.
        self.repair_attempts_used = 0
        parsed: Any = None
        reason = ""
        if self.config.output_schema is not None:
            parsed, reason = self._validate_structured(accumulated)
            while reason and self.repair_attempts_used < self.config.max_repair_attempts:
                self.repair_attempts_used += 1
                if span is not None:
                    span.record_event(
                        "structured.repair",
                        {"attempt": self.repair_attempts_used, "error": reason},
                    )
                messages.append(
                    _AM(
                        content=[TextContent(text=accumulated)],
                        api="",
                        provider="",
                        model="",
                    )
                )
                messages.append(
                    UserMessage(
                        content=(
                            "Your previous response did not match the required "
                            f"JSON Schema. Error: {reason}. Reply with ONLY the "
                            "corrected JSON — no prose, no markdown fences."
                        )
                    )
                )
                repair_ctx = Context(
                    system_prompt=system_prompt,
                    messages=list(messages),
                    tools=context_tools,
                )
                (
                    r_text,
                    _r_think,
                    _r_tcs,
                    r_usage,
                    r_error,
                    r_final,
                ) = await self._stream_one_round(repair_ctx, options, span)
                if r_usage.input or r_usage.output:
                    total_usage = Usage(
                        input=total_usage.input + r_usage.input,
                        output=total_usage.output + r_usage.output,
                        cache_read=total_usage.cache_read + r_usage.cache_read,
                        cache_write=total_usage.cache_write + r_usage.cache_write,
                        cache_write_1h=(
                            total_usage.cache_write_1h + r_usage.cache_write_1h
                        ),
                    )
                if r_error:
                    reason = r_error
                    break
                candidate = r_final or r_text
                if candidate:
                    accumulated = candidate
                parsed, reason = self._validate_structured(accumulated)

        # Final outputs. The raw text is always emitted (useful for tracing
        # and debugging); structured output is emitted separately on `json`.
        if accumulated:
            self.outs["text"].emit(
                TypedValue(type="text", payload=accumulated)
            )
        if self.config.output_schema is not None:
            self._route_structured(parsed, reason)
        if accumulated_thinking:
            self.outs["thinking"].emit(
                TypedValue(type="text", payload=accumulated_thinking)
            )
        if all_tool_calls:
            self.outs["tool_calls"].emit(
                TypedValue(
                    type="json",
                    payload=[
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in all_tool_calls
                    ],
                )
            )
        if total_usage.input or total_usage.output:
            usage_payload: dict[str, Any] = {
                "input": total_usage.input,
                "output": total_usage.output,
                "cache_read": total_usage.cache_read,
                "cache_write": total_usage.cache_write,
                "total": total_usage.total,
            }
            # Only structured-output agents report repair rounds, so agents
            # without a schema keep their historical usage payload (and digest).
            # Agent CI treats a rise here as a behavioral drift signal.
            if self.config.output_schema is not None:
                usage_payload["repair_attempts"] = self.repair_attempts_used
            self.outs["usage"].emit(
                TypedValue(type="json", payload=usage_payload)
            )

        # Update cumulative token/cost for the Director to read.
        self.total_tokens += total_usage.total
        # Cost = sum of input_cost + output_cost. catalog defines
        # these in $/million tokens; convert to dollars.
        cost = (total_usage.input * self.model.cost.input
                + total_usage.output * self.model.cost.output) / 1_000_000
        self.total_cost += cost

    def _validate_structured(self, text: str) -> tuple[Any, str]:
        """Parse `text` as JSON and validate it against the output schema.

        Pure: returns ``(parsed, reason)`` where ``reason == ""`` means valid.
        A non-empty reason is the parse/schema error, used both to steer a
        self-heal repair round and, if repair is exhausted, as the on_invalid
        failure message.
        """
        assert self.config.output_schema is not None
        try:
            parsed = self._extract_json(text)
        except ValueError as exc:
            return None, str(exc)
        errors = _validate_schema(parsed, self.config.output_schema)
        if errors:
            return parsed, "; ".join(errors)
        return parsed, ""

    def _route_structured(self, parsed: Any, reason: str) -> None:
        """Emit the validated object, or apply ``on_invalid`` on failure.

        On success the parsed object is emitted on the `json` port (which
        re-validates the shape as a boundary guarantee). On failure the
        behaviour follows `config.on_invalid`:
          * "raise"      -> raise PortContractViolation so a wrapping
            RetryEntity can re-fire, or the Director marks the run failed.
          * "error_port" -> emit the failure message on the `error` port,
            letting a downstream Router branch on it.
        """
        if not reason:
            self.outs["json"].emit(TypedValue(type="json", payload=parsed))
            return
        message = f"structured output invalid: {reason}"
        if self.config.on_invalid == "error_port":
            self.outs["error"].emit(TypedValue(type="text", payload=message))
            return
        raise PortContractViolation(f"{self.node_id}: {message}")

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Best-effort extraction of a JSON value from LLM text.

        Handles bare JSON, ```json fenced blocks, and prose that wraps a
        single JSON object/array. Raises ValueError if nothing parses.
        """
        candidate = text.strip()
        if candidate.startswith("```"):
            # Drop the opening fence line (```; ```json) and trailing fence.
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        try:
            return _json.loads(candidate)
        except (ValueError, TypeError):
            pass
        # Fallback: slice from the first opening bracket to the last matching
        # closing bracket and retry.
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = candidate.find(open_ch)
            end = candidate.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    return _json.loads(candidate[start : end + 1])
                except (ValueError, TypeError):
                    continue
        raise ValueError("response is not valid JSON")

    async def _stream_one_round(
        self,
        context: Context,
        options: StreamOptions,
        span: Any,
    ) -> tuple[str, str, list[ToolCall], Usage, str | None, str]:
        """Stream one LLM round. Returns (text_delta, think_delta,
        tool_calls, usage, error_message, final_text).

        `final_text` is the final text from the done event's
        AssistantMessage (may differ from the accumulated
        `text_delta` if the provider coalesces deltas).
        """
        assert self.provider is not None
        assert self.model is not None
        text_delta = ""
        think_delta = ""
        tool_calls: list[ToolCall] = []
        usage = Usage()
        error_msg: str | None = None
        final_text = ""

        try:
            async for ev in self.provider.stream(self.model, context, options):
                # Forward event to the trace span for replay support.
                if span is not None:
                    span.record_event(
                        ev.type,
                        {
                            "delta": ev.delta,
                            "content_index": ev.content_index,
                            "tool_call": (
                                ev.tool_call.model_dump()
                                if ev.tool_call
                                else None
                            ),
                            "message": (
                                ev.message.model_dump()
                                if ev.message
                                else None
                            ),
                            "error": ev.error,
                        },
                    )
                if ev.type == "text_delta" and ev.delta:
                    text_delta += ev.delta
                    # Live-stream: emit a partial token to `draft`
                    self.outs["draft"].emit(
                        TypedValue(type="text", payload=text_delta)
                    )
                elif ev.type == "thinking_delta" and ev.delta:
                    think_delta += ev.delta
                elif ev.type == "toolcall_end" and ev.tool_call is not None:
                    tool_calls.append(ev.tool_call)
                elif ev.type == "done" and ev.message is not None:
                    if ev.message.usage.input or ev.message.usage.output:
                        usage = ev.message.usage
                    for block in ev.message.content:
                        if isinstance(block, TextContent) and not final_text:
                            final_text = block.text
                elif ev.type == "error":
                    error_msg = ev.error
                    break
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

        return text_delta, think_delta, tool_calls, usage, error_msg, final_text


__all__ = ["LLMAgent", "LLMEntityConfig"]

"""RemoteEntity — execute a typed node through the Pharos worker protocol."""

from __future__ import annotations

import hashlib
import os

import httpx

from pharos.core.entity import Entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.observability.trace import current_span
from pharos.worker import WorkerInput, WorkerRequest, WorkerResponse


class RemoteEntity(Entity):
    """POST a fire to a language-neutral HTTP worker.

    Retries remain a graph concern: wrap this entity with ``retry:`` so the
    same idempotency key is reused and the worker can deduplicate side effects.
    """

    required_permissions = {"net:connect"}

    def __init__(
        self,
        node_id: str,
        endpoint: str,
        *,
        timeout: float = 120.0,
        headers_env: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers_env = dict(headers_env or {})
        self.transport = transport
        self.ins = {"input": InputPort("input", accepted_types=[])}
        self.outs = {
            "output": OutputPort("output", accepted_types=[]),
            "metadata": OutputPort("metadata", accepted_types=["json"]),
        }
        self._client: httpx.AsyncClient | None = None

    async def setup(self, ctx) -> None:
        headers: dict[str, str] = {}
        for header, env_name in self.headers_env.items():
            value = os.environ.get(env_name)
            if not value:
                raise ValueError(
                    f"remote entity {self.node_id!r} requires env {env_name!r}"
                )
            headers[header] = value
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=headers,
            transport=self.transport,
        )

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fire(self, ctx) -> None:
        tokens = self.ins["input"].consume()
        if not tokens:
            return
        if self._client is None:
            await self.setup(ctx)
        assert self._client is not None

        stable = f"{ctx.run_id}:{self.node_id}:{ctx.step_id}:{ctx.iter}"
        request = WorkerRequest(
            run_id=ctx.run_id,
            node_id=self.node_id,
            fire_id=ctx.step_id,
            iteration=ctx.iter,
            idempotency_key=hashlib.sha256(stable.encode()).hexdigest(),
            inputs=[
                WorkerInput(
                    port="input",
                    type=token.value.type,
                    payload=token.value.payload,
                    self_hash=token.self_hash,
                )
                for token in tokens
            ],
        )
        headers = {"Idempotency-Key": request.idempotency_key}
        span = current_span()
        if span is not None:
            headers["traceparent"] = (
                f"00-{span.trace_id.replace('-', '')[:32]:0<32}-"
                f"{span.id.replace('-', '')[:16]:0<16}-01"
            )
        response = await self._client.post(
            self.endpoint,
            json=request.model_dump(mode="json"),
            headers=headers,
        )
        response.raise_for_status()
        result = WorkerResponse.model_validate(response.json())
        for output in result.outputs:
            port = self.outs.get(output.port)
            if port is None:
                raise ValueError(
                    f"worker returned undeclared port {output.port!r} for {self.node_id!r}"
                )
            port.emit(TypedValue(type=output.type, payload=output.payload))
        if result.metadata:
            self.outs["metadata"].emit(
                TypedValue(type="json", payload=result.metadata)
            )


__all__ = ["RemoteEntity"]

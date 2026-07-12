"""ContainerEntity — execute a worker protocol fire in an OCI container."""

from __future__ import annotations

import asyncio
import json

from pharos.core.entity import Entity
from pharos.core.port import InputPort, OutputPort
from pharos.core.token import TypedValue
from pharos.worker import WorkerInput, WorkerRequest, WorkerResponse


class ContainerEntity(Entity):
    required_permissions = {"container:execute"}

    def __init__(
        self,
        node_id: str,
        image: str,
        *,
        command: list[str] | None = None,
        timeout: float = 300.0,
        network: bool = False,
        allow_unpinned: bool = False,
        runtime: str = "docker",
    ) -> None:
        super().__init__(node_id=node_id)
        if not allow_unpinned and "@sha256:" not in image:
            raise ValueError(
                "container image must be pinned by digest; set allow_unpinned "
                "only for local development"
            )
        self.image = image
        self.command = list(command or [])
        self.timeout = timeout
        self.network = network
        self.runtime = runtime
        self.ins = {"input": InputPort("input", accepted_types=[])}
        self.outs = {
            "output": OutputPort("output", accepted_types=[]),
            "metadata": OutputPort("metadata", accepted_types=["json"]),
            "stderr": OutputPort("stderr", accepted_types=["text"]),
        }
        if network:
            self.required_permissions = {"container:execute", "net:connect"}

    async def fire(self, ctx) -> None:
        tokens = self.ins["input"].consume()
        if not tokens:
            return
        request = WorkerRequest(
            run_id=ctx.run_id,
            node_id=self.node_id,
            fire_id=ctx.step_id,
            iteration=ctx.iter,
            idempotency_key=f"{ctx.run_id}:{self.node_id}:{ctx.step_id}",
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
        args = [self.runtime, "run", "--rm", "-i", "--read-only"]
        args.extend(["--network", "bridge" if self.network else "none"])
        args.extend([self.image, *self.command])
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(
                    json.dumps(request.model_dump(mode="json")).encode()
                ),
                timeout=self.timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"container entity {self.node_id!r} exceeded {self.timeout}s"
            ) from None
        if stderr:
            self.outs["stderr"].emit(
                TypedValue(type="text", payload=stderr.decode(errors="replace"))
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"container entity {self.node_id!r} exited {proc.returncode}: "
                f"{stderr.decode(errors='replace')[:500]}"
            )
        try:
            result = WorkerResponse.model_validate_json(stdout)
        except Exception as exc:
            raise ValueError(
                f"container entity {self.node_id!r} returned invalid worker JSON"
            ) from exc
        self._emit_result(result)

    def _emit_result(self, result: WorkerResponse) -> None:
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


__all__ = ["ContainerEntity"]

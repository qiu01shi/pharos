"""Language-neutral Pharos worker protocol models (v1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

WORKER_PROTOCOL_VERSION: Literal["pharos.worker/v1"] = "pharos.worker/v1"


class WorkerInput(BaseModel):
    port: str
    type: str
    payload: Any
    self_hash: str


class WorkerRequest(BaseModel):
    protocol: Literal["pharos.worker/v1"] = WORKER_PROTOCOL_VERSION
    run_id: str
    node_id: str
    fire_id: str
    iteration: int = 0
    idempotency_key: str
    inputs: list[WorkerInput] = Field(default_factory=list)


class WorkerOutput(BaseModel):
    port: str = "output"
    type: str = "json"
    payload: Any


class WorkerResponse(BaseModel):
    protocol: Literal["pharos.worker/v1"] = WORKER_PROTOCOL_VERSION
    outputs: list[WorkerOutput] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "WORKER_PROTOCOL_VERSION",
    "WorkerInput",
    "WorkerOutput",
    "WorkerRequest",
    "WorkerResponse",
]

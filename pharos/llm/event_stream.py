"""Stream helpers for the LLM layer.

Currently a thin module that re-exports the async helpers from `base`.
Kept separate so future additions (backpressure, batching) have a clear
home without touching the Protocol.
"""

from pharos.llm.base import acomplete_from_stream, collect_stream

__all__ = ["acomplete_from_stream", "collect_stream"]

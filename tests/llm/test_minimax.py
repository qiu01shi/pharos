"""Tests for the MiniMax provider (Anthropic-compatible)."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from pharos.llm.providers.minimax import MiniMaxProvider
from pharos.llm.types import Context, UserMessage


def _make_provider() -> MiniMaxProvider:
    # Pass a dummy key so unit tests run without MINIMAX_CN_API_KEY.
    # Streaming tests replace `_client` via `_attach_mock`, so no real
    # network / key is ever used.
    return MiniMaxProvider(api_key="test-key-not-real")


def _attach_mock(p: MiniMaxProvider, chunks: list[str]) -> None:
    """Replace p._client with one that returns the given SSE chunks."""

    async def handler(request: httpx.Request) -> httpx.Response:
        # Verify request uses the right base URL and headers
        assert "x-api-key" in request.headers
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
        )

    p._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url=p.base_url,
        headers={"x-api-key": p.api_key, "anthropic-version": "2023-06-01"},
        transport=httpx.MockTransport(handler),
    )


class TestMiniMaxProvider:
    def test_name(self):
        assert _make_provider().name == "minimax"

    def test_base_url(self):
        assert "minimaxi.com" in _make_provider().base_url
        assert "/anthropic" in _make_provider().base_url

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MINIMAX_CN_API_KEY"):
            MiniMaxProvider()

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-key-from-env")
        p = MiniMaxProvider()
        assert p.api_key == "test-key-from-env"

    def test_key_from_dotenv_loaded(self, monkeypatch):
        """After load_dotenv() runs, MINIMAX_CN_API_KEY should be set
        even if the shell didn't have it. Simulates the typical case."""
        # Don't actually re-import; just verify the provider can
        # read the env var if it's present.
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-key-from-dotenv")
        p = MiniMaxProvider()
        assert p.api_key == "test-key-from-dotenv"

    async def test_stream_text(self):
        chunks = [
            "data: " + json.dumps({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 5, "output_tokens": 0}},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "content_block_stop", "index": 0,
            }) + "\n\n",
            "data: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            }) + "\n\n",
            "data: " + json.dumps({"type": "message_stop"}) + "\n\n",
        ]
        p = _make_provider()
        _attach_mock(p, chunks)
        from pharos.llm.types import Model, ModelCost

        model = Model(
            id="MiniMax-Text-01", name="MiniMax", api="anthropic-messages",
            provider="minimax", base_url="https://api.minimaxi.com/anthropic",
            cost=ModelCost(input=1, output=8), context_window=1000000, max_tokens=64000,
        )
        events = []
        async for ev in p.stream(model, Context(messages=[UserMessage(content="hi")])):
            events.append(ev)
        msg = events[-1].message
        assert msg is not None
        assert msg.text() == "hi"

    async def test_list_models(self):
        p = _make_provider()
        models = await p.list_models()
        assert len(models) == 1
        assert models[0].id == "MiniMax-Text-01"
        assert models[0].api == "anthropic-messages"
        assert "minimaxi.com" in models[0].base_url


class TestEnvLoading:
    """Verify pharos auto-loads ~/.pharos/.env on import."""

    def test_dotenv_loader_works(self, tmp_path, monkeypatch):
        from pharos.env import load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text("# comment\nTEST_VAR=hello\n")
        monkeypatch.setenv("PHAROS_DOTENV", str(env_file))
        # Clear any existing value
        monkeypatch.delenv("TEST_VAR", raising=False)
        n = load_dotenv()
        assert n >= 1
        assert os.environ.get("TEST_VAR") == "hello"

    def test_shell_env_wins(self, tmp_path, monkeypatch):
        from pharos.env import load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text("CONFLICTING=from-file\n")
        monkeypatch.setenv("PHAROS_DOTENV", str(env_file))
        monkeypatch.setenv("CONFLICTING", "from-shell")
        load_dotenv()
        assert os.environ.get("CONFLICTING") == "from-shell"
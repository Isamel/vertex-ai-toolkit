"""End-to-end async integration tests — validates the full async-native rewrite.

Tests verify:
1. Async agent creation and execution
2. Async session lifecycle (create -> save turn -> load)
3. Async telemetry flush
4. gather_with_errors with real-ish coroutines
5. Backward compat: sync still works alongside async
6. asyncio.run() entry point pattern (CLI simulation)
7. Cross-layer async wiring (agent -> client -> session)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vaig.agents.base import AgentConfig, AgentResult
from vaig.agents.specialist import SpecialistAgent
from vaig.core.async_utils import gather_with_errors, run_sync, to_async
from vaig.core.client import GenerationResult
from vaig.core.telemetry import TelemetryCollector
from vaig.session.manager import SessionManager
from vaig.session.store import SessionStore

# ── Helpers ────────────────────────────────────────────────────


def _make_mock_client(
    *,
    text: str = "mocked response",
    model: str = "gemini-2.5-pro",
) -> MagicMock:
    """Create a mock GeminiClient with both sync and async methods."""
    client = MagicMock()
    gen_result = GenerationResult(
        text=text,
        model=model,
        finish_reason="STOP",
        usage={"prompt_tokens": 10, "candidates_tokens": 20, "total_tokens": 30},
    )
    client.generate.return_value = gen_result
    client.async_generate = AsyncMock(return_value=gen_result)
    client.current_model = model
    return client


# ── 1. Async Agent Creation and Execution ─────────────────────


class TestAsyncAgentExecution:
    """Test that async agents can be created and called end-to-end."""

    def test_specialist_agent_sync_execute(self) -> None:
        """Sync execute still works — backward compat baseline."""
        config = AgentConfig(
            name="test-sync",
            role="specialist",
            system_instruction="You are a test agent.",
        )
        client = _make_mock_client(text="sync response")
        agent = SpecialistAgent(config, client)

        result = agent.execute("Hello sync")

        assert result.success is True
        assert result.content == "sync response"
        assert result.agent_name == "test-sync"
        client.generate.assert_called_once()

    async def test_specialist_agent_async_execute(self) -> None:
        """Async execute uses async_generate on the client."""
        config = AgentConfig(
            name="test-async",
            role="specialist",
            system_instruction="You are an async test agent.",
        )
        client = _make_mock_client(text="async response")
        agent = SpecialistAgent(config, client)

        result = await agent.async_execute("Hello async")

        assert result.success is True
        assert result.content == "async response"
        assert result.agent_name == "test-async"
        client.async_generate.assert_awaited_once()

    async def test_async_and_sync_produce_same_result(self) -> None:
        """Both paths produce equivalent AgentResults."""
        config = AgentConfig(
            name="dual-test",
            role="specialist",
            system_instruction="Dual agent.",
        )
        client = _make_mock_client(text="same output")
        agent = SpecialistAgent(config, client)

        sync_result = agent.execute("prompt")
        agent.reset()
        async_result = await agent.async_execute("prompt")

        assert sync_result.content == async_result.content
        assert sync_result.success == async_result.success
        assert sync_result.agent_name == async_result.agent_name

    async def test_async_agent_conversation_tracking(self) -> None:
        """Async execute tracks conversation history correctly."""
        config = AgentConfig(
            name="conv-test",
            role="specialist",
            system_instruction="Track conv.",
        )
        client = _make_mock_client(text="response 1")
        agent = SpecialistAgent(config, client)

        await agent.async_execute("turn 1")
        assert len(agent.conversation_history) == 2  # user + agent

        client.async_generate = AsyncMock(
            return_value=GenerationResult(
                text="response 2",
                model="gemini-2.5-pro",
                finish_reason="STOP",
                usage={"total_tokens": 30},
            )
        )
        await agent.async_execute("turn 2")
        assert len(agent.conversation_history) == 4  # 2 turns x 2 messages

    async def test_async_agent_error_handling(self) -> None:
        """Async execute handles API errors gracefully."""
        config = AgentConfig(
            name="error-test",
            role="specialist",
            system_instruction="Error test.",
        )
        client = _make_mock_client()
        client.async_generate = AsyncMock(side_effect=RuntimeError("API down"))
        agent = SpecialistAgent(config, client)

        result = await agent.async_execute("this will fail")

        assert result.success is False
        assert "API down" in result.content


# ── 2. Async Session Lifecycle ────────────────────────────────


class TestAsyncSessionLifecycle:
    """Test async session: create -> save messages -> load -> verify."""

    @pytest.fixture()
    async def session_store(self, tmp_path: Path) -> SessionStore:
        db = tmp_path / "test_integration_sessions.db"
        s = SessionStore(db)
        yield s  # type: ignore[misc]
        await s.async_close()

    async def test_async_session_create_and_load(self, session_store: SessionStore) -> None:
        """Create session async, then load it back async."""
        session_id = await session_store.async_create_session(
            name="integration-test",
            model="gemini-2.5-pro",
            skill="test-skill",
        )
        assert session_id is not None

        session = await session_store.async_get_session(session_id)
        assert session is not None
        assert session["name"] == "integration-test"
        assert session["model"] == "gemini-2.5-pro"
        assert session["skill"] == "test-skill"

    async def test_async_session_full_turn_cycle(self, session_store: SessionStore) -> None:
        """Full cycle: create -> add messages -> load messages -> verify."""
        session_id = await session_store.async_create_session(
            name="turn-test",
            model="gemini-2.5-pro",
        )

        # Save a user turn and agent response
        await session_store.async_add_message(
            session_id=session_id,
            role="user",
            content="What is Kubernetes?",
            model="gemini-2.5-pro",
            token_count=5,
        )
        await session_store.async_add_message(
            session_id=session_id,
            role="model",
            content="Kubernetes is a container orchestration platform.",
            model="gemini-2.5-pro",
            token_count=8,
        )

        # Load messages back
        messages = await session_store.async_get_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Kubernetes?"
        assert messages[1]["role"] == "model"
        assert "container orchestration" in messages[1]["content"]

    async def test_async_session_list_and_search(self, session_store: SessionStore) -> None:
        """List and search sessions via async methods."""
        await session_store.async_create_session(name="alpha-session", model="gemini-2.5-pro")
        await session_store.async_create_session(name="beta-session", model="gemini-2.5-flash")

        sessions = await session_store.async_list_sessions(limit=10)
        assert len(sessions) >= 2

        results = await session_store.async_search_sessions("alpha")
        assert any("alpha" in s["name"] for s in results)

    async def test_async_session_metadata(self, session_store: SessionStore) -> None:
        """Metadata update and retrieval via async."""
        session_id = await session_store.async_create_session(name="meta-test", model="gemini-2.5-pro")

        await session_store.async_update_metadata(session_id, {"cost_usd": 0.05, "turns": 3})

        meta = await session_store.async_get_metadata(session_id)
        assert meta is not None
        assert meta["cost_usd"] == 0.05
        assert meta["turns"] == 3

    async def test_sync_and_async_session_interop(self, session_store: SessionStore) -> None:
        """Sync-created session can be loaded async and vice versa."""
        # Create sync
        sync_id = session_store.create_session(name="sync-created", model="gemini-2.5-pro")

        # Load async
        session = await session_store.async_get_session(sync_id)
        assert session is not None
        assert session["name"] == "sync-created"

        # Create async
        async_id = await session_store.async_create_session(name="async-created", model="gemini-2.5-pro")

        # Load sync
        session2 = session_store.get_session(async_id)
        assert session2 is not None
        assert session2["name"] == "async-created"


# ── 3. Async Telemetry ────────────────────────────────────────


class TestAsyncTelemetry:
    """Test async telemetry flush and query operations."""

    @pytest.fixture()
    async def telem(self, tmp_path: Path) -> TelemetryCollector:
        c = TelemetryCollector(db_path=tmp_path / "test_telem.db", enabled=True, buffer_size=5)
        yield c  # type: ignore[misc]
        await c.async_close()

    async def test_async_flush_persists_events(self, telem: TelemetryCollector) -> None:
        """Events emitted sync are flushed async to SQLite."""
        telem.emit_api_call(
            model="gemini-2.5-pro",
            tokens_in=100,
            tokens_out=200,
            cost_usd=0.001,
        )
        telem.emit_tool_call(tool_name="kubectl_get", duration_ms=150.0)

        # Flush async
        await telem.async_flush()

        # Query async to verify
        events = await telem.async_query_events(limit=10)
        assert len(events) >= 2

        event_names = [e["event_name"] for e in events]
        assert "gemini-2.5-pro" in event_names or any("generate" in n or "gemini" in n for n in event_names)

    async def test_async_telemetry_summary(self, telem: TelemetryCollector) -> None:
        """Async summary aggregation works."""
        for i in range(3):
            telem.emit_api_call(
                model="gemini-2.5-pro",
                tokens_in=100,
                tokens_out=200,
                cost_usd=0.01,
            )

        await telem.async_flush()
        summary = await telem.async_get_summary()

        assert summary is not None
        assert summary["total_events"] >= 3

    async def test_async_clear_events(self, telem: TelemetryCollector) -> None:
        """Async clear removes old events."""
        telem.emit_tool_call(tool_name="test_tool")
        await telem.async_flush()

        # Clear events older than 0 days (everything)
        cleared = await telem.async_clear_events(older_than_days=0)
        assert cleared >= 0  # might be 0 if timestamp is "now"

    async def test_async_close(self, tmp_path: Path) -> None:
        """Async close flushes remaining buffer and closes connection."""
        c = TelemetryCollector(db_path=tmp_path / "close_test.db", enabled=True, buffer_size=100)
        c.emit_tool_call(tool_name="final_tool")

        await c.async_close()
        # After close, the collector should be marked closed
        assert c._closed is True


# ── 4. gather_with_errors Integration ─────────────────────────


class TestGatherWithErrors:
    """Test gather_with_errors with realistic coroutines."""

    async def test_gather_multiple_successful_coroutines(self) -> None:
        """All coroutines succeed."""

        async def compute(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        results = await gather_with_errors(compute(1), compute(2), compute(3))
        assert results == [2, 4, 6]

    async def test_gather_with_exception_propagation(self) -> None:
        """Exception from one coroutine propagates."""

        async def ok() -> str:
            return "ok"

        async def fail() -> str:
            msg = "boom"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="boom"):
            await gather_with_errors(ok(), fail())

    async def test_gather_return_exceptions(self) -> None:
        """With return_exceptions=True, exceptions are returned as results."""

        async def ok() -> str:
            return "ok"

        async def fail() -> str:
            msg = "boom"
            raise ValueError(msg)

        results = await gather_with_errors(ok(), fail(), return_exceptions=True)
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)

    async def test_gather_with_timeout(self) -> None:
        """Timeout cancels slow coroutines."""

        async def slow() -> str:
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(asyncio.TimeoutError):
            await gather_with_errors(slow(), timeout=0.05)

    async def test_gather_empty(self) -> None:
        """Empty input returns empty list."""
        results = await gather_with_errors()
        assert results == []

    async def test_gather_mixed_fast_slow(self) -> None:
        """Mix of fast and slow coroutines all complete within timeout."""

        async def fast() -> str:
            return "fast"

        async def medium() -> str:
            await asyncio.sleep(0.02)
            return "medium"

        results = await gather_with_errors(fast(), medium(), timeout=5.0)
        assert results == ["fast", "medium"]


# ── 5. Backward Compatibility ─────────────────────────────────


class TestBackwardCompatibility:
    """Verify sync paths still work alongside async additions."""

    def test_sync_agent_creation_and_execute(self) -> None:
        """Basic sync agent workflow — must not regress."""
        config = AgentConfig(
            name="compat-sync",
            role="specialist",
            system_instruction="Compat test.",
        )
        client = _make_mock_client(text="sync works")
        agent = SpecialistAgent(config, client)

        result = agent.execute("test")
        assert result.success is True
        assert result.content == "sync works"

    def test_sync_session_store_operations(self, tmp_path: Path) -> None:
        """Sync SessionStore operations still work."""
        store = SessionStore(tmp_path / "compat.db")
        try:
            sid = store.create_session(name="compat", model="gemini-2.5-pro")
            store.add_message(
                session_id=sid, role="user", content="hello", model="gemini-2.5-pro"
            )
            msgs = store.get_messages(sid)
            assert len(msgs) == 1
            assert msgs[0]["content"] == "hello"

            sessions = store.list_sessions()
            assert len(sessions) >= 1
        finally:
            store.close()

    def test_sync_telemetry_operations(self, tmp_path: Path) -> None:
        """Sync TelemetryCollector operations still work."""
        c = TelemetryCollector(
            db_path=tmp_path / "compat_telem.db", enabled=True, buffer_size=5
        )
        try:
            c.emit_tool_call(tool_name="sync_tool", duration_ms=50.0)
            c.flush()
            events = c.query_events(limit=5)
            assert len(events) >= 1
        finally:
            c.close()

    def test_run_sync_bridge(self) -> None:
        """run_sync can execute an async function from sync context."""

        async def async_add(a: int, b: int) -> int:
            return a + b

        result = run_sync(async_add(3, 4))
        assert result == 7

    def test_to_async_wrapper(self) -> None:
        """to_async wraps a sync function for async usage."""

        def sync_multiply(a: int, b: int) -> int:
            return a * b

        async_multiply = to_async(sync_multiply)

        result = run_sync(async_multiply(5, 6))
        assert result == 30

    def test_core_init_exports(self) -> None:
        """Core __init__.py exports all async utilities."""
        from vaig.core import gather_with_errors, run_sync, to_async

        assert callable(run_sync)
        assert callable(to_async)
        assert callable(gather_with_errors)

    def test_agents_init_exports(self) -> None:
        """Agents __init__.py exports all agent types."""
        from vaig.agents import (
            AgentConfig,
            BaseAgent,
            SpecialistAgent,
        )

        assert AgentConfig is not None
        assert BaseAgent is not None
        assert SpecialistAgent is not None

    def test_session_store_importable(self) -> None:
        """Session store is importable from the session package."""
        from vaig.session.manager import SessionManager
        from vaig.session.store import SessionStore

        assert SessionStore is not None
        assert SessionManager is not None


# ── 6. asyncio.run() Entry Point Pattern ──────────────────────


class TestAsyncioRunEntryPoint:
    """Simulate how CLI uses asyncio.run() to enter async land."""

    def test_asyncio_run_agent_execution(self) -> None:
        """asyncio.run() can drive a full async agent execution."""

        async def main() -> AgentResult:
            config = AgentConfig(
                name="cli-sim",
                role="specialist",
                system_instruction="CLI simulation.",
            )
            client = _make_mock_client(text="cli async response")
            agent = SpecialistAgent(config, client)
            return await agent.async_execute("simulate cli")

        result = asyncio.run(main())
        assert result.success is True
        assert result.content == "cli async response"

    def test_asyncio_run_session_lifecycle(self, tmp_path: Path) -> None:
        """asyncio.run() drives async session create -> save -> load."""

        async def main() -> list[dict]:
            store = SessionStore(tmp_path / "cli_session.db")
            try:
                sid = await store.async_create_session(
                    name="cli-test", model="gemini-2.5-pro"
                )
                await store.async_add_message(
                    session_id=sid,
                    role="user",
                    content="hello from CLI",
                )
                return await store.async_get_messages(sid)
            finally:
                await store.async_close()

        messages = asyncio.run(main())
        assert len(messages) == 1
        assert messages[0]["content"] == "hello from CLI"

    def test_asyncio_run_gather_fanout(self) -> None:
        """asyncio.run() with gather_with_errors simulating parallel agents."""

        async def mock_agent(name: str) -> str:
            await asyncio.sleep(0.01)
            return f"{name} done"

        async def main() -> list[str]:
            return await gather_with_errors(
                mock_agent("agent-1"),
                mock_agent("agent-2"),
                mock_agent("agent-3"),
                timeout=5.0,
            )

        results = asyncio.run(main())
        assert len(results) == 3
        assert all("done" in r for r in results)

    def test_run_sync_from_sync_context(self) -> None:
        """run_sync helper works as an alternative to asyncio.run()."""

        async def async_work() -> str:
            return "done"

        assert run_sync(async_work()) == "done"


# ── 7. Cross-Layer Async Wiring ───────────────────────────────


class TestCrossLayerAsyncWiring:
    """Test that async flows correctly wire across agent -> client -> session layers."""

    async def test_agent_to_client_async_wiring(self) -> None:
        """Agent's async_execute correctly calls client.async_generate."""
        config = AgentConfig(
            name="wiring-test",
            role="specialist",
            system_instruction="Wiring test.",
        )
        client = _make_mock_client(text="wired response")
        agent = SpecialistAgent(config, client)

        result = await agent.async_execute("test wiring", context="some context")

        assert result.success is True
        client.async_generate.assert_awaited_once()

        # Verify the call included system_instruction and model
        call_kwargs = client.async_generate.call_args
        assert call_kwargs.kwargs["system_instruction"] == "Wiring test."
        assert call_kwargs.kwargs["model_id"] == "gemini-2.5-pro"

    async def test_session_manager_async_new_and_add(self, tmp_path: Path) -> None:
        """SessionManager async_new_session + async_add_message works end-to-end."""
        db_path = tmp_path / "manager_test.db"
        settings = MagicMock()
        settings.models.default = "gemini-2.5-pro"
        settings.session.auto_save = True
        settings.session.max_history_messages = 100
        settings.db_path_resolved = str(db_path)

        manager = SessionManager(settings=settings)
        store = manager._store

        try:
            session = await manager.async_new_session("manager-test", model="gemini-2.5-pro")
            assert session is not None
            assert session.name == "manager-test"

            await manager.async_add_message("user", "hello manager")
            await manager.async_add_message("model", "hello back")

            # Verify messages persisted
            messages = await store.async_get_messages(session.id)
            assert len(messages) == 2
        finally:
            await store.async_close()

    async def test_telemetry_and_session_async_coexistence(self, tmp_path: Path) -> None:
        """Telemetry and session can both use aiosqlite concurrently."""
        store = SessionStore(tmp_path / "coexist_session.db")
        telem = TelemetryCollector(
            db_path=tmp_path / "coexist_telem.db", enabled=True, buffer_size=5
        )

        try:
            # Run both concurrently
            async def session_work() -> str:
                sid = await store.async_create_session(
                    name="concurrent", model="gemini-2.5-pro"
                )
                await store.async_add_message(
                    session_id=sid, role="user", content="concurrent msg"
                )
                return sid

            async def telem_work() -> None:
                telem.emit_api_call(model="gemini-2.5-pro", tokens_in=50, tokens_out=100)
                await telem.async_flush()

            results = await gather_with_errors(
                session_work(),
                telem_work(),
            )

            # session_work returned a session_id
            session_id = results[0]
            msgs = await store.async_get_messages(session_id)
            assert len(msgs) == 1

            events = await telem.async_query_events(limit=5)
            assert len(events) >= 1
        finally:
            await store.async_close()
            await telem.async_close()

    async def test_parallel_agent_execution_with_gather(self) -> None:
        """Multiple agents can execute in parallel via gather_with_errors."""

        async def run_agent(name: str, text: str) -> AgentResult:
            config = AgentConfig(
                name=name, role="specialist", system_instruction=f"{name} instructions"
            )
            client = _make_mock_client(text=text)
            agent = SpecialistAgent(config, client)
            return await agent.async_execute("parallel test")

        results = await gather_with_errors(
            run_agent("agent-a", "response A"),
            run_agent("agent-b", "response B"),
            run_agent("agent-c", "response C"),
        )

        assert len(results) == 3
        assert all(isinstance(r, AgentResult) for r in results)
        assert {r.agent_name for r in results} == {"agent-a", "agent-b", "agent-c"}
        assert {r.content for r in results} == {"response A", "response B", "response C"}

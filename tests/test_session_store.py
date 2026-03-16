"""Tests for the session store (SQLite persistence)."""

from __future__ import annotations

from vaig.session.store import SessionStore

# store fixture is provided by conftest.py


class TestSessionStore:
    def test_create_session(self, store: SessionStore) -> None:
        sid = store.create_session(name="test", model="gemini-2.5-pro")
        assert sid is not None
        assert len(sid) == 36  # UUID format

    def test_get_session(self, store: SessionStore) -> None:
        sid = store.create_session(name="my-chat", model="gemini-2.5-flash", skill="rca")
        session = store.get_session(sid)
        assert session is not None
        assert session["name"] == "my-chat"
        assert session["model"] == "gemini-2.5-flash"
        assert session["skill"] == "rca"

    def test_get_session_not_found(self, store: SessionStore) -> None:
        result = store.get_session("nonexistent-id")
        assert result is None

    def test_list_sessions(self, store: SessionStore) -> None:
        store.create_session(name="session-1", model="gemini-2.5-pro")
        store.create_session(name="session-2", model="gemini-2.5-flash")
        store.create_session(name="session-3", model="gemini-2.5-pro")

        sessions = store.list_sessions()
        assert len(sessions) == 3

    def test_list_sessions_limit(self, store: SessionStore) -> None:
        for i in range(5):
            store.create_session(name=f"session-{i}", model="gemini-2.5-pro")

        sessions = store.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_delete_session(self, store: SessionStore) -> None:
        sid = store.create_session(name="to-delete", model="gemini-2.5-pro")
        store.delete_session(sid)
        assert store.get_session(sid) is None

    def test_add_message(self, store: SessionStore) -> None:
        sid = store.create_session(name="chat", model="gemini-2.5-pro")
        store.add_message(sid, role="user", content="Hello!")
        store.add_message(sid, role="assistant", content="Hi there!", model="gemini-2.5-pro", token_count=42)

        messages = store.get_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["token_count"] == 42

    def test_get_messages_order(self, store: SessionStore) -> None:
        sid = store.create_session(name="chat", model="gemini-2.5-pro")
        store.add_message(sid, role="user", content="First")
        store.add_message(sid, role="assistant", content="Second")
        store.add_message(sid, role="user", content="Third")

        messages = store.get_messages(sid)
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    def test_get_messages_limit(self, store: SessionStore) -> None:
        sid = store.create_session(name="chat", model="gemini-2.5-pro")
        for i in range(10):
            store.add_message(sid, role="user", content=f"Message {i}")

        messages = store.get_messages(sid, limit=3)
        assert len(messages) == 3

    def test_add_context_file(self, store: SessionStore) -> None:
        sid = store.create_session(name="chat", model="gemini-2.5-pro")
        store.add_context_file(sid, file_path="/tmp/test.py", file_type="code", size_bytes=1024)

        files = store.get_context_files(sid)
        assert len(files) == 1
        assert files[0]["file_path"] == "/tmp/test.py"
        assert files[0]["file_type"] == "code"
        assert files[0]["size_bytes"] == 1024

    def test_search_sessions(self, store: SessionStore) -> None:
        store.create_session(name="rca-investigation", model="gemini-2.5-pro")
        store.create_session(name="code-review", model="gemini-2.5-flash")
        store.create_session(name="rca-postmortem", model="gemini-2.5-pro")

        results = store.search_sessions("rca")
        assert len(results) == 2
        assert all("rca" in r["name"] for r in results)

    def test_cascade_delete(self, store: SessionStore) -> None:
        sid = store.create_session(name="chat", model="gemini-2.5-pro")
        store.add_message(sid, role="user", content="Hello")
        store.add_context_file(sid, file_path="/tmp/x.py", file_type="code")

        store.delete_session(sid)
        assert store.get_messages(sid) == []
        assert store.get_context_files(sid) == []

    def test_session_with_metadata(self, store: SessionStore) -> None:
        sid = store.create_session(
            name="custom",
            model="gemini-2.5-pro",
            metadata={"project": "odin", "env": "staging"},
        )
        session = store.get_session(sid)
        assert session is not None
        assert '"project"' in session["metadata"]

    def test_update_metadata(self, store: SessionStore) -> None:
        sid = store.create_session(name="test", model="gemini-2.5-pro")
        result = store.update_metadata(sid, {"cost_data": {"total_cost": 0.05}})

        assert result is True
        metadata = store.get_metadata(sid)
        assert metadata is not None
        assert metadata["cost_data"]["total_cost"] == 0.05

    def test_update_metadata_merges(self, store: SessionStore) -> None:
        """update_metadata should merge with existing metadata, not replace."""
        sid = store.create_session(
            name="test",
            model="gemini-2.5-pro",
            metadata={"project": "odin"},
        )
        store.update_metadata(sid, {"cost_data": {"total_cost": 0.01}})

        metadata = store.get_metadata(sid)
        assert metadata is not None
        assert metadata["project"] == "odin"  # preserved
        assert metadata["cost_data"]["total_cost"] == 0.01  # added

    def test_update_metadata_overwrites_existing_key(self, store: SessionStore) -> None:
        """If the same key exists, it should be overwritten."""
        sid = store.create_session(
            name="test",
            model="gemini-2.5-pro",
            metadata={"cost_data": {"total_cost": 0.01}},
        )
        store.update_metadata(sid, {"cost_data": {"total_cost": 0.05}})

        metadata = store.get_metadata(sid)
        assert metadata is not None
        assert metadata["cost_data"]["total_cost"] == 0.05

    def test_update_metadata_nonexistent_session(self, store: SessionStore) -> None:
        result = store.update_metadata("nonexistent-id", {"key": "value"})
        assert result is False

    def test_get_metadata(self, store: SessionStore) -> None:
        sid = store.create_session(
            name="test",
            model="gemini-2.5-pro",
            metadata={"project": "odin", "env": "prod"},
        )
        metadata = store.get_metadata(sid)
        assert metadata is not None
        assert metadata["project"] == "odin"
        assert metadata["env"] == "prod"

    def test_get_metadata_empty(self, store: SessionStore) -> None:
        sid = store.create_session(name="test", model="gemini-2.5-pro")
        metadata = store.get_metadata(sid)
        assert metadata is not None
        assert metadata == {}

    def test_get_metadata_nonexistent_session(self, store: SessionStore) -> None:
        metadata = store.get_metadata("nonexistent-id")
        assert metadata is None


class TestAsyncSessionStore:
    """Async versions of all SessionStore methods using aiosqlite."""

    async def test_async_create_session(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="async-test", model="gemini-2.5-pro")
        assert sid is not None
        assert len(sid) == 36

    async def test_async_get_session(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="async-chat", model="gemini-2.5-flash", skill="rca")
        session = await store.async_get_session(sid)
        assert session is not None
        assert session["name"] == "async-chat"
        assert session["model"] == "gemini-2.5-flash"
        assert session["skill"] == "rca"

    async def test_async_get_session_not_found(self, store: SessionStore) -> None:
        result = await store.async_get_session("nonexistent-id")
        assert result is None

    async def test_async_list_sessions(self, store: SessionStore) -> None:
        await store.async_create_session(name="s1", model="gemini-2.5-pro")
        await store.async_create_session(name="s2", model="gemini-2.5-flash")
        await store.async_create_session(name="s3", model="gemini-2.5-pro")

        sessions = await store.async_list_sessions()
        assert len(sessions) == 3

    async def test_async_list_sessions_limit(self, store: SessionStore) -> None:
        for i in range(5):
            await store.async_create_session(name=f"s-{i}", model="gemini-2.5-pro")

        sessions = await store.async_list_sessions(limit=2)
        assert len(sessions) == 2

    async def test_async_delete_session(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="to-delete", model="gemini-2.5-pro")
        deleted = await store.async_delete_session(sid)
        assert deleted is True
        assert await store.async_get_session(sid) is None

    async def test_async_delete_session_nonexistent(self, store: SessionStore) -> None:
        deleted = await store.async_delete_session("nonexistent-id")
        assert deleted is False

    async def test_async_add_message(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="chat", model="gemini-2.5-pro")
        await store.async_add_message(sid, role="user", content="Hello!")
        await store.async_add_message(sid, role="assistant", content="Hi!", model="gemini-2.5-pro", token_count=42)

        messages = await store.async_get_messages(sid)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["token_count"] == 42

    async def test_async_get_messages_order(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="chat", model="gemini-2.5-pro")
        await store.async_add_message(sid, role="user", content="First")
        await store.async_add_message(sid, role="assistant", content="Second")
        await store.async_add_message(sid, role="user", content="Third")

        messages = await store.async_get_messages(sid)
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    async def test_async_get_messages_limit(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="chat", model="gemini-2.5-pro")
        for i in range(10):
            await store.async_add_message(sid, role="user", content=f"Msg {i}")

        messages = await store.async_get_messages(sid, limit=3)
        assert len(messages) == 3

    async def test_async_add_context_file(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="chat", model="gemini-2.5-pro")
        await store.async_add_context_file(sid, file_path="/tmp/test.py", file_type="code", size_bytes=1024)

        files = await store.async_get_context_files(sid)
        assert len(files) == 1
        assert files[0]["file_path"] == "/tmp/test.py"
        assert files[0]["file_type"] == "code"
        assert files[0]["size_bytes"] == 1024

    async def test_async_search_sessions(self, store: SessionStore) -> None:
        await store.async_create_session(name="rca-investigation", model="gemini-2.5-pro")
        await store.async_create_session(name="code-review", model="gemini-2.5-flash")
        await store.async_create_session(name="rca-postmortem", model="gemini-2.5-pro")

        results = await store.async_search_sessions("rca")
        assert len(results) == 2
        assert all("rca" in r["name"] for r in results)

    async def test_async_cascade_delete(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="chat", model="gemini-2.5-pro")
        await store.async_add_message(sid, role="user", content="Hello")
        await store.async_add_context_file(sid, file_path="/tmp/x.py", file_type="code")

        await store.async_delete_session(sid)
        assert await store.async_get_messages(sid) == []
        assert await store.async_get_context_files(sid) == []

    async def test_async_update_metadata(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="test", model="gemini-2.5-pro")
        result = await store.async_update_metadata(sid, {"cost_data": {"total_cost": 0.05}})
        assert result is True

        metadata = await store.async_get_metadata(sid)
        assert metadata is not None
        assert metadata["cost_data"]["total_cost"] == 0.05

    async def test_async_update_metadata_merges(self, store: SessionStore) -> None:
        sid = await store.async_create_session(
            name="test",
            model="gemini-2.5-pro",
            metadata={"project": "odin"},
        )
        await store.async_update_metadata(sid, {"cost_data": {"total_cost": 0.01}})

        metadata = await store.async_get_metadata(sid)
        assert metadata is not None
        assert metadata["project"] == "odin"
        assert metadata["cost_data"]["total_cost"] == 0.01

    async def test_async_update_metadata_nonexistent(self, store: SessionStore) -> None:
        result = await store.async_update_metadata("nonexistent-id", {"key": "value"})
        assert result is False

    async def test_async_get_metadata(self, store: SessionStore) -> None:
        sid = await store.async_create_session(
            name="test",
            model="gemini-2.5-pro",
            metadata={"project": "odin", "env": "prod"},
        )
        metadata = await store.async_get_metadata(sid)
        assert metadata is not None
        assert metadata["project"] == "odin"
        assert metadata["env"] == "prod"

    async def test_async_get_metadata_empty(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="test", model="gemini-2.5-pro")
        metadata = await store.async_get_metadata(sid)
        assert metadata is not None
        assert metadata == {}

    async def test_async_get_metadata_nonexistent(self, store: SessionStore) -> None:
        metadata = await store.async_get_metadata("nonexistent-id")
        assert metadata is None

    async def test_async_rename_session(self, store: SessionStore) -> None:
        sid = await store.async_create_session(name="old-name", model="gemini-2.5-pro")
        result = await store.async_rename_session(sid, "new-name")
        assert result is True

        session = await store.async_get_session(sid)
        assert session is not None
        assert session["name"] == "new-name"

    async def test_async_rename_session_nonexistent(self, store: SessionStore) -> None:
        result = await store.async_rename_session("nonexistent-id", "name")
        assert result is False

    async def test_async_get_last_session(self, store: SessionStore) -> None:
        await store.async_create_session(name="first", model="gemini-2.5-pro")
        await store.async_create_session(name="second", model="gemini-2.5-flash")

        last = await store.async_get_last_session()
        assert last is not None
        assert last["name"] == "second"

    async def test_async_get_last_session_empty(self, store: SessionStore) -> None:
        last = await store.async_get_last_session()
        assert last is None

    async def test_async_close(self, store: SessionStore) -> None:
        """async_close should cleanly close the async connection."""
        sid = await store.async_create_session(name="test", model="gemini-2.5-pro")
        assert sid is not None
        await store.async_close()
        assert store._aconn is None

    async def test_sync_and_async_coexist(self, store: SessionStore) -> None:
        """Sync-created sessions should be visible via async and vice versa."""
        sid_sync = store.create_session(name="sync-session", model="gemini-2.5-pro")
        session = await store.async_get_session(sid_sync)
        assert session is not None
        assert session["name"] == "sync-session"

        sid_async = await store.async_create_session(name="async-session", model="gemini-2.5-flash")
        session2 = store.get_session(sid_async)
        assert session2 is not None
        assert session2["name"] == "async-session"

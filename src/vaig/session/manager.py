"""Session manager — high-level session management with chat history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vaig.core.client import ChatMessage
from vaig.core.config import Settings
from vaig.session.store import SessionStore
from vaig.session.summarizer import HistorySummarizer, estimate_history_tokens

if TYPE_CHECKING:
    from vaig.core.client import GeminiClient

logger = logging.getLogger(__name__)

# Number of recent messages to keep intact during summarization.
# These are the most relevant for the current conversation turn.
_RECENT_MESSAGES_TO_KEEP = 10


@dataclass
class ActiveSession:
    """An active chat session with history."""

    id: str
    name: str
    model: str
    skill: str | None = None
    history: list[ChatMessage] = field(default_factory=list)


class SessionManager:
    """Manages session lifecycle and chat history."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = SessionStore(settings.db_path_resolved)
        self._active: ActiveSession | None = None

    @property
    def active(self) -> ActiveSession | None:
        """Get the currently active session."""
        return self._active

    @property
    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        return self._active is not None

    def new_session(
        self,
        name: str,
        *,
        model: str | None = None,
        skill: str | None = None,
    ) -> ActiveSession:
        """Create and activate a new session."""
        model = model or self._settings.models.default

        session_id = self._store.create_session(
            name=name,
            model=model,
            skill=skill,
        )

        self._active = ActiveSession(
            id=session_id,
            name=name,
            model=model,
            skill=skill,
        )

        logger.info("New session started: %s (model=%s, skill=%s)", name, model, skill)

        # Telemetry: emit session_start event via EventBus
        try:
            from vaig.core.event_bus import EventBus
            from vaig.core.events import SessionStarted

            EventBus.get().emit(
                SessionStarted(
                    session_id=session_id,
                    name=name,
                    model=model,
                    skill=skill or "",
                )
            )
        except Exception:  # noqa: BLE001
            pass

        return self._active

    def add_message(self, role: str, content: str, *, model: str | None = None, token_count: int = 0) -> None:
        """Add a message to the active session."""
        if not self._active:
            msg = "No active session. Start one with /new or load an existing session."
            raise RuntimeError(msg)

        message = ChatMessage(role=role, content=content)
        self._active.history.append(message)

        # Persist
        if self._settings.session.auto_save:
            self._store.add_message(
                session_id=self._active.id,
                role=role,
                content=content,
                model=model,
                token_count=token_count,
            )

        # Trim history if needed
        max_msgs = self._settings.session.max_history_messages
        if len(self._active.history) > max_msgs:
            self._active.history = self._active.history[-max_msgs:]

    def get_history(self) -> list[ChatMessage]:
        """Get the chat history for the active session."""
        if not self._active:
            return []
        return self._active.history

    def get_token_estimate(self) -> int:
        """Return a rough token estimate for the active session's history.

        Uses the fast ``len(text) / 4`` heuristic — no API calls.
        Returns ``0`` when there is no active session.
        """
        if not self._active:
            return 0
        return estimate_history_tokens(self._active.history)

    def _check_and_summarize(self, client: GeminiClient) -> None:
        """Summarize older messages if the history token estimate exceeds the threshold.

        This method is **lazy** — it only triggers when the rough token
        estimate crosses ``summarization_threshold * max_history_tokens``.
        The most recent ``_RECENT_MESSAGES_TO_KEEP`` messages are always
        preserved intact.

        Args:
            client: An initialized ``GeminiClient`` used for the summarization call.
        """
        if not self._active:
            return

        session_cfg = self._settings.session
        threshold_tokens = int(session_cfg.summarization_threshold * session_cfg.max_history_tokens)

        # Fast path: rough estimate
        rough_estimate = self.get_token_estimate()
        if rough_estimate < threshold_tokens:
            return

        history = self._active.history
        if len(history) <= _RECENT_MESSAGES_TO_KEEP:
            # Not enough messages to split — nothing to summarize.
            logger.debug(
                "Token estimate (%d) exceeds threshold (%d) but only %d messages — skipping summarization",
                rough_estimate,
                threshold_tokens,
                len(history),
            )
            return

        # Split: older messages to summarize, recent to keep
        older = history[:-_RECENT_MESSAGES_TO_KEEP]
        recent = history[-_RECENT_MESSAGES_TO_KEEP:]

        logger.info(
            "History token estimate (%d) exceeds threshold (%d) — summarizing %d older messages",
            rough_estimate,
            threshold_tokens,
            len(older),
        )

        summarizer = HistorySummarizer(
            model_name=self._settings.models.fallback,
            summary_target_tokens=session_cfg.summary_target_tokens,
        )

        try:
            summary_message = summarizer.summarize(older, client)
            self._active.history = [summary_message, *recent]
            logger.info(
                "Summarization complete — history reduced from %d to %d messages",
                len(older) + len(recent),
                len(self._active.history),
            )
        except Exception:  # noqa: BLE001
            # Summarization is non-critical — if it fails, keep the original history.
            # The message-count trimming in add_message() is still the safety net.
            logger.warning("History summarization failed — keeping original history", exc_info=True)

    def load_session(self, session_id: str) -> ActiveSession | None:
        """Load an existing session and its history."""
        session_data = self._store.get_session(session_id)
        if not session_data:
            return None

        messages = self._store.get_messages(session_id)
        history = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

        self._active = ActiveSession(
            id=session_data["id"],
            name=session_data["name"],
            model=session_data["model"],
            skill=session_data.get("skill"),
            history=history,
        )

        logger.info("Loaded session: %s (%d messages)", session_data["name"], len(history))
        return self._active

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions."""
        return self._store.list_sessions(limit=limit)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if self._active and self._active.id == session_id:
            self._active = None
        return self._store.delete_session(session_id)

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session. Updates active session name if it matches."""
        result = self._store.rename_session(session_id, new_name)
        if result and self._active and self._active.id == session_id:
            self._active.name = new_name
        return result

    def search_sessions(self, query: str) -> list[dict[str, Any]]:
        """Search sessions by name or message content."""
        return self._store.search_sessions(query)

    def get_last_session(self) -> dict[str, Any] | None:
        """Get the most recently updated session."""
        return self._store.get_last_session()

    def resume_last_session(self) -> ActiveSession | None:
        """Load the most recently updated session."""
        last = self._store.get_last_session()
        if not last:
            return None
        return self.load_session(last["id"])

    def clear_history(self) -> None:
        """Clear the active session's history (memory only, DB preserved)."""
        if self._active:
            self._active.history.clear()

    def save_cost_data(self, cost_data: dict[str, Any]) -> bool:
        """Persist cost tracker data to the active session's metadata.

        Args:
            cost_data: Serialized cost tracker state (from ``CostTracker.to_dict()``).

        Returns:
            True if saved successfully, False if no active session or update failed.
        """
        if not self._active:
            return False
        return self._store.update_metadata(self._active.id, {"cost_data": cost_data})

    def load_cost_data(self, session_id: str) -> dict[str, Any] | None:
        """Load cost tracker data from a session's metadata.

        Args:
            session_id: The session to load cost data from.

        Returns:
            The cost data dict, or None if no cost data exists.
        """
        metadata = self._store.get_metadata(session_id)
        if metadata is None:
            return None
        return metadata.get("cost_data")

    def close(self) -> None:
        """Close the session store."""
        # Telemetry: emit session_end event via EventBus
        try:
            from vaig.core.event_bus import EventBus
            from vaig.core.events import SessionEnded

            session_id = self._active.id if self._active else ""
            EventBus.get().emit(SessionEnded(session_id=session_id))
        except Exception:  # noqa: BLE001
            pass

        # Flush telemetry collector
        try:
            from vaig.core.telemetry import get_telemetry_collector

            get_telemetry_collector().flush()
        except Exception:  # noqa: BLE001
            pass

        self._store.close()

    # ── Async public methods ─────────────────────────────────

    async def async_new_session(
        self,
        name: str,
        *,
        model: str | None = None,
        skill: str | None = None,
    ) -> ActiveSession:
        """Async version of :meth:`new_session`."""
        model = model or self._settings.models.default

        session_id = await self._store.async_create_session(
            name=name,
            model=model,
            skill=skill,
        )

        self._active = ActiveSession(
            id=session_id,
            name=name,
            model=model,
            skill=skill,
        )

        logger.info("New session started (async): %s (model=%s, skill=%s)", name, model, skill)

        # Telemetry: emit session_start event via EventBus
        try:
            from vaig.core.event_bus import EventBus
            from vaig.core.events import SessionStarted

            EventBus.get().emit(
                SessionStarted(
                    session_id=session_id,
                    name=name,
                    model=model,
                    skill=skill or "",
                )
            )
        except Exception:  # noqa: BLE001
            pass

        return self._active

    async def async_add_message(
        self, role: str, content: str, *, model: str | None = None, token_count: int = 0
    ) -> None:
        """Async version of :meth:`add_message`."""
        if not self._active:
            msg = "No active session. Start one with /new or load an existing session."
            raise RuntimeError(msg)

        message = ChatMessage(role=role, content=content)
        self._active.history.append(message)

        # Persist
        if self._settings.session.auto_save:
            await self._store.async_add_message(
                session_id=self._active.id,
                role=role,
                content=content,
                model=model,
                token_count=token_count,
            )

        # Trim history if needed
        max_msgs = self._settings.session.max_history_messages
        if len(self._active.history) > max_msgs:
            self._active.history = self._active.history[-max_msgs:]

    async def async_load_session(self, session_id: str) -> ActiveSession | None:
        """Async version of :meth:`load_session`."""
        session_data = await self._store.async_get_session(session_id)
        if not session_data:
            return None

        messages = await self._store.async_get_messages(session_id)
        history = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]

        self._active = ActiveSession(
            id=session_data["id"],
            name=session_data["name"],
            model=session_data["model"],
            skill=session_data.get("skill"),
            history=history,
        )

        logger.info("Loaded session (async): %s (%d messages)", session_data["name"], len(history))
        return self._active

    async def async_list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Async version of :meth:`list_sessions`."""
        return await self._store.async_list_sessions(limit=limit)

    async def async_get_session_messages(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Async version of store's ``get_messages`` — exposed at manager level."""
        return await self._store.async_get_messages(session_id, limit=limit)

    async def async_delete_session(self, session_id: str) -> bool:
        """Async version of :meth:`delete_session`."""
        if self._active and self._active.id == session_id:
            self._active = None
        return await self._store.async_delete_session(session_id)

    async def async_rename_session(self, session_id: str, new_name: str) -> bool:
        """Async version of :meth:`rename_session`."""
        result = await self._store.async_rename_session(session_id, new_name)
        if result and self._active and self._active.id == session_id:
            self._active.name = new_name
        return result

    async def async_search_sessions(self, query: str) -> list[dict[str, Any]]:
        """Async version of :meth:`search_sessions`."""
        return await self._store.async_search_sessions(query)

    async def async_get_last_session(self) -> dict[str, Any] | None:
        """Async version of :meth:`get_last_session`."""
        return await self._store.async_get_last_session()

    async def async_resume_last_session(self) -> ActiveSession | None:
        """Async version of :meth:`resume_last_session`."""
        last = await self._store.async_get_last_session()
        if not last:
            return None
        return await self.async_load_session(last["id"])

    async def async_save_cost_data(self, cost_data: dict[str, Any]) -> bool:
        """Async version of :meth:`save_cost_data`."""
        if not self._active:
            return False
        return await self._store.async_update_metadata(self._active.id, {"cost_data": cost_data})

    async def async_load_cost_data(self, session_id: str) -> dict[str, Any] | None:
        """Async version of :meth:`load_cost_data`."""
        metadata = await self._store.async_get_metadata(session_id)
        if metadata is None:
            return None
        return metadata.get("cost_data")

    async def async_close(self) -> None:
        """Async version of :meth:`close`."""
        # Telemetry: emit session_end event via EventBus
        try:
            from vaig.core.event_bus import EventBus
            from vaig.core.events import SessionEnded

            session_id = self._active.id if self._active else ""
            EventBus.get().emit(SessionEnded(session_id=session_id))
        except Exception:  # noqa: BLE001
            pass

        # Flush telemetry collector
        try:
            from vaig.core.telemetry import get_telemetry_collector

            await get_telemetry_collector().async_flush()
        except Exception:  # noqa: BLE001
            pass

        await self._store.async_close()

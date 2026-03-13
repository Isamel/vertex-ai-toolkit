"""Session manager — high-level session management with chat history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from vaig.core.client import ChatMessage
from vaig.core.config import Settings
from vaig.session.store import SessionStore

logger = logging.getLogger(__name__)


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

    def load_session(self, session_id: str) -> ActiveSession | None:
        """Load an existing session and its history."""
        session_data = self._store.get_session(session_id)
        if not session_data:
            return None

        messages = self._store.get_messages(session_id)
        history = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]

        self._active = ActiveSession(
            id=session_data["id"],
            name=session_data["name"],
            model=session_data["model"],
            skill=session_data.get("skill"),
            history=history,
        )

        logger.info("Loaded session: %s (%d messages)", session_data["name"], len(history))
        return self._active

    def list_sessions(self, *, limit: int = 20) -> list[dict]:
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

    def search_sessions(self, query: str) -> list[dict]:
        """Search sessions by name or message content."""
        return self._store.search_sessions(query)

    def get_last_session(self) -> dict | None:
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

    def close(self) -> None:
        """Close the session store."""
        self._store.close()

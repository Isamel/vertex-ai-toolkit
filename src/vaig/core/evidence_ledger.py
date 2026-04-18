"""Evidence Ledger — per-run structured log of tool call results.

Pydantic models for tracking what tools have been called, what they returned,
and whether their output supports or contradicts specific claims.

Zero internal vaig.* imports — only stdlib + Pydantic.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _hash_tool_args(tool_args: dict[str, Any]) -> str:
    """Return a 16-char SHA-256 prefix of the sorted JSON-encoded tool args."""
    return hashlib.sha256(
        json.dumps(tool_args, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


class EvidenceEntry(BaseModel):
    """A single tool-call result recorded in the evidence ledger.

    Attributes:
        id: Auto-generated UUID4 hex string.
        timestamp: Auto-generated ISO 8601 UTC string.
        source_agent: Name of the agent that executed the tool.
        tool_name: Name of the tool that was called.
        tool_args_hash: 16-char SHA-256 prefix of sorted JSON-encoded tool args.
        question: Optional question/query context (may be empty).
        answer_summary: First 500 characters of the tool output.
        raw_output_ref: ToolCallStore reference key (``run_id.jsonl``) for full record.
        supports: Tuple of claim strings this evidence supports.
        contradicts: Tuple of claim strings this evidence contradicts.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_uuid)
    timestamp: str = Field(default_factory=_utc_now_iso)
    source_agent: str = ""
    tool_name: str = ""
    tool_args_hash: str = ""
    question: str = ""
    answer_summary: str = ""
    raw_output_ref: str = ""
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()

    def model_post_init(self, __context: Any) -> None:
        """Truncate answer_summary to 500 chars at construction time."""
        if len(self.answer_summary) > 500:
            # frozen model: use object.__setattr__ to bypass immutability during init
            object.__setattr__(self, "answer_summary", self.answer_summary[:500])


class EvidenceLedger(BaseModel):
    """Immutable, append-only log of EvidenceEntry records for one pipeline run.

    Use ``append()`` to create a new ledger with one more entry.
    Use ``search()`` / ``already_answered()`` to query existing evidence.

    Attributes:
        entries: Immutable tuple of all evidence entries recorded so far.
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[EvidenceEntry, ...] = Field(default_factory=tuple)

    def append(self, entry: EvidenceEntry) -> EvidenceLedger:
        """Return a new EvidenceLedger with *entry* appended."""
        return self.model_copy(update={"entries": (*self.entries, entry)})

    def search(self, q: str) -> list[EvidenceEntry]:
        """Return entries where *q* matches tool_name, question, or answer_summary (case-insensitive).

        Empty *q* returns all entries.
        """
        if not q:
            return list(self.entries)
        q_lower = q.lower()
        return [
            e for e in self.entries
            if q_lower in e.tool_name.lower()
            or q_lower in e.question.lower()
            or q_lower in e.answer_summary.lower()
        ]

    def already_answered(self, question: str) -> list[EvidenceEntry]:
        """Return all entries matching *question* — non-empty means already answered."""
        return self.search(question)

    def to_summary(self, max_entries: int = 10) -> str:
        """Return a concise multi-line summary of the last *max_entries* entries.

        Each line has the format::

            - [{tool_name}] {question}: {answer_summary[:100]}

        Returns an empty string when the ledger has no entries.
        """
        if not self.entries:
            return ""
        lines = [
            f"- [{e.tool_name}] {e.question}: {e.answer_summary[:100]}"
            for e in self.entries[-max_entries:]
        ]
        return "\n".join(lines)


def new_ledger() -> EvidenceLedger:
    """Create a fresh empty EvidenceLedger."""
    return EvidenceLedger()

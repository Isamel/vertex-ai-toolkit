"""Domain models for remote code migration (Phase 6 — GH-03).

Provides:
- :class:`Phase8RequiredError` — stub sentinel for write operations.
- :class:`ProvenanceMetadata` — tracks the origin of migrated code.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Phase8RequiredError(Exception):
    """Raised when a feature requires Phase 8 (CM-05 Git Integration).

    Write-path operations (``--to-repo``, ``--push``) are stubbed in Phase 6
    and will be wired in Phase 8 (CM-05).  Callers MUST catch this exception
    and surface a clear message to the user rather than silently failing.

    Args:
        feature: Short description of the requested feature.
    """

    def __init__(self, feature: str = "git write operations") -> None:
        super().__init__(f"'{feature}' requires Phase 8 CM-05 Git Integration")


class ProvenanceMetadata(BaseModel):
    """Tracks the origin of migrated code from a remote repository.

    Attached to migration results when ``--from-repo`` is used so that
    downstream consumers can trace every migrated file back to its source.

    Attributes:
        source_repo: ``owner/repo`` identifier of the source repository.
        source_ref: Branch, tag, or commit SHA that was cloned.
        source_path: Repository-relative path of the source file.
        migrated_at: UTC timestamp when the migration was performed.
    """

    source_repo: str
    source_ref: str
    source_path: str
    migrated_at: datetime

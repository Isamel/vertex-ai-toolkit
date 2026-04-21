"""Migration state: tracks per-file progress across orchestrator iterations."""
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["FileStatus", "FileRecord", "MigrationState"]


class FileStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileRecord(BaseModel):
    source_path: str
    target_path: str | None = None
    status: FileStatus = FileStatus.PENDING
    gate_results: list[dict[str, object]] = Field(default_factory=list)  # serialized GateResult dicts
    error: str | None = None
    completed_at: str | None = None  # ISO datetime string


class MigrationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    change_id: str  # uuid
    created_at: str  # ISO datetime
    updated_at: str  # ISO datetime
    source_kind: str
    target_kind: str
    files: dict[str, FileRecord] = Field(default_factory=dict)  # keyed by source_path str

    def mark_completed(
        self, source_path: str, target_path: str, gate_results: list[object]
    ) -> None:
        now = datetime.now(UTC).isoformat()
        record = self.files.get(source_path)
        if record is None:
            record = FileRecord(source_path=source_path)
        record.status = FileStatus.COMPLETED
        record.target_path = target_path
        record.gate_results = [r for r in gate_results if isinstance(r, dict)]
        record.completed_at = now
        record.error = None
        self.files[source_path] = record
        self.updated_at = now

    def mark_failed(self, source_path: str, error: str) -> None:
        now = datetime.now(UTC).isoformat()
        record = self.files.get(source_path)
        if record is None:
            record = FileRecord(source_path=source_path)
        record.status = FileStatus.FAILED
        record.error = error
        self.files[source_path] = record
        self.updated_at = now

    def pending_files(self) -> list[str]:
        return [
            path
            for path, record in self.files.items()
            if record.status not in (FileStatus.COMPLETED, FileStatus.SKIPPED)
        ]

    def is_complete(self) -> bool:
        if not self.files:
            return False
        return all(
            r.status in (FileStatus.COMPLETED, FileStatus.SKIPPED)
            for r in self.files.values()
        )

    def save(self, path: Path) -> None:
        """Write state as JSON to path."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MigrationState":
        """Load state from JSON file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def new(cls, source_kind: str, target_kind: str) -> "MigrationState":
        """Create fresh state with new UUID and timestamps."""
        now = datetime.now(UTC).isoformat()
        return cls(
            change_id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            source_kind=source_kind,
            target_kind=target_kind,
        )

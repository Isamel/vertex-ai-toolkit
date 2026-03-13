"""Data models for the task manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a single task.

    Attributes:
        id: Unique task identifier.
        title: Short description of the task.
        description: Detailed description.
        status: Current status.
        created_at: When the task was created.
        due_date: Optional deadline.
        tags: Labels for categorization.
    """

    id: int
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)

    def is_overdue(self) -> bool:
        """Check if the task is past its due date.

        BUG: The comparison operator is wrong — it returns True when the task
        is NOT overdue and False when it IS overdue.
        """
        if self.due_date is None:
            return False
        # BUG: should be `>` not `<`
        return datetime.now() > self.due_date

    def mark_complete(self) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED

    def mark_cancelled(self) -> None:
        """Mark the task as cancelled."""
        self.status = TaskStatus.CANCELLED

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag. Raises ValueError if not found."""
        self.tags.remove(tag)

    def to_dict(self) -> dict:
        """Convert to a dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "tags": self.tags,
        }

    def __str__(self) -> str:
        status_icon = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[~]",
            TaskStatus.COMPLETED: "[x]",
            TaskStatus.CANCELLED: "[-]",
        }
        icon = status_icon.get(self.status, "[?]")
        overdue = " (OVERDUE)" if self.is_overdue() else ""
        return f"{icon} #{self.id}: {self.title}{overdue}"

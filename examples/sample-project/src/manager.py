"""Task manager — CRUD operations for tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.models import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages a collection of tasks with persistence.

    Attributes:
        tasks: In-memory list of tasks.
        storage_path: Path to the JSON file for persistence.
    """

    def __init__(self, storage_path: str = "tasks.json") -> None:
        self.tasks: list[Task] = []
        self.storage_path = Path(storage_path)
        self._next_id = 1

    def add_task(self, title: str, description: str = "", tags: list[str] | None = None) -> Task:
        """Create and add a new task.

        Args:
            title: Short description of the task.
            description: Detailed description.
            tags: Optional list of tags.

        Returns:
            The newly created Task.

        Raises:
            ValueError: If title is empty.
        """
        if not title.strip():
            raise ValueError("Task title cannot be empty")

        task = Task(
            id=self._next_id,
            title=title.strip(),
            description=description.strip(),
            tags=tags or [],
        )
        self.tasks.append(task)
        self._next_id += 1
        logger.info("Created task #%d: %s", task.id, task.title)
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        """Find a task by ID.

        Args:
            task_id: The task ID to search for.

        Returns:
            The Task if found, None otherwise.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def update_task(self, task_id: int, **kwargs) -> Task:
        """Update a task's fields.

        Args:
            task_id: The task ID to update.
            **kwargs: Fields to update (title, description, status, tags).

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task is not found.
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Task #{task_id} not found")

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
            else:
                raise ValueError(f"Unknown field: {key}")

        logger.info("Updated task #%d", task_id)
        return task

    def delete_task(self, task_id: int) -> None:
        """Delete a task by ID.

        Args:
            task_id: The ID of the task to delete.

        Raises:
            KeyError: If the task is not found.
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Task #{task_id} not found")

        self.tasks.remove(task)
        logger.info("Deleted task #%d", task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        tag: Optional[str] = None,
    ) -> list[Task]:
        """List tasks with optional filtering.

        Args:
            status: Filter by status.
            tag: Filter by tag.

        Returns:
            List of matching tasks.
        """
        result = self.tasks

        if status is not None:
            result = [t for t in result if t.status == status]

        if tag is not None:
            result = [t for t in result if tag in t.tags]

        return result

    def complete_task(self, task_id: int) -> Task:
        """Mark a task as completed.

        Raises:
            KeyError: If the task is not found.
        """
        task = self.get_task(task_id)
        if task is None:
            raise KeyError(f"Task #{task_id} not found")

        task.mark_complete()
        logger.info("Completed task #%d: %s", task.id, task.title)
        return task

    def search_tasks(self, query: str) -> list[Task]:
        """Search tasks by title or description.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching tasks.
        """
        query_lower = query.lower()
        return [
            t
            for t in self.tasks
            if query_lower in t.title.lower() or query_lower in t.description.lower()
        ]

    def save(self) -> None:
        """Save tasks to the JSON file."""
        data = [t.to_dict() for t in self.tasks]
        self.storage_path.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d tasks to %s", len(data), self.storage_path)

    def stats(self) -> dict[str, int]:
        """Get task statistics.

        Returns:
            Dictionary with counts per status.
        """
        counts: dict[str, int] = {}
        for status in TaskStatus:
            counts[status.value] = len(
                [t for t in self.tasks if t.status == status]
            )
        counts["total"] = len(self.tasks)
        counts["overdue"] = len([t for t in self.tasks if t.is_overdue()])
        return counts

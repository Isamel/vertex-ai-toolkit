"""Output formatting for task display."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import Task


def format_task_table(tasks: list[Task]) -> str:
    """Format a list of tasks as an ASCII table.

    Args:
        tasks: List of tasks to display.

    Returns:
        Formatted table string.
    """
    if not tasks:
        return "No tasks found."

    # TODO: Add column for due date and tags
    # TODO: Add color coding for overdue tasks
    header = f"{'ID':>4} {'Status':<12} {'Title':<40}"
    separator = "-" * len(header)
    rows = []

    for task in tasks:
        status_display = task.status.value.replace("_", " ").title()
        title = task.title[:40] if len(task.title) > 40 else task.title
        rows.append(f"{task.id:>4} {status_display:<12} {title:<40}")

    return "\n".join([header, separator, *rows])


def format_task_detail(task: Task) -> str:
    """Format a single task with full details.

    Args:
        task: The task to display.

    Returns:
        Formatted detail string.
    """
    lines = [
        f"Task #{task.id}",
        f"  Title:       {task.title}",
        f"  Status:      {task.status.value}",
        f"  Created:     {task.created_at.strftime('%Y-%m-%d %H:%M')}",
    ]

    if task.description:
        lines.append(f"  Description: {task.description}")

    if task.due_date:
        lines.append(f"  Due Date:    {task.due_date.strftime('%Y-%m-%d %H:%M')}")
        if task.is_overdue():
            lines.append("  *** OVERDUE ***")

    if task.tags:
        lines.append(f"  Tags:        {', '.join(task.tags)}")

    return "\n".join(lines)


def format_stats(stats: dict[str, int]) -> str:
    """Format task statistics.

    Args:
        stats: Dictionary of status counts.

    Returns:
        Formatted stats string.
    """
    lines = ["Task Statistics", "=" * 20]
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        lines.append(f"  {label:<15} {value:>4}")
    return "\n".join(lines)

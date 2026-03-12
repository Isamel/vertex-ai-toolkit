"""Tests for the TaskManager."""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.manager import TaskManager
from src.models import TaskStatus


def test_add_task():
    """Test adding a new task."""
    manager = TaskManager()
    task = manager.add_task("Buy groceries", description="Milk, eggs, bread")
    assert task.id == 1
    assert task.title == "Buy groceries"
    assert len(manager.tasks) == 1


def test_add_task_empty_title():
    """Test that empty title raises ValueError."""
    manager = TaskManager()
    try:
        manager.add_task("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_get_task():
    """Test finding a task by ID."""
    manager = TaskManager()
    manager.add_task("Task 1")
    manager.add_task("Task 2")
    task = manager.get_task(2)
    assert task is not None
    assert task.title == "Task 2"


def test_get_task_not_found():
    """Test finding a non-existent task."""
    manager = TaskManager()
    assert manager.get_task(999) is None


def test_update_task():
    """Test updating task fields."""
    manager = TaskManager()
    manager.add_task("Original title")
    updated = manager.update_task(1, title="New title")
    assert updated.title == "New title"


def test_list_tasks_filter_by_status():
    """Test filtering tasks by status."""
    manager = TaskManager()
    manager.add_task("Task 1")
    manager.add_task("Task 2")
    manager.complete_task(1)

    pending = manager.list_tasks(status=TaskStatus.PENDING)
    assert len(pending) == 1
    assert pending[0].title == "Task 2"


def test_list_tasks_filter_by_tag():
    """Test filtering tasks by tag."""
    manager = TaskManager()
    manager.add_task("Bug fix", tags=["bug"])
    manager.add_task("Feature", tags=["feature"])
    manager.add_task("Urgent bug", tags=["bug", "urgent"])

    bugs = manager.list_tasks(tag="bug")
    assert len(bugs) == 2


def test_search_tasks():
    """Test searching tasks by keyword."""
    manager = TaskManager()
    manager.add_task("Fix login bug", description="Users can't login")
    manager.add_task("Add dark mode")
    manager.add_task("Update login page")

    results = manager.search_tasks("login")
    assert len(results) == 2


def test_complete_task():
    """Test completing a task."""
    manager = TaskManager()
    manager.add_task("Test task")
    completed = manager.complete_task(1)
    assert completed.status == TaskStatus.COMPLETED


def test_complete_task_not_found():
    """Test completing a non-existent task."""
    manager = TaskManager()
    try:
        manager.complete_task(999)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# BUG: This test has wrong assertion — it tests delete_task which doesn't exist yet
def test_delete_task():
    """Test deleting a task."""
    manager = TaskManager()
    manager.add_task("To delete")
    manager.add_task("To keep")

    # BUG: delete_task method doesn't exist yet — this will raise AttributeError
    manager.delete_task(1)
    assert len(manager.tasks) == 1
    assert manager.tasks[0].title == "To keep"


def test_stats():
    """Test task statistics."""
    manager = TaskManager()
    manager.add_task("Task 1")
    manager.add_task("Task 2")
    manager.add_task("Task 3")
    manager.complete_task(1)

    stats = manager.stats()
    assert stats["total"] == 3
    assert stats["completed"] == 1
    assert stats["pending"] == 2

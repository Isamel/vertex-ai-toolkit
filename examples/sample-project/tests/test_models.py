"""Tests for the Task model."""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Task, TaskStatus


def test_task_creation():
    """Test basic task creation with defaults."""
    task = Task(id=1, title="Test task")
    assert task.id == 1
    assert task.title == "Test task"
    assert task.status == TaskStatus.PENDING
    assert task.tags == []
    assert task.due_date is None


def test_task_mark_complete():
    """Test marking a task as completed."""
    task = Task(id=1, title="Test")
    task.mark_complete()
    assert task.status == TaskStatus.COMPLETED


def test_task_mark_cancelled():
    """Test marking a task as cancelled."""
    task = Task(id=1, title="Test")
    task.mark_cancelled()
    assert task.status == TaskStatus.CANCELLED


def test_task_add_tag():
    """Test adding tags."""
    task = Task(id=1, title="Test")
    task.add_tag("urgent")
    assert "urgent" in task.tags

    # Adding same tag again should not duplicate
    task.add_tag("urgent")
    assert task.tags.count("urgent") == 1


def test_task_remove_tag():
    """Test removing tags."""
    task = Task(id=1, title="Test", tags=["urgent", "bug"])
    task.remove_tag("urgent")
    assert "urgent" not in task.tags
    assert "bug" in task.tags


def test_task_to_dict():
    """Test serialization to dictionary."""
    task = Task(id=1, title="Test", description="A test task")
    data = task.to_dict()
    assert data["id"] == 1
    assert data["title"] == "Test"
    assert data["status"] == "pending"


def test_task_str_representation():
    """Test string representation."""
    task = Task(id=1, title="My Task")
    assert "[ ] #1: My Task" in str(task)

    task.mark_complete()
    assert "[x] #1: My Task" in str(task)


def test_task_is_overdue_no_due_date():
    """Test is_overdue returns False when no due date is set."""
    task = Task(id=1, title="Test")
    assert task.is_overdue() is False


# NOTE: The test below will FAIL because of the bug in models.py
# The is_overdue method has the comparison operator reversed
def test_task_is_overdue_past_date():
    """Test is_overdue returns True when due date is in the past."""
    past = datetime.now() - timedelta(days=1)
    task = Task(id=1, title="Test", due_date=past)
    # This SHOULD be True, but the bug makes it False
    assert task.is_overdue() is True


def test_task_is_overdue_future_date():
    """Test is_overdue returns False when due date is in the future."""
    future = datetime.now() + timedelta(days=1)
    task = Task(id=1, title="Test", due_date=future)
    assert task.is_overdue() is False

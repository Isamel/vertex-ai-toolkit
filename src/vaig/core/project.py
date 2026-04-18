"""Project-local directory utilities — create and manage the `.vaig/` directory."""

from __future__ import annotations

from pathlib import Path


def ensure_project_dir() -> Path:
    """Create `.vaig/` in the current working directory if absent and return its path.

    This is idempotent — calling it multiple times is safe and returns the same path.

    Returns:
        Absolute path to the `.vaig/` directory.
    """
    project_dir = Path.cwd() / ".vaig"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def ensure_project_subdir(name: str) -> Path:
    """Create `.vaig/{name}/` in the current working directory and return its path.

    Args:
        name: Subdirectory name within `.vaig/`.

    Returns:
        Absolute path to the `.vaig/{name}/` directory.
    """
    subdir = ensure_project_dir() / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir

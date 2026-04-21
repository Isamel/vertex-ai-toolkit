"""ReadOnlyFilesystemJail: path-validation jail for read-only source directories."""
from __future__ import annotations

from pathlib import Path

__all__ = ["ReadOnlyFilesystemJail", "ReadOnlySourceError"]


class ReadOnlySourceError(PermissionError):
    """Raised when an agent attempts to write to a read-only source directory."""


class ReadOnlyFilesystemJail:
    """A read-only view of a directory.

    Does NOT mount anything — pure path validation + read helpers.
    Raises ReadOnlySourceError on any path that escapes the root.
    """

    def __init__(self, path: str | Path) -> None:
        self.root = Path(path).resolve()

    def __enter__(self) -> ReadOnlyFilesystemJail:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def _resolve_safe(self, relative_path: str) -> Path:
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            raise ReadOnlySourceError(
                f"Path '{relative_path}' escapes read-only jail '{self.root}'"
            ) from None
        return target

    def safe_read(self, relative_path: str) -> str:
        """Read a file within the jail root. Raises ReadOnlySourceError on traversal."""
        target = self._resolve_safe(relative_path)
        return target.read_text(encoding="utf-8")

    def check_write_blocked(self, absolute_path: str | Path) -> None:
        """Raise ReadOnlySourceError if absolute_path falls inside this jail."""
        target = Path(absolute_path).resolve()
        try:
            target.relative_to(self.root)
            raise ReadOnlySourceError(
                f"Write to '{absolute_path}' is blocked — path is inside read-only source jail '{self.root}'"
            )
        except ReadOnlySourceError:
            raise
        except ValueError:
            pass  # outside jail — fine

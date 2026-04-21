"""Read-only filesystem jail — path validation and safe file reading."""

from __future__ import annotations

from pathlib import Path


class ReadOnlyFilesystemJail:
    """Context manager that constrains file access to a single directory tree.

    Only provides read access; write operations are intentionally absent.

    Usage::

        with ReadOnlyFilesystemJail("/some/path") as jail:
            content = jail.safe_read("subdir/file.py")
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.root: Path  # set on __enter__

    def __enter__(self) -> ReadOnlyFilesystemJail:
        self.root = self._path.resolve()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass  # nothing to tear down — we never mount anything

    def safe_read(self, relative_path: str) -> str:
        """Read a file relative to the jail root.

        Raises
        ------
        PermissionError
            If the resolved path escapes the jail root (path traversal).
        FileNotFoundError
            If the file does not exist inside the jail.
        """
        target = (self.root / relative_path).resolve()
        # Path traversal check: target must be equal to or inside self.root
        try:
            target.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                f"Path traversal detected: '{relative_path}' resolves outside jail root '{self.root}'"
            ) from None
        return target.read_text(encoding="utf-8")

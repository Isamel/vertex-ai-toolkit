"""Context builder — ingests directories and files into a unified context for Gemini."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from google.genai import types
from rich.console import Console
from rich.table import Table

from vaig.context.filters import build_file_filter, should_include_file
from vaig.context.loader import FileType, LoadedFile, load_file
from vaig.core.config import DEFAULT_CHARS_PER_TOKEN, Settings

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ContextBundle:
    """A bundle of files ready to be sent to Gemini as context."""

    files: list[LoadedFile] = field(default_factory=list)
    total_tokens_estimate: int = 0
    total_size_bytes: int = 0

    def add_file(self, loaded_file: LoadedFile) -> None:
        """Add a loaded file to the bundle."""
        self.files.append(loaded_file)
        self.total_tokens_estimate += loaded_file.token_estimate
        self.total_size_bytes += loaded_file.size_bytes

    def to_parts(self) -> list[types.Part | str]:
        """Convert all files to a list of Gemini-compatible parts.

        Text/code files are concatenated as a single string part.
        Binary files (images, audio, PDF) are individual Part objects.
        """
        parts: list[types.Part | str] = []
        text_sections: list[str] = []

        for f in self.files:
            if f.content is not None:
                # Text-based file — accumulate
                text_sections.append(f"## File: {f.path}\n\n{f.content}")
            elif f.part is not None:
                # Flush accumulated text first
                if text_sections:
                    parts.append("\n\n---\n\n".join(text_sections))
                    text_sections.clear()
                # Add binary part with context
                parts.append(f"[Attached file: {f.path} ({f.mime_type})]")
                parts.append(f.part)

        # Flush remaining text
        if text_sections:
            parts.append("\n\n---\n\n".join(text_sections))

        return parts

    def to_context_string(self) -> str:
        """Get a text-only representation (for skills/agents that need raw text)."""
        sections: list[str] = []
        for f in self.files:
            if f.content is not None:
                sections.append(f"## File: {f.path}\n\n{f.content}")
            else:
                sections.append(f"## File: {f.path} [{f.mime_type}, {f.size_bytes} bytes] (binary — not shown)")
        return "\n\n---\n\n".join(sections)

    def summary_table(self) -> Table:
        """Generate a rich Table summarizing the loaded context."""
        table = Table(title="📁 Context Bundle", show_lines=False)
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Size", justify="right", style="green")
        table.add_column("Tokens (est.)", justify="right", style="yellow")

        for f in self.files:
            size_str = _format_size(f.size_bytes)
            table.add_row(str(f.path), f.file_type.value, size_str, f"{f.token_estimate:,}")

        table.add_section()
        table.add_row(
            f"[bold]{len(self.files)} files[/bold]",
            "",
            f"[bold]{_format_size(self.total_size_bytes)}[/bold]",
            f"[bold]{self.total_tokens_estimate:,}[/bold]",
        )
        return table

    @property
    def file_count(self) -> int:
        """Number of files in the bundle."""
        return len(self.files)

    def clear(self) -> None:
        """Remove all files from the bundle."""
        self.files.clear()
        self.total_tokens_estimate = 0
        self.total_size_bytes = 0


class ContextBuilder:
    """Builds a ContextBundle from directories, files, and raw text."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bundle = ContextBundle()

    @property
    def bundle(self) -> ContextBundle:
        """Get the current context bundle."""
        return self._bundle

    def add_directory(self, dir_path: str | Path, *, recursive: bool = True) -> int:
        """Add all supported files from a directory.

        Returns the number of files added.
        """
        dir_path = Path(dir_path).resolve()
        if not dir_path.is_dir():
            msg = f"Not a directory: {dir_path}"
            raise FileNotFoundError(msg)

        spec = build_file_filter(self._settings, dir_path)
        added = 0

        pattern = "**/*" if recursive else "*"
        for filepath in sorted(dir_path.glob(pattern)):
            if not filepath.is_file():
                continue

            if should_include_file(filepath, settings=self._settings, spec=spec, root_dir=dir_path):
                try:
                    loaded = load_file(filepath)
                    # Use relative path for display
                    loaded.path = filepath.relative_to(dir_path)
                    self._bundle.add_file(loaded)
                    added += 1
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to load file: %s", filepath, exc_info=True)

        logger.info("Added %d files from %s", added, dir_path)
        return added

    def add_file(self, file_path: str | Path) -> LoadedFile:
        """Add a single file to the context."""
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            msg = f"Not a file: {file_path}"
            raise FileNotFoundError(msg)

        loaded = load_file(file_path)
        self._bundle.add_file(loaded)
        logger.info("Added file: %s (%s)", file_path, loaded.file_type)
        return loaded

    def add_text(self, text: str, *, label: str = "inline") -> LoadedFile:
        """Add raw text as context (e.g., pasted logs, error messages)."""
        loaded = LoadedFile(
            path=Path(label),
            file_type=FileType.TEXT,
            content=text,
            size_bytes=len(text.encode()),
            token_estimate=int(len(text) / DEFAULT_CHARS_PER_TOKEN),
        )
        self._bundle.add_file(loaded)
        return loaded

    def clear(self) -> None:
        """Clear all context."""
        self._bundle.clear()

    def show_summary(self) -> None:
        """Print a summary table of the current context."""
        if self._bundle.file_count == 0:
            console.print("[yellow]No files loaded in context.[/yellow]")
            return
        console.print(self._bundle.summary_table())

    # ── Async Methods ─────────────────────────────────────────
    # These mirror the sync methods above but use ``asyncio.to_thread()``
    # for blocking file I/O so the event loop stays unblocked.

    async def async_add_directory(self, dir_path: str | Path, *, recursive: bool = True) -> int:
        """Async version of :meth:`add_directory`.

        Runs the blocking file I/O in a thread via ``asyncio.to_thread()``.

        Returns the number of files added.
        """
        return await asyncio.to_thread(self.add_directory, dir_path, recursive=recursive)

    async def async_add_file(self, file_path: str | Path) -> LoadedFile:
        """Async version of :meth:`add_file`.

        Runs the blocking file I/O in a thread via ``asyncio.to_thread()``.
        """
        return await asyncio.to_thread(self.add_file, file_path)

    async def async_add_text(self, text: str, *, label: str = "inline") -> LoadedFile:
        """Async version of :meth:`add_text`.

        While ``add_text()`` doesn't do disk I/O, this async wrapper
        exists for API symmetry so callers can ``await`` any add method
        consistently.
        """
        return self.add_text(text, label=label)


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    size: float = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

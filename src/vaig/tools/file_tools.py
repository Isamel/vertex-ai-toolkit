"""File tools — read, write, edit, list, search, and verify files within a workspace."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from vaig.tools.base import ToolDef, ToolParam, ToolResult
from vaig.tools.categories import CODING

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE = 1_048_576  # 1 MB

_IGNORED_NAMES: frozenset[str] = frozenset({
    "__pycache__",
    "node_modules",
    ".git",
    ".venv",
})

_IGNORED_SUFFIXES: frozenset[str] = frozenset({
    ".pyc",
})


def _is_ignored(name: str) -> bool:
    """Return True if the name should be skipped during listing/search."""
    if name.startswith("."):
        return True
    if name in _IGNORED_NAMES:
        return True
    if any(name.endswith(s) for s in _IGNORED_SUFFIXES):
        return True
    if name.endswith(".egg-info"):
        return True
    return False


# ── Task 2.7 — Path safety utility ──────────────────────────


def _resolve_safe_path(path: str, workspace: Path) -> Path | None:
    """Resolve *path* relative to *workspace* and reject traversal attempts.

    Returns the resolved ``Path`` on success, or ``None`` if the resulting
    path falls outside the workspace root.
    """
    resolved_workspace = workspace.resolve()
    # Reject absolute paths that are clearly outside workspace
    candidate = (resolved_workspace / path).resolve()
    if not candidate.is_relative_to(resolved_workspace):
        return None
    return candidate


# ── Task 2.1 — read_file ────────────────────────────────────


def read_file(path: str, *, workspace: Path) -> ToolResult:
    """Read the contents of a file within the workspace."""
    logger.debug("read_file: path=%s workspace=%s", path, workspace)

    resolved = _resolve_safe_path(path, workspace)
    if resolved is None:
        return ToolResult(
            output=f"Path safety error: '{path}' resolves outside the workspace.",
            error=True,
        )

    try:
        size = resolved.stat().st_size
    except FileNotFoundError:
        return ToolResult(output=f"File not found: {path}", error=True)

    if size > _MAX_FILE_SIZE:
        size_kb = size / 1024
        return ToolResult(
            output=f"File too large: {path} ({size_kb:.1f} KB). Maximum supported size is 1 MB.",
            error=True,
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(
            output=f"Cannot read binary file: {path}",
            error=True,
        )
    except OSError as exc:
        return ToolResult(output=f"Error reading {path}: {exc}", error=True)

    return ToolResult(output=content)


# ── Task 2.2 — write_file ───────────────────────────────────


def write_file(path: str, content: str, *, workspace: Path) -> ToolResult:
    """Write *content* to a file, creating parent directories as needed."""
    logger.debug("write_file: path=%s workspace=%s", path, workspace)

    resolved = _resolve_safe_path(path, workspace)
    if resolved is None:
        return ToolResult(
            output=f"Path safety error: '{path}' resolves outside the workspace.",
            error=True,
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        written = len(content.encode("utf-8"))
    except OSError as exc:
        return ToolResult(output=f"Error writing {path}: {exc}", error=True)

    return ToolResult(output=f"Wrote {written} bytes to {path}")


# ── Task 2.3 — edit_file ────────────────────────────────────


def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    *,
    workspace: Path,
) -> ToolResult:
    """Apply an exact string replacement in a file (Claude Code pattern).

    Fails if *old_string* is not found or matches more than once.
    """
    logger.debug("edit_file: path=%s workspace=%s", path, workspace)

    resolved = _resolve_safe_path(path, workspace)
    if resolved is None:
        return ToolResult(
            output=f"Path safety error: '{path}' resolves outside the workspace.",
            error=True,
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult(output=f"File not found: {path}", error=True)
    except UnicodeDecodeError:
        return ToolResult(output=f"Cannot edit binary file: {path}", error=True)
    except OSError as exc:
        return ToolResult(output=f"Error reading {path}: {exc}", error=True)

    count = content.count(old_string)

    if count == 0:
        return ToolResult(
            output="old_string not found in file",
            error=True,
        )

    if count > 1:
        return ToolResult(
            output=(
                f"Found {count} matches for old_string. "
                "Provide more surrounding context to identify the correct match."
            ),
            error=True,
        )

    new_content = content.replace(old_string, new_string, 1)

    try:
        resolved.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        return ToolResult(output=f"Error writing {path}: {exc}", error=True)

    return ToolResult(output=f"Successfully edited {path}")


# ── Task 2.4 — list_files ───────────────────────────────────


def list_files(path: str = ".", *, workspace: Path) -> ToolResult:
    """List directory contents, skipping hidden / ignored entries."""
    logger.debug("list_files: path=%s workspace=%s", path, workspace)

    resolved = _resolve_safe_path(path, workspace)
    if resolved is None:
        return ToolResult(
            output=f"Path safety error: '{path}' resolves outside the workspace.",
            error=True,
        )

    if not resolved.is_dir():
        return ToolResult(output=f"Not a directory: {path}", error=True)

    try:
        entries: list[str] = []
        for entry in sorted(resolved.iterdir()):
            if _is_ignored(entry.name):
                continue
            suffix = "/" if entry.is_dir() else ""
            entries.append(f"{entry.name}{suffix}")
    except OSError as exc:
        return ToolResult(output=f"Error listing {path}: {exc}", error=True)

    if not entries:
        return ToolResult(output="(empty directory)")

    return ToolResult(output="\n".join(entries))


# ── Task 2.5 — search_files ─────────────────────────────────

_MAX_MATCHES = 100


def search_files(pattern: str, path: str = ".", *, workspace: Path) -> ToolResult:
    """Search file contents using a regex pattern (recursive grep)."""
    logger.debug("search_files: pattern=%r path=%s workspace=%s", pattern, path, workspace)

    resolved = _resolve_safe_path(path, workspace)
    if resolved is None:
        return ToolResult(
            output=f"Path safety error: '{path}' resolves outside the workspace.",
            error=True,
        )

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(output=f"Invalid regex pattern: {exc}", error=True)

    matches: list[str] = []
    total_found = 0
    resolved_workspace = workspace.resolve()

    for dirpath, dirnames, filenames in os.walk(resolved):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if not _is_ignored(d)]

        for fname in sorted(filenames):
            if _is_ignored(fname):
                continue

            fpath = Path(dirpath) / fname

            try:
                size = fpath.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_SIZE:
                continue

            try:
                content = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            rel = fpath.resolve().relative_to(resolved_workspace)

            for line_no, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    total_found += 1
                    if len(matches) < _MAX_MATCHES:
                        matches.append(f"{rel}:{line_no}:{line}")

    if not matches:
        return ToolResult(output=f"No matches found for pattern: {pattern}")

    result = "\n".join(matches)
    if total_found > _MAX_MATCHES:
        result += f"\n... and {total_found - _MAX_MATCHES} more matches"

    return ToolResult(output=result)


# ── verify_completeness ──────────────────────────────────────

# Patterns that indicate incomplete/placeholder code
_INCOMPLETE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("TODO", re.compile(r"\bTODO\b")),
    ("FIXME", re.compile(r"\bFIXME\b")),
    ("HACK", re.compile(r"\bHACK\b")),
    ("XXX", re.compile(r"\bXXX\b")),
    ("bare pass", re.compile(r"^\s*pass\s*(?:#.*)?$")),
    ("ellipsis body", re.compile(r"^\s*\.\.\.\s*(?:#.*)?$")),
    ("NotImplementedError", re.compile(r"raise\s+NotImplementedError")),
]


def verify_completeness(paths: list[str], *, workspace: Path) -> ToolResult:
    """Scan files for incomplete placeholder patterns.

    Checks each file in *paths* for common indicators of unfinished code:
    ``TODO``, ``FIXME``, ``HACK``, ``XXX``, bare ``pass`` statements,
    ellipsis bodies (``...``), and ``raise NotImplementedError`` expressions.

    Args:
        paths: Relative paths to files to scan (from workspace root).
        workspace: Resolved workspace root path used for path safety checks.

    Returns:
        ToolResult whose ``output`` is a human-readable report in the format
        ``file:line: pattern — 'matched text'`` for each hit, or a
        "No issues found" message when all files are clean.  ``error`` is
        ``True`` only when a path safety or read error occurs; placeholder
        findings do NOT set ``error=True`` — they are informational.
    """
    logger.debug("verify_completeness: paths=%s workspace=%s", paths, workspace)

    if not paths:
        return ToolResult(output="No paths provided — nothing to scan.")

    findings: list[str] = []
    errors: list[str] = []

    for raw_path in paths:
        resolved = _resolve_safe_path(raw_path, workspace)
        if resolved is None:
            errors.append(f"Path safety error: '{raw_path}' resolves outside the workspace.")
            continue

        if not resolved.exists():
            errors.append(f"File not found: {raw_path}")
            continue

        if not resolved.is_file():
            errors.append(f"Not a file: {raw_path}")
            continue

        try:
            size = resolved.stat().st_size
        except OSError as exc:
            errors.append(f"Error reading {raw_path}: {exc}")
            continue

        if size > _MAX_FILE_SIZE:
            size_kb = size / 1024
            errors.append(
                f"File too large: {raw_path} ({size_kb:.1f} KB). "
                f"Maximum supported size is 1 MB."
            )
            continue

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Cannot read binary file: {raw_path}")
            continue
        except OSError as exc:
            errors.append(f"Error reading {raw_path}: {exc}")
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            for pattern_name, pattern_re in _INCOMPLETE_PATTERNS:
                if pattern_re.search(line):
                    snippet = line.strip()[:120]
                    findings.append(
                        f"{raw_path}:{line_no}: {pattern_name} — '{snippet}'"
                    )
                    break  # One match per line is enough

    parts: list[str] = []
    if findings:
        parts.append(f"Found {len(findings)} incomplete pattern(s):\n" + "\n".join(findings))
    else:
        parts.append("No incomplete patterns found — all scanned files look complete.")

    if errors:
        parts.append("Errors:\n" + "\n".join(errors))

    has_errors = bool(errors)
    return ToolResult(output="\n\n".join(parts), error=has_errors)


# ── Task CM-09 — patch_file ─────────────────────────────────

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunks(patch_str: str) -> list[dict[str, Any]] | str:
    """Parse a unified diff string into a list of hunk descriptors.

    Returns a list of dicts (one per hunk) on success, or an error string.
    Each hunk dict has: ``old_start``, ``old_lines``, ``new_start``,
    ``new_lines``, and ``ops`` (list of ``(op, text)`` tuples where
    ``op`` is ``" "``, ``"-"``, or ``"+"``).
    """
    lines = patch_str.splitlines()

    # Require at least one @@ header somewhere in the patch
    has_hunk = any(_HUNK_HEADER_RE.match(ln) for ln in lines)
    if not has_hunk:
        return "Invalid unified diff: missing hunk header"

    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            current = {
                "old_start": old_start,
                "old_lines": old_count,
                "new_start": new_start,
                "new_lines": new_count,
                "ops": [],
            }
        elif current is not None:
            if line.startswith(("-", "+", " ")):
                current["ops"].append((line[0], line[1:]))
            # Lines that are pure "\ No newline at end of file" — skip silently

    if current is not None:
        hunks.append(current)

    if not hunks:
        return "Invalid unified diff: no hunks found"

    return hunks


def _apply_patch(
    file_path: Path,
    patch_str: str,
    *,
    backup_enabled: bool = False,
) -> ToolResult:
    """Apply a unified diff patch to *file_path* atomically.

    All hunks succeed or none are applied (atomic).  Returns a
    :class:`~vaig.tools.base.ToolResult` whose ``output`` is a JSON
    string conforming to the CM-09 result contract::

        {"success": true, "path": "<relative_or_abs_path>"}
        {"success": false, "error": "<message>", "conflicts": [...]}

    Args:
        file_path: Absolute path to the target file.
        patch_str: Unified diff string (must contain ``@@`` hunk headers).
        backup_enabled: When True, write ``<file_path>.orig`` before patching.
    """
    path_str = str(file_path)

    if not file_path.exists():
        payload = json.dumps({"success": False, "error": f"File not found: {path_str}"})
        return ToolResult(output=payload, error=True)

    try:
        original = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        payload = json.dumps({"success": False, "error": f"Cannot read file: {exc}"})
        return ToolResult(output=payload, error=True)

    parsed = _parse_hunks(patch_str)
    if isinstance(parsed, str):
        payload = json.dumps({"success": False, "error": parsed})
        return ToolResult(output=payload, error=True)

    hunks = parsed
    file_lines = original.splitlines(keepends=True)
    # Work on a mutable copy; apply hunks sequentially with offset tracking
    result_lines: list[str] = list(file_lines)
    offset = 0  # cumulative line shift from prior hunk insertions/deletions
    conflicts: list[dict[str, Any]] = []

    for hunk in hunks:
        old_start = hunk["old_start"] - 1 + offset  # convert to 0-based
        ops: list[tuple[str, str]] = hunk["ops"]

        # Collect context+removal lines to verify against file
        expected: list[str] = []
        additions: list[str] = []
        for op, text in ops:
            line_with_nl = text if text.endswith("\n") else text + "\n"
            if op in (" ", "-"):
                expected.append(line_with_nl)
            if op in (" ", "+"):
                additions.append(line_with_nl)

        actual_slice = result_lines[old_start : old_start + len(expected)]

        # Strip trailing newline from last line for comparison tolerance
        def _norm(lines: list[str]) -> list[str]:
            if not lines:
                return lines
            normed = list(lines)
            normed[-1] = normed[-1].rstrip("\n")
            return normed

        if _norm(actual_slice) != _norm(expected):
            conflicts.append({
                "hunk_start": hunk["old_start"],
                "expected": "".join(expected),
                "found": "".join(actual_slice),
            })

    if conflicts:
        payload = json.dumps({
            "success": False,
            "error": "Hunk context mismatch",
            "conflicts": conflicts,
        })
        return ToolResult(output=payload, error=True)

    # All hunks validated — apply atomically
    offset = 0
    final_lines: list[str] = list(file_lines)

    for hunk in hunks:
        old_start = hunk["old_start"] - 1 + offset
        hunk_ops: list[tuple[str, str]] = hunk["ops"]

        expected_lines: list[str] = []
        replacement: list[str] = []
        for op, text in hunk_ops:
            line_with_nl = text if text.endswith("\n") else text + "\n"
            if op in (" ", "-"):
                expected_lines.append(line_with_nl)
            if op in (" ", "+"):
                replacement.append(line_with_nl)

        final_lines[old_start : old_start + len(expected_lines)] = replacement
        delta = len(replacement) - len(expected_lines)
        offset += delta

    new_content = "".join(final_lines)

    if backup_enabled:
        backup_path = file_path.with_suffix(file_path.suffix + ".orig")
        try:
            backup_path.write_text(original, encoding="utf-8")
        except OSError as exc:
            logger.warning("patch_file: could not write backup %s: %s", backup_path, exc)

    try:
        file_path.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        payload = json.dumps({"success": False, "error": f"Write failed: {exc}"})
        return ToolResult(output=payload, error=True)

    payload = json.dumps({"success": True, "path": path_str})
    return ToolResult(output=payload)


# ── Task 2.8 — Tool factory ─────────────────────────────────


def create_file_tools(workspace: Path) -> list[ToolDef]:
    """Create all file tool definitions bound to a workspace."""
    return [
        ToolDef(
            name="read_file",
            description="Read the contents of a file. Returns the file content as text.",
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="path",
                    type="string",
                    description="Relative path to the file from workspace root",
                ),
            ],
            execute=lambda path, _ws=workspace: read_file(path, workspace=_ws),
        ),
        ToolDef(
            name="write_file",
            description="Write content to a file. Creates parent directories if needed.",
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="path",
                    type="string",
                    description="Relative path to the file from workspace root",
                ),
                ToolParam(
                    name="content",
                    type="string",
                    description="The content to write to the file",
                ),
            ],
            execute=lambda path, content, _ws=workspace: write_file(
                path, content, workspace=_ws
            ),
        ),
        ToolDef(
            name="edit_file",
            description=(
                "Apply an exact string replacement in a file. "
                "The old_string must appear exactly once."
            ),
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="path",
                    type="string",
                    description="Relative path to the file from workspace root",
                ),
                ToolParam(
                    name="old_string",
                    type="string",
                    description="The exact string to find (must match exactly once)",
                ),
                ToolParam(
                    name="new_string",
                    type="string",
                    description="The replacement string",
                ),
            ],
            execute=lambda path, old_string, new_string, _ws=workspace: edit_file(
                path, old_string, new_string, workspace=_ws
            ),
        ),
        ToolDef(
            name="list_files",
            description=(
                "List directory contents. Returns one entry per line. "
                "Directories end with /. Skips hidden files and common ignores."
            ),
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="path",
                    type="string",
                    description="Relative directory path (defaults to workspace root)",
                    required=False,
                ),
            ],
            execute=lambda path=".", _ws=workspace: list_files(path, workspace=_ws),
        ),
        ToolDef(
            name="search_files",
            description=(
                "Search file contents using a regex pattern (recursive). "
                "Returns matches in filepath:line:content format."
            ),
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="pattern",
                    type="string",
                    description="Regex pattern to search for in file contents",
                ),
                ToolParam(
                    name="path",
                    type="string",
                    description="Directory to search in (defaults to workspace root)",
                    required=False,
                ),
            ],
            execute=lambda pattern, path=".", _ws=workspace: search_files(
                pattern, path, workspace=_ws
            ),
        ),
        ToolDef(
            name="verify_completeness",
            description=(
                "Scan files for incomplete placeholder patterns: TODO, FIXME, HACK, XXX, "
                "bare `pass`, ellipsis `...`, and NotImplementedError. "
                "Returns a report of findings with file:line references."
            ),
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="paths",
                    type="string",
                    description=(
                        "Comma-separated relative file paths (from workspace root) to scan. "
                        "Pass the files you just created or modified, e.g. "
                        "'src/foo.py,src/bar.py'."
                    ),
                ),
            ],
            execute=lambda paths, _ws=workspace: verify_completeness(
                [p.strip() for p in paths.split(",") if p.strip()],
                workspace=_ws,
            ),
        ),
        ToolDef(
            name="patch_file",
            description=(
                "Apply a unified diff patch to an existing file. "
                "Prefer this over write_file when modifying existing files — it is safer "
                "and more precise. The patch must contain unified diff hunk headers (@@ ... @@). "
                "All hunks are applied atomically: if any hunk fails, no changes are made. "
                "Returns JSON: {\"success\": true, \"path\": \"...\"} or "
                "{\"success\": false, \"error\": \"...\", \"conflicts\": [...]}."
            ),
            categories=frozenset({CODING}),
            parameters=[
                ToolParam(
                    name="path",
                    type="string",
                    description="Relative path to the file to patch (from workspace root)",
                ),
                ToolParam(
                    name="patch",
                    type="string",
                    description=(
                        "Unified diff string. Must contain @@ hunk headers. "
                        "Example: '@@ -1,3 +1,4 @@\\n context\\n-old line\\n+new line\\n context'"
                    ),
                ),
                ToolParam(
                    name="backup",
                    type="string",
                    description=(
                        "Set to 'true' to create a .orig backup before patching. "
                        "Defaults to 'false'."
                    ),
                    required=False,
                ),
            ],
            execute=lambda path, patch, backup="false", _ws=workspace: _apply_patch(
                _resolve_safe_path(path, _ws) or (_ws / path),
                patch,
                backup_enabled=backup.lower() == "true",
            ),
        ),
    ]

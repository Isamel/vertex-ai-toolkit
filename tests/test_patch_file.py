"""Tests for the patch_file tool (CM-09)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaig.tools.file_tools import _apply_patch, create_file_tools

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def tmp_file(tmp_path: Path) -> Path:
    """Return a temporary file pre-populated with 5 numbered lines."""
    f = tmp_path / "sample.py"
    f.write_text(
        "line1\n"
        "line2\n"
        "line3\n"
        "line4\n"
        "line5\n",
        encoding="utf-8",
    )
    return f


# ── Single-hunk patch ─────────────────────────────────────────


def test_single_hunk_replaces_line(tmp_file: Path) -> None:
    """A single-hunk patch that replaces one line applies correctly."""
    patch = (
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+LINE2_PATCHED\n"
        " line3\n"
    )
    result = _apply_patch(tmp_file, patch)
    data = json.loads(result.output)

    assert data["success"] is True
    content = tmp_file.read_text(encoding="utf-8")
    assert "LINE2_PATCHED" in content
    assert "line2" not in content


# ── Multi-hunk patch ──────────────────────────────────────────


def test_multi_hunk_both_applied(tmp_file: Path) -> None:
    """Two hunks in one patch are both applied atomically."""
    patch = (
        "@@ -1,3 +1,3 @@\n"
        "-line1\n"
        "+LINE1\n"
        " line2\n"
        " line3\n"
        "@@ -4,2 +4,2 @@\n"
        " line4\n"
        "-line5\n"
        "+LINE5\n"
    )
    result = _apply_patch(tmp_file, patch)
    data = json.loads(result.output)

    assert data["success"] is True
    content = tmp_file.read_text(encoding="utf-8")
    assert "LINE1" in content
    assert "LINE5" in content
    assert "line1\n" not in content
    assert "line5\n" not in content


# ── Malformed patch ───────────────────────────────────────────


def test_malformed_patch_returns_error(tmp_file: Path) -> None:
    """A patch without @@ headers returns a JSON error and does not modify the file."""
    original = tmp_file.read_text(encoding="utf-8")
    result = _apply_patch(tmp_file, "not a valid patch at all")

    assert result.error is True
    data = json.loads(result.output)
    assert data["success"] is False
    assert "error" in data
    # File must be unchanged
    assert tmp_file.read_text(encoding="utf-8") == original


# ── Context mismatch ──────────────────────────────────────────


def test_context_mismatch_returns_conflicts(tmp_file: Path) -> None:
    """When context lines do not match, patch returns conflicts and leaves file intact."""
    original = tmp_file.read_text(encoding="utf-8")
    patch = (
        "@@ -1,3 +1,3 @@\n"
        " WRONG_CONTEXT\n"
        "-line2\n"
        "+REPLACED\n"
        " line3\n"
    )
    result = _apply_patch(tmp_file, patch)

    assert result.error is True
    data = json.loads(result.output)
    assert data["success"] is False
    assert "conflicts" in data
    assert len(data["conflicts"]) >= 1
    # File must be unchanged
    assert tmp_file.read_text(encoding="utf-8") == original


# ── File not found ────────────────────────────────────────────


def test_file_not_found_returns_error(tmp_path: Path) -> None:
    """Patching a non-existent file returns a JSON error with success=False."""
    missing = tmp_path / "does_not_exist.py"
    patch = "@@ -1,1 +1,1 @@\n-old\n+new\n"
    result = _apply_patch(missing, patch)

    assert result.error is True
    data = json.loads(result.output)
    assert data["success"] is False
    assert "not found" in data["error"].lower()


# ── Backup creation ───────────────────────────────────────────


def test_backup_enabled_creates_orig(tmp_file: Path) -> None:
    """When backup_enabled=True, a .orig file is created before patching."""
    original_content = tmp_file.read_text(encoding="utf-8")
    patch = (
        "@@ -1,3 +1,3 @@\n"
        "-line1\n"
        "+LINE1\n"
        " line2\n"
        " line3\n"
    )
    result = _apply_patch(tmp_file, patch, backup_enabled=True)
    data = json.loads(result.output)

    assert data["success"] is True
    backup = tmp_file.with_suffix(tmp_file.suffix + ".orig")
    assert backup.exists(), "Backup .orig file was not created"
    assert backup.read_text(encoding="utf-8") == original_content


# ── ToolDef registration ──────────────────────────────────────


def test_patch_file_tool_registered(tmp_path: Path) -> None:
    """patch_file ToolDef must be present in create_file_tools()."""
    tools = create_file_tools(tmp_path)
    names = [t.name for t in tools]
    assert "patch_file" in names


def test_patch_file_tool_execute_via_tooldef(tmp_file: Path) -> None:
    """patch_file ToolDef execute lambda applies a patch correctly end-to-end."""
    workspace = tmp_file.parent
    tools = {t.name: t for t in create_file_tools(workspace)}
    patch_tool = tools["patch_file"]

    rel_path = tmp_file.name
    patch = (
        "@@ -2,3 +2,3 @@\n"
        " line2\n"
        "-line3\n"
        "+line3_patched\n"
        " line4\n"
    )
    result = patch_tool.execute(rel_path, patch)  # type: ignore[call-arg]
    data = json.loads(result.output)

    assert data["success"] is True
    assert "line3_patched" in tmp_file.read_text(encoding="utf-8")

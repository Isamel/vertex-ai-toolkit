"""Tests for verify_completeness — placeholder pattern scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaig.tools.file_tools import verify_completeness

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Provide a temporary directory as the workspace root."""
    return tmp_path


def _write(workspace: Path, relative: str, content: str) -> str:
    """Write *content* to *relative* path inside *workspace*, return relative path."""
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return relative


# ===========================================================================
# TestVerifyCompletenessCleanFiles
# ===========================================================================


class TestVerifyCompletenessCleanFiles:
    """Tests with files that have no incomplete patterns."""

    def test_single_clean_file(self, workspace: Path) -> None:
        """A file with no placeholders returns a clean message."""
        path = _write(workspace, "module.py", "def add(a: int, b: int) -> int:\n    return a + b\n")

        result = verify_completeness([path], workspace=workspace)

        assert result.error is False
        assert "No incomplete patterns found" in result.output

    def test_multiple_clean_files(self, workspace: Path) -> None:
        """Multiple clean files all return clean."""
        path_a = _write(workspace, "a.py", "x = 1\n")
        path_b = _write(workspace, "b.py", "y = 2\n")

        result = verify_completeness([path_a, path_b], workspace=workspace)

        assert result.error is False
        assert "No incomplete patterns found" in result.output

    def test_empty_file_is_clean(self, workspace: Path) -> None:
        """An empty file has no placeholders."""
        path = _write(workspace, "empty.py", "")

        result = verify_completeness([path], workspace=workspace)

        assert result.error is False
        assert "No incomplete patterns found" in result.output


# ===========================================================================
# TestVerifyCompletenessPatterns
# ===========================================================================


class TestVerifyCompletenessPatterns:
    """Tests that each incomplete pattern is detected."""

    def test_detects_todo(self, workspace: Path) -> None:
        """TODO comment is detected."""
        path = _write(workspace, "a.py", "# TODO: implement this\nx = 1\n")

        result = verify_completeness([path], workspace=workspace)

        assert "TODO" in result.output
        assert "a.py:1" in result.output

    def test_detects_fixme(self, workspace: Path) -> None:
        """FIXME comment is detected."""
        path = _write(workspace, "a.py", "x = 1\n# FIXME: broken logic\n")

        result = verify_completeness([path], workspace=workspace)

        assert "FIXME" in result.output
        assert "a.py:2" in result.output

    def test_detects_hack(self, workspace: Path) -> None:
        """HACK comment is detected."""
        path = _write(workspace, "a.py", "# HACK: workaround for issue #42\n")

        result = verify_completeness([path], workspace=workspace)

        assert "HACK" in result.output

    def test_detects_xxx(self, workspace: Path) -> None:
        """XXX marker is detected."""
        path = _write(workspace, "a.py", "# XXX: review this logic\n")

        result = verify_completeness([path], workspace=workspace)

        assert "XXX" in result.output

    def test_detects_bare_pass(self, workspace: Path) -> None:
        """Bare pass statement is detected."""
        path = _write(workspace, "a.py", "def f():\n    pass\n")

        result = verify_completeness([path], workspace=workspace)

        assert "bare pass" in result.output
        assert "a.py:2" in result.output

    def test_detects_ellipsis_body(self, workspace: Path) -> None:
        """Ellipsis as function body is detected."""
        path = _write(workspace, "a.py", "def f():\n    ...\n")

        result = verify_completeness([path], workspace=workspace)

        assert "ellipsis body" in result.output
        assert "a.py:2" in result.output

    def test_detects_not_implemented_error(self, workspace: Path) -> None:
        """NotImplementedError raise is detected."""
        path = _write(workspace, "a.py", "def f():\n    raise NotImplementedError\n")

        result = verify_completeness([path], workspace=workspace)

        assert "NotImplementedError" in result.output

    def test_todo_inline_in_code(self, workspace: Path) -> None:
        """TODO inline with code is detected."""
        path = _write(workspace, "a.py", "x = compute()  # TODO: cache result\n")

        result = verify_completeness([path], workspace=workspace)

        assert "TODO" in result.output

    def test_pass_with_comment_is_detected(self, workspace: Path) -> None:
        """Bare pass with trailing comment is still detected."""
        path = _write(workspace, "a.py", "    pass  # placeholder\n")

        result = verify_completeness([path], workspace=workspace)

        assert "bare pass" in result.output

    def test_ellipsis_with_comment_is_detected(self, workspace: Path) -> None:
        """Ellipsis with trailing comment is still detected."""
        path = _write(workspace, "a.py", "    ...  # stub\n")

        result = verify_completeness([path], workspace=workspace)

        assert "ellipsis body" in result.output


# ===========================================================================
# TestVerifyCompletenessMultipleFindings
# ===========================================================================


class TestVerifyCompletenessMultipleFindings:
    """Tests with multiple findings across files."""

    def test_multiple_patterns_in_one_file(self, workspace: Path) -> None:
        """Multiple patterns in one file are all reported."""
        content = (
            "# TODO: implement\n"
            "def f():\n"
            "    pass\n"
            "# FIXME: check edge cases\n"
        )
        path = _write(workspace, "a.py", content)

        result = verify_completeness([path], workspace=workspace)

        assert "Found 3 incomplete pattern(s)" in result.output

    def test_findings_across_multiple_files(self, workspace: Path) -> None:
        """Findings are reported from all files."""
        path_a = _write(workspace, "a.py", "# TODO: fix\n")
        path_b = _write(workspace, "b.py", "def g():\n    pass\n")

        result = verify_completeness([path_a, path_b], workspace=workspace)

        assert "Found 2 incomplete pattern(s)" in result.output
        assert "a.py:1" in result.output
        assert "b.py:2" in result.output

    def test_finding_count_in_output(self, workspace: Path) -> None:
        """Output states the exact number of findings."""
        content = "# TODO: a\n# TODO: b\n# FIXME: c\n"
        path = _write(workspace, "a.py", content)

        result = verify_completeness([path], workspace=workspace)

        assert "Found 3 incomplete pattern(s)" in result.output

    def test_output_includes_file_and_line(self, workspace: Path) -> None:
        """Each finding includes the file path and line number."""
        path = _write(workspace, "sub/module.py", "# TODO: check this on line one\n")

        result = verify_completeness([path], workspace=workspace)

        assert "sub/module.py:1" in result.output

    def test_output_includes_matched_text(self, workspace: Path) -> None:
        """Each finding includes a snippet of the matched line."""
        path = _write(workspace, "a.py", "# TODO: implement auth\n")

        result = verify_completeness([path], workspace=workspace)

        assert "implement auth" in result.output


# ===========================================================================
# TestVerifyCompletenessEdgeCases
# ===========================================================================


class TestVerifyCompletenessEdgeCases:
    """Edge cases: empty input, path safety, missing files."""

    def test_empty_paths_list(self, workspace: Path) -> None:
        """Empty paths list returns a no-paths message."""
        result = verify_completeness([], workspace=workspace)

        assert result.error is False
        assert "No paths provided" in result.output

    def test_path_outside_workspace_sets_error(self, workspace: Path) -> None:
        """Path traversal outside workspace sets error=True."""
        result = verify_completeness(["../../etc/passwd"], workspace=workspace)

        assert result.error is True
        assert "Path safety error" in result.output

    def test_nonexistent_file_sets_error(self, workspace: Path) -> None:
        """Non-existent file path sets error=True."""
        result = verify_completeness(["ghost.py"], workspace=workspace)

        assert result.error is True
        assert "File not found" in result.output

    def test_directory_path_sets_error(self, workspace: Path) -> None:
        """Passing a directory path (not a file) sets error=True."""
        subdir = workspace / "subdir"
        subdir.mkdir()

        result = verify_completeness(["subdir"], workspace=workspace)

        assert result.error is True
        assert "Not a file" in result.output

    def test_only_one_match_per_line(self, workspace: Path) -> None:
        """A line matching multiple patterns is only counted once."""
        # This line has both TODO and FIXME
        path = _write(workspace, "a.py", "# TODO FIXME: handle this\n")

        result = verify_completeness([path], workspace=workspace)

        # Only 1 finding (first match wins)
        assert "Found 1 incomplete pattern(s)" in result.output

    def test_pass_in_string_not_detected(self, workspace: Path) -> None:
        """'pass' inside a string is not a bare pass statement."""
        path = _write(workspace, "a.py", 'msg = "pass the butter"\n')

        result = verify_completeness([path], workspace=workspace)

        assert "No incomplete patterns found" in result.output

    def test_ellipsis_in_slice_not_detected(self, workspace: Path) -> None:
        """Ellipsis used in array slicing is not flagged."""
        # This is valid numpy-style slicing — not a standalone ellipsis body
        path = _write(workspace, "a.py", "arr[..., 0] = 1\n")

        result = verify_completeness([path], workspace=workspace)

        assert "No incomplete patterns found" in result.output

    def test_error_and_findings_coexist(self, workspace: Path) -> None:
        """When some paths error and others have findings, both appear in output."""
        path_ok = _write(workspace, "ok.py", "# TODO: something\n")

        result = verify_completeness([path_ok, "ghost.py"], workspace=workspace)

        assert "TODO" in result.output
        assert "Errors" in result.output
        assert result.error is True

    def test_mixed_clean_and_incomplete(self, workspace: Path) -> None:
        """A mix of clean and incomplete files only reports incomplete ones."""
        clean = _write(workspace, "clean.py", "x = 1\n")
        dirty = _write(workspace, "dirty.py", "# TODO: add logic\n")

        result = verify_completeness([clean, dirty], workspace=workspace)

        assert "Found 1 incomplete pattern(s)" in result.output
        assert "clean.py" not in result.output
        assert "dirty.py" in result.output


# ===========================================================================
# TestVerifyCompletenessIntegration
# ===========================================================================


class TestVerifyCompletenessIntegration:
    """Integration tests — verifying verify_completeness is in the tool registry."""

    def test_tool_registered_in_create_file_tools(self, workspace: Path) -> None:
        """verify_completeness is registered in create_file_tools output."""
        from vaig.tools.file_tools import create_file_tools

        tools = create_file_tools(workspace)
        tool_names = [t.name for t in tools]

        assert "verify_completeness" in tool_names

    def test_tool_has_paths_parameter(self, workspace: Path) -> None:
        """verify_completeness tool definition has a 'paths' parameter."""
        from vaig.tools.file_tools import create_file_tools

        tools = create_file_tools(workspace)
        verify_tool = next(t for t in tools if t.name == "verify_completeness")
        param_names = [p.name for p in verify_tool.parameters]

        assert "paths" in param_names

    def test_tool_execute_callable(self, workspace: Path) -> None:
        """verify_completeness ToolDef.execute is callable and returns ToolResult."""
        from vaig.tools.base import ToolResult
        from vaig.tools.file_tools import create_file_tools

        path = _write(workspace, "test.py", "x = 1\n")
        tools = create_file_tools(workspace)
        verify_tool = next(t for t in tools if t.name == "verify_completeness")

        result = verify_tool.execute(paths=[path])

        assert isinstance(result, ToolResult)

    def test_tool_registered_in_coding_agent(self, workspace: Path) -> None:
        """CodingAgent registers verify_completeness as an available tool."""
        from unittest.mock import patch

        from vaig.agents.coding import CodingAgent
        from vaig.core.config import CodingConfig

        config = CodingConfig(workspace_root=str(workspace))
        client_mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        client_mock.current_model = "gemini-2.5-pro"

        with patch("vaig.agents.coding.create_shell_tools", return_value=[]):
            agent = CodingAgent(client_mock, config)

        tool_names = [t.name for t in agent.registry.list_tools()]
        assert "verify_completeness" in tool_names

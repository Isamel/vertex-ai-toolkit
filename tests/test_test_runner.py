"""Tests for TestRunnerTool — TestExecutionResult, _detect_test_command, _parse_pytest_output, _run_tests, create_test_runner_tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from vaig.tools.test_runner import (
    TestExecutionResult,
    _detect_test_command,
    _parse_pytest_output,
    _run_tests,
    create_test_runner_tool,
)

# ── TestExecutionResult ───────────────────────────────────────


def test_success_flag_set_for_tests_passed() -> None:
    result = TestExecutionResult(state="tests_passed", passed=3)
    assert result.success is True


def test_success_flag_false_for_tests_failed() -> None:
    result = TestExecutionResult(state="tests_failed", failed=2)
    assert result.success is False


def test_success_flag_false_for_execution_error() -> None:
    result = TestExecutionResult(state="execution_error")
    assert result.success is False


def test_success_flag_false_for_no_tests_found() -> None:
    result = TestExecutionResult(state="no_tests_found")
    assert result.success is False


def test_default_fields() -> None:
    result = TestExecutionResult()
    assert result.passed == 0
    assert result.failed == 0
    assert result.errors == 0
    assert result.output == ""
    assert result.duration_seconds == 0.0
    assert result.command == ""


# ── _detect_test_command ──────────────────────────────────────


def test_detect_pyproject_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    assert _detect_test_command(tmp_path) == "pytest"


def test_detect_setup_cfg(tmp_path: Path) -> None:
    (tmp_path / "setup.cfg").write_text("[tool:pytest]\n")
    assert _detect_test_command(tmp_path) == "pytest"


def test_detect_conftest_py(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text("# conftest\n")
    assert _detect_test_command(tmp_path) == "pytest"


def test_detect_setup_py(tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text("from setuptools import setup\n")
    assert _detect_test_command(tmp_path) == "pytest"


def test_detect_conftest_one_level_deep(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "conftest.py").write_text("# conftest\n")
    assert _detect_test_command(tmp_path) == "pytest"


def test_detect_no_markers_returns_empty(tmp_path: Path) -> None:
    assert _detect_test_command(tmp_path) == ""


# ── _parse_pytest_output ──────────────────────────────────────


def test_parse_all_passed() -> None:
    stdout = "3 passed in 0.42s"
    passed, failed, errors, state = _parse_pytest_output(stdout, 0)
    assert passed == 3
    assert failed == 0
    assert errors == 0
    assert state == "tests_passed"


def test_parse_mixed_pass_fail() -> None:
    stdout = "2 passed, 1 failed in 0.5s"
    passed, failed, errors, state = _parse_pytest_output(stdout, 1)
    assert passed == 2
    assert failed == 1
    assert state == "tests_failed"


def test_parse_errors() -> None:
    stdout = "0 passed, 2 errors in 0.1s"
    passed, failed, errors, state = _parse_pytest_output(stdout, 1)
    assert errors == 2
    assert state == "tests_failed"


def test_parse_no_tests_ran() -> None:
    stdout = "no tests ran"
    passed, failed, errors, state = _parse_pytest_output(stdout, 5)
    assert state == "no_tests_found"
    assert passed == 0


def test_parse_collected_zero_items() -> None:
    stdout = "collected 0 items\n"
    _, _, _, state = _parse_pytest_output(stdout, 5)
    assert state == "no_tests_found"


def test_parse_nonzero_return_code_no_counts() -> None:
    stdout = "something went wrong"
    _, _, _, state = _parse_pytest_output(stdout, 1)
    assert state == "tests_failed"


# ── _run_tests ────────────────────────────────────────────────


def _make_completed_proc(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def test_run_tests_success(tmp_path: Path) -> None:
    proc = _make_completed_proc(stdout="3 passed in 0.5s", returncode=0)
    with patch("vaig.tools.test_runner.subprocess.run", return_value=proc) as mock_run:
        result = _run_tests(tmp_path, "pytest", timeout=30)

    mock_run.assert_called_once()
    assert result.state == "tests_passed"
    assert result.passed == 3
    assert result.command == "pytest"
    assert result.success is True


def test_run_tests_failure(tmp_path: Path) -> None:
    proc = _make_completed_proc(stdout="1 passed, 2 failed in 1.0s", returncode=1)
    with patch("vaig.tools.test_runner.subprocess.run", return_value=proc):
        result = _run_tests(tmp_path, "pytest -x", timeout=30)

    assert result.state == "tests_failed"
    assert result.failed == 2
    assert result.success is False


def test_run_tests_timeout(tmp_path: Path) -> None:
    with patch(
        "vaig.tools.test_runner.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=5),
    ):
        result = _run_tests(tmp_path, "pytest", timeout=5)

    assert result.state == "execution_error"
    assert "timed out" in result.output.lower()
    assert result.success is False


def test_run_tests_file_not_found(tmp_path: Path) -> None:
    with patch(
        "vaig.tools.test_runner.subprocess.run",
        side_effect=FileNotFoundError("No such file or directory: 'pytest'"),
    ):
        result = _run_tests(tmp_path, "pytest", timeout=30)

    assert result.state == "execution_error"
    assert result.success is False


def test_run_tests_output_truncated(tmp_path: Path) -> None:
    long_output = "x" * 5000
    proc = _make_completed_proc(stdout=long_output, returncode=0)
    with patch("vaig.tools.test_runner.subprocess.run", return_value=proc):
        result = _run_tests(tmp_path, "pytest", timeout=30)

    assert len(result.output) <= 2000


# ── create_test_runner_tool ───────────────────────────────────


def test_create_tool_returns_none_when_no_markers(tmp_path: Path) -> None:
    tool = create_test_runner_tool(tmp_path)
    assert tool is None


def test_create_tool_detects_pytest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]\n")
    tool = create_test_runner_tool(tmp_path)
    assert tool is not None
    assert tool.name == "run_tests"


def test_create_tool_with_explicit_command(tmp_path: Path) -> None:
    tool = create_test_runner_tool(tmp_path, test_command="python -m pytest")
    assert tool is not None
    assert tool.name == "run_tests"


def test_create_tool_execute_returns_tool_result(tmp_path: Path) -> None:
    proc = _make_completed_proc(stdout="5 passed in 0.8s", returncode=0)
    with patch("vaig.tools.test_runner.subprocess.run", return_value=proc):
        tool = create_test_runner_tool(tmp_path, test_command="pytest")
        assert tool is not None
        tool_result = tool.execute()

    assert tool_result.error is False
    # Output should be JSON-parseable TestExecutionResult
    import json
    data = json.loads(tool_result.output)
    assert data["state"] == "tests_passed"
    assert data["passed"] == 5


def test_create_tool_execute_error_flag_on_failure(tmp_path: Path) -> None:
    proc = _make_completed_proc(stdout="execution_error", returncode=2)
    with patch("vaig.tools.test_runner.subprocess.run", return_value=proc):
        tool = create_test_runner_tool(tmp_path, test_command="pytest")
        assert tool is not None
        tool_result = tool.execute()

    # error flag is based on state == "execution_error", not on test failures
    # Here returncode=2 with no parsed counts → tests_failed, so error=False
    assert tool_result.error is False


def test_create_tool_description_contains_key_fields(tmp_path: Path) -> None:
    tool = create_test_runner_tool(tmp_path, test_command="pytest")
    assert tool is not None
    assert "state" in tool.description
    assert "passed" in tool.description

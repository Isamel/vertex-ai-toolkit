"""TestRunnerTool — run tests inside the jailed workspace and return structured results.

Executes the project's test suite (auto-detected or configured) and parses the
output into a :class:`TestExecutionResult` that the Verifier can use as
structured evidence.

Usage::

    tool = create_test_runner_tool(workspace, timeout=120, test_command="pytest -x")
    result = tool.execute()

The tool is registered in the coding pipeline's :class:`~vaig.tools.base.ToolRegistry`
when ``test_command`` is set or pytest is detected in the workspace.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vaig.tools.base import ToolDef, ToolResult

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 2000
"""Maximum characters of test output to include in the result."""

TestState = Literal["no_tests_found", "tests_passed", "tests_failed", "execution_error"]


class TestExecutionResult(BaseModel):
    """Structured result from running the project's test suite.

    Attributes:
        state: High-level outcome — one of ``no_tests_found``,
            ``tests_passed``, ``tests_failed``, or ``execution_error``.
        passed: Number of tests that passed.
        failed: Number of tests that failed.
        errors: Number of tests that errored.
        output: Truncated test output (max 2000 chars).
        duration_seconds: Wall-clock time for the test run.
        command: The command that was executed.
        success: Convenience flag — True when ``state == "tests_passed"``.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    state: TestState = "no_tests_found"
    passed: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""
    duration_seconds: float = 0.0
    command: str = ""
    success: bool = False

    def model_post_init(self, _context: object) -> None:
        """Sync ``success`` with ``state``."""
        object.__setattr__(self, "success", self.state == "tests_passed")


def _detect_test_command(workspace: Path) -> str:
    """Heuristic: detect pytest by scanning the workspace for marker files.

    Checks for ``pyproject.toml``, ``setup.cfg``, and ``conftest.py``
    (including one level deep).  Falls back to an empty string when no
    marker is found.

    Args:
        workspace: The directory to scan.

    Returns:
        ``"pytest"`` when a pytest project is detected, otherwise ``""``.
    """
    if not workspace.is_dir():
        logger.debug("_detect_test_command: workspace does not exist — skipping detection")
        return ""

    markers = [
        workspace / "pyproject.toml",
        workspace / "setup.cfg",
        workspace / "conftest.py",
        workspace / "setup.py",
    ]
    for marker in markers:
        if marker.exists():
            logger.debug("_detect_test_command: found %s → using pytest", marker.name)
            return "pytest"

    # One-level-deep conftest.py scan
    try:
        children = list(workspace.iterdir())
    except PermissionError:
        logger.debug("_detect_test_command: permission denied listing workspace — skipping scan")
        return ""

    for child in children:
        try:
            if child.is_dir() and (child / "conftest.py").exists():
                logger.debug(
                    "_detect_test_command: found %s/conftest.py → using pytest", child.name
                )
                return "pytest"
        except PermissionError:
            continue

    return ""


def _parse_pytest_output(stdout: str, return_code: int) -> tuple[int, int, int, TestState]:
    """Parse pytest output to extract pass/fail/error counts and state.

    Args:
        stdout: Combined stdout + stderr from the test run.
        return_code: Process return code.

    Returns:
        Tuple of (passed, failed, errors, state).
    """
    # Typical pytest summary lines:
    # "5 passed, 2 failed, 1 error in 1.23s"
    # "no tests ran"
    # "collected 0 items"
    passed = 0
    failed = 0
    errors = 0

    if re.search(r"no tests ran|collected 0 items", stdout, re.IGNORECASE):
        return 0, 0, 0, "no_tests_found"

    m_passed = re.search(r"(\d+) passed", stdout)
    m_failed = re.search(r"(\d+) failed", stdout)
    m_error = re.search(r"(\d+) error", stdout)

    if m_passed:
        passed = int(m_passed.group(1))
    if m_failed:
        failed = int(m_failed.group(1))
    if m_error:
        errors = int(m_error.group(1))

    if failed > 0 or errors > 0:
        return passed, failed, errors, "tests_failed"
    if passed > 0 and return_code == 0:
        return passed, 0, 0, "tests_passed"
    if return_code != 0:
        return passed, failed, errors, "tests_failed"
    # Return code 0 with no parsed counts — treat as passed if no error markers
    return passed, 0, 0, "tests_passed"


def _run_tests(
    workspace: Path,
    command: str,
    timeout: int,
) -> TestExecutionResult:
    """Execute the test command and return a structured result.

    Args:
        workspace: Working directory for the command.
        command: Shell command to run (e.g. ``"pytest -x"``).
        timeout: Maximum seconds before killing the process.

    Returns:
        :class:`TestExecutionResult` with parsed state and counts.
    """
    start = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S602 S603
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        combined = (proc.stdout or "") + (proc.stderr or "")
        snippet = combined[-_MAX_OUTPUT_CHARS:] if len(combined) > _MAX_OUTPUT_CHARS else combined

        passed, failed, errors, state = _parse_pytest_output(combined, proc.returncode)

        return TestExecutionResult(
            state=state,
            passed=passed,
            failed=failed,
            errors=errors,
            output=snippet,
            duration_seconds=round(duration, 3),
            command=command,
        )

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        logger.warning("TestRunnerTool: command timed out after %ss: %r", timeout, command)
        return TestExecutionResult(
            state="execution_error",
            output=f"Execution timed out after {timeout}s",
            duration_seconds=round(duration, 3),
            command=command,
        )
    except FileNotFoundError as exc:
        duration = time.monotonic() - start
        logger.warning("TestRunnerTool: command not found: %r — %s", command, exc)
        return TestExecutionResult(
            state="execution_error",
            output=f"Command not found: {exc}",
            duration_seconds=round(duration, 3),
            command=command,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        logger.warning("TestRunnerTool: unexpected error: %s", exc, exc_info=True)
        return TestExecutionResult(
            state="execution_error",
            output=f"Unexpected error: {exc}",
            duration_seconds=round(duration, 3),
            command=command,
        )


def create_test_runner_tool(
    workspace: Path,
    *,
    timeout: int = 120,
    test_command: str = "",
) -> ToolDef | None:
    """Create a :class:`~vaig.tools.base.ToolDef` that runs the project's test suite.

    If neither *test_command* is given nor pytest can be auto-detected in
    *workspace*, returns ``None`` so the caller can skip registration.

    Args:
        workspace: The workspace directory (or jail effective_path).
        timeout: Maximum seconds allowed for the test run.
        test_command: Explicit command override.  When empty, auto-detection
            via :func:`_detect_test_command` is used.

    Returns:
        A :class:`~vaig.tools.base.ToolDef` wrapping :func:`_run_tests`,
        or ``None`` when no test runner is available.
    """
    resolved_command = test_command or _detect_test_command(workspace)
    if not resolved_command:
        logger.debug("create_test_runner_tool: no test command detected — tool not registered")
        return None

    def _execute(**_kwargs: object) -> ToolResult:
        """Run the detected test suite and return structured JSON output."""
        result = _run_tests(workspace, resolved_command, timeout)
        return ToolResult(output=result.model_dump_json(), error=result.state == "execution_error")

    return ToolDef(
        name="run_tests",
        description=(
            "Execute the project's test suite and return a structured JSON result. "
            "Fields: state (no_tests_found|tests_passed|tests_failed|execution_error), "
            "passed, failed, errors, output (truncated), duration_seconds, command."
        ),
        parameters=None,
        execute=_execute,
        cacheable=False,
    )

"""Shell tools — run commands within a workspace."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from vaig.tools.base import ToolDef, ToolParam, ToolResult

logger = logging.getLogger(__name__)


# ── Task 2.6 — run_command ───────────────────────────────────


def run_command(
    command: str,
    *,
    workspace: Path,
    allowed_commands: list[str] | None = None,
) -> ToolResult:
    """Run a shell command inside *workspace*.

    If *allowed_commands* is provided and non-empty, the first token of the
    command must appear in the allowlist.
    """
    logger.debug("run_command: command=%r workspace=%s", command, workspace)

    try:
        args = shlex.split(command)
    except ValueError as exc:
        return ToolResult(output=f"Failed to parse command: {exc}", error=True)

    if not args:
        return ToolResult(output="Empty command", error=True)

    if allowed_commands and args[0] not in allowed_commands:
        return ToolResult(
            output=(
                f"Command '{args[0]}' is not in the allowed commands list. "
                f"Allowed: {', '.join(allowed_commands)}"
            ),
            error=True,
        )

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=workspace,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            output=f"Command timed out after 30 seconds: {command}",
            error=True,
        )
    except FileNotFoundError:
        return ToolResult(
            output=f"Command not found: {args[0]}",
            error=True,
        )
    except OSError as exc:
        return ToolResult(output=f"Error running command: {exc}", error=True)

    output_parts: list[str] = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(result.stderr)

    combined = "\n".join(output_parts).rstrip("\n") if output_parts else "(no output)"

    if result.returncode != 0:
        return ToolResult(
            output=f"{combined}\n(exit code {result.returncode})",
            error=True,
        )

    return ToolResult(output=combined)


# ── Task 2.8 — Tool factory ─────────────────────────────────


def create_shell_tools(
    workspace: Path,
    allowed_commands: list[str] | None = None,
) -> list[ToolDef]:
    """Create shell tool definitions bound to a workspace."""
    return [
        ToolDef(
            name="run_command",
            description=(
                "Run a shell command in the workspace directory. "
                "Returns stdout and stderr."
            ),
            parameters=[
                ToolParam(
                    name="command",
                    type="string",
                    description="The shell command to execute",
                ),
            ],
            execute=lambda command, _ws=workspace, _ac=allowed_commands: run_command(
                command, workspace=_ws, allowed_commands=_ac
            ),
        ),
    ]

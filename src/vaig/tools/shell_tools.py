"""Shell tools — run commands within a workspace."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from pathlib import Path

from vaig.tools.base import ToolDef, ToolParam, ToolResult

logger = logging.getLogger(__name__)

# Maximum output size in characters to prevent memory exhaustion from
# commands that produce unbounded output (e.g. ``cat /dev/urandom``).
_MAX_OUTPUT_CHARS = 100_000

# Default command timeout in seconds.  Can be overridden via the
# ``timeout`` parameter on ``run_command``.
_DEFAULT_TIMEOUT = 30

# Argument patterns that are blocked regardless of the allowed-commands
# allowlist.  These prevent path traversal, reading sensitive files, and
# shell meta-character abuse.
_BLOCKED_ARG_PATTERNS: tuple[str, ...] = (
    "/etc/shadow",
    "/etc/passwd",
    "/etc/sudoers",
    "~root",
)


def _check_denied_command(
    command: str,
    denied_patterns: list[str],
) -> str | None:
    """Check *command* against a list of denied regex patterns.

    Returns a human-readable reason string if the command matches any
    denied pattern, or ``None`` if it is allowed through.

    The check runs against the **raw command string** (before ``shlex``
    splitting) so that patterns can match shell operators, pipes, and
    multi-token sequences.
    """
    for pattern in denied_patterns:
        try:
            if re.search(pattern, command):
                return (
                    f"Command denied — matches blocked pattern: {pattern!r}. "
                    "This command is not allowed for security reasons."
                )
        except re.error:
            logger.warning("Invalid denied_commands regex pattern: %r", pattern)
    return None


def _is_arg_safe(arg: str, workspace: Path) -> tuple[bool, str]:
    """Validate that a single command argument is safe.

    Returns ``(True, "")`` when the arg is allowed, or
    ``(False, reason)`` when it should be rejected.
    """
    # Block known sensitive paths
    for pattern in _BLOCKED_ARG_PATTERNS:
        if pattern in arg:
            return False, f"Blocked argument pattern: {pattern}"

    # For arguments that look like file paths, ensure they resolve
    # inside the workspace directory (prevent traversal via ``../``).
    if "/" in arg or arg.startswith(".."):
        try:
            resolved = (workspace / arg).resolve()
            if not str(resolved).startswith(str(workspace.resolve())):
                return False, f"Path escapes workspace: {arg}"
        except (OSError, ValueError):
            pass  # Non-path argument — allow through

    return True, ""


# ── Task 2.6 — run_command ───────────────────────────────────


def run_command(
    command: str,
    *,
    workspace: Path,
    allowed_commands: list[str] | None = None,
    denied_commands: list[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> ToolResult:
    """Run a shell command inside *workspace*.

    Security checks are applied in order:

    1. **Denylist** — if *denied_commands* is provided and the raw command
       string matches any regex pattern, the command is rejected immediately.
    2. **Allowlist** — if *allowed_commands* is provided and non-empty, the
       first token must appear in the allowlist.
    3. **Argument safety** — each argument is validated to prevent path
       traversal outside the workspace and access to sensitive system files.
    """
    logger.debug("run_command: command=%r workspace=%s", command, workspace)

    # ── Denylist check (runs on raw string BEFORE parsing) ────
    if denied_commands:
        denied_reason = _check_denied_command(command, denied_commands)
        if denied_reason:
            return ToolResult(output=denied_reason, error=True)

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

    # Validate all arguments for safety
    for arg in args[1:]:
        safe, reason = _is_arg_safe(arg, workspace)
        if not safe:
            return ToolResult(output=f"Argument rejected: {reason}", error=True)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            output=f"Command timed out after {timeout} seconds: {command}",
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

    # Truncate excessive output to prevent memory issues
    if len(combined) > _MAX_OUTPUT_CHARS:
        combined = combined[:_MAX_OUTPUT_CHARS] + f"\n... (truncated — output exceeded {_MAX_OUTPUT_CHARS} chars)"

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
    denied_commands: list[str] | None = None,
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
            execute=lambda command, _ws=workspace, _ac=allowed_commands, _dc=denied_commands: run_command(
                command, workspace=_ws, allowed_commands=_ac, denied_commands=_dc
            ),
        ),
    ]

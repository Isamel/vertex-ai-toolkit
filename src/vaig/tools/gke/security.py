"""Security tools — exec command, RBAC checks, deny/allow lists."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from vaig.tools.base import ToolResult

from . import _clients, _resources

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig

logger = logging.getLogger(__name__)

# ── Lazy import guard (mirrors _clients.py) ──────────────────
# Needed locally for except clauses and k8s_client usage in check_rbac.
_K8S_AVAILABLE = True
try:
    from kubernetes import client as k8s_client  # noqa: WPS433
    from kubernetes.client import exceptions as k8s_exceptions  # noqa: WPS433
except ImportError:
    _K8S_AVAILABLE = False


# ── exec_command security model ──────────────────────────────

# Denylist — checked FIRST.  If any pattern matches, the command is rejected.
DENIED_PATTERNS: list[str] = [
    r"[;&|`$]",       # shell metacharacters / chaining / substitution
    r">\s*",           # output redirection (overwrite)
    r">>\s*",          # output redirection (append)
    r"\brm\b",         # file deletion
    r"\bkill\b",       # process killing
    r"\bshutdown\b",   # system shutdown
    r"\breboot\b",     # system reboot
    r"\bdd\b",         # raw disk write
    r"\bmkfs\b",       # filesystem creation
    r"\bfdisk\b",      # partition editing
    r"\bchmod\b",      # permission changes
    r"\bchown\b",      # ownership changes
    r"\bsudo\b",       # privilege escalation
]
_COMPILED_DENY = [re.compile(p) for p in DENIED_PATTERNS]

# Allowlist — checked SECOND.  The command must start with one of these prefixes.
ALLOWED_EXEC_COMMANDS: list[str] = [
    "cat", "head", "tail", "ls", "env", "printenv",
    "whoami", "id", "hostname", "date",
    "ps", "top -bn1",
    "df", "du",
    "mount",
    "ip", "ifconfig", "netstat", "ss", "nslookup", "dig", "ping", "curl", "wget", "nc",
    "java -version", "python --version", "node --version",
    "cat /etc/resolv.conf", "cat /etc/hosts",
]

_MAX_EXEC_OUTPUT = 10_000


def _check_denied(command: str) -> str | None:
    """Return a reason string if *command* matches any deny pattern, else ``None``."""
    for pattern, compiled in zip(DENIED_PATTERNS, _COMPILED_DENY):
        if compiled.search(command):
            return f"Command denied — matches blocked pattern: {pattern}"
    return None


def _check_allowed(command: str) -> bool:
    """Return ``True`` if *command* starts with an allowed prefix."""
    cmd_stripped = command.strip()
    return any(cmd_stripped == prefix or cmd_stripped.startswith(prefix + " ") for prefix in ALLOWED_EXEC_COMMANDS)


# ── exec_command ─────────────────────────────────────────────


def exec_command(
    pod_name: str,
    command: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    container: str | None = None,
    timeout: int = 30,
) -> ToolResult:
    """Execute a diagnostic command inside a running container.

    Uses ``kubernetes.stream.stream()`` with
    ``CoreV1Api.connect_get_namespaced_pod_exec``.

    Security model:
    1. ``gke_config.exec_enabled`` must be ``True`` (disabled by default).
    2. Command is checked against a **denylist** of dangerous patterns.
    3. Command must start with an **allowed prefix** (read-only diagnostics).
    4. Output is truncated to *_MAX_EXEC_OUTPUT* characters.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # ── Gate: exec must be explicitly enabled ──────────────
    if not gke_config.exec_enabled:
        return ToolResult(
            output=(
                "exec_command is disabled. Set gke.exec_enabled=true in your config "
                "to enable container command execution."
            ),
            error=True,
        )

    # ── Validate command ──────────────────────────────────
    if not command or not command.strip():
        return ToolResult(output="Command cannot be empty.", error=True)

    denied_reason = _check_denied(command)
    if denied_reason:
        return ToolResult(output=denied_reason, error=True)

    if not _check_allowed(command):
        return ToolResult(
            output=(
                f"Command not in allowlist. Allowed prefixes: {', '.join(ALLOWED_EXEC_COMMANDS)}. "
                f"Got: '{command.split()[0]}'"
            ),
            error=True,
        )

    # ── Validate timeout ──────────────────────────────────
    if timeout < 1 or timeout > 300:
        return ToolResult(
            output=f"Timeout must be between 1 and 300 seconds. Got: {timeout}",
            error=True,
        )

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    core_v1, _, _, _ = result

    try:
        # Lazy-import stream to keep it optional
        from kubernetes.stream import stream as k8s_stream  # noqa: WPS433

        exec_kwargs: dict[str, Any] = {
            "name": pod_name,
            "namespace": ns,
            "command": ["/bin/sh", "-c", command],
            "stderr": True,
            "stdin": False,
            "stdout": True,
            "tty": False,
            "_request_timeout": timeout,
        }
        if container:
            exec_kwargs["container"] = container

        resp = k8s_stream(
            core_v1.connect_get_namespaced_pod_exec,
            **exec_kwargs,
        )

        # resp is the full combined output as a string
        stdout = resp if isinstance(resp, str) else str(resp)
        stderr = ""  # stream() merges stderr into the response string

        # Truncate if needed
        if len(stdout) > _MAX_EXEC_OUTPUT:
            stdout = stdout[:_MAX_EXEC_OUTPUT] + f"\n... [truncated at {_MAX_EXEC_OUTPUT} chars]"

        lines: list[str] = []
        lines.append(f"=== exec_command: {command} ===")
        lines.append(f"Pod: {pod_name} | Namespace: {ns}")
        if container:
            lines.append(f"Container: {container}")
        lines.append("")
        if stdout.strip():
            lines.append("--- stdout ---")
            lines.append(stdout)
        if stderr.strip():
            lines.append("--- stderr ---")
            lines.append(stderr)
        if not stdout.strip() and not stderr.strip():
            lines.append("(no output)")

        return ToolResult(output="\n".join(lines), error=False)

    except k8s_exceptions.ApiException as exc:
        if exc.status == 404:
            msg = f"Pod '{pod_name}' not found in namespace '{ns}'."
            if container:
                msg += f" (container: {container})"
            return ToolResult(output=msg, error=True)
        if exc.status in (401, 403):
            return ToolResult(
                output=f"Permission denied executing command on pod '{pod_name}': {exc.reason}",
                error=True,
            )
        return ToolResult(output=f"K8s API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(output=f"Error executing command: {exc}", error=True)


# ── check_rbac ───────────────────────────────────────────────


def check_rbac(
    verb: str,
    resource: str,
    *,
    gke_config: GKEConfig,
    namespace: str = "default",
    service_account: str | None = None,
    resource_name: str | None = None,
) -> ToolResult:
    """Check whether a service account has permission to perform a specific action.

    Uses ``AuthorizationV1Api.create_namespaced_subject_access_review()``
    for specific service accounts, or
    ``create_self_subject_access_review()`` for the current user.

    Read-only — only checks permissions, does not modify any resources.
    """
    if not _K8S_AVAILABLE:
        return _clients._k8s_unavailable()

    # Normalise resource alias (e.g. "po" → "pods")
    normalised = _resources._normalise_resource(resource)

    ns = namespace or gke_config.default_namespace

    result = _clients._create_k8s_clients(gke_config)
    if isinstance(result, ToolResult):
        return result
    _, _, _, api_client = result

    try:
        auth_v1 = k8s_client.AuthorizationV1Api(api_client)

        if service_account:
            # SubjectAccessReview — check for a specific service account
            sar_spec = k8s_client.V1SubjectAccessReviewSpec(
                user=f"system:serviceaccount:{ns}:{service_account}",
                resource_attributes=k8s_client.V1ResourceAttributes(
                    namespace=ns,
                    verb=verb,
                    resource=normalised,
                    name=resource_name or "",
                ),
            )
            sar = k8s_client.V1SubjectAccessReview(spec=sar_spec)
            review = auth_v1.create_subject_access_review(body=sar)
        else:
            # SelfSubjectAccessReview — check for the current user
            self_spec = k8s_client.V1SelfSubjectAccessReviewSpec(
                resource_attributes=k8s_client.V1ResourceAttributes(
                    namespace=ns,
                    verb=verb,
                    resource=normalised,
                    name=resource_name or "",
                ),
            )
            self_sar = k8s_client.V1SelfSubjectAccessReview(spec=self_spec)
            review = auth_v1.create_self_subject_access_review(body=self_sar)

        status = review.status
        allowed = status.allowed if status else False
        reason = status.reason or "" if status else ""
        denied = getattr(status, "denied", False) if status else False
        evaluation_error = getattr(status, "evaluation_error", "") if status else ""

        lines: list[str] = []
        lines.append("=== RBAC Permission Check ===")
        if service_account:
            lines.append(f"Subject: system:serviceaccount:{ns}:{service_account}")
        else:
            lines.append("Subject: current user (self)")
        lines.append(f"Action: {verb} {normalised}")
        if resource_name:
            lines.append(f"Resource Name: {resource_name}")
        lines.append(f"Namespace: {ns}")
        lines.append("")
        lines.append(f"Allowed: {'YES' if allowed else 'NO'}")
        if denied:
            lines.append("Explicitly Denied: YES")
        if reason:
            lines.append(f"Reason: {reason}")
        if evaluation_error:
            lines.append(f"Evaluation Error: {evaluation_error}")

        return ToolResult(output="\n".join(lines), error=False)

    except k8s_exceptions.ApiException as exc:
        if exc.status in (401, 403):
            return ToolResult(
                output=f"Permission denied performing RBAC check: {exc.reason}",
                error=True,
            )
        return ToolResult(output=f"K8s API error ({exc.status}): {exc.reason}", error=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(output=f"Error checking RBAC: {exc}", error=True)

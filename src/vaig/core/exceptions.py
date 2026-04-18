"""Custom exceptions for the VAIG toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaig.agents.chunked import ChunkResult

# ── Shared context-window error keywords ─────────────────────
# Canonical list of keywords (case-insensitive) that identify a context /
# token-limit error from the Gemini API.  Imported by both ``client.py``
# (low-level retry layer) and ``mixins.py`` (tool-loop layer) so the
# detection logic stays in one place.
CONTEXT_WINDOW_ERROR_KEYWORDS: tuple[str, ...] = (
    "context window",
    "token limit",
    "max tokens",
    "maximum tokens",
    "prompt is too long",
    "too many tokens",
    "exceeds the maximum",
    "exceeds the maximum allowed size",
    "content too large",
    "resource_exhausted",
    "request payload size exceeds",
)


class VAIGError(Exception):
    """Base exception for all VAIG errors."""


class GeminiClientError(VAIGError):
    """Error raised when the Gemini client encounters an unrecoverable failure.

    Wraps the original exception and includes retry context when applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: Exception | None = None,
        retries_attempted: int = 0,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.retries_attempted = retries_attempted


class GeminiRateLimitError(GeminiClientError):
    """Raised when the API returns 429 (ResourceExhausted) after all retries."""


class GeminiConnectionError(GeminiClientError):
    """Raised when the API returns a server/connection error (500/502/503/504) after all retries."""


class ToolExecutionError(VAIGError):
    """Raised when a tool fails to execute."""

    def __init__(self, message: str, *, tool_name: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class HelmError(ToolExecutionError):
    """Raised when a Helm tool operation fails.

    Covers K8s API errors, Helm secret decoding failures (base64/gzip/JSON),
    and other Helm-specific runtime issues.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, tool_name="helm")


class ArgoCDError(ToolExecutionError):
    """Raised when an ArgoCD tool operation fails.

    Covers K8s API errors, kubeconfig loading failures, ArgoCD CRD access
    issues, and other ArgoCD-specific runtime problems.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, tool_name="argocd")


class MaxIterationsError(VAIGError):
    """Raised when the tool-use loop exceeds the configured maximum iterations.

    Attributes:
        iterations: The iteration count at which the limit was hit.
        partial_output: Any LLM text accumulated before the limit was hit.
            Defaults to empty string for backward compatibility.
    """

    def __init__(self, message: str, *, iterations: int, partial_output: str = "") -> None:
        super().__init__(message)
        self.iterations = iterations
        self.partial_output = partial_output


class ChunkedProcessingError(VAIGError):
    """Raised when chunked file processing fails (partial or total failure).

    Carries context about which chunks failed and any partial results
    collected before the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        total_chunks: int = 0,
        failed_chunks: list[int] | None = None,
        partial_results: list[ChunkResult] | None = None,
    ) -> None:
        super().__init__(message)
        self.total_chunks = total_chunks
        self.failed_chunks = failed_chunks or []
        self.partial_results = partial_results or []


class VaigAuthError(VAIGError):
    """Authentication failure — missing credentials, expired tokens, etc."""


class GCPAuthError(VaigAuthError):
    """GCP-specific authentication error."""

    def __init__(self, message: str, fix_suggestion: str = "") -> None:
        self.fix_suggestion = fix_suggestion
        super().__init__(message)


class GCPPermissionError(VaigAuthError):
    """GCP permission denied — missing IAM roles."""

    def __init__(
        self,
        message: str,
        required_permissions: list[str] | None = None,
        fix_suggestion: str = "",
    ) -> None:
        self.required_permissions = required_permissions or []
        self.fix_suggestion = fix_suggestion
        super().__init__(message)


class K8sAuthError(VaigAuthError):
    """Kubernetes authentication/authorization error."""


class QuotaExceededError(VAIGError):
    """Raised when a user exceeds their rate-limit quota for a dimension.

    Attributes:
        dimension: Which quota was exceeded (``"requests_per_day"``, ``"tokens_per_day"``, ``"executions_per_day"``).
        used: Current usage count at the time of rejection.
        limit: The configured limit for this dimension.
        user_key: The composite user key that was rate-limited.
    """

    def __init__(
        self,
        *,
        dimension: str,
        used: int,
        limit: int,
        user_key: str,
    ) -> None:
        self.dimension = dimension
        self.used = used
        self.limit = limit
        self.user_key = user_key
        super().__init__(
            f"Quota exceeded for '{user_key}': {dimension} usage {used:,}/{limit:,} per day"
        )


class BudgetExhaustedError(VAIGError):
    """Raised when the global budget (tokens, cost, tool calls, or wall time) is exhausted.

    Attributes:
        dimension: Which limit was hit (``"tokens"``, ``"cost_usd"``, ``"tool_calls"``, ``"wall_seconds"``).
        used: Current usage at the time of rejection.
        limit: The configured limit for this dimension.
    """

    def __init__(self, *, dimension: str, used: float, limit: float) -> None:
        self.dimension = dimension
        self.used = used
        self.limit = limit
        super().__init__(
            f"Global budget exhausted: {dimension} usage {used}/{limit}"
        )


class CircuitBreakerOpenError(VAIGError):
    """Raised when the circuit breaker is in the OPEN state and rejects a request.

    Attributes:
        failure_count: Number of consecutive failures that tripped the breaker.
        recovery_timeout: Seconds until the breaker enters HALF-OPEN state.
    """

    def __init__(self, *, failure_count: int, recovery_timeout: float) -> None:
        self.failure_count = failure_count
        self.recovery_timeout = recovery_timeout
        super().__init__(
            f"Circuit breaker is OPEN after {failure_count} failures; "
            f"retry after {recovery_timeout:.1f}s"
        )


class TokenBudgetError(VAIGError):
    """Raised when the token budget cannot be computed or is invalid.

    Examples: context_window is too small to fit even the system prompt,
    or count_tokens() fails and no fallback is possible.
    """


class ContextWindowExceededError(VAIGError):
    """Raised when the API rejects a request because the context window is exceeded.

    Wraps the ``google.api_core.exceptions.InvalidArgument`` (HTTP 400) that
    Gemini returns when the prompt is too large for the model's context window.

    Attributes:
        message: Human-readable description.
        context_pct: Estimated context window percentage at the time of failure.
    """

    def __init__(
        self,
        message: str,
        *,
        context_pct: float = 0.0,
        usage: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.context_pct = context_pct
        self.usage: dict[str, int] = usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def format_error_for_user(exc: Exception, *, debug: bool = False) -> str:
    """Format an exception for user display using Rich markup.

    In normal mode: shows a clean, actionable message with fix suggestions.
    In debug mode: additionally includes the full traceback.
    """
    import traceback

    lines: list[str] = []

    if isinstance(exc, GCPPermissionError):
        lines.append(f"[red]Permission Denied:[/red] {exc}")
        if exc.required_permissions:
            lines.append(f"[yellow]Required permissions:[/yellow] {', '.join(exc.required_permissions)}")
        if exc.fix_suggestion:
            lines.append(f"[yellow]Fix:[/yellow] {exc.fix_suggestion}")
    elif isinstance(exc, GCPAuthError):
        lines.append(f"[red]Authentication Error:[/red] {exc}")
        if exc.fix_suggestion:
            lines.append(f"[yellow]Fix:[/yellow] {exc.fix_suggestion}")
    elif isinstance(exc, K8sAuthError):
        lines.append(f"[red]Kubernetes Auth Error:[/red] {exc}")
        lines.append("[yellow]Fix:[/yellow] Check your kubeconfig: kubectl config current-context")
    elif isinstance(exc, VaigAuthError):
        lines.append(f"[red]Authentication Error:[/red] {exc}")
    elif isinstance(exc, ContextWindowExceededError):
        lines.append(f"[red]Context Window Exceeded:[/red] {exc}")
        lines.append(
            f"[yellow]Context usage:[/yellow] {exc.context_pct:.1f}% of context window was consumed. "
            "Reduce prompt size or use a model with a larger context window."
        )
    elif isinstance(exc, VAIGError):
        lines.append(f"[red]Error:[/red] {exc}")
    else:
        lines.append(f"[red]Unexpected Error:[/red] {type(exc).__name__}: {exc}")

    if debug:
        lines.append("")
        lines.append("[dim]Full traceback:[/dim]")
        lines.append(traceback.format_exc())
    else:
        lines.append("[dim]Use --debug for full traceback[/dim]")

    return "\n".join(lines)

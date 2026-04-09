"""Remediation engine — command classification and safety-tiered execution.

Provides the foundation types, classifier, and executor for the runbook
execution engine:

- :class:`SafetyTier` — three-level safety classification (SAFE/REVIEW/BLOCKED).
- :class:`ClassifiedCommand` — frozen dataclass representing a parsed command
  with its safety tier assignment.
- :class:`CommandClassifier` — classifies command strings into safety tiers
  using ordered regex rules with a default-BLOCKED policy.
- :class:`ToolResult` — frozen dataclass capturing execution outcome.
- :class:`RemediationExecutor` — dispatches classified commands to native K8s
  functions or subprocess, with safety gating and audit event emission.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from vaig.tools.shell_tools import _check_denied_command

if TYPE_CHECKING:
    from vaig.core.config import GKEConfig, RemediationConfig, ReviewConfig
    from vaig.core.event_bus import EventBus
    from vaig.core.review_store import ReviewStore
    from vaig.skills.service_health.schema import RecommendedAction
    from vaig.tools.base import ToolResult

logger = logging.getLogger(__name__)


class SafetyTier(StrEnum):
    """Three-level safety classification for remediation commands.

    - ``SAFE``: Auto-executable with ``--approve`` (e.g. kubectl annotate).
    - ``REVIEW``: Requires explicit ``--execute`` after plan review.
    - ``BLOCKED``: Never executed — explains why and suggests alternatives.
    """

    SAFE = "safe"
    REVIEW = "review"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ClassifiedCommand:
    """A parsed command with safety tier assignment.

    Attributes:
        tool: The CLI tool (e.g. ``"kubectl"``, ``"helm"``).
        subcommand: The subcommand (e.g. ``"scale"``, ``"upgrade"``).
        args: Remaining arguments after tool and subcommand.
        tier: Safety tier assignment from the classifier.
        raw_command: Original command string before parsing.
        block_reason: Explanation when ``tier`` is ``BLOCKED``, else ``None``.
    """

    tool: str
    subcommand: str
    args: tuple[str, ...]
    tier: SafetyTier
    raw_command: str
    block_reason: str | None = None


# ── Dangerous shell meta-characters ──────────────────────────────

# Any command containing these patterns is BLOCKED immediately — before
# any tier rule matching.  This prevents shell injection attacks via
# pipes, redirects, subshells, and command chaining.
_DANGEROUS_PATTERNS: re.Pattern[str] = re.compile(
    r"[|><;`]"        # pipe, redirect, semicolon, backtick
    r"|&&"             # logical AND chaining
    r"|\|\|"           # logical OR chaining
    r"|\$\("           # subshell via $(...)
)


# ── Tier rule definitions ────────────────────────────────────────

# Each rule is a (tool_regex, subcommand_regex, tier) tuple.
# Matched top-to-bottom — first match wins.
# BLOCKED rules come first so they can't be bypassed by SAFE/REVIEW rules.
TierRule = tuple[str, str, SafetyTier]

_TIER_RULES: list[TierRule] = [
    # ── BLOCKED: destructive operations ──
    (r"^kubectl$", r"^delete$", SafetyTier.BLOCKED),       # kubectl delete (anything) — default blocked
    (r"^helm$", r"^uninstall$", SafetyTier.BLOCKED),        # helm uninstall
    (r"^helm$", r"^delete$", SafetyTier.BLOCKED),           # helm delete (alias)

    # ── SAFE: low-risk metadata & scaling ──
    (r"^kubectl$", r"^annotate$", SafetyTier.SAFE),
    (r"^kubectl$", r"^label$", SafetyTier.SAFE),
    (r"^kubectl$", r"^scale$", SafetyTier.SAFE),
    (r"^kubectl$", r"^rollout$", SafetyTier.SAFE),          # rollout restart/status

    # ── REVIEW: impactful but legitimate operations ──
    (r"^kubectl$", r"^apply$", SafetyTier.REVIEW),
    (r"^kubectl$", r"^patch$", SafetyTier.REVIEW),
    (r"^kubectl$", r"^set$", SafetyTier.REVIEW),
    (r"^kubectl$", r"^cordon$", SafetyTier.REVIEW),
    (r"^kubectl$", r"^uncordon$", SafetyTier.REVIEW),
    (r"^kubectl$", r"^drain$", SafetyTier.REVIEW),
    (r"^kubectl$", r"^taint$", SafetyTier.REVIEW),
    (r"^helm$", r"^upgrade$", SafetyTier.REVIEW),
    (r"^helm$", r"^install$", SafetyTier.REVIEW),
    (r"^helm$", r"^rollback$", SafetyTier.REVIEW),

    # ── Read-only kubectl: SAFE ──
    (r"^kubectl$", r"^get$", SafetyTier.SAFE),
    (r"^kubectl$", r"^describe$", SafetyTier.SAFE),
    (r"^kubectl$", r"^logs$", SafetyTier.SAFE),
    (r"^kubectl$", r"^top$", SafetyTier.SAFE),

    # ── Read-only helm: SAFE ──
    (r"^helm$", r"^list$", SafetyTier.SAFE),
    (r"^helm$", r"^status$", SafetyTier.SAFE),
    (r"^helm$", r"^get$", SafetyTier.SAFE),
    (r"^helm$", r"^history$", SafetyTier.SAFE),
]

# Special rule: ``kubectl delete pod`` is REVIEW (restart semantics),
# but ``kubectl delete deployment/namespace/...`` stays BLOCKED.
# This is handled in the classifier logic, not in _TIER_RULES.
_KUBECTL_DELETE_POD_PATTERN = re.compile(
    r"^pod(/|\s|$)",
    re.IGNORECASE,
)

# ``kubectl rollout restart`` and ``rollout status`` stay SAFE (the tier-rule
# default), but ``rollout undo``, ``rollout pause``, and ``rollout resume``
# are promoted to REVIEW because they mutate workload state.
_KUBECTL_ROLLOUT_REVIEW_SUBCMDS = re.compile(
    r"^(undo|pause|resume)$",
    re.IGNORECASE,
)


def _load_coding_denied_commands() -> list[str]:
    """Load denied_commands from Settings().coding — best-effort.

    Returns an empty list if Settings can't be loaded (e.g. during
    testing without config files).
    """
    try:
        from vaig.core.config import get_settings

        return get_settings().coding.denied_commands
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:  # noqa: BLE001
        return []


class CommandClassifier:
    """Classify command strings into safety tiers.

    Uses an ordered list of ``(tool_regex, subcommand_regex, tier)``
    rules with a default-BLOCKED policy: any command that doesn't match
    an explicit rule is classified as BLOCKED.

    Integrates with :func:`~vaig.tools.shell_tools._check_denied_command`
    to respect the coding config's deny list, and supports per-config
    ``blocked_commands`` and ``tier_overrides``.

    Args:
        config: Remediation configuration.
        coding_denied_commands: Optional explicit deny list.  When
            ``None`` (the default), the list is loaded from
            ``Settings().coding.denied_commands``.  Pass an explicit
            list to override (useful in tests or when Settings is
            already available).
    """

    def __init__(
        self,
        config: RemediationConfig,
        coding_denied_commands: list[str] | None = None,
    ) -> None:
        self._config = config
        self._tier_rules = list(_TIER_RULES)
        if coding_denied_commands is not None:
            self._coding_denied = coding_denied_commands
        else:
            self._coding_denied = _load_coding_denied_commands()

    def classify(self, command: str) -> ClassifiedCommand:
        """Parse and classify a command string into a :class:`ClassifiedCommand`.

        Classification order (first match wins):

        1. Empty / unparseable → BLOCKED
        2. Dangerous shell meta-characters (``|``, ``>``, ``;``, etc.) → BLOCKED
        3. ``config.blocked_commands`` regex patterns → BLOCKED
        4. ``_check_denied_command()`` from shell_tools → BLOCKED
        5. Tier rules (ordered list) → matched tier
        6. Special case: ``kubectl delete pod/...`` → REVIEW
        7. ``config.tier_overrides`` (post-rule override) → overridden tier
        8. No rule matched → BLOCKED (default-deny)
        """
        raw = command.strip()

        # ── Step 1: empty command ──
        if not raw:
            return ClassifiedCommand(
                tool="",
                subcommand="",
                args=(),
                tier=SafetyTier.BLOCKED,
                raw_command=command,
                block_reason="Empty command",
            )

        # ── Step 2: dangerous shell meta-characters ──
        if _DANGEROUS_PATTERNS.search(raw):
            tool, subcommand, args = self._parse_best_effort(raw)
            return ClassifiedCommand(
                tool=tool,
                subcommand=subcommand,
                args=args,
                tier=SafetyTier.BLOCKED,
                raw_command=command,
                block_reason="Command contains dangerous shell operators "
                "(pipe, redirect, subshell, or chaining)",
            )

        # ── Step 3: config blocked_commands ──
        if self._config.blocked_commands:
            reason = _check_denied_command(raw, self._config.blocked_commands)
            if reason:
                tool, subcommand, args = self._parse_best_effort(raw)
                return ClassifiedCommand(
                    tool=tool,
                    subcommand=subcommand,
                    args=args,
                    tier=SafetyTier.BLOCKED,
                    raw_command=command,
                    block_reason=reason,
                )

        # ── Step 4: parse with shlex ──
        try:
            parts = shlex.split(raw)
        except ValueError:
            # Malformed quoting — can't parse safely
            tool, subcommand, args = self._parse_best_effort(raw)
            return ClassifiedCommand(
                tool=tool,
                subcommand=subcommand,
                args=args,
                tier=SafetyTier.BLOCKED,
                raw_command=command,
                block_reason="Malformed command — failed to parse",
            )

        if not parts:
            return ClassifiedCommand(
                tool="",
                subcommand="",
                args=(),
                tier=SafetyTier.BLOCKED,
                raw_command=command,
                block_reason="Empty command after parsing",
            )

        tool = parts[0]
        subcommand = parts[1] if len(parts) > 1 else ""
        args = tuple(parts[2:]) if len(parts) > 2 else ()

        # ── Step 5: _check_denied_command from shell_tools ──
        # Uses CodingConfig.denied_commands patterns (loaded at init)
        # to catch globally blocked commands (rm -rf, sudo, etc.)
        if self._coding_denied:
            reason = _check_denied_command(raw, self._coding_denied)
            if reason:
                return ClassifiedCommand(
                    tool=tool,
                    subcommand=subcommand,
                    args=args,
                    tier=SafetyTier.BLOCKED,
                    raw_command=command,
                    block_reason=reason,
                )

        # ── Step 6: match against tier rules ──
        matched_tier: SafetyTier | None = None
        for tool_pattern, subcmd_pattern, tier in self._tier_rules:
            if re.match(tool_pattern, tool, re.IGNORECASE) and re.match(
                subcmd_pattern, subcommand, re.IGNORECASE
            ):
                matched_tier = tier
                break

        # ── Step 7: special case — kubectl delete pod → REVIEW ──
        if (
            matched_tier == SafetyTier.BLOCKED
            and tool.lower() == "kubectl"
            and subcommand.lower() == "delete"
            and args
            and _KUBECTL_DELETE_POD_PATTERN.match(args[0])
        ):
            matched_tier = SafetyTier.REVIEW

        # ── Step 7b: special case — kubectl rollout undo/pause/resume → REVIEW ──
        if (
            matched_tier == SafetyTier.SAFE
            and tool.lower() == "kubectl"
            and subcommand.lower() == "rollout"
            and args
            and _KUBECTL_ROLLOUT_REVIEW_SUBCMDS.match(args[0])
        ):
            matched_tier = SafetyTier.REVIEW

        # ── Step 8: config tier_overrides ──
        if self._config.tier_overrides:
            for pattern, tier_name in self._config.tier_overrides.items():
                try:
                    if re.search(pattern, raw, re.IGNORECASE):
                        try:
                            matched_tier = SafetyTier(tier_name.lower())
                        except ValueError:
                            logger.warning(
                                "Invalid tier_override value %r for pattern %r — ignoring",
                                tier_name,
                                pattern,
                            )
                        break
                except re.error:
                    logger.warning(
                        "Invalid tier_override regex pattern: %r — ignoring",
                        pattern,
                    )

        # ── Step 9: default-deny — no rule matched → BLOCKED ──
        if matched_tier is None:
            return ClassifiedCommand(
                tool=tool,
                subcommand=subcommand,
                args=args,
                tier=SafetyTier.BLOCKED,
                raw_command=command,
                block_reason=f"No tier rule matched for {tool} {subcommand}",
            )

        return ClassifiedCommand(
            tool=tool,
            subcommand=subcommand,
            args=args,
            tier=matched_tier,
            raw_command=command,
            block_reason=(
                f"Matched BLOCKED rule for {tool} {subcommand}"
                if matched_tier == SafetyTier.BLOCKED
                else None
            ),
        )

    @staticmethod
    def _parse_best_effort(raw: str) -> tuple[str, str, tuple[str, ...]]:
        """Best-effort parse when shlex fails — split on whitespace."""
        parts = raw.split()
        tool = parts[0] if parts else ""
        subcommand = parts[1] if len(parts) > 1 else ""
        args = tuple(parts[2:]) if len(parts) > 2 else ()
        return tool, subcommand, args


# ── Native dispatch map ──────────────────────────────────────
# Maps (tool, subcommand) → async wrapper function name.
# Used by RemediationExecutor to dispatch kubectl mutations
# through the native Kubernetes client instead of subprocess.

_NATIVE_DISPATCH: dict[tuple[str, str], str] = {
    ("kubectl", "scale"): "async_kubectl_scale",
    ("kubectl", "rollout"): "async_kubectl_restart",  # rollout restart
    ("kubectl", "annotate"): "async_kubectl_annotate",
    ("kubectl", "label"): "async_kubectl_label",
}


class RemediationExecutor:
    """Execute classified remediation commands with safety gating.

    Dispatches commands to either native Kubernetes client functions
    (for ``kubectl scale/rollout/annotate/label``) or to subprocess
    execution (for ``helm`` and other tools).

    Safety gating:
    - ``BLOCKED`` commands are never executed — an error ToolResult is
      returned and a :class:`RemediationExecuted` event is emitted.
    - ``REVIEW`` commands require explicit ``approved=True`` to execute;
      otherwise a plan ToolResult is returned (no execution, no event).
    - ``SAFE`` commands execute directly when ``approved=True``.
    - ``dry_run=True`` returns a description without executing.

    An :class:`asyncio.Lock` ensures sequential execution — only one
    remediation command runs at a time.

    Args:
        config: Remediation configuration (timeouts, dry_run defaults).
        bus: Event bus for emitting :class:`RemediationExecuted` events.
    """

    def __init__(
        self,
        config: RemediationConfig,
        bus: EventBus,
        *,
        review_store: ReviewStore | None = None,
        review_config: ReviewConfig | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._lock = asyncio.Lock()
        self._review_store = review_store
        self._review_config = review_config

    def validate_context(
        self, report_cluster: str, gke_config: GKEConfig
    ) -> bool:
        """Check that the report's cluster matches the live GKE config.

        Returns ``True`` if the clusters match, ``False`` otherwise.
        This prevents accidentally running remediation commands against
        a different cluster than the one the health report was generated for.
        """
        return report_cluster == gke_config.cluster_name

    async def execute(
        self,
        action: RecommendedAction,
        classified: ClassifiedCommand,
        gke_config: GKEConfig,
        *,
        dry_run: bool = False,
        approved: bool = False,
        run_id: str | None = None,
    ) -> ToolResult:
        """Execute a classified remediation command.

        Args:
            action: The recommended action from the health report.
            classified: The classified command to execute.
            gke_config: GKE cluster configuration for native dispatch.
            dry_run: If ``True``, return what *would* run without executing.
            approved: If ``True``, the command is approved for execution.
            run_id: Health-report run ID used to check review approval.

        Returns:
            A :class:`ToolResult` with the command output or plan.
        """
        from vaig.core.events import RemediationExecuted
        from vaig.tools.base import ToolResult

        # ── Wire config-level overrides ──
        if self._config.dry_run:
            dry_run = True
        if self._config.auto_approve_safe and classified.tier == SafetyTier.SAFE:
            approved = True

        async with self._lock:
            cluster = gke_config.cluster_name

            # ── REVIEW GATE: block if review required but not approved ──
            if (
                self._review_config
                and self._review_config.enabled
                and self._review_config.require_review_for_remediation
                and self._review_store
                and run_id
                and not self._review_store.is_approved(run_id)
            ):
                reason = (
                    f"Review not approved for run_id {run_id!r}. "
                    "Submit a review approval before executing remediation."
                )
                return ToolResult(output=reason, error=True)

            # ── BLOCKED: never execute ──
            if classified.tier == SafetyTier.BLOCKED:
                reason = classified.block_reason or "Command is BLOCKED"
                self._bus.emit(
                    RemediationExecuted(
                        action_title=action.title,
                        command=classified.raw_command,
                        tier=classified.tier.value,
                        result_output="",
                        error=reason,
                        dry_run=dry_run,
                        cluster=cluster,
                    )
                )
                return ToolResult(output=reason, error=True)

            # ── REVIEW without approval: return plan only ──
            if classified.tier == SafetyTier.REVIEW and not approved:
                plan = (
                    f"[REVIEW REQUIRED] {action.title}\n"
                    f"Command: {classified.raw_command}\n"
                    f"Risk: {action.risk or 'Not specified'}\n"
                    f"Re-run with --execute to approve."
                )
                return ToolResult(output=plan, error=False)

            # ── SAFE without approval: return plan unless auto_approve_safe ──
            if classified.tier == SafetyTier.SAFE and not approved:
                if not self._config.auto_approve_safe:
                    plan = (
                        f"[APPROVAL NEEDED] {action.title}\n"
                        f"Command: {classified.raw_command}\n"
                        f"Tier: SAFE\n"
                        f"Re-run with --approve to execute."
                    )
                    return ToolResult(output=plan, error=False)

            # ── dry_run: describe without executing ──
            if dry_run:
                description = (
                    f"[DRY RUN] Would execute: {classified.raw_command}\n"
                    f"Tier: {classified.tier.value}\n"
                    f"Action: {action.title}"
                )
                return ToolResult(output=description, error=False)

            # ── Dispatch and execute ──
            try:
                result = await self._dispatch(classified, gke_config)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                error_msg = f"Execution failed: {exc}"
                logger.exception(
                    "Remediation execution failed for %r",
                    classified.raw_command,
                )
                self._bus.emit(
                    RemediationExecuted(
                        action_title=action.title,
                        command=classified.raw_command,
                        tier=classified.tier.value,
                        result_output="",
                        error=error_msg,
                        dry_run=False,
                        cluster=cluster,
                    )
                )
                return ToolResult(output=error_msg, error=True)

            # ── Emit success/failure event ──
            self._bus.emit(
                RemediationExecuted(
                    action_title=action.title,
                    command=classified.raw_command,
                    tier=classified.tier.value,
                    result_output=result.output,
                    error="" if not result.error else result.output,
                    dry_run=False,
                    cluster=cluster,
                )
            )
            return result

    async def _dispatch(
        self, classified: ClassifiedCommand, gke_config: GKEConfig
    ) -> ToolResult:
        """Route to native K8s client or subprocess execution.

        Native dispatch is used for known kubectl mutations (scale,
        rollout restart, annotate, label).  Everything else goes through
        ``async_run_command`` with the tool in ``allowed_commands``.
        """
        key = (classified.tool.lower(), classified.subcommand.lower())

        if key in _NATIVE_DISPATCH:
            return await self._dispatch_native(key, classified, gke_config)

        # Subprocess dispatch for helm and other tools
        return await self._dispatch_subprocess(classified, gke_config)

    async def _dispatch_native(
        self,
        key: tuple[str, str],
        classified: ClassifiedCommand,
        gke_config: GKEConfig,
    ) -> ToolResult:
        """Dispatch to native async kubectl wrappers."""
        from vaig.tools.gke.mutations import (
            async_kubectl_annotate,
            async_kubectl_label,
            async_kubectl_restart,
            async_kubectl_scale,
        )

        args = classified.args

        if key == ("kubectl", "scale"):
            # Expected: kubectl scale <resource/name> --replicas=N
            resource, name = self._parse_resource_name(args)
            replicas = self._extract_replicas(args)
            namespace = self._extract_namespace(args, gke_config)
            return await async_kubectl_scale(
                resource, name, replicas,
                gke_config=gke_config, namespace=namespace,
            )

        if key == ("kubectl", "rollout"):
            # Expected: kubectl rollout restart <resource/name>
            # Only "restart" is handled natively; other subsubcommands
            # (status, undo, pause, resume) fall through to subprocess.
            remaining = args
            if remaining and remaining[0].lower() == "restart":
                remaining = remaining[1:]
                resource, name = self._parse_resource_name(remaining)
                namespace = self._extract_namespace(args, gke_config)
                return await async_kubectl_restart(
                    resource, name,
                    gke_config=gke_config, namespace=namespace,
                )
            # Not "restart" — fall through to subprocess dispatch
            return await self._dispatch_subprocess(classified, gke_config)

        if key == ("kubectl", "annotate"):
            # Expected: kubectl annotate <resource/name> key=value
            resource, name = self._parse_resource_name(args)
            annotations = self._extract_kv_pairs(args)
            namespace = self._extract_namespace(args, gke_config)
            return await async_kubectl_annotate(
                resource, name, annotations,
                gke_config=gke_config, namespace=namespace,
            )

        if key == ("kubectl", "label"):
            # Expected: kubectl label <resource/name> key=value
            resource, name = self._parse_resource_name(args)
            labels = self._extract_kv_pairs(args)
            namespace = self._extract_namespace(args, gke_config)
            return await async_kubectl_label(
                resource, name, labels,
                gke_config=gke_config, namespace=namespace,
            )

        # Should never reach here — _NATIVE_DISPATCH is exhaustive
        from vaig.tools.base import ToolResult

        return ToolResult(
            output=f"No native handler for {key}", error=True,
        )

    async def _dispatch_subprocess(
        self, classified: ClassifiedCommand, gke_config: GKEConfig,
    ) -> ToolResult:
        """Dispatch via async_run_command with tool in allowed_commands."""
        from vaig.tools.shell_tools import async_run_command

        workspace = Path.cwd()
        return await async_run_command(
            classified.raw_command,
            workspace=workspace,
            allowed_commands=[classified.tool],
            timeout=self._config.timeout,
        )

    # ── Argument parsing helpers ─────────────────────────────

    @staticmethod
    def _parse_resource_name(
        args: tuple[str, ...],
    ) -> tuple[str, str]:
        """Extract resource type and name from command args.

        Handles both ``resource/name`` and ``resource name`` formats.
        Returns ``(resource_type, name)`` or ``("unknown", "unknown")``.
        """
        if not args:
            return "unknown", "unknown"

        # Skip flag args (e.g. --namespace, --replicas)
        positional = [a for a in args if not a.startswith("-")]
        if not positional:
            return "unknown", "unknown"

        first = positional[0]
        if "/" in first:
            parts = first.split("/", 1)
            return parts[0], parts[1]

        # resource name as two separate args
        if len(positional) >= 2:
            return positional[0], positional[1]

        return first, "unknown"

    @staticmethod
    def _extract_replicas(args: tuple[str, ...]) -> int:
        """Extract --replicas=N from args, defaulting to 1."""
        for arg in args:
            if arg.startswith("--replicas="):
                try:
                    return int(arg.split("=", 1)[1])
                except ValueError:
                    return 1
        return 1

    @staticmethod
    def _extract_namespace(
        args: tuple[str, ...], gke_config: GKEConfig,
    ) -> str:
        """Extract -n/--namespace from args, falling back to gke_config."""
        args_list = list(args)
        for i, arg in enumerate(args_list):
            if arg in ("-n", "--namespace") and i + 1 < len(args_list):
                return args_list[i + 1]
            if arg.startswith("--namespace="):
                return arg.split("=", 1)[1]
            if arg.startswith("-n") and len(arg) > 2:
                return arg[2:]
        return gke_config.default_namespace

    @staticmethod
    def _extract_kv_pairs(args: tuple[str, ...]) -> str:
        """Extract key=value pairs from args (for labels/annotations).

        Joins all non-flag, non-resource/name args that contain ``=``.
        """
        pairs: list[str] = []
        for arg in args:
            if "=" in arg and not arg.startswith("-"):
                pairs.append(arg)
        return ",".join(pairs) if pairs else ""

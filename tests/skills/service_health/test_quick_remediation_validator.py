"""Tests for the quick_remediation placeholder validator on Finding.

AUDIT-01 — Ban "investigate…" placeholders in quick_remediation.

Covers:
- 10 Spanish placeholder variants raise ValueError
- 10 English placeholder variants raise ValueError
- Valid remediation strings are accepted
- None / empty string are accepted
- Exact safe fallback literal is accepted
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vaig.skills.service_health.schema import Finding


def _make_finding(quick_remediation: str | None) -> Finding:
    """Helper: build a minimal Finding with the given quick_remediation."""
    return Finding(
        id="test-finding",
        title="Test Finding",
        severity="INFO",
        quick_remediation=quick_remediation,
    )


# ── Spanish placeholder variants ─────────────────────────────

SPANISH_PLACEHOLDERS = [
    "Investigar la causa raíz",
    "Investigar el problema",
    "Investigar más a fondo",
    "investigar el origen",
    "INVESTIGAR la falla",
    "Investigación requerida",
    "Revisa la causa del problema",
    "Revisar la causa raíz",
    "Analyze the issue detected",
    "investiga el pod crasheando",
]

# ── English placeholder variants ─────────────────────────────

ENGLISH_PLACEHOLDERS = [
    "Investigate the root cause",
    "investigate further",
    "Investigate the issue",
    "Look into the problem",
    "look into the root cause",
    "Check the root cause",
    "Check the issue directly",
    "Analyze the issue",
    "Investigating the failure",
    "investigate",
]


class TestSpanishPlaceholders:
    @pytest.mark.parametrize("placeholder", SPANISH_PLACEHOLDERS)
    def test_spanish_placeholder_raises(self, placeholder: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _make_finding(placeholder)
        assert "quick_remediation must be an actionable command" in str(exc_info.value)


class TestEnglishPlaceholders:
    @pytest.mark.parametrize("placeholder", ENGLISH_PLACEHOLDERS)
    def test_english_placeholder_raises(self, placeholder: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _make_finding(placeholder)
        assert "quick_remediation must be an actionable command" in str(exc_info.value)


class TestValidRemediations:
    def test_none_is_accepted(self) -> None:
        f = _make_finding(None)
        assert f.quick_remediation is None

    def test_empty_string_is_accepted(self) -> None:
        f = _make_finding("")
        assert f.quick_remediation == ""

    def test_safe_fallback_literal_accepted(self) -> None:
        f = _make_finding("(see Recommended Actions section)")
        assert f.quick_remediation == "(see Recommended Actions section)"

    def test_kubectl_command_accepted(self) -> None:
        cmd = "kubectl rollout restart deployment/payment-svc -n production"
        f = _make_finding(cmd)
        assert f.quick_remediation == cmd

    def test_helm_command_accepted(self) -> None:
        cmd = "helm upgrade my-release ./charts/app --set replicas=3"
        f = _make_finding(cmd)
        assert f.quick_remediation == cmd

    def test_concrete_config_change_accepted(self) -> None:
        cmd = "Set replicas: 3 in charts/payment/values-prd.yaml"
        f = _make_finding(cmd)
        assert f.quick_remediation == cmd

    def test_argocd_command_accepted(self) -> None:
        cmd = "argocd app sync payment-svc --force"
        f = _make_finding(cmd)
        assert f.quick_remediation == cmd

    def test_gcloud_command_accepted(self) -> None:
        cmd = "gcloud container clusters upgrade my-cluster --node-pool=pool-1"
        f = _make_finding(cmd)
        assert f.quick_remediation == cmd

    def test_sentence_with_investigate_not_at_start_accepted(self) -> None:
        """Patterns are anchored at start — mid-sentence 'investigate' should pass."""
        cmd = "kubectl logs deployment/foo -n bar | grep -i investigate"
        f = _make_finding(cmd)
        assert f.quick_remediation == cmd

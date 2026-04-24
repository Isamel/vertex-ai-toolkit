from __future__ import annotations

"""Unit tests for ATT-10 G3: attachment_context propagation through skill.py config builders."""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARKER = "MARKER_ATT10_G3"
ATTACHMENT_HEADER = "## Attached Context"


def _make_skill() -> ServiceHealthSkill:  # noqa: F821
    """Return a ServiceHealthSkill instance with minimal mocking."""
    from vaig.skills.service_health.skill import ServiceHealthSkill

    skill = ServiceHealthSkill.__new__(ServiceHealthSkill)
    # Minimal state init — avoid __init__ which triggers K8s client warm-up
    skill._prefetched_metrics = {"pods": "", "nodes": ""}
    from vaig.skills.service_health.skill import DDResolutionResult

    skill._prefetched_dd_resolution = DDResolutionResult()
    skill._enrichment_pool = None
    skill._gemini_client = None
    return skill


def _fake_settings() -> MagicMock:
    """Build a minimal settings mock that avoids real I/O."""
    settings = MagicMock()
    # GKE — use real strings everywhere a str is expected
    gke = MagicMock()
    gke.default_namespace = "default"
    gke.cluster_name = "test-cluster"
    gke.location = "us-central1"
    gke.helm_enabled = False
    gke.argocd_enabled = False
    gke.argo_rollouts_enabled = False

    # model_copy must return an object with the same string attributes
    def _model_copy(update: dict) -> MagicMock:  # noqa: ANN001
        copy = MagicMock()
        copy.default_namespace = update.get("default_namespace", gke.default_namespace)
        copy.cluster_name = update.get("cluster_name", gke.cluster_name)
        copy.location = update.get("location", gke.location)
        copy.helm_enabled = gke.helm_enabled
        copy.argocd_enabled = gke.argocd_enabled
        copy.argo_rollouts_enabled = gke.argo_rollouts_enabled
        return copy

    gke.model_copy = _model_copy
    settings.gke = gke
    settings.datadog.enabled = False
    settings.investigation.enabled = False
    return settings


# ---------------------------------------------------------------------------
# Parameterized config-function list
# ---------------------------------------------------------------------------

CONFIG_FNS = ["get_parallel_agents_config", "get_sequential_agents_config"]


# ---------------------------------------------------------------------------
# T3.2-A: identity without attachment_context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_fn_name", CONFIG_FNS)
def test_configs_identity_without_context(config_fn_name: str) -> None:
    """Calling config builders with no attachment_context → no ATTACHMENT_HEADER in prompts."""
    skill = _make_skill()
    fake_settings = _fake_settings()

    with (
        patch("vaig.core.config.get_settings", return_value=fake_settings),
        patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        patch.object(skill, "_detect_argocd", return_value=False),
        patch.object(skill, "_detect_argo_rollouts", return_value=False),
    ):
        config_fn = getattr(skill, config_fn_name)
        configs = config_fn()

    for cfg in configs:
        si = cfg.get("system_instruction", "")
        assert ATTACHMENT_HEADER not in si, (
            f"Agent '{cfg.get('name')}' has ATTACHMENT_HEADER in system_instruction "
            f"even though attachment_context was not passed (config_fn={config_fn_name})"
        )


# ---------------------------------------------------------------------------
# T3.2-B: propagate attachment_context to every agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_fn_name", CONFIG_FNS)
def test_configs_propagate_context(config_fn_name: str) -> None:
    """Calling config builders with attachment_context → every agent's system_instruction
    contains both the ATTACHMENT_HEADER and the marker text."""
    skill = _make_skill()
    fake_settings = _fake_settings()

    with (
        patch("vaig.core.config.get_settings", return_value=fake_settings),
        patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        patch.object(skill, "_detect_argocd", return_value=False),
        patch.object(skill, "_detect_argo_rollouts", return_value=False),
    ):
        config_fn = getattr(skill, config_fn_name)
        configs = config_fn(attachment_context=MARKER)

    assert configs, f"{config_fn_name} returned empty list"

    for cfg in configs:
        si = cfg.get("system_instruction", "")
        assert ATTACHMENT_HEADER in si, (
            f"Agent '{cfg.get('name')}' is missing ATTACHMENT_HEADER (config_fn={config_fn_name})"
        )
        assert MARKER in si, f"Agent '{cfg.get('name')}' is missing marker '{MARKER}' (config_fn={config_fn_name})"


# ---------------------------------------------------------------------------
# T3.2-C: get_agents_config forwards attachment_context
# ---------------------------------------------------------------------------


def test_get_agents_config_forwards_attachment_context() -> None:
    """get_agents_config(**kwargs) must forward attachment_context to get_parallel_agents_config."""
    skill = _make_skill()
    fake_settings = _fake_settings()

    with (
        patch("vaig.core.config.get_settings", return_value=fake_settings),
        patch("vaig.tools.gke._clients.detect_autopilot", return_value=False),
        patch.object(skill, "_detect_argocd", return_value=False),
        patch.object(skill, "_detect_argo_rollouts", return_value=False),
    ):
        configs = skill.get_agents_config(attachment_context=MARKER)

    assert configs, "get_agents_config returned empty list"
    for cfg in configs:
        si = cfg.get("system_instruction", "")
        assert ATTACHMENT_HEADER in si, f"Agent '{cfg.get('name')}' missing ATTACHMENT_HEADER via get_agents_config"
        assert MARKER in si, f"Agent '{cfg.get('name')}' missing marker via get_agents_config"

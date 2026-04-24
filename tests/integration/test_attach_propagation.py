"""Integration test: end-to-end propagation of --attach content into prompt context.

This is the end-to-end proof that ATT-10 is fixed: content from an attachment
(a local file) reaches the attachment_context argument that every prompt builder
receives, so the LLM actually sees the attached content.

The test calls execute_skill_headless directly with a real SingleFileAdapter
pointing to a tmp_path file containing the sentinel token
``ATT10_FIXTURE_TOKEN_XYZ``.  The orchestrator layer is mocked so no Vertex AI
calls are made.  We assert the captured attachment_context contains the token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ── Patch targets ─────────────────────────────────────────────────────────────

_P_REGISTER = "vaig.core.gke.register_live_tools"
_P_ORCHESTRATOR = "vaig.agents.orchestrator.Orchestrator"
_P_CLIENT = "vaig.core.client.GeminiClient"
_P_CREDS = "vaig.core.auth.get_gke_credentials"

# ── Sentinel ──────────────────────────────────────────────────────────────────

FIXTURE_TOKEN = "ATT10_FIXTURE_TOKEN_XYZ"

# ── Minimal fakes ─────────────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    skill_name: str = "service-health"
    phase: str = "execute"
    synthesized_output: str = "ok"
    success: bool = True
    run_cost_usd: float = 0.001
    structured_report: Any = None
    agent_results: list[Any] = field(default_factory=list)


class _FakeSkillMeta:
    name: str = "service-health"
    display_name: str = "Service Health"


class _FakeSkill:
    def get_metadata(self) -> _FakeSkillMeta:
        return _FakeSkillMeta()


class _FakeToolRegistry:
    def list_tools(self) -> list[str]:
        return ["kubectl_get_pods"]


def _make_gke_config():
    from vaig.core.config import GKEConfig

    return GKEConfig(
        cluster_name="test-cluster",
        location="us-central1",
        project="test-project",
        default_namespace="default",
    )


def _make_settings():
    from vaig.core.config import Settings

    return Settings()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestAttachPropagationEndToEnd:
    """T7.1 — Integration proof that attachment content reaches prompt context."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_fixture_token_reaches_attachment_context(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
        tmp_path: Path,
    ):
        """The sentinel token written to a temp file is visible in attachment_context
        forwarded to orchestrator.execute_with_tools.

        This proves the full plumbing:
          SingleFileAdapter(tmp_file) → execute_skill_headless
          → RepoIndex.build_from_attachments → _render_attachment_context
          → orchestrator.execute_with_tools(attachment_context=...)
        """
        # Arrange: write fixture file
        fixture_file = tmp_path / "fixture.md"
        fixture_file.write_text(f"# Fixture\n\n{FIXTURE_TOKEN}\n")

        # Arrange: build a real SingleFileAdapter pointing to the fixture
        from vaig.core.attachment_adapter import AttachmentKind, AttachmentSpec, SingleFileAdapter

        spec = AttachmentSpec(source=str(fixture_file), kind=AttachmentKind.single_file)
        adapter = SingleFileAdapter(path=fixture_file, spec=spec)

        # Arrange: mock infra layer
        fake_result = _FakeResult()
        mock_register.return_value = _FakeToolRegistry()
        mock_orch = MagicMock()
        captured: dict[str, Any] = {}

        def _capture_execute_with_tools(**kwargs: Any) -> _FakeResult:
            captured.update(kwargs)
            return fake_result

        mock_orch.execute_with_tools.side_effect = _capture_execute_with_tools
        mock_orch_cls.return_value = mock_orch
        mock_client_cls.return_value = MagicMock()

        # Act
        from vaig.core.headless import execute_skill_headless

        execute_skill_headless(
            settings=_make_settings(),
            skill=_FakeSkill(),
            query="test",
            gke_config=_make_gke_config(),
            attachment_adapters=[adapter],
        )

        # Assert: attachment_context was forwarded
        assert "attachment_context" in captured, "execute_with_tools was not called with attachment_context kwarg"
        ctx = captured["attachment_context"]
        assert ctx is not None, "attachment_context must not be None when adapters provided"
        assert FIXTURE_TOKEN in ctx, (
            f"Sentinel token '{FIXTURE_TOKEN}' not found in attachment_context.\nCaptured context:\n{ctx!r}"
        )

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_no_adapters_attachment_context_is_none(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        """Baseline: without attachment_adapters, attachment_context is None."""
        fake_result = _FakeResult()
        mock_register.return_value = _FakeToolRegistry()
        mock_orch = MagicMock()
        captured: dict[str, Any] = {}

        def _capture(**kwargs: Any) -> _FakeResult:
            captured.update(kwargs)
            return fake_result

        mock_orch.execute_with_tools.side_effect = _capture
        mock_orch_cls.return_value = mock_orch
        mock_client_cls.return_value = MagicMock()

        from vaig.core.headless import execute_skill_headless

        execute_skill_headless(
            settings=_make_settings(),
            skill=_FakeSkill(),
            query="test",
            gke_config=_make_gke_config(),
        )

        assert captured.get("attachment_context") is None

"""Unit tests for ATT-10 G5: attachment_adapters → RepoIndex → attachment_context in headless.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Patch targets (lazy imports inside headless.py) ───────────────────────────
_P_REGISTER = "vaig.core.gke.register_live_tools"
_P_ORCHESTRATOR = "vaig.agents.orchestrator.Orchestrator"
_P_CLIENT = "vaig.core.client.GeminiClient"
_P_CREDS = "vaig.core.auth.get_gke_credentials"
_P_REPO_INDEX = "vaig.core.repo_index.RepoIndex"
_P_RENDER = "vaig.core.headless._render_attachment_context"

# ── Minimal fakes ─────────────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    skill_name: str = "discovery"
    phase: str = "execute"
    synthesized_output: str = "test output"
    success: bool = True
    run_cost_usd: float = 0.001
    structured_report: Any = None
    agent_results: list[Any] = field(default_factory=list)


class _FakeSkill:
    def get_metadata(self) -> _FakeSkillMeta:
        return _FakeSkillMeta()


class _FakeSkillMeta:
    name: str = "discovery"
    display_name: str = "Discovery"


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


def _patched_base(mock_register, mock_orch_cls, mock_client_cls, fake_result=None):
    """Configure the standard base mocks for headless execution."""
    fake_result = fake_result or _FakeResult()
    mock_register.return_value = _FakeToolRegistry()
    mock_orch = MagicMock()
    mock_orch.execute_with_tools.return_value = fake_result
    mock_orch_cls.return_value = mock_orch
    mock_client_cls.return_value = MagicMock()
    return mock_orch


# ── T5.1 + baseline: no adapters → unchanged behaviour ───────────────────────


class TestHeadlessIdentityWithoutAdapters:
    """T5.4 test_headless_identity_without_adapters: when attachment_adapters is None,
    execute_with_tools is called WITHOUT attachment_context (or with None)."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_no_attachment_context_when_adapters_none(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        from vaig.core.headless import execute_skill_headless

        mock_orch = _patched_base(mock_register, mock_orch_cls, mock_client_cls)

        execute_skill_headless(
            settings=_make_settings(),
            skill=_FakeSkill(),
            query="test",
            gke_config=_make_gke_config(),
        )

        call_kwargs = mock_orch.execute_with_tools.call_args.kwargs
        # Either not present or explicitly None
        assert call_kwargs.get("attachment_context") is None

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_repoindex_not_called_when_no_adapters(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        with patch(_P_REPO_INDEX) as mock_ri:
            from vaig.core.headless import execute_skill_headless

            _patched_base(mock_register, mock_orch_cls, mock_client_cls)

            execute_skill_headless(
                settings=_make_settings(),
                skill=_FakeSkill(),
                query="test",
                gke_config=_make_gke_config(),
            )

            mock_ri.build_from_attachments.assert_not_called()


# ── T5.4: empty adapter list treated as None ─────────────────────────────────


class TestHeadlessEmptyAdaptersList:
    """T5.4 test_headless_empty_adapters_list: [] → no RepoIndex build."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_empty_adapters_no_repoindex(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        with patch(_P_REPO_INDEX) as mock_ri:
            from vaig.core.headless import execute_skill_headless

            mock_orch = _patched_base(mock_register, mock_orch_cls, mock_client_cls)

            execute_skill_headless(
                settings=_make_settings(),
                skill=_FakeSkill(),
                query="test",
                gke_config=_make_gke_config(),
                attachment_adapters=[],
            )

            mock_ri.build_from_attachments.assert_not_called()
            call_kwargs = mock_orch.execute_with_tools.call_args.kwargs
            assert call_kwargs.get("attachment_context") is None


# ── T5.4: adapters present → context forwarded ───────────────────────────────


class TestHeadlessPassesContextWhenAdaptersPresent:
    """T5.4 test_headless_passes_context_when_adapters_present."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_context_forwarded_to_orchestrator(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        from vaig.core.headless import execute_skill_headless

        mock_orch = _patched_base(mock_register, mock_orch_cls, mock_client_cls)

        fake_adapter = MagicMock()
        fake_index = MagicMock()
        fake_index._chunks = []  # no chunks needed for render test

        expected_context = "### fixture.md\n```\nATT10_FIXTURE_TOKEN_XYZ\n```\n"

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_RENDER, return_value=expected_context):
                execute_skill_headless(
                    settings=_make_settings(),
                    skill=_FakeSkill(),
                    query="test",
                    gke_config=_make_gke_config(),
                    attachment_adapters=[fake_adapter],
                )

        mock_ri.build_from_attachments.assert_called_once()
        call_kwargs = mock_orch.execute_with_tools.call_args.kwargs
        assert call_kwargs.get("attachment_context") == expected_context


# ── T5.4: cap enforced ────────────────────────────────────────────────────────


class TestHeadlessCapEnforced:
    """T5.4 test_headless_cap_enforced: rendered output > 32 KB is truncated."""

    def test_render_truncates_at_32kb(self):
        """_render_attachment_context truncates at MAX_ATTACHMENT_CONTEXT_BYTES."""
        from vaig.core.headless import MAX_ATTACHMENT_CONTEXT_BYTES, _render_attachment_context
        from vaig.core.repo_chunkers import Chunk

        # Build a RepoIndex with a single large chunk
        big_content = "x" * (MAX_ATTACHMENT_CONTEXT_BYTES + 5_000)
        chunk = Chunk(
            file_path="bigfile.txt",
            start_line=1,
            end_line=1,
            content=big_content,
            token_estimate=9999,
            kind="text",
            outline="bigfile.txt",
        )

        from vaig.core.repo_index import RepoIndex

        index = RepoIndex([chunk])
        result = _render_attachment_context(index)

        assert len(result.encode("utf-8")) <= MAX_ATTACHMENT_CONTEXT_BYTES
        assert "[... truncated at 32 KB ...]" in result

    def test_render_no_truncation_small_content(self):
        """Small index renders without truncation marker."""
        from vaig.core.headless import MAX_ATTACHMENT_CONTEXT_BYTES, _render_attachment_context
        from vaig.core.repo_chunkers import Chunk
        from vaig.core.repo_index import RepoIndex

        chunk = Chunk(
            file_path="small.txt",
            start_line=1,
            end_line=1,
            content="hello world",
            token_estimate=3,
            kind="text",
            outline="small.txt",
        )
        index = RepoIndex([chunk])
        result = _render_attachment_context(index)

        assert len(result.encode("utf-8")) <= MAX_ATTACHMENT_CONTEXT_BYTES
        assert "[... truncated at 32 KB ...]" not in result
        assert "hello world" in result


# ── T5.4: RepoIndex error is graceful ────────────────────────────────────────


class TestHeadlessRepoIndexErrorIsGraceful:
    """T5.4 test_headless_repoindex_error_is_graceful."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_runtime_error_logs_warning_and_continues(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        from vaig.core.headless import execute_skill_headless

        mock_orch = _patched_base(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.side_effect = RuntimeError("boom")
            with patch("vaig.core.headless.logger") as mock_logger:
                result = execute_skill_headless(
                    settings=_make_settings(),
                    skill=_FakeSkill(),
                    query="test",
                    gke_config=_make_gke_config(),
                    attachment_adapters=[fake_adapter],
                )

        # Warning was logged
        mock_logger.warning.assert_called()
        warning_args = mock_logger.warning.call_args_list
        assert any("boom" in str(c) or "Failed" in str(c) for c in warning_args)

        # Orchestrator still invoked with attachment_context=None
        assert result is not None
        call_kwargs = mock_orch.execute_with_tools.call_args.kwargs
        assert call_kwargs.get("attachment_context") is None

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_keyboard_interrupt_propagates(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        """KeyboardInterrupt must NOT be swallowed."""
        from vaig.core.headless import execute_skill_headless

        _patched_base(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.side_effect = KeyboardInterrupt()
            with pytest.raises(KeyboardInterrupt):
                execute_skill_headless(
                    settings=_make_settings(),
                    skill=_FakeSkill(),
                    query="test",
                    gke_config=_make_gke_config(),
                    attachment_adapters=[fake_adapter],
                )

"""Tests for att-chunked-context-b1: truncation flag, gap recovery, budget override.

Covers:
- D1: Unit tests for ``_render_attachment_context`` (S1, S2, S4 renderer layer)
- D2: Unit tests for ``OrchestratorResult`` defaults + ``RepoInvestigationConfig`` validation
- D3: Integration-style tests via caplog + fake adapters (S3–S6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Patch targets ─────────────────────────────────────────────────────────────
_P_REGISTER = "vaig.core.gke.register_live_tools"
_P_ORCHESTRATOR = "vaig.agents.orchestrator.Orchestrator"
_P_CLIENT = "vaig.core.client.GeminiClient"
_P_CREDS = "vaig.core.auth.get_gke_credentials"
_P_REPO_INDEX = "vaig.core.repo_index.RepoIndex"


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
    attachment_truncated: bool = False
    attachment_gaps: list[str] = field(default_factory=list)


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


def _make_chunk(file_path: str, content: str):
    """Create a minimal Chunk object for testing."""
    from vaig.core.repo_chunkers import Chunk

    return Chunk(
        file_path=file_path,
        start_line=1,
        end_line=1,
        content=content,
        token_estimate=len(content) // 4,
        kind="text",
        outline=file_path,
    )


# ── D1: Unit tests for _render_attachment_context ────────────────────────────


class TestRenderAttachmentContext:
    """D1 — unit tests for the renderer."""

    def test_fit_path_returns_false_and_no_marker(self):
        """S1: chunks fit within budget → (str, False), marker NOT in text."""
        from vaig.core.headless import _TRUNCATION_MARKER, _render_attachment_context
        from vaig.core.repo_index import RepoIndex

        chunk = _make_chunk("hello.py", "print('hello')")
        index = RepoIndex([chunk])

        rendered, truncated = _render_attachment_context(index)

        assert truncated is False
        assert _TRUNCATION_MARKER not in rendered
        assert "hello.py" in rendered
        assert "print('hello')" in rendered

    def test_overflow_path_returns_true_and_marker(self):
        """S2: chunks exceed default budget → (str_with_marker, True)."""
        from vaig.core.headless import (
            _TRUNCATION_MARKER,
            MAX_ATTACHMENT_CONTEXT_BYTES,
            _render_attachment_context,
        )
        from vaig.core.repo_index import RepoIndex

        # Build content larger than the full budget
        big_content = "y" * (MAX_ATTACHMENT_CONTEXT_BYTES + 10_000)
        chunk = _make_chunk("big.txt", big_content)
        index = RepoIndex([chunk])

        rendered, truncated = _render_attachment_context(index)

        assert truncated is True
        assert _TRUNCATION_MARKER in rendered
        # Rendered output must be within budget
        assert len(rendered.encode("utf-8")) <= MAX_ATTACHMENT_CONTEXT_BYTES

    def test_marker_is_suffix_when_truncated(self):
        """When truncated, _TRUNCATION_MARKER must be the final part appended."""
        from vaig.core.headless import _TRUNCATION_MARKER, _render_attachment_context
        from vaig.core.repo_index import RepoIndex

        big_content = "z" * 200_000
        chunk = _make_chunk("overflow.txt", big_content)
        index = RepoIndex([chunk])

        rendered, truncated = _render_attachment_context(index)

        assert truncated is True
        assert rendered.endswith(_TRUNCATION_MARKER)

    def test_custom_budget_respected(self):
        """S4 renderer: custom budget_bytes overrides module default."""
        from vaig.core.headless import _render_attachment_context
        from vaig.core.repo_index import RepoIndex

        # 600-byte content that fits in default (128 KB) but not in 512-byte custom budget
        content = "a" * 600
        chunk = _make_chunk("medium.txt", content)
        index = RepoIndex([chunk])

        # With default budget → no truncation
        _, truncated_default = _render_attachment_context(index)
        assert truncated_default is False

        # With 512-byte custom budget → truncation fires
        _, truncated_custom = _render_attachment_context(index, budget_bytes=512)
        assert truncated_custom is True

    def test_empty_index_no_truncation(self):
        """Empty index returns empty string with truncated=False."""
        from vaig.core.headless import _render_attachment_context
        from vaig.core.repo_index import RepoIndex

        index = RepoIndex([])
        rendered, truncated = _render_attachment_context(index)

        assert truncated is False
        assert rendered == ""

    def test_multiple_chunks_all_fit(self):
        """Multiple small chunks all fit → no truncation, all rendered."""
        from vaig.core.headless import _TRUNCATION_MARKER, _render_attachment_context
        from vaig.core.repo_index import RepoIndex

        chunks = [_make_chunk(f"file{i}.py", f"content {i}") for i in range(5)]
        index = RepoIndex(chunks)

        rendered, truncated = _render_attachment_context(index)

        assert truncated is False
        assert _TRUNCATION_MARKER not in rendered
        for i in range(5):
            assert f"file{i}.py" in rendered


# ── D2: Unit tests for OrchestratorResult + RepoInvestigationConfig ──────────


class TestOrchestratorResultDefaults:
    """D2 — OrchestratorResult new fields have correct defaults."""

    def test_default_attachment_truncated_is_false(self):
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.skills.base import SkillPhase

        result = OrchestratorResult(skill_name="test", phase=SkillPhase.EXECUTE)
        assert result.attachment_truncated is False

    def test_default_attachment_gaps_is_empty_list(self):
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.skills.base import SkillPhase

        result = OrchestratorResult(skill_name="test", phase=SkillPhase.EXECUTE)
        assert result.attachment_gaps == []

    def test_attachment_gaps_is_not_shared_mutable(self):
        """Each instance gets its own list (no shared mutable default)."""
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.skills.base import SkillPhase

        r1 = OrchestratorResult(skill_name="test", phase=SkillPhase.EXECUTE)
        r2 = OrchestratorResult(skill_name="test", phase=SkillPhase.EXECUTE)
        r1.attachment_gaps.append("gap1")
        assert r2.attachment_gaps == []

    def test_fields_can_be_set(self):
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.skills.base import SkillPhase

        result = OrchestratorResult(skill_name="test", phase=SkillPhase.EXECUTE)
        result.attachment_truncated = True
        result.attachment_gaps = ["[WARNING] missing_file: path/to/file — details here"]

        assert result.attachment_truncated is True
        assert len(result.attachment_gaps) == 1


class TestRepoInvestigationConfigBudgetField:
    """D2 — RepoInvestigationConfig.attachment_context_budget_bytes validation."""

    def test_default_is_none(self):
        from vaig.core.config import RepoInvestigationConfig

        cfg = RepoInvestigationConfig()
        assert cfg.attachment_context_budget_bytes is None

    def test_positive_value_accepted(self):
        from vaig.core.config import RepoInvestigationConfig

        cfg = RepoInvestigationConfig(attachment_context_budget_bytes=65536)
        assert cfg.attachment_context_budget_bytes == 65536

    def test_value_1_accepted(self):
        from vaig.core.config import RepoInvestigationConfig

        cfg = RepoInvestigationConfig(attachment_context_budget_bytes=1)
        assert cfg.attachment_context_budget_bytes == 1

    def test_zero_raises_validation_error(self):
        from pydantic import ValidationError

        from vaig.core.config import RepoInvestigationConfig

        with pytest.raises(ValidationError):
            RepoInvestigationConfig(attachment_context_budget_bytes=0)

    def test_negative_raises_validation_error(self):
        from pydantic import ValidationError

        from vaig.core.config import RepoInvestigationConfig

        with pytest.raises(ValidationError):
            RepoInvestigationConfig(attachment_context_budget_bytes=-1)

    def test_extra_field_forbidden(self):
        """ConfigDict(extra='forbid') is still enforced after our addition."""
        from pydantic import ValidationError

        from vaig.core.config import RepoInvestigationConfig

        with pytest.raises(ValidationError):
            RepoInvestigationConfig(unknown_field="x")  # type: ignore[call-arg]


# ── D3: Integration tests for execute_skill_headless scenarios S3–S6 ─────────


class TestExecuteSkillHeadlessIntegration:
    """D3 — integration-style tests using caplog + fake adapters."""

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_s3_gap_recovery(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
    ):
        """S3: gaps returned by build_from_attachments are stringified + stored."""
        from vaig.core.headless import execute_skill_headless

        fake_result = _FakeResult()
        mock_orch = _patched_base(mock_register, mock_orch_cls, mock_client_cls, fake_result)

        # Build fake gaps with the attributes _stringify_gap reads
        gap1 = MagicMock()
        gap1.level = "WARNING"
        gap1.kind = "missing_file"
        gap1.path = "src/missing.py"
        gap1.details = "file not found in git tree"

        gap2 = MagicMock()
        gap2.level = "ERROR"
        gap2.kind = "parse_error"
        gap2.path = None
        gap2.details = "syntax error in chunk"

        fake_index = MagicMock()
        fake_index.chunks = []

        fake_adapter = MagicMock()

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [gap1, gap2])
            result = execute_skill_headless(
                settings=_make_settings(),
                skill=_FakeSkill(),
                query="test",
                gke_config=_make_gke_config(),
                attachment_adapters=[fake_adapter],
            )

        assert len(result.attachment_gaps) == 2
        assert any("missing_file" in g for g in result.attachment_gaps)
        assert any("parse_error" in g for g in result.attachment_gaps)

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_s4_budget_override_forces_truncation(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
        caplog,
    ):
        """S4: repo_config with small budget overrides default → truncation fires."""
        from vaig.core.config import RepoInvestigationConfig
        from vaig.core.headless import execute_skill_headless
        from vaig.core.repo_chunkers import Chunk
        from vaig.core.repo_index import RepoIndex

        fake_result = _FakeResult()
        _patched_base(mock_register, mock_orch_cls, mock_client_cls, fake_result)

        # Create content that fits default (128 KB) but not our custom 512-byte budget
        content = "b" * 1000
        chunk = Chunk(
            file_path="medium.txt",
            start_line=1,
            end_line=1,
            content=content,
            token_estimate=250,
            kind="text",
            outline="medium.txt",
        )
        real_index = RepoIndex([chunk])

        fake_adapter = MagicMock()
        custom_cfg = RepoInvestigationConfig(attachment_context_budget_bytes=512)

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (real_index, [])
            with caplog.at_level(logging.WARNING, logger="vaig.core.headless"):
                result = execute_skill_headless(
                    settings=_make_settings(),
                    skill=_FakeSkill(),
                    query="test",
                    gke_config=_make_gke_config(),
                    attachment_adapters=[fake_adapter],
                    repo_config=custom_cfg,
                )

        assert result.attachment_truncated is True
        # Warning must reference the custom budget (512)
        assert any("512" in r.message for r in caplog.records)

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_s5_zero_attachments_no_truncation_no_warning(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
        caplog,
    ):
        """S5: no attachment_adapters → truncated=False, gaps=[], no warning."""
        from vaig.core.headless import execute_skill_headless

        fake_result = _FakeResult()
        _patched_base(mock_register, mock_orch_cls, mock_client_cls, fake_result)

        with caplog.at_level(logging.WARNING, logger="vaig.core.headless"):
            result = execute_skill_headless(
                settings=_make_settings(),
                skill=_FakeSkill(),
                query="test",
                gke_config=_make_gke_config(),
                # no attachment_adapters
            )

        assert result.attachment_truncated is False
        assert result.attachment_gaps == []
        truncation_warnings = [
            r for r in caplog.records if "attachment context truncated" in r.message
        ]
        assert len(truncation_warnings) == 0

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_s6_combined_truncation_and_gaps(
        self,
        _mock_creds,
        mock_register,
        mock_orch_cls,
        mock_client_cls,
        caplog,
    ):
        """S6: truncation + gap both present → both reflected in result + warning logged."""
        from vaig.core.config import RepoInvestigationConfig
        from vaig.core.headless import _TRUNCATION_MARKER, execute_skill_headless
        from vaig.core.repo_chunkers import Chunk
        from vaig.core.repo_index import RepoIndex

        fake_result = _FakeResult()
        mock_orch = _patched_base(mock_register, mock_orch_cls, mock_client_cls, fake_result)

        # Content that overflows a small budget
        big_content = "c" * 2000
        chunk = Chunk(
            file_path="overflow.txt",
            start_line=1,
            end_line=1,
            content=big_content,
            token_estimate=500,
            kind="text",
            outline="overflow.txt",
        )
        real_index = RepoIndex([chunk])

        gap = MagicMock()
        gap.level = "WARNING"
        gap.kind = "missing_file"
        gap.path = "src/gone.py"
        gap.details = "file gone from git"

        fake_adapter = MagicMock()
        custom_cfg = RepoInvestigationConfig(attachment_context_budget_bytes=512)

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (real_index, [gap])
            with caplog.at_level(logging.WARNING, logger="vaig.core.headless"):
                result = execute_skill_headless(
                    settings=_make_settings(),
                    skill=_FakeSkill(),
                    query="test",
                    gke_config=_make_gke_config(),
                    attachment_adapters=[fake_adapter],
                    repo_config=custom_cfg,
                )

        assert result.attachment_truncated is True
        assert len(result.attachment_gaps) >= 1
        assert any("missing_file" in g for g in result.attachment_gaps)

        # Warning must have been logged
        truncation_warnings = [
            r for r in caplog.records if "attachment context truncated" in r.message
        ]
        assert len(truncation_warnings) >= 1

        # The prompt passed to orchestrator should contain _TRUNCATION_MARKER
        call_kwargs = mock_orch.execute_with_tools.call_args.kwargs
        ctx = call_kwargs.get("attachment_context", "")
        assert _TRUNCATION_MARKER in (ctx or "")

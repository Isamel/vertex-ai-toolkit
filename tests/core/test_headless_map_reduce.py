"""Unit tests for Map-Reduce attachment analysis in execute_skill_headless (F1–F12).

Covers: window slicing, fast path, MAP loop, REDUCE phase, error handling,
window cap, empty windows, map_reduce_windows_used field.
"""

from __future__ import annotations

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
_P_SLICE = "vaig.core.headless._slice_attachment_windows"
_P_RENDER = "vaig.core.headless._render_attachment_context"
_P_REDUCE = "vaig.core.headless._reduce_window_results"


# ── Fakes ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeResult:
    skill_name: str = "discovery"
    phase: str = "execute"
    synthesized_output: str = "output"
    success: bool = True
    run_cost_usd: float = 0.01
    total_usage: dict = field(default_factory=dict)
    structured_report: Any = None
    agent_results: list = field(default_factory=list)
    attachment_truncated: bool = False
    attachment_gaps: list = field(default_factory=list)
    map_reduce_windows_used: int = 1


class _FakeSkill:
    def get_metadata(self):
        m = MagicMock()
        m.name = "discovery"
        return m


class _FakeToolRegistry:
    def list_tools(self):
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


def _fake_chunk(path: str = "file.py", content: str = "x" * 100) -> MagicMock:
    c = MagicMock()
    c.file_path = path
    c.content = content
    c.outline = path
    return c


def _fake_index(n_chunks: int = 2) -> MagicMock:
    idx = MagicMock()
    idx.chunks = [_fake_chunk(f"file{i}.py") for i in range(n_chunks)]
    return idx


def _fake_window(n_chunks: int = 1) -> MagicMock:
    w = MagicMock()
    w.chunks = [_fake_chunk() for _ in range(n_chunks)]
    return w


def _base_patches(mock_register, mock_orch_cls, mock_client_cls, fake_result=None):
    fake_result = fake_result or _FakeResult()
    mock_register.return_value = _FakeToolRegistry()
    mock_orch = MagicMock()
    mock_orch.execute_with_tools.return_value = fake_result
    mock_orch_cls.return_value = mock_orch
    mock_client_cls.return_value = MagicMock()
    return mock_orch, fake_result


# ── F1: _slice_attachment_windows — basic partitioning ───────────────────────


class TestSliceAttachmentWindows:
    def test_empty_index_returns_empty(self):
        from vaig.core.headless import _slice_attachment_windows
        from vaig.core.repo_index import RepoIndex

        idx = RepoIndex([])
        assert _slice_attachment_windows(idx, 128_000) == []

    def test_single_chunk_fits_one_window(self):
        from vaig.core.headless import _slice_attachment_windows
        from vaig.core.repo_chunkers import Chunk
        from vaig.core.repo_index import RepoIndex

        chunk = Chunk(
            file_path="small.py",
            start_line=1,
            end_line=1,
            content="hello",
            token_estimate=1,
            kind="text",
            outline="small.py",
        )
        idx = RepoIndex([chunk])
        windows = _slice_attachment_windows(idx, 128_000)
        assert len(windows) == 1
        assert len(windows[0].chunks) == 1

    def test_oversized_chunk_gets_own_window(self):
        """A chunk larger than the budget must be placed alone — render will truncate."""
        from vaig.core.headless import _slice_attachment_windows
        from vaig.core.repo_chunkers import Chunk
        from vaig.core.repo_index import RepoIndex

        big = Chunk(
            file_path="big.txt",
            start_line=1,
            end_line=1,
            content="x" * 200_000,
            token_estimate=9999,
            kind="text",
            outline="big.txt",
        )
        small = Chunk(
            file_path="small.txt",
            start_line=1,
            end_line=1,
            content="tiny",
            token_estimate=1,
            kind="text",
            outline="small.txt",
        )
        idx = RepoIndex([big, small])
        windows = _slice_attachment_windows(idx, 128_000)
        # big chunk alone, small chunk in second window
        assert len(windows) >= 2
        assert any(len(w.chunks) == 1 and w.chunks[0].file_path == "big.txt" for w in windows)

    def test_multiple_small_chunks_fit_one_window(self):
        from vaig.core.headless import _slice_attachment_windows
        from vaig.core.repo_chunkers import Chunk
        from vaig.core.repo_index import RepoIndex

        chunks = [
            Chunk(
                file_path=f"f{i}.py",
                start_line=1,
                end_line=1,
                content="x" * 10,
                token_estimate=1,
                kind="text",
                outline=f"f{i}.py",
            )
            for i in range(5)
        ]
        idx = RepoIndex(chunks)
        windows = _slice_attachment_windows(idx, 128_000)
        assert len(windows) == 1
        assert len(windows[0].chunks) == 5


# ── F2: fast path (0 or 1 window) ────────────────────────────────────────────


class TestFastPath:
    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_no_adapters_uses_fast_path_windows_used_0(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        mock_orch, fake_result = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        result = execute_skill_headless(
            settings=_make_settings(),
            skill=_FakeSkill(),
            query="q",
            gke_config=_make_gke_config(),
        )
        assert result.map_reduce_windows_used == 0

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_single_window_sets_windows_used_1(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        mock_orch, fake_result = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(1)
        fake_window = _fake_window(1)

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=[fake_window]):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    result = execute_skill_headless(
                        settings=_make_settings(),
                        skill=_FakeSkill(),
                        query="q",
                        gke_config=_make_gke_config(),
                        attachment_adapters=[fake_adapter],
                    )
        assert result.map_reduce_windows_used == 1

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_fast_path_does_not_call_reduce(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(0)

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=[]):
                with patch(_P_REDUCE) as mock_reduce:
                    execute_skill_headless(
                        settings=_make_settings(),
                        skill=_FakeSkill(),
                        query="q",
                        gke_config=_make_gke_config(),
                        attachment_adapters=[fake_adapter],
                    )
        mock_reduce.assert_not_called()


# ── F3: MAP loop calls orchestrator once per window ──────────────────────────


class TestMapLoop:
    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_orchestrator_called_once_per_window(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        mock_orch, _ = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(4)
        windows = [_fake_window() for _ in range(3)]

        reduce_result = _FakeResult()
        reduce_result.map_reduce_windows_used = 3

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=windows):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with patch(_P_REDUCE, return_value=reduce_result):
                        execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                        )

        assert mock_orch.execute_with_tools.call_count == 3

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_map_reduce_windows_used_set_correctly(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(4)
        windows = [_fake_window() for _ in range(2)]

        reduce_result = _FakeResult()

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=windows):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with patch(_P_REDUCE, return_value=reduce_result):
                        result = execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                        )

        assert result.map_reduce_windows_used == 2


# ── F4: window-level exception → EvidenceGap, not re-raised ──────────────────


class TestMapWindowError:
    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_window_exception_produces_gap_not_crash(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        mock_orch, _ = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(2)
        windows = [_fake_window(), _fake_window()]

        # First window raises, second succeeds
        good_result = _FakeResult()
        mock_orch.execute_with_tools.side_effect = [RuntimeError("window 1 exploded"), good_result]

        reduce_result = _FakeResult()
        reduce_calls: list = []

        def fake_reduce(window_results, extra_gaps, windows_attempted, skill):
            reduce_calls.append((window_results, extra_gaps))
            return reduce_result

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=windows):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with patch(_P_REDUCE, side_effect=fake_reduce):
                        result = execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                        )

        # reduce was still called with the one good result
        assert len(reduce_calls) == 1
        window_results, gaps = reduce_calls[0]
        assert window_results == [good_result]
        # gap was recorded for the failing window
        assert len(gaps) == 1
        assert "window 1 exploded" in gaps[0].details

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_keyboard_interrupt_propagates_in_map_loop(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        mock_orch, _ = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(2)
        windows = [_fake_window(), _fake_window()]

        mock_orch.execute_with_tools.side_effect = KeyboardInterrupt()

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=windows):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with pytest.raises(KeyboardInterrupt):
                        execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                        )


# ── F5: window cap enforced ───────────────────────────────────────────────────


class TestWindowCap:
    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_window_cap_limits_orchestrator_calls(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.config import RepoInvestigationConfig
        from vaig.core.headless import execute_skill_headless

        mock_orch, _ = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(30)
        # 25 windows returned but cap is 3
        all_windows = [_fake_window() for _ in range(25)]
        reduce_result = _FakeResult()

        repo_cfg = RepoInvestigationConfig(map_reduce_max_windows=3)

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=all_windows):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with patch(_P_REDUCE, return_value=reduce_result):
                        execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                            repo_config=repo_cfg,
                        )

        # Only 3 windows processed despite 25 returned from slice
        assert mock_orch.execute_with_tools.call_count == 3

    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_window_cap_sets_attachment_truncated(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.config import RepoInvestigationConfig
        from vaig.core.headless import execute_skill_headless

        _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(30)
        all_windows = [_fake_window() for _ in range(5)]
        reduce_result = _FakeResult()

        repo_cfg = RepoInvestigationConfig(map_reduce_max_windows=2)

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=all_windows):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with patch(_P_REDUCE, return_value=reduce_result):
                        result = execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                            repo_config=repo_cfg,
                        )

        assert result.attachment_truncated is True


# ── F6: empty window skipped ─────────────────────────────────────────────────


class TestEmptyWindowSkipped:
    @patch(_P_CLIENT)
    @patch(_P_ORCHESTRATOR)
    @patch(_P_REGISTER)
    @patch(_P_CREDS, return_value=None)
    def test_empty_window_not_dispatched(self, _creds, mock_register, mock_orch_cls, mock_client_cls):
        from vaig.core.headless import execute_skill_headless

        mock_orch, _ = _base_patches(mock_register, mock_orch_cls, mock_client_cls)
        fake_adapter = MagicMock()
        fake_index = _fake_index(2)

        empty_window = MagicMock()
        empty_window.chunks = []
        real_window = _fake_window()
        reduce_result = _FakeResult()

        with patch(_P_REPO_INDEX) as mock_ri:
            mock_ri.build_from_attachments.return_value = (fake_index, [])
            with patch(_P_SLICE, return_value=[empty_window, real_window]):
                with patch(_P_RENDER, return_value=("ctx", False)):
                    with patch(_P_REDUCE, return_value=reduce_result):
                        execute_skill_headless(
                            settings=_make_settings(),
                            skill=_FakeSkill(),
                            query="q",
                            gke_config=_make_gke_config(),
                            attachment_adapters=[fake_adapter],
                        )

        # Only the non-empty window dispatched
        assert mock_orch.execute_with_tools.call_count == 1


# ── F7: _reduce_window_results unit tests ────────────────────────────────────


class TestReduceWindowResults:
    def _make_result(self, synthesized: str = "out", cost: float = 0.01) -> _FakeResult:
        r = _FakeResult(synthesized_output=synthesized, run_cost_usd=cost)
        return r

    def test_empty_returns_failure(self):
        from vaig.core.headless import _reduce_window_results

        skill = _FakeSkill()
        result = _reduce_window_results(
            window_results=[],
            extra_gaps=[],
            windows_attempted=2,
            skill=skill,
        )
        assert result.success is False

    def test_costs_summed(self):
        # Use real OrchestratorResult
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.core.headless import _reduce_window_results

        r1 = OrchestratorResult(skill_name="s", phase="p", success=True)
        r1.run_cost_usd = 0.05
        r1.total_usage = {"input_tokens": 100}
        r1.synthesized_output = "part1"
        r1.structured_report = None

        r2 = OrchestratorResult(skill_name="s", phase="p", success=True)
        r2.run_cost_usd = 0.03
        r2.total_usage = {"input_tokens": 50}
        r2.synthesized_output = "part2"
        r2.structured_report = None

        result = _reduce_window_results(
            window_results=[r1, r2],
            extra_gaps=[],
            windows_attempted=2,
            skill=_FakeSkill(),
        )
        assert abs(result.run_cost_usd - 0.08) < 1e-9
        assert result.total_usage["input_tokens"] == 150

    def test_synthesized_joined_with_separator(self):
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.core.headless import _reduce_window_results

        r1 = OrchestratorResult(skill_name="s", phase="p", success=True)
        r1.synthesized_output = "window-1-output"
        r1.structured_report = None
        r1.run_cost_usd = 0.0
        r1.total_usage = {}

        r2 = OrchestratorResult(skill_name="s", phase="p", success=True)
        r2.synthesized_output = "window-2-output"
        r2.structured_report = None
        r2.run_cost_usd = 0.0
        r2.total_usage = {}

        result = _reduce_window_results(
            window_results=[r1, r2],
            extra_gaps=[],
            windows_attempted=2,
            skill=_FakeSkill(),
        )
        assert "window-1-output" in result.synthesized_output
        assert "window-2-output" in result.synthesized_output
        assert "---" in result.synthesized_output

    def test_success_true_if_any_window_succeeded(self):
        from vaig.agents.orchestrator import OrchestratorResult
        from vaig.core.headless import _reduce_window_results

        r_fail = OrchestratorResult(skill_name="s", phase="p", success=False)
        r_fail.structured_report = None
        r_fail.run_cost_usd = 0.0
        r_fail.total_usage = {}
        r_fail.synthesized_output = ""

        r_ok = OrchestratorResult(skill_name="s", phase="p", success=True)
        r_ok.structured_report = None
        r_ok.run_cost_usd = 0.0
        r_ok.total_usage = {}
        r_ok.synthesized_output = "ok"

        result = _reduce_window_results(
            window_results=[r_fail, r_ok],
            extra_gaps=[],
            windows_attempted=2,
            skill=_FakeSkill(),
        )
        assert result.success is True

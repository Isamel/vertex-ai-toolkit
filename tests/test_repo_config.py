"""Tests for SPEC-V2-REPO-01 — Multi --repo-path CLI + RepoInvestigationConfig."""

from __future__ import annotations

import logging

import pytest

from vaig.core.config import _DEFAULT_EXCLUDE_GLOBS, RepoInvestigationConfig

# The "vaig" parent logger has propagate=False (set by setup_logging()),
# which prevents caplog from capturing records. Temporarily enable it
# for all tests in this module so caplog works correctly.
_vaig_logger = logging.getLogger("vaig")


@pytest.fixture(autouse=True)
def _enable_vaig_propagation() -> None:  # type: ignore[misc]
    """Temporarily set vaig logger propagate=True for caplog compatibility."""
    orig = _vaig_logger.propagate
    _vaig_logger.propagate = True
    yield  # type: ignore[misc]
    _vaig_logger.propagate = orig

# ── Unit tests: RepoInvestigationConfig model ─────────────────────────────────


class TestRepoInvestigationConfig:
    """RepoInvestigationConfig Pydantic model tests."""

    def test_default_paths_are_empty(self) -> None:
        cfg = RepoInvestigationConfig()
        assert cfg.paths == []

    def test_default_include_globs_are_empty(self) -> None:
        cfg = RepoInvestigationConfig()
        assert cfg.include_globs == []

    def test_default_exclude_globs_match_constants(self) -> None:
        cfg = RepoInvestigationConfig()
        assert cfg.exclude_globs == _DEFAULT_EXCLUDE_GLOBS

    def test_default_max_files(self) -> None:
        cfg = RepoInvestigationConfig()
        assert cfg.max_files == 500

    def test_default_streaming_threshold(self) -> None:
        cfg = RepoInvestigationConfig()
        assert cfg.streaming_threshold_bytes == 2_000_000

    def test_default_redaction_enabled(self) -> None:
        cfg = RepoInvestigationConfig()
        assert cfg.redaction_enabled is True

    def test_multi_paths_preserved(self) -> None:
        """AC-1: paths=['a','b'] is stored correctly."""
        cfg = RepoInvestigationConfig(repo="owner/repo", paths=["a", "b"])
        assert cfg.paths == ["a", "b"]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            RepoInvestigationConfig(unknown_field="x")  # type: ignore[call-arg]

    def test_exclude_globs_can_be_overridden(self) -> None:
        cfg = RepoInvestigationConfig(exclude_globs=["**/foo/**"])
        assert cfg.exclude_globs == ["**/foo/**"]

    def test_exclude_globs_default_is_independent_copy(self) -> None:
        """Mutating one instance's list must not affect another."""
        cfg1 = RepoInvestigationConfig()
        cfg2 = RepoInvestigationConfig()
        cfg1.exclude_globs.append("**/extra/**")
        assert "**/extra/**" not in cfg2.exclude_globs

    def test_no_repo_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when repo is None (back-compat path)."""
        with caplog.at_level(logging.WARNING, logger="vaig.core.config"):
            RepoInvestigationConfig()
        assert not caplog.records

    def test_repo_without_paths_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """AC-3: repo set but no paths/globs → warning about full-scan."""
        with caplog.at_level(logging.WARNING, logger="vaig.core.config"):
            RepoInvestigationConfig(repo="owner/repo")
        assert any("full-scan" in r.message for r in caplog.records)

    def test_repo_with_paths_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="vaig.core.config"):
            RepoInvestigationConfig(repo="owner/repo", paths=["apps/"])
        assert not caplog.records

    def test_streaming_threshold_alias_maps_correctly(self) -> None:
        """streaming_threshold_bytes field exists and accepts ints."""
        cfg = RepoInvestigationConfig(streaming_threshold_bytes=200_000)
        assert cfg.streaming_threshold_bytes == 200_000


# ── CLI flag tests using _build_repo_investigation_config ────────────────────


class TestBuildRepoInvestigationConfig:
    """Tests for the CLI-layer builder helper."""

    def _build(self, **kwargs):  # type: ignore[no-untyped-def]
        from vaig.cli.commands.live import _build_repo_investigation_config
        defaults: dict[str, object] = {
            "repo": None,
            "repo_ref": "HEAD",
            "repo_paths": [],
            "include_globs": [],
            "exclude_globs": None,
            "max_files": None,
            "streaming_threshold_bytes": None,
        }
        defaults.update(kwargs)
        return _build_repo_investigation_config(**defaults)

    def test_multi_repo_paths_yielded(self) -> None:
        """AC-1: --repo-path a --repo-path b yields paths == ['a', 'b']."""
        cfg = self._build(repo="owner/repo", repo_paths=["a", "b"])
        assert cfg.paths == ["a", "b"]

    def test_no_repo_path_empty_list(self) -> None:
        """AC-3: no --repo-path → paths is empty (back-compat)."""
        cfg = self._build(repo="owner/repo")
        assert cfg.paths == []

    def test_invalid_glob_raises_bad_parameter(self) -> None:
        """AC-2: malformed glob raises typer.BadParameter."""
        import typer
        with pytest.raises(typer.BadParameter, match="malformed"):
            self._build(include_globs=["[invalid"])

    def test_valid_glob_does_not_raise(self) -> None:
        cfg = self._build(include_globs=["**/values*.yaml"])
        assert cfg.include_globs == ["**/values*.yaml"]

    def test_exclude_globs_override(self) -> None:
        cfg = self._build(exclude_globs=["**/secrets/**"])
        assert cfg.exclude_globs == ["**/secrets/**"]

    def test_exclude_globs_none_uses_defaults(self) -> None:
        cfg = self._build()
        assert cfg.exclude_globs == _DEFAULT_EXCLUDE_GLOBS

    def test_max_files_override(self) -> None:
        cfg = self._build(max_files=100)
        assert cfg.max_files == 100

    def test_streaming_threshold_maps_to_field(self) -> None:
        """--repo-max-bytes-per-file maps to streaming_threshold_bytes."""
        cfg = self._build(streaming_threshold_bytes=200_000)
        assert cfg.streaming_threshold_bytes == 200_000

    def test_back_compat_no_flags(self) -> None:
        """AC-3: using no repo flags produces a valid config with all defaults."""
        cfg = self._build()
        assert cfg.repo is None
        assert cfg.paths == []
        assert cfg.max_files == 500

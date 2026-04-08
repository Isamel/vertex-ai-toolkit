"""Tests for skill loader — directory, entry-point, and package-based skill discovery."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vaig.core.config import SkillsConfig
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase
from vaig.skills.loader import (
    load_from_directories,
    load_from_entry_points,
    load_from_packages,
)

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

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_DUMMY_SKILL_PY = """\
from vaig.skills.base import BaseSkill, SkillMetadata, SkillPhase


class DummyExternalSkill(BaseSkill):
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dummy-external",
            display_name="Dummy External",
            description="A dummy external skill for testing",
            version="0.1.0",
            tags=["test"],
            supported_phases=[SkillPhase.ANALYZE],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return "You are a dummy external skill."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {context} | {user_input}"
"""


def _make_skill_dir(parent: Path, name: str, content: str = _DUMMY_SKILL_PY) -> Path:
    """Create a subdirectory with a skill.py file inside ``parent``."""
    skill_dir = parent / name
    skill_dir.mkdir()
    (skill_dir / "skill.py").write_text(content)
    return skill_dir


class _DummySkillForEP(BaseSkill):
    """In-memory dummy skill used for entry-point mocking."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ep-dummy",
            display_name="EP Dummy",
            description="Entry-point dummy skill",
            version="1.0.0",
            tags=["test"],
            supported_phases=[SkillPhase.ANALYZE],
            recommended_model="gemini-2.5-flash",
        )

    def get_system_instruction(self) -> str:
        return "EP dummy."

    def get_phase_prompt(self, phase: SkillPhase, context: str, user_input: str) -> str:
        return f"[{phase.value}] {context} | {user_input}"


# ═══════════════════════════════════════════════════════════════
# load_from_directories  (Task 5.1)
# ═══════════════════════════════════════════════════════════════


class TestLoadFromDirectories:
    """Tests for load_from_directories()."""

    def test_loads_valid_skill(self, tmp_path: Path) -> None:
        """Should load a valid BaseSkill from a subdir with skill.py."""
        _make_skill_dir(tmp_path, "my_skill")

        skills = load_from_directories([str(tmp_path)])

        assert len(skills) == 1
        assert skills[0].get_metadata().name == "dummy-external"

    def test_nonexistent_directory_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log warning and return [] for a non-existent directory."""
        with caplog.at_level(logging.WARNING):
            skills = load_from_directories(["/nonexistent/path/xyz"])

        assert skills == []
        assert "not found" in caplog.text.lower()

    def test_subdir_without_skill_py_skipped(self, tmp_path: Path) -> None:
        """Should skip subdirectories that don't contain skill.py."""
        (tmp_path / "empty_dir").mkdir()

        skills = load_from_directories([str(tmp_path)])
        assert skills == []

    def test_import_error_logged_and_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning and skip a skill whose skill.py has an import error."""
        _make_skill_dir(
            tmp_path,
            "broken_skill",
            content="raise SyntaxError('bad module')\n",
        )

        with caplog.at_level(logging.WARNING):
            skills = load_from_directories([str(tmp_path)])

        assert skills == []
        assert "failed to load" in caplog.text.lower()

    def test_multiple_dirs_aggregated(self, tmp_path: Path) -> None:
        """Should aggregate skills from multiple directories."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        _make_skill_dir(dir_a, "skill_a")
        _make_skill_dir(dir_b, "skill_b")

        skills = load_from_directories([str(dir_a), str(dir_b)])
        assert len(skills) == 2

    def test_empty_list_returns_empty(self) -> None:
        """Should return [] when no directories are provided."""
        assert load_from_directories([]) == []

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        """Should return [] when directory exists but has no subdirs."""
        skills = load_from_directories([str(tmp_path)])
        assert skills == []


# ═══════════════════════════════════════════════════════════════
# load_from_entry_points  (Task 5.2)
# ═══════════════════════════════════════════════════════════════


class TestLoadFromEntryPoints:
    """Tests for load_from_entry_points()."""

    def _mock_ep(self, cls: type = _DummySkillForEP, name: str = "ep-dummy") -> MagicMock:
        """Create a mock EntryPoint whose .load() returns ``cls``."""
        ep = MagicMock()
        ep.name = name
        ep.value = "some.module:SomeClass"
        ep.load.return_value = cls
        return ep

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_loads_valid_entry_point(self, mock_eps: MagicMock) -> None:
        """Should load a valid BaseSkill from an entry point."""
        mock_eps.return_value = [self._mock_ep()]

        skills = load_from_entry_points()

        assert len(skills) == 1
        assert skills[0].get_metadata().name == "ep-dummy"

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_broken_entry_point_skipped(
        self, mock_eps: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning and skip an entry point that raises on load."""
        bad_ep = self._mock_ep()
        bad_ep.load.side_effect = ImportError("missing module")
        mock_eps.return_value = [bad_ep]

        with caplog.at_level(logging.WARNING):
            skills = load_from_entry_points()

        assert skills == []
        assert "failed to load" in caplog.text.lower()

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_no_entry_points_returns_empty(self, mock_eps: MagicMock) -> None:
        """Should return [] when no entry points are registered."""
        mock_eps.return_value = []
        assert load_from_entry_points() == []

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_non_baseskill_entry_point_skipped(
        self, mock_eps: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should skip entry points that don't reference a BaseSkill subclass."""
        bad_ep = self._mock_ep()
        bad_ep.load.return_value = str  # Not a BaseSkill subclass
        mock_eps.return_value = [bad_ep]

        with caplog.at_level(logging.WARNING):
            skills = load_from_entry_points()

        assert skills == []
        assert "does not reference a baseskill subclass" in caplog.text.lower()


# ═══════════════════════════════════════════════════════════════
# load_from_packages  (Task 5.3)
# ═══════════════════════════════════════════════════════════════


class TestLoadFromPackages:
    """Tests for load_from_packages()."""

    def _mock_ep_with_dist(
        self,
        dist_name: str,
        cls: type = _DummySkillForEP,
        ep_name: str = "ep-skill",
    ) -> MagicMock:
        """Create a mock EntryPoint with a dist.name attribute."""
        ep = MagicMock()
        ep.name = ep_name
        ep.value = "pkg.mod:Cls"
        ep.load.return_value = cls
        ep.dist = MagicMock()
        ep.dist.name = dist_name
        return ep

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_filters_by_package_name(self, mock_eps: MagicMock) -> None:
        """Should only load entry points whose dist name is in the filter list."""
        ep_good = self._mock_ep_with_dist("vaig-security-skills")
        ep_bad = self._mock_ep_with_dist("vaig-other-skills")
        mock_eps.return_value = [ep_good, ep_bad]

        skills = load_from_packages(["vaig-security-skills"])

        assert len(skills) == 1
        assert skills[0].get_metadata().name == "ep-dummy"

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_empty_names_returns_empty(self, mock_eps: MagicMock) -> None:
        """Should return [] when names list is empty (no filtering)."""
        mock_eps.return_value = [self._mock_ep_with_dist("some-pkg")]

        skills = load_from_packages([])
        assert skills == []
        # entry_points should not even be called
        mock_eps.assert_not_called()

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_no_matching_packages_returns_empty(self, mock_eps: MagicMock) -> None:
        """Should return [] when no entry points match the filter."""
        ep = self._mock_ep_with_dist("unrelated-package")
        mock_eps.return_value = [ep]

        skills = load_from_packages(["vaig-specific-skills"])
        assert skills == []

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_pep503_normalization_matches_case_insensitive(self, mock_eps: MagicMock) -> None:
        """PEP 503: dist names with underscores/mixed case should match hyphenated filter."""
        ep = self._mock_ep_with_dist("Vaig_Security_Skills")
        mock_eps.return_value = [ep]

        skills = load_from_packages(["vaig-security-skills"])

        assert len(skills) == 1
        assert skills[0].get_metadata().name == "ep-dummy"

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_pep503_normalization_dots(self, mock_eps: MagicMock) -> None:
        """PEP 503: dist names with dots should normalize to hyphens."""
        ep = self._mock_ep_with_dist("vaig.security.skills")
        mock_eps.return_value = [ep]

        skills = load_from_packages(["vaig-security-skills"])

        assert len(skills) == 1

    @patch("vaig.skills.loader.importlib.metadata.entry_points")
    def test_pep503_normalization_filter_side(self, mock_eps: MagicMock) -> None:
        """PEP 503: filter names with underscores should match hyphenated dist names."""
        ep = self._mock_ep_with_dist("vaig-security-skills")
        mock_eps.return_value = [ep]

        skills = load_from_packages(["Vaig_Security_Skills"])

        assert len(skills) == 1


# ═══════════════════════════════════════════════════════════════
# _migrate_custom_dir validator  (Task 5.4)
# ═══════════════════════════════════════════════════════════════


class TestMigrateCustomDir:
    """Tests for SkillsConfig._migrate_custom_dir model validator."""

    def test_custom_dir_migrates_to_external_dirs(self) -> None:
        """custom_dir set + external_dirs empty → migrate with DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="custom_dir is deprecated"):
            cfg = SkillsConfig(custom_dir="/old/path")

        assert cfg.external_dirs == ["/old/path"]

    def test_both_set_external_dirs_wins(self) -> None:
        """custom_dir + external_dirs both set → external_dirs wins, no DeprecationWarning."""
        cfg = SkillsConfig(custom_dir="/old", external_dirs=["/new"])
        assert cfg.external_dirs == ["/new"]

    def test_custom_dir_none_no_warning(self) -> None:
        """custom_dir=None → no warning, external_dirs stays []."""
        cfg = SkillsConfig(custom_dir=None)
        assert cfg.external_dirs == []

    def test_no_custom_dir_at_all(self) -> None:
        """No custom_dir provided → external_dirs stays []."""
        cfg = SkillsConfig()
        assert cfg.external_dirs == []
        assert cfg.packages == []

    def test_custom_dir_with_explicit_null_external_dirs(self) -> None:
        """custom_dir set + external_dirs explicitly None → migrate without AttributeError."""
        with pytest.warns(DeprecationWarning, match="custom_dir is deprecated"):
            cfg = SkillsConfig.model_validate(
                {"custom_dir": "/old/path", "external_dirs": None}
            )

        assert cfg.external_dirs == ["/old/path"]


# ═══════════════════════════════════════════════════════════════
# _import_skill_from_path — module name sanitization  (Fix 6)
# ═══════════════════════════════════════════════════════════════


class TestImportSkillFromPathSanitization:
    """Tests for hyphen-to-underscore sanitization in _import_skill_from_path."""

    def test_hyphenated_dir_name_produces_valid_module(self, tmp_path: Path) -> None:
        """Directory names with hyphens should not cause import errors."""
        skill_dir = tmp_path / "my-custom-skills"
        skill_dir.mkdir()
        (skill_dir / "skill.py").write_text(_DUMMY_SKILL_PY)

        # load_from_directories calls _import_skill_from_path under the hood
        skills = load_from_directories([str(tmp_path)])

        assert len(skills) == 1
        assert skills[0].get_metadata().name == "dummy-external"


# ═══════════════════════════════════════════════════════════════
# Registry: _load_external_skills — no double loading  (Fix 1)
# ═══════════════════════════════════════════════════════════════


class TestRegistryNoDuplicateLoading:
    """Ensure _load_external_skills calls only ONE of entry_points/packages."""

    @patch("vaig.skills.loader.load_from_packages")
    @patch("vaig.skills.loader.load_from_entry_points")
    @patch("vaig.skills.loader.load_from_directories", return_value=[])
    def test_packages_configured_skips_entry_points(
        self,
        _mock_dirs: MagicMock,
        mock_ep: MagicMock,
        mock_pkg: MagicMock,
    ) -> None:
        """When packages is non-empty, load_from_entry_points must NOT be called."""
        from vaig.core.config import Settings

        mock_pkg.return_value = []

        settings = Settings()  # type: ignore[call-arg]
        settings.skills.packages = ["vaig-security-skills"]

        from vaig.skills.registry import SkillRegistry

        registry = SkillRegistry(settings)
        registry._load_external_skills()

        mock_pkg.assert_called_once_with(["vaig-security-skills"])
        mock_ep.assert_not_called()

    @patch("vaig.skills.loader.load_from_packages")
    @patch("vaig.skills.loader.load_from_entry_points")
    @patch("vaig.skills.loader.load_from_directories", return_value=[])
    def test_no_packages_uses_entry_points(
        self,
        _mock_dirs: MagicMock,
        mock_ep: MagicMock,
        mock_pkg: MagicMock,
    ) -> None:
        """When packages is empty, load_from_entry_points should be called."""
        from vaig.core.config import Settings

        mock_ep.return_value = []

        settings = Settings()  # type: ignore[call-arg]
        settings.skills.packages = []

        from vaig.skills.registry import SkillRegistry

        registry = SkillRegistry(settings)
        registry._load_external_skills()

        mock_ep.assert_called_once()
        mock_pkg.assert_not_called()

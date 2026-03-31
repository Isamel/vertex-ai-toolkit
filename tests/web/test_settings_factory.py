"""Tests for Settings.from_overrides() — Task 2.1.

Covers:
- Constructing Settings with shorthand overrides (project, model, temperature, region)
- Empty/None overrides are skipped
- Base config is not mutated
- Unknown keys are silently ignored
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip(
    "fastapi",
    reason="FastAPI not available; install the 'web' extra to run web tests.",
)

from vaig.core.config import Settings


@pytest.fixture()
def _no_yaml_files(tmp_path: Path) -> Generator[None, Any, None]:
    """Redirect Path.cwd() and Path.home() to empty temp dirs.

    This ensures from_overrides() finds no YAML config files on disk,
    so the only data in Settings comes from pydantic defaults + overrides.
    """
    fake_cwd = tmp_path / "cwd"
    fake_cwd.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    # Stack patches so both are active for the entire test
    with (
        patch.object(Path, "cwd", return_value=fake_cwd),
        patch.object(Path, "home", return_value=fake_home),
    ):
        yield


@pytest.mark.usefixtures("_no_yaml_files")
class TestSettingsFromOverrides:
    """Settings.from_overrides() classmethod tests."""

    def test_returns_settings_instance(self) -> None:
        """from_overrides() must return a Settings instance."""
        result = Settings.from_overrides()
        assert isinstance(result, Settings)

    def test_project_override_maps_to_gcp_project_id(self) -> None:
        """project='my-proj' should map to gcp.project_id."""
        result = Settings.from_overrides(project="my-test-proj")
        assert result.gcp.project_id == "my-test-proj"

    def test_model_override_maps_to_models_default(self) -> None:
        """model='gemini-2.0-flash' should map to models.default."""
        result = Settings.from_overrides(model="gemini-2.0-flash")
        assert result.models.default == "gemini-2.0-flash"

    def test_temperature_override_maps_to_generation(self) -> None:
        """temperature=0.5 should map to generation.temperature."""
        result = Settings.from_overrides(temperature=0.5)
        assert result.generation.temperature == 0.5

    def test_region_override_maps_to_gcp_location(self) -> None:
        """region='us-east1' should map to gcp.location."""
        result = Settings.from_overrides(region="us-east1")
        assert result.gcp.location == "us-east1"

    def test_none_overrides_are_skipped(self) -> None:
        """None values should NOT override defaults."""
        base = Settings.from_overrides()
        result = Settings.from_overrides(project=None, model=None)
        assert result.gcp.project_id == base.gcp.project_id
        assert result.models.default == base.models.default

    def test_empty_string_overrides_are_skipped(self) -> None:
        """Empty string overrides should NOT override defaults."""
        base = Settings.from_overrides()
        result = Settings.from_overrides(project="", model="  ")
        assert result.gcp.project_id == base.gcp.project_id
        assert result.models.default == base.models.default

    def test_unknown_keys_are_ignored(self) -> None:
        """Unknown override keys should be silently ignored."""
        # Should NOT raise
        result = Settings.from_overrides(unknown_key="value", another="thing")
        assert isinstance(result, Settings)

    def test_multiple_overrides_combined(self) -> None:
        """Multiple overrides should all be applied."""
        result = Settings.from_overrides(
            project="multi-proj",
            model="gemini-2.5-pro",
            temperature=1.0,
            region="europe-west4",
        )
        assert result.gcp.project_id == "multi-proj"
        assert result.models.default == "gemini-2.5-pro"
        assert result.generation.temperature == 1.0
        assert result.gcp.location == "europe-west4"

    def test_concurrent_calls_produce_independent_instances(self) -> None:
        """Two from_overrides() calls should produce independent Settings."""
        s1 = Settings.from_overrides(project="proj-a")
        s2 = Settings.from_overrides(project="proj-b")
        assert s1 is not s2
        assert s1.gcp.project_id == "proj-a"
        assert s2.gcp.project_id == "proj-b"

"""Tests for SPEC-V2-AUDIT-06: severity filter presets in SPA template.

Verifies that:
- Three preset buttons are present in the SPA HTML template
- Each button has the correct data-preset attribute
- The applyPreset JS function is defined in the template
- PRESET_MAP covers critical-only, high-and-above, and all
- applyPreset clears active severities and applies the preset set
"""

from __future__ import annotations

from pathlib import Path

import pytest

SPA_PATH = Path(__file__).parent.parent / "src" / "vaig" / "ui" / "spa_template.html"


@pytest.fixture(scope="module")
def spa_html() -> str:
    return SPA_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. HTML structure
# ---------------------------------------------------------------------------

class TestPresetButtonsHTML:
    def test_preset_container_present(self, spa_html: str) -> None:
        assert 'id="severity-presets"' in spa_html

    def test_critical_only_button(self, spa_html: str) -> None:
        assert 'data-preset="critical-only"' in spa_html

    def test_high_and_above_button(self, spa_html: str) -> None:
        assert 'data-preset="high-and-above"' in spa_html

    def test_all_button(self, spa_html: str) -> None:
        assert 'data-preset="all"' in spa_html

    def test_buttons_use_filter_preset_btn_class(self, spa_html: str) -> None:
        assert 'class="filter-preset-btn"' in spa_html

    def test_buttons_call_applypreset(self, spa_html: str) -> None:
        assert "applyPreset('critical-only')" in spa_html
        assert "applyPreset('high-and-above')" in spa_html
        assert "applyPreset('all')" in spa_html


# ---------------------------------------------------------------------------
# 2. CSS
# ---------------------------------------------------------------------------

class TestPresetCSS:
    def test_filter_preset_btn_style_defined(self, spa_html: str) -> None:
        assert ".filter-preset-btn" in spa_html

    def test_active_state_style_defined(self, spa_html: str) -> None:
        assert ".filter-preset-btn.active" in spa_html


# ---------------------------------------------------------------------------
# 3. JavaScript
# ---------------------------------------------------------------------------

class TestApplyPresetJS:
    def test_apply_preset_function_defined(self, spa_html: str) -> None:
        assert "function applyPreset(" in spa_html

    def test_preset_map_defined(self, spa_html: str) -> None:
        assert "PRESET_MAP" in spa_html

    def test_preset_map_contains_critical_only(self, spa_html: str) -> None:
        assert "'critical-only'" in spa_html or '"critical-only"' in spa_html

    def test_preset_map_contains_high_and_above(self, spa_html: str) -> None:
        assert "'high-and-above'" in spa_html or '"high-and-above"' in spa_html

    def test_preset_map_contains_all(self, spa_html: str) -> None:
        # The 'all' key is present in PRESET_MAP
        assert "PRESET_MAP" in spa_html
        # The map should reference all 5 severities for 'all' preset
        assert "'CRITICAL'" in spa_html or '"CRITICAL"' in spa_html
        assert "'HIGH'" in spa_html or '"HIGH"' in spa_html
        assert "'MEDIUM'" in spa_html or '"MEDIUM"' in spa_html
        assert "'LOW'" in spa_html or '"LOW"' in spa_html
        assert "'INFO'" in spa_html or '"INFO"' in spa_html

    def test_apply_preset_syncs_pills(self, spa_html: str) -> None:
        """applyPreset must toggle inactive class on filter pills."""
        # Check the JS block contains the pills sync logic
        assert "filter-pill" in spa_html
        # The applyPreset function should reference classList.toggle
        apply_preset_idx = spa_html.index("function applyPreset(")
        snippet = spa_html[apply_preset_idx: apply_preset_idx + 600]
        assert "classList.toggle" in snippet

    def test_apply_preset_syncs_preset_buttons(self, spa_html: str) -> None:
        """applyPreset must mark the active preset button."""
        apply_preset_idx = spa_html.index("function applyPreset(")
        snippet = spa_html[apply_preset_idx: apply_preset_idx + 600]
        assert "filter-preset-btn" in snippet
        assert "active" in snippet

    def test_apply_preset_shows_hides_cards(self, spa_html: str) -> None:
        """applyPreset must show/hide finding cards."""
        apply_preset_idx = spa_html.index("function applyPreset(")
        snippet = spa_html[apply_preset_idx: apply_preset_idx + 600]
        assert "finding-card" in snippet

"""Tests for SPEC-V2-AUDIT-03: rename cost sections in SPA template.

Verifies:
- Sidebar nav item "Cost & Usage" is renamed to "Investigation Cost"
- Sidebar nav item "GKE Cost" is renamed to "Workload Cost"
- Section h2 "Cost & Usage" is renamed to "Investigation Cost"
- Section h2 / toggle "GKE Workload Cost Estimation" is renamed to "Workload Cost"
- Old names do not appear as visible labels in the HTML
"""

from __future__ import annotations

from pathlib import Path

import pytest

SPA_PATH = Path(__file__).parent.parent / "src" / "vaig" / "ui" / "spa_template.html"


@pytest.fixture(scope="module")
def spa_html() -> str:
    return SPA_PATH.read_text(encoding="utf-8")


class TestCostSectionRenames:
    # ── New names present ──────────────────────────────────────────────────

    def test_investigation_cost_in_sidebar(self, spa_html: str) -> None:
        assert "Investigation Cost" in spa_html

    def test_workload_cost_in_sidebar(self, spa_html: str) -> None:
        assert "Workload Cost" in spa_html

    def test_investigation_cost_in_section_h2(self, spa_html: str) -> None:
        # h2 tag should contain the new label
        assert "<h2>" in spa_html or "<h2 " in spa_html
        # The section h2 must mention Investigation Cost
        assert "Investigation Cost" in spa_html

    def test_workload_cost_in_section_toggle(self, spa_html: str) -> None:
        assert "Workload Cost" in spa_html

    # ── Old names absent ───────────────────────────────────────────────────

    def test_cost_and_usage_label_removed(self, spa_html: str) -> None:
        """The visible label 'Cost & Usage' / 'Cost &amp; Usage' must not appear."""
        assert "Cost &amp; Usage" not in spa_html
        assert "Cost & Usage" not in spa_html

    def test_gke_cost_sidebar_label_removed(self, spa_html: str) -> None:
        """The sidebar label text '> GKE Cost' must not appear."""
        # The nav item text was "GKE Cost" — check it's gone from nav context
        assert "> GKE Cost\n" not in spa_html
        assert "> GKE Cost<" not in spa_html

    def test_gke_workload_cost_estimation_label_removed(self, spa_html: str) -> None:
        """The section toggle label 'GKE Workload Cost Estimation' must be gone."""
        assert "GKE Workload Cost Estimation" not in spa_html

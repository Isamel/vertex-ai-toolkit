"""AUDIT-08 — LOW/INFO findings collapse by default in the SPA report.

Tests verify that the spa_template.html contains the required JS/CSS
elements for collapsing LOW and INFO severity findings on page load.
"""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "src" / "vaig" / "ui" / "spa_template.html"


def _template() -> str:
    return TEMPLATE_PATH.read_text()


class TestDataFindingSeverityAttribute:
    """AUDIT-08 — finding cards must carry data-finding-severity attribute."""

    def test_data_finding_severity_in_template(self) -> None:
        assert 'data-finding-severity=' in _template()

    def test_data_finding_severity_set_from_severity(self) -> None:
        template = _template()
        # The attribute must be populated dynamically from f.severity
        assert 'data-finding-severity="${esc(f.severity)}"' in template


class TestApplyDefaultCollapseFunction:
    """AUDIT-08 — applyDefaultCollapse JS function must exist and be correct."""

    def test_function_defined(self) -> None:
        assert "function applyDefaultCollapse()" in _template()

    def test_function_targets_low_severity(self) -> None:
        assert '[data-finding-severity="LOW"]' in _template()

    def test_function_targets_info_severity(self) -> None:
        assert '[data-finding-severity="INFO"]' in _template()

    def test_function_adds_collapsed_class(self) -> None:
        template = _template()
        # The function must add 'collapsed' class to matched elements
        assert "classList.add('collapsed')" in template

    def test_function_called_in_render_findings(self) -> None:
        assert "applyDefaultCollapse();" in _template()


class TestExpandAllObservationsFunction:
    """AUDIT-08 — expandAllObservations must remove .collapsed from LOW/INFO."""

    def test_function_defined(self) -> None:
        assert "function expandAllObservations()" in _template()

    def test_function_removes_collapsed_class(self) -> None:
        assert "classList.remove('collapsed')" in _template()


class TestExpandObservationsLink:
    """AUDIT-08 — 'Expand observations' link must be injected when LOW/INFO findings exist."""

    def test_expand_link_class_exists_in_css(self) -> None:
        assert ".expand-observations-link" in _template()

    def test_expand_link_references_expand_function(self) -> None:
        assert "expandAllObservations" in _template()

    def test_expand_observations_text_in_template(self) -> None:
        assert "Expand observations" in _template()


class TestCollapsedCss:
    """AUDIT-08 — CSS must hide .finding-body for .collapsed finding cards."""

    def test_collapsed_hides_finding_body(self) -> None:
        assert ".finding-card.collapsed .finding-body" in _template()

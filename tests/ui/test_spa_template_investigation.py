"""Tests for the Autonomous Investigation section in spa_template.html.

Verifies that the Investigation section DOM IDs and renderInvestigation()
function exist and use the correct InvestigationEvidenceSnapshot field names.
"""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "src" / "vaig" / "ui" / "spa_template.html"


def _template() -> str:
    return TEMPLATE_PATH.read_text()


class TestInvestigationSectionDOMIDs:
    """The Investigation section must contain expected DOM IDs."""

    def test_section_investigation_id_exists(self) -> None:
        assert 'id="section-investigation"' in _template()

    def test_investigation_steps_tbody_id_exists(self) -> None:
        assert 'id="investigation-steps-tbody"' in _template()

    def test_investigation_summary_id_exists(self) -> None:
        assert 'id="investigation-summary"' in _template()


class TestRenderInvestigationFunction:
    """renderInvestigation() must be defined and use correct field names."""

    def test_function_defined(self) -> None:
        assert "function renderInvestigation()" in _template()

    def test_uses_verdict_field(self) -> None:
        """Must read e.verdict, not e.supports/e.contradicts."""
        template = _template()
        assert "e.verdict" in template

    def test_uses_hypothesis_field(self) -> None:
        """Must read e.hypothesis, not e.question."""
        template = _template()
        assert "e.hypothesis" in template

    def test_uses_iteration_field(self) -> None:
        """Must read e.iteration, not e.replan_iteration."""
        template = _template()
        assert "e.iteration" in template

    def test_does_not_use_wrong_fields(self) -> None:
        """Must not reference deprecated field names."""
        template = _template()
        # These are the wrong field names from before the fix
        assert "e.supports" not in template
        assert "e.contradicts" not in template
        assert "e.question" not in template
        assert "e.replan_iteration" not in template

    def test_uses_tool_name_field(self) -> None:
        assert "e.tool_name" in _template()

    def test_uses_target_field(self) -> None:
        assert "e.target" in _template()

    def test_function_called_in_init(self) -> None:
        assert "renderInvestigation();" in _template()

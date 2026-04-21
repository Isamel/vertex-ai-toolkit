"""Tests for the Root Cause Hypotheses section in spa_template.html.

Verifies that renderRootCause() uses the correct RootCauseHypothesis field
names from schema.py and does NOT reference any stale/renamed fields.
"""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "src" / "vaig" / "ui" / "spa_template.html"


def _template() -> str:
    return TEMPLATE_PATH.read_text()


class TestRootCauseHypothesesDOMIDs:
    """The Root Cause section must contain expected DOM IDs."""

    def test_rootcause_list_id_exists(self) -> None:
        assert 'id="rootcause-list"' in _template()


class TestRenderRootCauseFunction:
    """renderRootCause() must use correct RootCauseHypothesis field names."""

    def test_function_defined(self) -> None:
        assert "function renderRootCause()" in _template()

    def test_uses_label_field(self) -> None:
        """Must read h.label (not h.finding_title)."""
        assert "h.label" in _template()

    def test_uses_probability_field(self) -> None:
        """Must read h.probability (not h.confidence)."""
        assert "h.probability" in _template()

    def test_uses_status_field(self) -> None:
        """Must read h.status."""
        assert "h.status" in _template()

    def test_uses_supporting_evidence_field(self) -> None:
        """Must read h.supporting_evidence."""
        assert "h.supporting_evidence" in _template()

    def test_uses_confirms_if_field(self) -> None:
        """Must read h.confirms_if (not h.what_would_confirm)."""
        assert "h.confirms_if" in _template()

    def test_uses_refutes_if_field(self) -> None:
        """Must read h.refutes_if."""
        assert "h.refutes_if" in _template()

    def test_does_not_use_stale_fields(self) -> None:
        """Must not reference any renamed/removed field names."""
        template = _template()
        assert "h.finding_title" not in template
        assert "h.mechanism" not in template
        assert "h.what_would_confirm" not in template
        assert "h.confidence" not in template

    def test_function_called_in_init(self) -> None:
        assert "renderRootCause();" in _template()

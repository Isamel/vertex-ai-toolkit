"""T11–T13 — Verify that spa_template.html contains evidence gaps rendering elements."""

from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "src" / "vaig" / "ui" / "spa_template.html"


def _template() -> str:
    return TEMPLATE_PATH.read_text()


class TestInvestigationCoverageBadge:
    """T11 — Coverage badge element exists in summary section."""

    def test_badge_element_exists(self) -> None:
        assert 'id="investigation-coverage-badge"' in _template()

    def test_badge_starts_hidden(self) -> None:
        template = _template()
        # The badge div should start with display:none
        idx = template.find('id="investigation-coverage-badge"')
        assert idx != -1
        surrounding = template[max(0, idx - 100):idx + 200]
        assert "display:none" in surrounding


class TestEvidenceGapsSection:
    """T12 — Evidence Gaps section exists in the HTML."""

    def test_section_element_exists(self) -> None:
        assert 'id="section-evidence-gaps"' in _template()

    def test_section_starts_hidden(self) -> None:
        template = _template()
        idx = template.find('id="section-evidence-gaps"')
        assert idx != -1
        surrounding = template[max(0, idx - 10):idx + 100]
        assert "display:none" in surrounding

    def test_coverage_div_exists(self) -> None:
        assert 'id="evidence-gaps-coverage"' in _template()

    def test_gaps_list_div_exists(self) -> None:
        assert 'id="evidence-gaps-list"' in _template()


class TestRenderEvidenceGapsFunction:
    """T12 — renderEvidenceGaps JS function exists and is called in init."""

    def test_render_function_defined(self) -> None:
        assert "function renderEvidenceGaps()" in _template()

    def test_render_function_called_in_init(self) -> None:
        assert "renderEvidenceGaps();" in _template()

    def test_render_function_reads_evidence_gaps(self) -> None:
        assert "REPORT_DATA.evidence_gaps" in _template()

    def test_render_function_reads_investigation_coverage(self) -> None:
        assert "REPORT_DATA.investigation_coverage" in _template()

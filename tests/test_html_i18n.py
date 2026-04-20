"""Tests for SPEC-V2-AUDIT-09 — SPA chrome i18n localisation layer.

Strategy: render the SPA template with a minimal HealthReport whose
``metadata.detected_language`` is set to "es" or "en", then assert that
the generated HTML contains the expected localised strings embedded in the
i18n JS dictionary and that the data-i18n attributes are present on the
expected chrome elements.
"""

from __future__ import annotations

import json
import re

import pytest

from vaig.skills.service_health.schema import (
    ExecutiveSummary,
    HealthReport,
    OverallStatus,
    ReportMetadata,
)
from vaig.ui.html_report import render_health_report_html

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_report(lang: str = "en") -> HealthReport:
    return HealthReport(
        executive_summary=ExecutiveSummary(
            overall_status=OverallStatus.HEALTHY,
            scope="test-ns",
            summary_text="All services healthy.",
            critical_count=0,
            warning_count=0,
            issues_found=0,
            services_checked=3,
        ),
        metadata=ReportMetadata(
            generated_at="2026-04-20T12:00:00Z",
            detected_language=lang,
        ),
    )


# ── i18n JS block presence ────────────────────────────────────────────────────


class TestI18NBlockPresent:
    """The generated HTML must contain the I18N dictionary and t() helper."""

    def test_i18n_dict_present(self) -> None:
        html = render_health_report_html(_make_report())
        assert "const I18N = {" in html, "I18N dict not found in rendered HTML"

    def test_t_helper_present(self) -> None:
        html = render_health_report_html(_make_report())
        assert "function t(key)" in html, "t() helper not found in rendered HTML"

    def test_apply_i18n_called(self) -> None:
        html = render_health_report_html(_make_report())
        assert "applyI18n()" in html, "applyI18n() call not found in rendered HTML"


# ── Spanish locale strings ────────────────────────────────────────────────────


class TestSpanishLocaleStrings:
    """The I18N dict must contain the correct Spanish strings for key AC terms."""

    @pytest.fixture(scope="class")
    def html(self) -> str:
        return render_health_report_html(_make_report("es"))

    def test_servicios_in_i18n_dict(self, html: str) -> None:
        assert "Servicios" in html, "Spanish 'Servicios' not found in I18N dict"

    def test_hallazgos_in_i18n_dict(self, html: str) -> None:
        assert "Hallazgos" in html, "Spanish 'Hallazgos' not found in I18N dict"

    def test_costo_investigacion_in_i18n_dict(self, html: str) -> None:
        assert "Costo de Investigación" in html or "Costo" in html, (
            "Spanish cost string not found in I18N dict"
        )

    def test_resumen_ejecutivo_in_i18n_dict(self, html: str) -> None:
        assert "Resumen Ejecutivo" in html, "Spanish 'Resumen Ejecutivo' not found in I18N dict"

    def test_acciones_recomendadas_in_i18n_dict(self, html: str) -> None:
        assert "Acciones Recomendadas" in html, (
            "Spanish 'Acciones Recomendadas' not found in I18N dict"
        )


# ── English locale strings ────────────────────────────────────────────────────


class TestEnglishLocaleStrings:
    """The I18N dict must contain the correct English strings."""

    @pytest.fixture(scope="class")
    def html(self) -> str:
        return render_health_report_html(_make_report("en"))

    def test_services_in_i18n_dict(self, html: str) -> None:
        assert "'Services'" in html or '"Services"' in html, (
            "English 'Services' not found in I18N dict"
        )

    def test_findings_in_i18n_dict(self, html: str) -> None:
        assert "'Findings'" in html or '"Findings"' in html, (
            "English 'Findings' not found in I18N dict"
        )

    def test_cost_in_i18n_dict(self, html: str) -> None:
        assert "Investigation Cost" in html, (
            "English 'Investigation Cost' not found in I18N dict"
        )


# ── data-i18n attributes ──────────────────────────────────────────────────────


class TestDataI18nAttributes:
    """Chrome HTML elements must carry data-i18n attributes for the key nav items."""

    @pytest.fixture(scope="class")
    def html(self) -> str:
        return render_health_report_html(_make_report())

    def test_sidebar_nav_label_attribute(self, html: str) -> None:
        assert 'data-i18n="nav_label"' in html, (
            "data-i18n='nav_label' not found on sidebar label element"
        )

    def test_nav_services_attribute(self, html: str) -> None:
        assert 'data-i18n="nav_services"' in html, (
            "data-i18n='nav_services' not found in sidebar"
        )

    def test_nav_findings_attribute(self, html: str) -> None:
        assert 'data-i18n="nav_findings"' in html, (
            "data-i18n='nav_findings' not found in sidebar"
        )

    def test_filter_label_attribute(self, html: str) -> None:
        assert 'data-i18n="nav_filter_label"' in html, (
            "data-i18n='nav_filter_label' not found on filter section label"
        )

    def test_h2_summary_attribute(self, html: str) -> None:
        assert 'data-i18n="h2_summary"' in html, (
            "data-i18n='h2_summary' not found on Executive Summary heading"
        )

    def test_h2_findings_attribute(self, html: str) -> None:
        assert 'data-i18n="h2_findings"' in html, (
            "data-i18n='h2_findings' not found on Findings heading"
        )

    def test_h2_cost_attribute(self, html: str) -> None:
        assert 'data-i18n="h2_cost"' in html, (
            "data-i18n='h2_cost' not found on Investigation Cost heading"
        )

    def test_pill_crit_attribute(self, html: str) -> None:
        assert 'data-i18n="pill_crit"' in html, (
            "data-i18n='pill_crit' not found on CRIT filter pill"
        )

    def test_preset_all_attribute(self, html: str) -> None:
        assert 'data-i18n="preset_all"' in html, (
            "data-i18n='preset_all' not found on All preset button"
        )


# ── detected_language field on ReportMetadata ─────────────────────────────────


class TestDetectedLanguageField:
    """ReportMetadata must expose a detected_language field that serialises into
    the REPORT_DATA JSON blob consumed by the SPA."""

    def test_field_defaults_to_en(self) -> None:
        meta = ReportMetadata()
        assert meta.detected_language == "en"

    def test_field_accepts_es(self) -> None:
        meta = ReportMetadata(detected_language="es")
        assert meta.detected_language == "es"

    def test_serialised_in_report_json(self) -> None:
        report = _make_report("es")
        html = render_health_report_html(report)
        # The REPORT_DATA JSON is embedded as:
        #   const REPORT_DATA = /*{{REPORT_DATA_JSON}}*/null;  →  const REPORT_DATA = {...};
        match = re.search(r"const REPORT_DATA = (.*?);", html, re.DOTALL)
        assert match, "REPORT_DATA assignment not found in rendered HTML"
        data = json.loads(match.group(1))
        assert data["metadata"]["detected_language"] == "es"

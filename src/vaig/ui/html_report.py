"""Self-contained HTML renderer for HealthReport objects.

Generates a complete, single-file HTML dashboard using the Kanagawa dark theme.
No external dependencies — pure Python f-strings, no Jinja2.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime

import vaig
from vaig.skills.service_health.schema import (
    ActionUrgency,
    HealthReport,
    OverallStatus,
    ServiceHealthStatus,
    Severity,
)

# ── Kanagawa dark theme colours ──────────────────────────────────────────────

_BG = "#1F1F28"
_BG_CARD = "#2A2A37"
_BG_TERMINAL = "#16161D"
_BG_TABLE_ALT = "#252530"
_FG = "#DCD7BA"
_FG_DIM = "#727169"
_ACCENT_BLUE = "#7FB4CA"
_ACCENT_GREEN = "#98BB6C"
_ACCENT_RED = "#E82424"
_ACCENT_ORANGE = "#FF9E3B"
_ACCENT_YELLOW = "#E6C384"
_ACCENT_PURPLE = "#957FB8"
_ACCENT_TEAL = "#6A9589"
_BORDER = "#363646"

# ── Severity → colour / label mappings ──────────────────────────────────────

_SEVERITY_COLOUR: dict[Severity, str] = {
    Severity.CRITICAL: _ACCENT_RED,
    Severity.HIGH: _ACCENT_ORANGE,
    Severity.MEDIUM: _ACCENT_YELLOW,
    Severity.LOW: _ACCENT_BLUE,
    Severity.INFO: _FG_DIM,
}

_SEVERITY_BG: dict[Severity, str] = {
    Severity.CRITICAL: "#3D1515",
    Severity.HIGH: "#3D2810",
    Severity.MEDIUM: "#2E2B14",
    Severity.LOW: "#152130",
    Severity.INFO: "#232330",
}

_STATUS_COLOUR: dict[OverallStatus, str] = {
    OverallStatus.HEALTHY: _ACCENT_GREEN,
    OverallStatus.DEGRADED: _ACCENT_YELLOW,
    OverallStatus.CRITICAL: _ACCENT_RED,
    OverallStatus.UNKNOWN: _FG_DIM,
}

_SERVICE_STATUS_COLOUR: dict[ServiceHealthStatus, str] = {
    ServiceHealthStatus.HEALTHY: _ACCENT_GREEN,
    ServiceHealthStatus.DEGRADED: _ACCENT_YELLOW,
    ServiceHealthStatus.FAILED: _ACCENT_RED,
    ServiceHealthStatus.UNKNOWN: _FG_DIM,
}

_URGENCY_COLOUR: dict[ActionUrgency, str] = {
    ActionUrgency.IMMEDIATE: _ACCENT_RED,
    ActionUrgency.SHORT_TERM: _ACCENT_ORANGE,
    ActionUrgency.LONG_TERM: _ACCENT_BLUE,
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _e(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text), quote=True)


def _badge(text: str, colour: str, bg: str = "") -> str:
    bg_style = f"background:{bg};" if bg else f"background:{colour}22;"
    return (
        f'<span class="badge" style="color:{colour};{bg_style}'
        f'border:1px solid {colour}44;">{_e(text)}</span>'
    )


def _severity_badge(severity: Severity) -> str:
    colour = _SEVERITY_COLOUR.get(severity, _FG_DIM)
    bg = _SEVERITY_BG.get(severity, "#232330")
    return _badge(severity.value, colour, bg)


def _status_badge(status: OverallStatus) -> str:
    colour = _STATUS_COLOUR.get(status, _FG_DIM)
    return _badge(status.value, colour)


def _service_status_badge(status: ServiceHealthStatus) -> str:
    colour = _SERVICE_STATUS_COLOUR.get(status, _FG_DIM)
    return _badge(status.value, colour)


def _urgency_badge(urgency: ActionUrgency) -> str:
    colour = _URGENCY_COLOUR.get(urgency, _ACCENT_BLUE)
    return _badge(urgency.value.replace("_", " "), colour)


def _card(title: str, content: str, accent_colour: str = _ACCENT_BLUE) -> str:
    return f"""
<div class="card">
  <div class="card-header" style="border-left:3px solid {accent_colour};">
    <h2>{_e(title)}</h2>
  </div>
  <div class="card-body">
    {content}
  </div>
</div>"""


# ── Section renderers ─────────────────────────────────────────────────────────


def _render_executive_summary(report: HealthReport) -> str:
    es = report.executive_summary
    colour = _STATUS_COLOUR.get(es.overall_status, _FG_DIM)
    status_badge = _status_badge(es.overall_status)

    stats_html = f"""
<div class="stats-row">
  <div class="stat-box" style="border-color:{_ACCENT_RED};">
    <div class="stat-value" style="color:{_ACCENT_RED};">{_e(str(es.critical_count))}</div>
    <div class="stat-label">Critical</div>
  </div>
  <div class="stat-box" style="border-color:{_ACCENT_ORANGE};">
    <div class="stat-value" style="color:{_ACCENT_ORANGE};">{_e(str(es.warning_count))}</div>
    <div class="stat-label">Warnings</div>
  </div>
  <div class="stat-box" style="border-color:{_ACCENT_BLUE};">
    <div class="stat-value" style="color:{_ACCENT_BLUE};">{_e(str(es.issues_found))}</div>
    <div class="stat-label">Total Issues</div>
  </div>
  <div class="stat-box" style="border-color:{_ACCENT_TEAL};">
    <div class="stat-value" style="color:{_ACCENT_TEAL};">{_e(str(es.services_checked))}</div>
    <div class="stat-label">Services Checked</div>
  </div>
</div>"""

    content = f"""
<div class="exec-summary">
  <div class="exec-status">
    <span class="status-label">Overall Status</span>
    {status_badge}
  </div>
  <div class="exec-scope" style="color:{_FG_DIM};">
    <strong style="color:{_ACCENT_TEAL};">Scope:</strong> {_e(es.scope)}
  </div>
  <p class="exec-text" style="color:{colour}88;">{_e(es.summary_text)}</p>
  {stats_html}
</div>"""
    return _card("Executive Summary", content, colour)


def _render_cluster_overview(report: HealthReport) -> str:
    if not report.cluster_overview:
        return ""
    rows = "".join(
        f'<tr><td class="metric-name">{_e(m.metric)}</td>'
        f'<td class="metric-value">{_e(m.value)}</td></tr>'
        for m in report.cluster_overview
    )
    content = f"""
<table class="data-table">
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""
    return _card("Cluster Overview", content, _ACCENT_TEAL)


def _render_service_statuses(report: HealthReport) -> str:
    if not report.service_statuses:
        return ""
    rows = ""
    for svc in report.service_statuses:
        badge = _service_status_badge(svc.status)
        issues_cell = f'<td class="issues-cell">{_e(svc.issues)}</td>' if svc.issues else "<td>—</td>"
        rows += (
            f"<tr>"
            f'<td class="svc-name"><code>{_e(svc.service)}</code></td>'
            f"<td>{badge}</td>"
            f'<td class="ns-cell">{_e(svc.namespace) if svc.namespace else "—"}</td>'
            f'<td class="pods-cell">{_e(svc.pods_ready)}</td>'
            f'<td class="restart-cell">{_e(svc.restarts_1h)}</td>'
            f"{issues_cell}"
            f"</tr>"
        )
    content = f"""
<table class="data-table">
  <thead>
    <tr>
      <th>Service</th><th>Status</th><th>Namespace</th>
      <th>Pods</th><th>Restarts/1h</th><th>Issues</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""
    return _card("Service Status", content, _ACCENT_GREEN)


def _render_findings(report: HealthReport) -> str:
    if not report.findings:
        return _card(
            "Technical Findings",
            f'<p style="color:{_ACCENT_GREEN};text-align:center;">No issues found. ✓</p>',
            _ACCENT_GREEN,
        )

    items = ""
    for f in report.findings:
        sev_badge = _severity_badge(f.severity)
        colour = _SEVERITY_COLOUR.get(f.severity, _FG_DIM)
        bg = _SEVERITY_BG.get(f.severity, "#232330")

        evidence_html = ""
        if f.evidence:
            ev_items = "".join(f"<li>{_e(e)}</li>" for e in f.evidence)
            evidence_html = f'<ul class="evidence-list">{ev_items}</ul>'

        remediation_html = ""
        if f.remediation:
            remediation_html = (
                f'<div class="remediation">'
                f'<strong style="color:{_ACCENT_TEAL};">Remediation:</strong> '
                f"{_e(f.remediation)}</div>"
            )

        items += f"""
<div class="finding" style="border-left:3px solid {colour};background:{bg};">
  <div class="finding-header">
    {sev_badge}
    <span class="finding-title" style="color:{_FG};">{_e(f.title)}</span>
    {f'<span class="finding-service" style="color:{_ACCENT_TEAL};">{_e(f.service)}</span>' if f.service else ""}
  </div>
  {f'<p class="finding-desc">{_e(f.description)}</p>' if f.description else ""}
  {f'<p class="finding-cause"><strong style="color:{_ACCENT_ORANGE};">Root cause:</strong> {_e(f.root_cause)}</p>' if f.root_cause else ""}
  {evidence_html}
  {remediation_html}
</div>"""

    return _card("Technical Findings", items, _ACCENT_ORANGE)


def _render_recommendations(report: HealthReport) -> str:
    if not report.recommendations:
        return ""
    items = ""
    for rec in sorted(report.recommendations, key=lambda r: r.priority):
        urgency_badge = _urgency_badge(rec.urgency)
        command_block = ""
        if rec.command:
            command_block = (
                f'<div class="terminal-block"><pre><code>{_e(rec.command)}</code></pre></div>'
            )
        why_html = f'<p class="rec-why"><em>{_e(rec.why)}</em></p>' if rec.why else ""
        risk_html = (
            f'<p class="rec-risk" style="color:{_ACCENT_ORANGE};">'
            f"⚠ {_e(rec.risk)}</p>"
            if rec.risk
            else ""
        )
        items += f"""
<div class="recommendation">
  <div class="rec-header">
    <span class="rec-priority" style="color:{_ACCENT_PURPLE};">#{_e(str(rec.priority))}</span>
    {urgency_badge}
    <span class="rec-title">{_e(rec.title)}</span>
  </div>
  {f'<p class="rec-desc">{_e(rec.description)}</p>' if rec.description else ""}
  {why_html}
  {command_block}
  {risk_html}
</div>"""

    return _card("Action Plan", items, _ACCENT_PURPLE)


def _render_timeline(report: HealthReport) -> str:
    if not report.timeline:
        return ""
    rows = ""
    for ev in report.timeline:
        colour = _SEVERITY_COLOUR.get(ev.severity, _FG_DIM)
        badge = _severity_badge(ev.severity)
        rows += (
            f"<tr>"
            f'<td class="time-cell" style="color:{_FG_DIM};">{_e(ev.time)}</td>'
            f"<td>{badge}</td>"
            f'<td style="color:{colour};">{_e(ev.event)}</td>'
            f'<td style="color:{_FG_DIM};">{_e(ev.service)}</td>'
            f"</tr>"
        )
    content = f"""
<table class="data-table">
  <thead><tr><th>Time</th><th>Severity</th><th>Event</th><th>Service</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""
    return _card("Timeline", content, _FG_DIM)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: {_BG};
  color: {_FG};
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  padding: 1.5rem;
}}

a {{ color: {_ACCENT_BLUE}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* ── Header ── */
.report-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid {_BORDER};
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}}
.report-title {{
  font-size: 1.4rem;
  color: {_ACCENT_BLUE};
  font-weight: bold;
  letter-spacing: 0.05em;
}}
.report-meta {{
  color: {_FG_DIM};
  font-size: 0.8rem;
  text-align: right;
}}

/* ── Cards ── */
.card {{
  background: {_BG_CARD};
  border: 1px solid {_BORDER};
  border-radius: 6px;
  margin-bottom: 1.5rem;
  overflow: hidden;
}}
.card-header {{
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid {_BORDER};
}}
.card-header h2 {{
  font-size: 1rem;
  font-weight: bold;
  color: {_FG};
  padding-left: 0.5rem;
}}
.card-body {{
  padding: 1rem 1.25rem;
}}

/* ── Badges ── */
.badge {{
  display: inline-block;
  padding: 0.15em 0.55em;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: bold;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  vertical-align: middle;
  margin-right: 0.4em;
}}

/* ── Executive Summary ── */
.exec-summary {{ }}
.exec-status {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}}
.status-label {{
  color: {_FG_DIM};
  font-size: 0.85rem;
}}
.exec-scope {{
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}}
.exec-text {{
  margin-bottom: 1rem;
  font-style: italic;
}}
.stats-row {{
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}}
.stat-box {{
  background: {_BG};
  border: 1px solid {_BORDER};
  border-radius: 4px;
  padding: 0.5rem 1rem;
  text-align: center;
  min-width: 90px;
}}
.stat-value {{
  font-size: 1.6rem;
  font-weight: bold;
  line-height: 1.2;
}}
.stat-label {{
  font-size: 0.75rem;
  color: {_FG_DIM};
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

/* ── Tables ── */
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}}
.data-table th {{
  background: {_BG};
  color: {_FG_DIM};
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.75rem;
  padding: 0.5rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid {_BORDER};
}}
.data-table td {{
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid {_BORDER}44;
  vertical-align: top;
}}
.data-table tr:nth-child(even) td {{
  background: {_BG_TABLE_ALT};
}}
.data-table tr:hover td {{
  background: {_BORDER}66;
}}
.svc-name code {{
  color: {_ACCENT_BLUE};
  background: {_BG};
  padding: 0.1em 0.4em;
  border-radius: 3px;
}}
.metric-name {{ color: {_FG_DIM}; }}
.metric-value {{ color: {_ACCENT_YELLOW}; font-weight: bold; }}

/* ── Findings ── */
.finding {{
  border-radius: 4px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.75rem;
}}
.finding:last-child {{ margin-bottom: 0; }}
.finding-header {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.4rem;
}}
.finding-title {{
  font-weight: bold;
  flex: 1;
}}
.finding-service {{
  font-size: 0.8rem;
  font-style: italic;
}}
.finding-desc {{ color: {_FG_DIM}; margin-bottom: 0.3rem; font-size: 0.9rem; }}
.finding-cause {{ font-size: 0.9rem; margin-bottom: 0.3rem; }}
.evidence-list {{
  margin: 0.4rem 0 0.4rem 1.2rem;
  font-size: 0.85rem;
  color: {_FG_DIM};
}}
.evidence-list li {{ margin-bottom: 0.2rem; }}
.remediation {{
  font-size: 0.85rem;
  margin-top: 0.3rem;
  padding: 0.3rem 0.6rem;
  background: {_BG};
  border-radius: 3px;
}}

/* ── Recommendations / Action Plan ── */
.recommendation {{
  padding: 0.75rem 0;
  border-bottom: 1px solid {_BORDER};
}}
.recommendation:last-child {{ border-bottom: none; }}
.rec-header {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.4rem;
}}
.rec-priority {{
  font-weight: bold;
  font-size: 0.9rem;
  min-width: 1.5rem;
}}
.rec-title {{ font-weight: bold; }}
.rec-desc {{ font-size: 0.9rem; margin-bottom: 0.4rem; color: {_FG_DIM}; }}
.rec-why {{ font-size: 0.85rem; color: {_FG_DIM}; margin-bottom: 0.3rem; }}
.rec-risk {{ font-size: 0.85rem; margin-top: 0.2rem; }}

/* ── Terminal block ── */
.terminal-block {{
  background: {_BG_TERMINAL};
  border: 1px solid {_BORDER};
  border-radius: 4px;
  padding: 0.6rem 1rem;
  margin: 0.4rem 0;
  overflow-x: auto;
}}
.terminal-block pre {{
  margin: 0;
}}
.terminal-block code {{
  color: {_ACCENT_GREEN};
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.85rem;
  white-space: pre-wrap;
  word-break: break-all;
}}

/* ── Timeline ── */
.time-cell {{ white-space: nowrap; font-size: 0.8rem; }}

/* ── Footer ── */
.report-footer {{
  text-align: center;
  color: {_FG_DIM};
  font-size: 0.75rem;
  padding-top: 1rem;
  border-top: 1px solid {_BORDER};
  margin-top: 0.5rem;
}}
"""


# ── Public API ────────────────────────────────────────────────────────────────


def render_health_report_html(report: HealthReport) -> str:
    """Render a HealthReport as a self-contained HTML dashboard.

    Returns a complete HTML5 document string with inline CSS using the
    Kanagawa dark theme.  No external dependencies required.

    Args:
        report: A fully populated (or partially populated) HealthReport.

    Returns:
        A complete HTML string suitable for writing to a ``.html`` file.
    """
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    vaig_version = vaig.__version__

    # Render all sections (skip empty ones)
    sections = [
        _render_executive_summary(report),
        _render_cluster_overview(report),
        _render_service_statuses(report),
        _render_findings(report),
        _render_recommendations(report),
        _render_timeline(report),
    ]
    body_content = "\n".join(s for s in sections if s)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>vaig — Service Health Report</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <header class="report-header">
    <div class="report-title">⚡ vaig — Service Health Report</div>
    <div class="report-meta">
      <div>Generated: {_e(generated_at)}</div>
      <div>vaig v{_e(vaig_version)}</div>
    </div>
  </header>

  <main>
    {body_content}
  </main>

  <footer class="report-footer">
    Generated by vaig v{_e(vaig_version)} · {_e(generated_at)}
  </footer>
</body>
</html>"""

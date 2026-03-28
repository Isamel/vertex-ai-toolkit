"""Self-contained HTML renderer for HealthReport objects.

Generates a complete, single-file HTML SPA dashboard using a bundled template.
The template is loaded once per process via importlib.resources and the report
data is injected by replacing a sentinel placeholder with serialised JSON.

Also provides :func:`render_watch_session_html` for exporting a full watch
session (last report + accumulated diff timeline) as a single HTML file.
"""

from __future__ import annotations

import importlib.resources
import json

from vaig.skills.service_health.diff import WatchSessionData
from vaig.skills.service_health.schema import HealthReport

# Sentinel that lives inside the <script> block of spa_template.html.
# It is a JS comment wrapping a null literal so the template is valid JS
# even when the placeholder has not been replaced.
_SENTINEL = "/*{{REPORT_DATA_JSON}}*/null"

# Second sentinel for watch-session timeline data.
_WATCH_SENTINEL = "/*{{WATCH_SESSION_JSON}}*/null"

# Module-level cache so the file is only read once per process.
_TEMPLATE_CACHE: str | None = None


def _load_template() -> str:
    """Load spa_template.html from the vaig.ui package, caching the result."""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = (
            importlib.resources.files("vaig.ui")
            .joinpath("spa_template.html")
            .read_text(encoding="utf-8")
        )
    return _TEMPLATE_CACHE


def _escape_for_script(payload: str) -> str:
    """Escape a JSON string for safe embedding inside a ``<script>`` tag."""
    # Prevent </script> injection: escape the closing tag sequence
    payload = payload.replace("</", "<\\/")
    # Escape U+2028/U+2029 (line/paragraph separators) — they break JS parsing
    # inside <script> blocks even though they are valid JSON string content.
    payload = payload.replace("\u2028", "\\u2028")
    payload = payload.replace("\u2029", "\\u2029")
    return payload


def render_health_report_html(report: HealthReport) -> str:
    """Render a HealthReport as a self-contained HTML SPA dashboard.

    Loads the bundled spa_template.html, serialises *report* to JSON, and
    injects it into the template by replacing the sentinel placeholder.

    Args:
        report: A fully or partially populated HealthReport.

    Returns:
        A complete HTML string suitable for writing to a ``.html`` file.
    """
    template = _load_template()
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    payload = _escape_for_script(payload)
    return template.replace(_SENTINEL, payload)


def render_watch_session_html(
    report: HealthReport,
    session_data: WatchSessionData,
) -> str:
    """Render a watch-session HTML report with diff timeline.

    Replaces **both** sentinels in the SPA template:

    - ``/*{{REPORT_DATA_JSON}}*/null`` → last HealthReport JSON
    - ``/*{{WATCH_SESSION_JSON}}*/null`` → WatchSessionData JSON

    Args:
        report: The *last* HealthReport from the watch session.
        session_data: Accumulated diff timeline and session metadata.

    Returns:
        A complete HTML string suitable for writing to a ``.html`` file.
    """
    template = _load_template()

    report_payload = json.dumps(
        report.model_dump(mode="json"), ensure_ascii=False
    )
    report_payload = _escape_for_script(report_payload)

    session_payload = json.dumps(session_data.to_dict(), ensure_ascii=False)
    session_payload = _escape_for_script(session_payload)

    html = template.replace(_SENTINEL, report_payload)
    html = html.replace(_WATCH_SENTINEL, session_payload)
    return html

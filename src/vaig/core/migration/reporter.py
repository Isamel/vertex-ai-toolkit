"""MigrationReporter: builds structured reports and HTML output from migration state."""
from __future__ import annotations

import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vaig.core.migration.budget import MigrationBudgetManager
from vaig.core.migration.state import FileStatus, MigrationState

__all__ = ["MigrationReport", "MigrationReporter"]


class MigrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    change_id: str
    source_kind: str
    target_kind: str
    generated_at: str
    files_total: int
    files_completed: int
    files_failed: int
    files_skipped: int
    gate_summary: dict[str, int] = Field(default_factory=dict)  # gate_name → fail_count
    budget_summary: dict[str, object] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class MigrationReporter:
    def __init__(
        self,
        state: MigrationState,
        budget: MigrationBudgetManager | None = None,
    ) -> None:
        self._state = state
        self._budget = budget

    def build_report(self) -> MigrationReport:
        files = list(self._state.files.values())
        files_completed = sum(1 for f in files if f.status == FileStatus.COMPLETED)
        files_failed = sum(1 for f in files if f.status == FileStatus.FAILED)
        files_skipped = sum(1 for f in files if f.status == FileStatus.SKIPPED)

        # Aggregate gate failures
        gate_summary: dict[str, int] = {}
        errors: list[str] = []
        for record in files:
            if record.error:
                errors.append(f"{record.source_path}: {record.error}")
            for gate_result in record.gate_results:
                if not gate_result.get("passed", True):
                    gate_name = str(gate_result.get("gate", "unknown"))
                    gate_summary[gate_name] = gate_summary.get(gate_name, 0) + 1

        budget_summary: dict[str, object] = (
            self._budget.summary() if self._budget is not None else {}
        )

        return MigrationReport(
            change_id=self._state.change_id,
            source_kind=self._state.source_kind,
            target_kind=self._state.target_kind,
            generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
            files_total=len(files),
            files_completed=files_completed,
            files_failed=files_failed,
            files_skipped=files_skipped,
            gate_summary=gate_summary,
            budget_summary=budget_summary,
            errors=errors,
        )

    def to_html(self, report: MigrationReport) -> str:
        """Generate a minimal but readable HTML report string."""

        def _row(label: str, value: object) -> str:
            return f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>"

        gate_rows = "".join(
            f"<tr><td>{name}</td><td>{count}</td></tr>"
            for name, count in report.gate_summary.items()
        ) or "<tr><td colspan='2'>No gate failures</td></tr>"

        budget_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in report.budget_summary.items()
        ) or "<tr><td colspan='2'>No budget data</td></tr>"

        error_items = "".join(f"<li>{e}</li>" for e in report.errors) or "<li>None</li>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Migration Report — {report.change_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: left; }}
    th {{ background: #f0f0f0; }}
    h1 {{ color: #333; }}
    h2 {{ color: #555; }}
  </style>
</head>
<body>
  <h1>Migration Report</h1>
  <p><strong>Change ID:</strong> {report.change_id}</p>
  <p><strong>Generated at:</strong> {report.generated_at}</p>

  <h2>Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {_row("Source kind", report.source_kind)}
    {_row("Target kind", report.target_kind)}
    {_row("Total files", report.files_total)}
    {_row("Completed", report.files_completed)}
    {_row("Failed", report.files_failed)}
    {_row("Skipped", report.files_skipped)}
  </table>

  <h2>Gate Failures</h2>
  <table>
    <tr><th>Gate</th><th>Fail Count</th></tr>
    {gate_rows}
  </table>

  <h2>Budget</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {budget_rows}
  </table>

  <h2>Errors</h2>
  <ul>
    {error_items}
  </ul>
</body>
</html>"""
        return html

    def save_html(self, report: MigrationReport, path: Path) -> None:
        """Write HTML to path."""
        path.write_text(self.to_html(report), encoding="utf-8")

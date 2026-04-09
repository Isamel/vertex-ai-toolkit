"""CheckOutput — stable, string-only schema for Terraform/CI consumers.

This model is the **external contract** between ``vaig check`` and
Terraform's ``external`` data source.  Every value is a ``str`` to
satisfy the ``external`` data source requirement that all map values
are strings.

The schema is intentionally decoupled from the internal
``HealthReport`` / ``ExecutiveSummary`` models — changes to those
MUST NOT break ``CheckOutput`` output.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from vaig import __version__


class CheckOutput(BaseModel):
    """Flat, string-only health check result for Terraform ``external`` data source.

    Every field is a ``str`` so that ``json.dumps(self.model_dump())``
    produces a map of ``string → string``, which is the only value type
    accepted by Terraform's ``external`` data source.
    """

    status: str
    """``HEALTHY``, ``DEGRADED``, ``CRITICAL``, ``UNKNOWN``, ``ERROR``, or ``TIMEOUT``."""

    critical_count: str
    """Number of critical findings (as string, e.g. ``"3"``)."""

    warning_count: str
    """Number of warning-level findings (as string)."""

    issues_found: str
    """Total number of issues found (as string)."""

    services_checked: str
    """Number of services checked (as string)."""

    summary_text: str
    """Human-readable 1-2 sentence summary of the health status."""

    scope: str
    """Blast radius — e.g. ``"Namespace: production"`` or ``"Cluster-wide"``."""

    timestamp: str
    """ISO 8601 UTC timestamp of when the check ran."""

    version: str
    """``vaig`` version string for schema tracking."""

    cached: str
    """``"true"`` or ``"false"`` — whether this result came from cache."""

    @classmethod
    def from_health_report(
        cls,
        report: object,
        *,
        cached: bool = False,
    ) -> CheckOutput:
        """Build a ``CheckOutput`` from a ``HealthReport``.

        Accesses ``report.executive_summary`` fields and converts every
        value to a string.  This is the ONLY coupling point between the
        internal schema and the external contract — if internal fields
        are renamed, only this method needs updating.

        Args:
            report: A ``HealthReport`` instance (typed as ``object`` to
                avoid importing the heavy schema module at module level).
            cached: Whether this result was served from the file cache.
        """
        es = report.executive_summary  # type: ignore[attr-defined]
        return cls(
            status=str(es.overall_status.value),
            critical_count=str(es.critical_count),
            warning_count=str(es.warning_count),
            issues_found=str(es.issues_found),
            services_checked=str(es.services_checked),
            summary_text=str(es.summary_text),
            scope=str(es.scope),
            timestamp=datetime.now(UTC).isoformat(),
            version=__version__,
            cached=str(cached).lower(),
        )

    @classmethod
    def from_error(cls, error_type: str, message: str) -> CheckOutput:
        """Build a ``CheckOutput`` representing an error or timeout.

        Args:
            error_type: Status string — ``"ERROR"`` or ``"TIMEOUT"``.
            message: Human-readable error description.
        """
        return cls(
            status=error_type.upper(),
            critical_count="0",
            warning_count="0",
            issues_found="0",
            services_checked="0",
            summary_text=message,
            scope="",
            timestamp=datetime.now(UTC).isoformat(),
            version=__version__,
            cached="false",
        )

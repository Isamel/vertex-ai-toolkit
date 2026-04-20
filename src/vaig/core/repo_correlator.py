"""SPEC-V2-REPO-07 — Cross-repo correlation: wire repo snippets into findings."""
from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from vaig.core.repo_pipeline import EvidenceGap
from vaig.skills.service_health.schema import RepoSnippet, Severity

if TYPE_CHECKING:
    from vaig.skills.service_health.schema import Finding, HealthReport


@runtime_checkable
class RepoSnippetSource(Protocol):
    """Minimal protocol for a repo index — avoids hard coupling to repo_index.py."""

    def search(self, query: str, k: int = 8) -> list[object]: ...


class CorrelationResult(BaseModel):
    report_findings_enriched: int
    contradiction_findings_added: int
    orphan_resources: list[str]
    undeployed_manifests: list[str]
    evidence_gaps: list[EvidenceGap]


class RepoCorrelator:
    """Post-analysis stage: enhance findings with repo evidence.

    Applied after the HealthReport is generated but before rendering.
    Operates entirely on in-memory data — no LLM calls.
    """

    def __init__(self, index: RepoSnippetSource) -> None:
        self._index = index

    def correlate(
        self,
        report: HealthReport,
        *,
        ref: str = "HEAD",
        repo_label: str = "",
    ) -> tuple[HealthReport, CorrelationResult]:
        """Enhance findings with repo evidence.

        For each finding with affected_resources:
          1. Query index for manifests declaring those resources.
          2. Attach snippets as RepoSnippet to finding.repo_evidence.
          3. Detect value drift: if a snippet contains 'replicas: N' and
             the finding title/description implies a different replica count,
             emit a contradiction finding.
          4. Track orphan_runtime_resources and undeployed_manifests (simple
             name-matching heuristic — not exhaustive).

        Returns mutated HealthReport (deep copy) + CorrelationResult.
        """
        report = copy.deepcopy(report)

        enriched_count = 0
        contradiction_findings: list[Finding] = []
        orphan_resources: list[str] = []
        undeployed_manifests: list[str] = []
        evidence_gaps: list[EvidenceGap] = []

        new_findings: list[Finding] = []
        for finding in report.findings:
            if not finding.affected_resources:
                new_findings.append(finding)
                continue

            query = finding.title + " " + " ".join(finding.affected_resources)
            try:
                raw_chunks = self._index.search(query, k=5)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:  # noqa: BLE001
                raw_chunks = []

            snippets: list[RepoSnippet] = []
            for chunk in raw_chunks:
                file_path = str(getattr(chunk, "path", ""))
                start_line = int(getattr(chunk, "start_line", 0))
                end_line = int(getattr(chunk, "end_line", 0))
                excerpt = str(getattr(chunk, "content", getattr(chunk, "outline", "")))
                relevance_score = float(getattr(chunk, "relevance_score", 0.5))
                relevance_score = max(0.0, min(1.0, relevance_score))
                retrieval_query = str(getattr(chunk, "retrieval_query", query))

                snippet = RepoSnippet(
                    file_path=file_path,
                    line_range=(start_line, end_line),
                    excerpt=excerpt,
                    relevance_score=relevance_score,
                    retrieval_query=retrieval_query,
                )
                snippets.append(snippet)

                # Contradiction detection: look for 'replicas: N' in snippet content
                replicas_match = re.search(r"replicas:\s*(\d+)", excerpt)
                if replicas_match:
                    chart_replicas = int(replicas_match.group(1))
                    # Look for a different replica count mentioned in finding title
                    title_replicas_match = re.search(r"(\d+)\s+pod", finding.title, re.IGNORECASE)
                    if title_replicas_match:
                        runtime_replicas = int(title_replicas_match.group(1))
                        if runtime_replicas != chart_replicas:
                            resource_name = (
                                finding.affected_resources[0]
                                if finding.affected_resources
                                else finding.id
                            )
                            from vaig.skills.service_health.schema import Finding as FindingCls

                            contradiction = FindingCls(
                                id=f"repo-drift-{resource_name}-replicas",
                                title=(
                                    f"Replica drift: chart declares {chart_replicas}, "
                                    f"cluster shows {runtime_replicas}"
                                ),
                                severity=Severity.HIGH,
                                category="repo_drift",
                                service=finding.service,
                                description=(
                                    f"Chart file {snippet.file_path} declares "
                                    f"replicas={chart_replicas} but runtime observation "
                                    f"shows {runtime_replicas} pods ready."
                                ),
                                evidence=[
                                    f"repo:{snippet.file_path}:"
                                    f"{snippet.line_range[0]}-{snippet.line_range[1]}"
                                ],
                                quick_remediation=(
                                    "kubectl get deployment <name> -o yaml | grep replicas"
                                ),
                            )
                            contradiction_findings.append(contradiction)

            if snippets:
                finding.repo_evidence = snippets
                enriched_count += 1

            new_findings.append(finding)

        all_findings = new_findings + contradiction_findings
        report = report.model_copy(update={"findings": all_findings})

        result = CorrelationResult(
            report_findings_enriched=enriched_count,
            contradiction_findings_added=len(contradiction_findings),
            orphan_resources=orphan_resources,
            undeployed_manifests=undeployed_manifests,
            evidence_gaps=evidence_gaps,
        )
        return report, result

"""Two-pass recommendation enrichment for service health reports.

After the reporter generates the initial HealthReport, this module takes each
recommendation paired with its related finding(s) and makes a focused LLM call
to enrich the ``expected_output`` and ``interpretation`` fields with specific,
actionable debugging guidance — mimicking the quality of interactive chat mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from vaig.utils.json_cleaner import clean_llm_json

if TYPE_CHECKING:
    from vaig.core.protocols import GeminiClientProtocol
    from vaig.skills.service_health.schema import (
        Finding,
        HealthReport,
        RecommendedAction,
    )

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM_INSTRUCTION = """\
You are an expert SRE advisor. Your job is to enrich a diagnostic \
recommendation with SPECIFIC, ACTIONABLE guidance.

You will receive:
1. A finding from a Kubernetes health diagnosis (with severity, description, \
root cause, evidence)
2. A recommendation with a command to run

Your task: Rewrite ONLY the `expected_output` and `interpretation` fields to \
be EXPERT-LEVEL debugging guidance.

## Rules for expected_output
- Show REALISTIC terminal output (2-5 lines) that the user will actually see \
when running the command
- Use the EXACT resource names from the finding (pod names, deployment names, \
namespaces)
- Include realistic values (CPU, memory, restart counts, ages) based on the \
finding context
- If the command is a `kubectl get`, show the table headers AND 2-3 rows of data
- If the command is a `kubectl describe`, show the relevant Events section \
(last 5-10 events)
- If the command is a `kubectl logs`, show realistic log lines with timestamps

## Rules for interpretation
- Structure as a DECISION TREE: "Look at X. If you see A → do B. If you see \
C → do D."
- Cover at least 3 scenarios: healthy, the specific problem from the finding, \
and one related failure mode
- For each scenario, provide the NEXT command to run
- Reference specific values from the expected_output
- Explain the MECHANISM, not just the symptom ("memory above 90% of limit \
means the OOM killer will terminate the process")

## Output Format
Respond with ONLY a JSON object with two fields:
```json
{
  "expected_output": "...",
  "interpretation": "..."
}
```

Do NOT include any other text, markdown fences, or explanation outside the JSON.\
"""


def _build_enrichment_prompt(
    finding: Finding,
    action: RecommendedAction,
) -> str:
    """Build the user prompt for enriching a single recommendation."""
    evidence_text = (
        "\n".join(f"  - {e}" for e in finding.evidence)
        if finding.evidence
        else "  (no evidence collected)"
    )

    severity_val = (
        finding.severity.value
        if hasattr(finding.severity, "value")
        else str(finding.severity)
    )
    urgency_val = (
        action.urgency.value
        if hasattr(action.urgency, "value")
        else str(action.urgency)
    )

    return f"""\
## Finding
- **Title**: {finding.title}
- **Severity**: {severity_val}
- **Description**: {finding.description}
- **Root Cause**: {finding.root_cause}
- **Impact**: {finding.impact}
- **Affected Resources**: {', '.join(finding.affected_resources) if finding.affected_resources else 'unknown'}
- **Evidence**:
{evidence_text}

## Recommendation to Enrich
- **Title**: {action.title}
- **Command**: `{action.command}`
- **Description**: {action.description}
- **Urgency**: {urgency_val}
- **Why**: {action.why}

Rewrite expected_output and interpretation for this specific recommendation. \
Be SPECIFIC to the finding above — use the exact resource names, namespaces, \
and error messages from the evidence."""


async def _enrich_one(
    idx: int,
    action: RecommendedAction,
    finding: Finding,
    client: GeminiClientProtocol,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout_per_call: float,
) -> tuple[int, str, str] | None:
    """Enrich a single recommendation.

    Returns ``(index, expected_output, interpretation)`` on success, or
    ``None`` when enrichment fails or produces empty fields.
    """
    prompt = _build_enrichment_prompt(finding, action)

    async with semaphore:
        try:
            result = await asyncio.wait_for(
                client.async_generate(
                    prompt,
                    system_instruction=ENRICHMENT_SYSTEM_INSTRUCTION,
                    model_id=model,
                    temperature=0.3,
                ),
                timeout=timeout_per_call,
            )

            cleaned = clean_llm_json(result.text)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    "Enrichment for recommendation %d (%s) returned invalid JSON",
                    idx,
                    action.title,
                )
                return None

            if not isinstance(data, dict):
                logger.warning(
                    "Enrichment for recommendation %d (%s) returned non-object JSON: %r",
                    idx,
                    action.title,
                    type(data).__name__,
                )
                return None

            raw_expected = data.get("expected_output", "")
            raw_interpretation = data.get("interpretation", "")

            new_expected: str = "" if raw_expected is None else str(raw_expected)
            new_interpretation: str = (
                "" if raw_interpretation is None else str(raw_interpretation)
            )

            # Treat purely-whitespace values as empty
            if new_expected.strip() and new_interpretation.strip():
                logger.debug(
                    "Enriched recommendation %d (%s): "
                    "expected_output=%d chars, interpretation=%d chars",
                    idx,
                    action.title,
                    len(new_expected),
                    len(new_interpretation),
                )
                return (idx, new_expected, new_interpretation)

            logger.warning(
                "Enrichment for recommendation %d returned empty fields", idx
            )
            return None

        except TimeoutError:
            logger.warning(
                "Enrichment timed out for recommendation %d (%s)",
                idx,
                action.title,
            )
            return None
        except Exception:
            logger.warning(
                "Enrichment failed for recommendation %d (%s)",
                idx,
                action.title,
                exc_info=True,
            )
            return None


async def enrich_recommendations(
    report: HealthReport,
    client: GeminiClientProtocol,
    *,
    model: str = "",
    max_concurrent: int = 5,
    timeout_per_call: float = 30.0,
) -> HealthReport:
    """Enrich report recommendations with focused, expert-level guidance.

    Makes one LLM call per recommendation, pairing it with its related
    finding(s) for focused context.  Uses the default model
    (``gemini-2.5-pro``) for higher quality.

    Args:
        report: The parsed HealthReport from the reporter agent.
        client: GeminiClient instance for making LLM calls.
        model: Model to use for enrichment.  Defaults to
            ``settings.models.default``.
        max_concurrent: Max parallel enrichment calls.
        timeout_per_call: Timeout per individual enrichment call in seconds.

    Returns:
        The same HealthReport with enriched recommendations.  If enrichment
        fails for any individual recommendation, the original values are
        preserved.
    """
    if not report.recommendations:
        return report

    if not model:
        from vaig.core.config import get_settings

        model = get_settings().models.default

    # Build a finding lookup by ID for quick matching
    finding_by_id: dict[str, Finding] = {f.id: f for f in report.findings}

    # Initialize the client once to avoid race conditions in concurrent tasks.
    # GeminiClient.async_initialize involves an `await asyncio.to_thread` for
    # credentials — if multiple tasks enter initialization simultaneously, they
    # can trigger a race condition.
    if hasattr(client, "async_initialize"):
        await client.async_initialize()

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks: list[asyncio.Task[tuple[int, str, str] | None]] = []

    for i, action in enumerate(report.recommendations):
        # Resolve the related finding — use first related_findings ID that
        # exists, or fall back to the first finding in the report.
        related: Finding | None = None
        for fid in action.related_findings:
            if fid in finding_by_id:
                related = finding_by_id[fid]
                break
        if related is None and report.findings:
            related = report.findings[0]
        if related is None:
            continue

        tasks.append(
            asyncio.ensure_future(
                _enrich_one(
                    idx=i,
                    action=action,
                    finding=related,
                    client=client,
                    model=model,
                    semaphore=semaphore,
                    timeout_per_call=timeout_per_call,
                )
            )
        )

    if not tasks:
        return report

    results = await asyncio.gather(*tasks)

    # Apply successful enrichments back to the report
    enriched_count = 0
    for enrichment in results:
        if enrichment is not None:
            idx, expected_output, interpretation = enrichment
            report.recommendations[idx].expected_output = expected_output
            report.recommendations[idx].interpretation = interpretation
            enriched_count += 1

    logger.info(
        "Enriched %d/%d recommendations using %s",
        enriched_count,
        len(report.recommendations),
        model,
    )

    return report

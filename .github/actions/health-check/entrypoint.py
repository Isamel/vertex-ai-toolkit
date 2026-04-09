"""GitHub Actions entrypoint for VAIG health-check action.

Parses ``INPUT_*`` environment variables, runs the discovery pipeline via
``execute_skill_headless()``, posts a PR comment with the health report,
sets ``GITHUB_OUTPUT`` values, and exits 0 (pass) or 1 (fail) based on
severity threshold.

This is a standalone script — NOT a CLI wrapper.  It imports directly from
the ``vaig`` package (installed in the Docker image) and does NOT import Rich.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import uuid
from dataclasses import dataclass
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("vaig-gha")

# ── Severity mapping ─────────────────────────────────────────────

SEVERITY_LEVELS: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "INFO": 0,
}

VALID_FAIL_ON = frozenset(SEVERITY_LEVELS.keys())

# ── Comment marker for idempotent updates ────────────────────────

COMMENT_MARKER = "<!-- vaig-health-check -->"


# ── Input dataclass ──────────────────────────────────────────────


@dataclass
class ActionInputs:
    """Typed representation of GitHub Action inputs."""

    cluster: str
    project_id: str
    location: str
    namespace: str = "default"
    fail_on: str = "CRITICAL"
    model: str = "gemini-2.5-flash"
    comment: bool = True
    timeout: int = 300


# ── Core functions ───────────────────────────────────────────────


def parse_inputs() -> ActionInputs:
    """Read ``INPUT_*`` env vars into a typed :class:`ActionInputs`.

    Validates that required fields are present and ``fail-on`` is a valid
    severity level.  Exits with code 1 on validation failure.
    """
    cluster = os.environ.get("INPUT_CLUSTER", "").strip()
    project_id = os.environ.get("INPUT_PROJECT-ID", "").strip()
    location = os.environ.get("INPUT_LOCATION", "").strip()

    missing: list[str] = []
    if not cluster:
        missing.append("cluster")
    if not project_id:
        missing.append("project-id")
    if not location:
        missing.append("location")

    if missing:
        for name in missing:
            print(f"Required input '{name}' not provided", file=sys.stderr)
        sys.exit(1)

    fail_on = os.environ.get("INPUT_FAIL-ON", "CRITICAL").strip().upper()
    if fail_on not in VALID_FAIL_ON:
        print(
            f"Invalid fail-on value '{fail_on}'. Must be: CRITICAL, HIGH, MEDIUM, LOW, INFO",
            file=sys.stderr,
        )
        sys.exit(1)

    comment_raw = os.environ.get("INPUT_COMMENT", "true").strip().lower()
    timeout_raw = os.environ.get("INPUT_TIMEOUT", "300").strip()
    try:
        timeout = int(timeout_raw)
    except ValueError:
        timeout = 300

    return ActionInputs(
        cluster=cluster,
        project_id=project_id,
        location=location,
        namespace=os.environ.get("INPUT_NAMESPACE", "default").strip(),
        fail_on=fail_on,
        model=os.environ.get("INPUT_MODEL", "gemini-2.5-flash").strip(),
        comment=comment_raw == "true",
        timeout=timeout,
    )


def build_config(inputs: ActionInputs) -> tuple[Any, Any]:
    """Construct :class:`Settings` and :class:`GKEConfig` from action inputs.

    Returns:
        ``(settings, gke_config)`` tuple of Pydantic models.
    """
    from vaig.core.config import GCPConfig, GKEConfig, ModelsConfig, Settings

    settings = Settings(
        gcp=GCPConfig(project_id=inputs.project_id, location=inputs.location),
        models=ModelsConfig(default=inputs.model),
    )
    gke_config = GKEConfig(
        cluster_name=inputs.cluster,
        project_id=inputs.project_id,
        location=inputs.location,
        default_namespace=inputs.namespace,
    )
    return settings, gke_config


def extract_severity(result: Any) -> tuple[str, int, int]:
    """Extract max severity from an :class:`OrchestratorResult`.

    Iterates ``result.structured_report.findings`` and returns the highest
    severity found.

    Returns:
        ``(max_severity_str, severity_level, findings_count)``
        — ``("NONE", -1, 0)`` when there are no findings or no report.
    """
    report = getattr(result, "structured_report", None)
    if report is None:
        return ("NONE", -1, 0)

    findings = getattr(report, "findings", None)
    if not findings:
        return ("NONE", -1, 0)

    max_level = -1
    max_name = "NONE"
    for finding in findings:
        sev_str = str(finding.severity).upper()
        level = SEVERITY_LEVELS.get(sev_str, 0)
        if level > max_level:
            max_level = level
            max_name = sev_str

    return (max_name, max_level, len(findings))


def format_comment(
    report_md: str,
    status: str,
    findings_count: int,
    cost: float,
) -> str:
    """Wrap the health report in a collapsible PR comment.

    The comment includes a hidden marker for idempotent updates and a
    one-line summary outside the ``<details>`` block.
    """
    summary_line = f"**VAIG Health Check** — Status: `{status}` | Findings: {findings_count} | Cost: ${cost:.4f}"
    body = (
        f"{COMMENT_MARKER}\n"
        f"{summary_line}\n\n"
        f"<details>\n"
        f"<summary>Full Health Report</summary>\n\n"
        f"{report_md}\n\n"
        f"</details>\n"
    )
    return body


def post_comment(comment: str) -> None:
    """Post or update a PR comment via the GitHub REST API.

    Reads ``GITHUB_EVENT_PATH`` to extract the PR number.  Uses
    ``GITHUB_TOKEN`` for authentication.  If a previous comment with
    :data:`COMMENT_MARKER` exists, it is updated via PATCH; otherwise
    a new comment is created via POST.

    Falls back to printing the report to stdout if the API call fails
    (e.g. insufficient permissions on fork PRs).
    """
    import requests

    token = os.environ.get("GITHUB_TOKEN", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if not event_path or not repo or not token:
        logger.warning("Missing GitHub context — printing report to stdout")
        print(comment)
        return

    # Extract PR number from event payload
    try:
        with open(event_path) as f:
            event = json.load(f)
        pr_number = (
            event.get("pull_request", {}).get("number")
            or event.get("issue", {}).get("number")
            or event.get("number")
        )
    except (OSError, json.JSONDecodeError, TypeError):
        pr_number = None

    if not pr_number:
        logger.warning("No PR number found in event payload — printing report to stdout")
        print(comment)
        return

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    api_base = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    comments_url = f"{api_base}/repos/{repo}/issues/{pr_number}/comments"

    # Look for existing comment with our marker (paginated)
    existing_comment_id = None
    try:
        page_url: str | None = f"{comments_url}?per_page=100"
        while page_url:
            resp = requests.get(page_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                break
            for c in resp.json():
                if COMMENT_MARKER in c.get("body", ""):
                    existing_comment_id = c["id"]
                    break
            if existing_comment_id:
                break
            # Follow Link header for next page
            page_url = resp.links.get("next", {}).get("url")
    except requests.RequestException:
        pass  # proceed to create new comment

    try:
        if existing_comment_id:
            patch_url = f"{api_base}/repos/{repo}/issues/comments/{existing_comment_id}"
            resp = requests.patch(
                patch_url,
                headers=headers,
                json={"body": comment},
                timeout=30,
            )
        else:
            resp = requests.post(
                comments_url,
                headers=headers,
                json={"body": comment},
                timeout=30,
            )

        if resp.status_code not in (200, 201):
            logger.warning(
                "Could not post PR comment — %s %s",
                resp.status_code,
                resp.text[:200],
            )
            print(comment)
    except requests.RequestException as exc:
        logger.warning("Could not post PR comment — %s", exc)
        print(comment)


def set_outputs(
    status: str,
    findings_count: int,
    max_severity: str,
    report: str,
) -> None:
    """Append action outputs to the ``$GITHUB_OUTPUT`` file.

    Uses the multiline delimiter syntax for the ``report`` field.
    """
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        logger.warning("GITHUB_OUTPUT not set — skipping output writing")
        return

    with open(output_file, "a") as f:
        f.write(f"status={status}\n")
        f.write(f"findings-count={findings_count}\n")
        f.write(f"max-severity={max_severity}\n")
        # Multiline output using unique heredoc-style delimiter to prevent collisions
        delimiter = f"VAIG_EOF_{uuid.uuid4().hex}"
        f.write(f"report<<{delimiter}\n")
        f.write(report)
        if not report.endswith("\n"):
            f.write("\n")
        f.write(f"{delimiter}\n")


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    """Signal handler for pipeline timeout."""
    raise TimeoutError("Health check timed out")


def main() -> int:
    """Orchestrate the full health-check flow.

    Returns 0 (pass) or 1 (fail).
    """
    # Step 1: Parse inputs
    inputs = parse_inputs()

    # Step 2: Build config
    settings, gke_config = build_config(inputs)

    # Step 3: Set timeout alarm
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(inputs.timeout)

    try:
        # Step 4: Resolve skill + execute
        from vaig.skills.discovery.skill import DiscoverySkill

        skill = DiscoverySkill()
        query = f"Discover and report on the health of namespace '{inputs.namespace}'"

        from vaig.core.headless import execute_skill_headless

        result = execute_skill_headless(settings, skill, query, gke_config)

        # Cancel timeout
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        # Step 5: Extract severity
        max_severity, severity_level, findings_count = extract_severity(result)

        # Step 6: Determine pass/fail
        threshold_level = SEVERITY_LEVELS.get(inputs.fail_on, 4)
        if severity_level >= threshold_level:
            status = "fail"
            exit_code = 1
        else:
            status = "pass"
            exit_code = 0

        # Step 7: Format report
        report = getattr(result, "structured_report", None)
        if report is not None and hasattr(report, "to_markdown"):
            report_md = report.to_markdown()
        else:
            report_md = getattr(result, "synthesized_output", "No report available")

        cost = getattr(result, "run_cost_usd", 0.0)

        # Step 8: Post PR comment
        if inputs.comment:
            comment_body = format_comment(report_md, status, findings_count, cost)
            post_comment(comment_body)

        # Step 9: Set outputs
        set_outputs(status, findings_count, max_severity, report_md)

        return exit_code

    except TimeoutError:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        msg = f"Health check timed out after {inputs.timeout}s"
        print(msg, file=sys.stderr)
        if inputs.comment:
            error_comment = format_comment(
                f"⏱️ {msg}", "error", 0, 0.0,
            )
            post_comment(error_comment)
        set_outputs("error", 0, "UNKNOWN", msg)
        return 1

    except Exception as exc:  # noqa: BLE001
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        # Detect auth failures
        exc_str = str(exc).lower()
        if "credential" in exc_str or "auth" in exc_str or "permission" in exc_str:
            msg = "Authentication failed — ensure google-github-actions/auth runs before this action"
        else:
            msg = f"Health check failed: {exc}"
        print(msg, file=sys.stderr)

        if inputs.comment:
            error_comment = format_comment(
                f"❌ {msg}", "error", 0, 0.0,
            )
            try:
                post_comment(error_comment)
            except Exception:  # noqa: BLE001
                pass  # best-effort — don't mask the original error
        set_outputs("error", 0, "UNKNOWN", msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Fine-tuning pipeline — BQ data extraction, JSONL transformation, and tuning job submission."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import bigquery, storage
    from google.genai import Client as GenaiClient

from vaig.core.config import DEFAULT_CHARS_PER_TOKEN, ExportConfig, TrainingConfig

logger = logging.getLogger(__name__)

# ── Retry helpers (shared with export.py pattern) ────────────

_RETRYABLE_GCP_ERROR_NAMES = {
    "TooManyRequests",
    "ServiceUnavailable",
    "InternalServerError",
    "BadGateway",
    "GatewayTimeout",
}


def _is_transient_gcp_error(exc: Exception) -> bool:
    """Return True when *exc* looks retryable for GCP network/service failures."""
    return exc.__class__.__name__ in _RETRYABLE_GCP_ERROR_NAMES


# ── Dependency guard ─────────────────────────────────────────


def _require_rag_deps() -> None:
    """Raise ImportError with install instructions if [rag] deps are missing."""
    try:
        import google.cloud.bigquery  # noqa: F401
        import google.cloud.storage  # noqa: F401
    except ImportError:
        raise ImportError(
            "Training features require the [rag] extras. "
            "Install with: pip install 'vertex-ai-toolkit[rag]'"
        ) from None


# ── Shared retry helper ──────────────────────────────────────


def _run_with_retry(operation_name: str, func: Any) -> Any:
    """Execute *func* with exponential backoff for transient GCP failures."""
    delays = (0.5, 1.0, 2.0)

    for attempt, delay in enumerate(delays, start=1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt == len(delays) or not _is_transient_gcp_error(exc):
                raise
            logger.warning(
                "%s failed with transient error (%s). Retrying in %.1fs [attempt %d/%d]",
                operation_name,
                exc.__class__.__name__,
                delay,
                attempt,
                len(delays),
            )
            time.sleep(delay)

    # Unreachable: for-loop always returns or raises. Defensive guard only.
    raise RuntimeError(f"{operation_name} retry loop exited unexpectedly")  # pragma: no cover


# ── Result dataclasses ───────────────────────────────────────


@dataclass
class PrepareResult:
    """Result metadata for a training data preparation operation."""

    jsonl_path: Path
    total_examples: int
    avg_rating: float
    estimated_tokens: int


@dataclass
class SubmitResult:
    """Result metadata for a tuning job submission."""

    job_name: str
    gcs_uri: str
    base_model: str
    status: str


# ── Training Data Preparer ───────────────────────────────────


class TrainingDataPreparer:
    """Extracts rated examples from BigQuery and transforms them into Gemini JSONL.

    Uses lazy client initialization and constructor injection for testing,
    following the DataExporter pattern from ``export.py``.

    Usage::

        preparer = TrainingDataPreparer(export_config, training_config)
        result = preparer.prepare(output_path=Path("train.jsonl"))
    """

    def __init__(
        self,
        config: ExportConfig,
        training_config: TrainingConfig,
        *,
        bq_client: bigquery.Client | None = None,
    ) -> None:
        """Create a TrainingDataPreparer.

        Args:
            config: Export configuration (project, dataset, etc.).
            training_config: Training pipeline configuration.
            bq_client: Optional pre-built BigQuery client for testing.
        """
        self._config = config
        self._training_config = training_config
        self._bq_client_override = bq_client
        self._bq_client: bigquery.Client | None = None
        self._bq_lock = threading.Lock()

    # ── Private client accessor (lazy) ───────────────────────

    def _get_bq_client(self) -> bigquery.Client:
        """Return the BigQuery client, creating it on first use."""
        if self._bq_client_override is not None:
            return self._bq_client_override

        if self._bq_client is None:
            with self._bq_lock:
                if self._bq_client is None:
                    _require_rag_deps()
                    from google.cloud import bigquery as bq_mod

                    project = self._config.gcp_project_id.strip()
                    self._bq_client = bq_mod.Client(project=project)
        return self._bq_client

    # ── Public API ───────────────────────────────────────────

    def extract_pairs(self, min_rating: int, max_examples: int) -> list[dict[str, Any]]:
        """Query BigQuery for feedback rows joined with context tables.

        Args:
            min_rating: Minimum feedback rating to include.
            max_examples: Maximum number of rows to return.

        Returns:
            List of dicts with keys: tool_name, input_params, report_markdown,
            event_type, cluster_name, namespace, rating.
        """
        client = self._get_bq_client()
        dataset = self._config.bigquery_dataset

        query = f"""
            SELECT
                tc.tool_name,
                tc.input_params,
                hr.report_markdown,
                te.event_type,
                hr.cluster_name,
                hr.namespace,
                f.rating
            FROM `{dataset}.feedback` AS f
            LEFT JOIN `{dataset}.health_reports` AS hr
                ON f.run_id = hr.run_id
            LEFT JOIN `{dataset}.tool_calls` AS tc
                ON f.run_id = tc.run_id
            LEFT JOIN `{dataset}.telemetry_events` AS te
                ON f.run_id = te.run_id
            WHERE f.rating >= @min_rating
            ORDER BY f.timestamp DESC
            LIMIT @max_examples
        """

        from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

        job_config = QueryJobConfig(
            query_parameters=[
                ScalarQueryParameter("min_rating", "INT64", min_rating),
                ScalarQueryParameter("max_examples", "INT64", max_examples),
            ]
        )

        result = _run_with_retry(
            "BQ training data query",
            lambda: client.query(query, job_config=job_config).result(),
        )
        return [dict(row) for row in result]

    def transform_to_jsonl(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform BQ rows into Gemini supervised-tuning JSONL format.

        Each row becomes:
        ``{"contents": [{"role": "user", "parts": [{"text": ...}]},
                        {"role": "model", "parts": [{"text": ...}]}]}``

        Args:
            rows: Raw BQ rows from extract_pairs().

        Returns:
            List of JSONL-ready dicts.
        """
        entries: list[dict[str, Any]] = []
        for row in rows:
            tool_name = row.get("tool_name", "unknown_tool")
            input_params = row.get("input_params", "{}")
            report_markdown = row.get("report_markdown", "")
            cluster_name = row.get("cluster_name", "")
            namespace = row.get("namespace", "")

            if not report_markdown:
                continue

            context = f" for {cluster_name}/{namespace}" if cluster_name or namespace else ""
            user_text = f"Analyze {tool_name} with params {input_params}{context}"

            entry = {
                "contents": [
                    {"role": "user", "parts": [{"text": user_text}]},
                    {"role": "model", "parts": [{"text": report_markdown}]},
                ]
            }
            entries.append(entry)
        return entries

    def prepare(
        self,
        output_path: Path | None = None,
        dry_run: bool = False,
    ) -> PrepareResult:
        """Orchestrate: extract → transform → write JSONL file.

        Args:
            output_path: Path for the output JSONL file.  Defaults to
                ``training_config.output_dir / training_YYYYMMDD_HHMMSS.jsonl``.
            dry_run: When True, report stats without writing a file.

        Returns:
            PrepareResult with statistics.

        Raises:
            ValueError: When fewer than ``min_examples`` rows are found.
        """
        tc = self._training_config
        rows = self.extract_pairs(tc.min_rating, tc.max_examples)
        entries = self.transform_to_jsonl(rows)

        if len(entries) < tc.min_examples:
            raise ValueError(
                f"Insufficient examples: {len(entries)} found, "
                f"{tc.min_examples} required"
            )

        # Estimate tokens
        total_chars = sum(len(json.dumps(e)) for e in entries)
        estimated_tokens = int(total_chars / DEFAULT_CHARS_PER_TOKEN)

        avg_rating = 0.0
        if rows:
            ratings = [r.get("rating", 0) for r in rows]
            avg_rating = sum(ratings) / len(ratings)

        if output_path is None:
            from datetime import UTC, datetime

            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = tc.output_dir / f"training_{ts}.jsonl"

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info("Wrote %d training examples to %s", len(entries), output_path)

        return PrepareResult(
            jsonl_path=output_path,
            total_examples=len(entries),
            avg_rating=round(avg_rating, 2),
            estimated_tokens=estimated_tokens,
        )


# ── Tuning Job Submitter ─────────────────────────────────────


class TuningJobSubmitter:
    """Validates JSONL, uploads to GCS, and submits Vertex AI tuning jobs.

    Uses lazy client initialization and constructor injection for testing,
    following the DataExporter pattern from ``export.py``.

    Usage::

        submitter = TuningJobSubmitter(export_config, training_config)
        result = submitter.submit(Path("train.jsonl"))
    """

    def __init__(
        self,
        config: ExportConfig,
        training_config: TrainingConfig,
        *,
        gcs_client: storage.Client | None = None,
        genai_client: GenaiClient | None = None,
    ) -> None:
        """Create a TuningJobSubmitter.

        Args:
            config: Export configuration (project, bucket, etc.).
            training_config: Training pipeline configuration.
            gcs_client: Optional pre-built GCS client for testing.
            genai_client: Optional pre-built genai client for testing.
        """
        self._config = config
        self._training_config = training_config
        self._gcs_client_override = gcs_client
        self._genai_client_override = genai_client
        self._gcs_client: storage.Client | None = None
        self._genai_client: GenaiClient | None = None
        self._gcs_lock = threading.Lock()
        self._genai_lock = threading.Lock()

    # ── Private client accessors (lazy) ──────────────────────

    def _get_gcs_client(self) -> storage.Client:
        """Return the GCS client, creating it on first use."""
        if self._gcs_client_override is not None:
            return self._gcs_client_override

        if self._gcs_client is None:
            with self._gcs_lock:
                if self._gcs_client is None:
                    _require_rag_deps()
                    from google.cloud import storage as gcs_mod

                    self._gcs_client = gcs_mod.Client(
                        project=self._config.gcp_project_id.strip() or None
                    )
        return self._gcs_client

    def _get_genai_client(self) -> GenaiClient:
        """Return the genai client, creating it on first use."""
        if self._genai_client_override is not None:
            return self._genai_client_override

        if self._genai_client is None:
            with self._genai_lock:
                if self._genai_client is None:
                    _require_rag_deps()
                    import google.auth
                    from google import genai

                    credentials, default_project = google.auth.default()
                    configured_project = self._config.gcp_project_id.strip() or None
                    project = configured_project or default_project

                    self._genai_client = genai.Client(
                        vertexai=True,
                        project=project,
                        credentials=credentials,
                    )
        return self._genai_client

    # ── Public API ───────────────────────────────────────────

    def validate(self, jsonl_path: Path) -> dict[str, Any]:
        """Validate a JSONL file and return statistics.

        Args:
            jsonl_path: Path to the JSONL file.

        Returns:
            Dict with keys: count, avg_tokens, valid.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file has fewer than min_examples lines
                or contains malformed JSON.
        """
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        count = 0
        total_chars = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {jsonl_path}: {exc.msg}"
                    ) from exc
                count += 1
                total_chars += len(json.dumps(entry))

        if count < self._training_config.min_examples:
            raise ValueError(
                f"Insufficient examples: {count} found, "
                f"{self._training_config.min_examples} required"
            )

        avg_tokens = int((total_chars / DEFAULT_CHARS_PER_TOKEN) / max(count, 1))

        return {
            "count": count,
            "avg_tokens": avg_tokens,
            "valid": True,
        }

    def upload_to_gcs(self, jsonl_path: Path) -> str:
        """Upload a local JSONL file to GCS.

        Args:
            jsonl_path: Path to the local JSONL file.

        Returns:
            The ``gs://`` URI of the uploaded blob.
        """
        client = self._get_gcs_client()
        bucket_name = self._config.gcs_bucket
        prefix = self._training_config.gcs_staging_prefix
        blob_path = f"{prefix}{jsonl_path.name}"

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        _run_with_retry(
            "GCS upload",
            lambda: blob.upload_from_filename(str(jsonl_path)),
        )

        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        logger.info("Uploaded %s to %s", jsonl_path, gcs_uri)
        return gcs_uri

    def submit(
        self,
        jsonl_path: Path,
        dry_run: bool = False,
    ) -> SubmitResult:
        """Validate, upload, and create a tuning job.

        Args:
            jsonl_path: Path to the JSONL training file.
            dry_run: When True, validate and report stats without
                uploading or creating a job.

        Returns:
            SubmitResult with job details (or dry-run stats).
        """
        stats = self.validate(jsonl_path)
        tc = self._training_config

        if dry_run:
            return SubmitResult(
                job_name="dry-run",
                gcs_uri="",
                base_model=tc.base_model,
                status=f"dry-run: {stats['count']} examples, ~{stats['avg_tokens']} avg tokens",
            )

        gcs_uri = self.upload_to_gcs(jsonl_path)

        genai_client = self._get_genai_client()
        tuning_job = _run_with_retry(
            "Vertex AI tuning job creation",
            lambda: genai_client.tunings.tune(
                base_model=tc.base_model,
                training_dataset=gcs_uri,
                config={
                    "epoch_count": tc.epochs,
                    "learning_rate_multiplier": tc.learning_rate_multiplier,
                },
            ),
        )

        job_name = getattr(tuning_job, "name", str(tuning_job))
        status = getattr(tuning_job, "state", "SUBMITTED")

        logger.info("Tuning job created: %s (status: %s)", job_name, status)

        return SubmitResult(
            job_name=job_name,
            gcs_uri=gcs_uri,
            base_model=tc.base_model,
            status=str(status),
        )

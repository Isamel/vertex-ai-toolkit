"""Hypothesis prompt library for autonomous investigation (SPEC-X-03).

Provides a catalogue of ``HypothesisTemplate`` objects that map symptom
patterns (keywords or regexes) to investigation strategies.  The library
ships with ≥10 built-in templates covering common Kubernetes failure modes
and can be extended with user-defined templates loaded from YAML.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from vaig.core.config import HypothesisConfig

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────


class HypothesisTemplate(BaseModel):
    """A single hypothesis template mapping a symptom pattern to an investigation strategy."""

    id: str
    """Unique identifier for this template (e.g. ``oom``, ``high_latency``)."""
    symptom_pattern: str
    """Keyword substring or regex pattern matched against symptom strings (case-insensitive)."""
    hypothesis_text: str
    """Human-readable hypothesis explaining the likely root cause."""
    investigation_strategy: str
    """Suggested investigation strategy / tool to confirm or disprove the hypothesis."""
    confidence_modifier: float = 0.0
    """Modifier applied to finding confidence when this template is matched (-1.0 … +1.0)."""


# ── Built-in templates ────────────────────────────────────────────────────────

_BUILTIN_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "oom",
        "symptom_pattern": r"(?i)(oomkill|out.?of.?memory|oom)",
        "hypothesis_text": "Container was killed by the OOM killer due to exceeding its memory limit.",
        "investigation_strategy": "kubectl_describe",
        "confidence_modifier": 0.2,
    },
    {
        "id": "high_latency",
        "symptom_pattern": r"(?i)(high.?latency|slow.?response|p99|p95|latency.?spike)",
        "hypothesis_text": "Service is experiencing elevated response latency, possibly due to resource contention or upstream dependency issues.",
        "investigation_strategy": "kubectl_top",
        "confidence_modifier": 0.1,
    },
    {
        "id": "pod_restarts",
        "symptom_pattern": r"(?i)(pod.*restart|crashloopbackoff|restart.?count|back.?off)",
        "hypothesis_text": "Pod is restarting repeatedly, indicating a crash loop or liveness probe failure.",
        "investigation_strategy": "kubectl_logs",
        "confidence_modifier": 0.2,
    },
    {
        "id": "5xx_errors",
        "symptom_pattern": r"(?i)(5xx|500|502|503|504|server.?error|internal.?error)",
        "hypothesis_text": "Service is returning 5xx HTTP errors, suggesting an application crash or overload.",
        "investigation_strategy": "kubectl_logs",
        "confidence_modifier": 0.15,
    },
    {
        "id": "cert_expiry",
        "symptom_pattern": r"(?i)(cert.*expir|tls.*expir|ssl.*expir|certificate.*expir|x509)",
        "hypothesis_text": "TLS certificate has expired or is about to expire, causing handshake failures.",
        "investigation_strategy": "kubectl_describe",
        "confidence_modifier": 0.25,
    },
    {
        "id": "dns_failure",
        "symptom_pattern": r"(?i)(dns.*fail|name.*resolution|nxdomain|could.?not.?resolve|dns.*error)",
        "hypothesis_text": "DNS resolution is failing, possibly due to CoreDNS issues or missing service records.",
        "investigation_strategy": "kubectl_exec",
        "confidence_modifier": 0.2,
    },
    {
        "id": "resource_quota",
        "symptom_pattern": r"(?i)(resource.?quota|quota.?exceed|forbidden.*quota|quota.*forbidden)",
        "hypothesis_text": "Namespace resource quota has been exceeded, preventing new pods or resources from being created.",
        "investigation_strategy": "kubectl_describe",
        "confidence_modifier": 0.3,
    },
    {
        "id": "node_pressure",
        "symptom_pattern": r"(?i)(node.?pressure|disk.?pressure|memory.?pressure|pid.?pressure|not.?schedulable)",
        "hypothesis_text": "Node is under resource pressure (disk, memory, or PID), causing evictions or scheduling failures.",
        "investigation_strategy": "kubectl_describe",
        "confidence_modifier": 0.2,
    },
    {
        "id": "deployment_rollback",
        "symptom_pattern": r"(?i)(rollback|roll.?back|deployment.*fail|failed.?deploy|image.*pull|imagepull)",
        "hypothesis_text": "Recent deployment introduced a regression; image pull failure or misconfiguration is causing unavailability.",
        "investigation_strategy": "kubectl_rollout",
        "confidence_modifier": 0.15,
    },
    {
        "id": "network_policy",
        "symptom_pattern": r"(?i)(network.?policy|netpol|connection.?refused|traffic.?block|egress.*block|ingress.*block)",
        "hypothesis_text": "A NetworkPolicy rule is blocking traffic between services or to external endpoints.",
        "investigation_strategy": "kubectl_get",
        "confidence_modifier": 0.1,
    },
]


# ── Library ───────────────────────────────────────────────────────────────────


class HypothesisLibrary:
    """A catalogue of hypothesis templates for autonomous investigation.

    On construction the library loads the built-in templates and optionally
    merges user-defined templates from a YAML file.  User templates with the
    same ``id`` as a built-in override the built-in.

    Usage::

        library = HypothesisLibrary.default()                # built-ins only
        library = HypothesisLibrary(config)                  # with user YAML
        matches = library.match(["OOMKilled", "pod restart"])
    """

    def __init__(self, config: HypothesisConfig) -> None:
        self._templates: dict[str, HypothesisTemplate] = {}
        self._load_builtins()
        if config.custom_templates_path is not None:
            self._load_yaml(config.custom_templates_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def match(self, symptoms: list[str]) -> list[HypothesisTemplate]:
        """Return templates whose ``symptom_pattern`` matches ANY symptom string.

        Matching is case-insensitive substring OR regex match.  A single template
        is returned at most once even if it matches multiple symptoms.

        Args:
            symptoms: List of symptom strings extracted from a HealthReport.

        Returns:
            Ordered list of matching ``HypothesisTemplate`` objects (insertion
            order — built-ins first, user overrides in declaration order).
        """
        matched: list[HypothesisTemplate] = []
        seen: set[str] = set()
        for tmpl in self._templates.values():
            if tmpl.id in seen:
                continue
            for symptom in symptoms:
                if self._matches_pattern(tmpl.symptom_pattern, symptom):
                    matched.append(tmpl)
                    seen.add(tmpl.id)
                    break
        return matched

    def register(self, template: HypothesisTemplate) -> None:
        """Add or override a template.

        If a template with the same ``id`` already exists (built-in or
        previously registered), it is replaced.

        Args:
            template: The ``HypothesisTemplate`` to register.
        """
        self._templates[template.id] = template

    @classmethod
    def default(cls) -> HypothesisLibrary:
        """Construct a library with built-in templates only (no config required).

        Returns:
            A ``HypothesisLibrary`` containing the 10 built-in templates.
        """
        return cls(HypothesisConfig())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_builtins(self) -> None:
        for raw in _BUILTIN_TEMPLATES:
            tmpl = HypothesisTemplate.model_validate(raw)
            self._templates[tmpl.id] = tmpl

    def _load_yaml(self, path: Path) -> None:
        """Lazy, fail-safe YAML loader.  Missing or malformed files are logged
        as warnings and the library falls back to built-ins only."""
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(
                "HypothesisLibrary: custom_templates_path %s not found — using built-ins only.",
                path,
            )
            return
        except OSError as exc:
            logger.warning(
                "HypothesisLibrary: could not read %s (%s) — using built-ins only.",
                path,
                exc,
            )
            return

        try:
            data: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            logger.warning(
                "HypothesisLibrary: YAML parse error in %s (%s) — using built-ins only.",
                path,
                exc,
            )
            return

        if not isinstance(data, list):
            logger.warning(
                "HypothesisLibrary: %s must contain a YAML list of templates — using built-ins only.",
                path,
            )
            return

        loaded = 0
        for item in data:
            try:
                tmpl = HypothesisTemplate.model_validate(item)
                self._templates[tmpl.id] = tmpl  # override built-in if same id
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HypothesisLibrary: skipping malformed template entry (%s): %s",
                    item,
                    exc,
                )
        logger.debug("HypothesisLibrary: loaded %d user templates from %s", loaded, path)

    @staticmethod
    def _matches_pattern(pattern: str, symptom: str) -> bool:
        """Return True if ``pattern`` matches ``symptom`` via regex or case-insensitive substring."""
        # Try regex first
        try:
            if re.search(pattern, symptom, re.IGNORECASE):
                return True
        except re.error:
            pass
        # Fall back to case-insensitive substring
        return pattern.lower() in symptom.lower()

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._templates)

    def __repr__(self) -> str:
        return f"HypothesisLibrary(templates={len(self._templates)})"

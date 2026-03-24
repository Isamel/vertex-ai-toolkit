"""Tool category constants for dynamic tool selection.

These constants are used to tag ``ToolDef`` instances with their domain
and to filter the ``ToolRegistry`` per-agent via
``ToolRegistry.filter_by_categories()``.
"""

from __future__ import annotations

#: Kubernetes core operations (kubectl get/describe/logs/top/scale/etc.)
KUBERNETES: str = "kubernetes"

#: Helm release management tools.
HELM: str = "helm"

#: ArgoCD GitOps tools.
ARGOCD: str = "argocd"

#: Service mesh tools (Istio/ASM — overview, config, security, sidecars).
MESH: str = "mesh"

#: Logging and observability tools (Cloud Logging queries, log retrieval).
LOGGING: str = "logging"

#: Scaling and autoscaling tools (HPA, VPA, scaling status).
SCALING: str = "scaling"

#: Shell command execution tools.
SHELL: str = "shell"

#: File system tools (read, write, edit, list, search files).
CODING: str = "coding"

#: Datadog APM and monitoring tools.
DATADOG: str = "datadog"

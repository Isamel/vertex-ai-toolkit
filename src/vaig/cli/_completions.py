"""Shell autocompletion callbacks for CLI options.

All functions here must be FAST and fail silently — they run on every
<TAB> press and must never crash the CLI or produce visible output.
"""

from __future__ import annotations


def complete_namespace(incomplete: str) -> list[str]:
    """List available Kubernetes namespaces for shell completion.

    Uses the kubernetes client to query the cluster.  Falls back to an
    empty list when:
    - ``kubernetes`` package is not installed
    - No valid kubeconfig / in-cluster config exists
    - The API server is unreachable or times out (3 s hard limit)

    Args:
        incomplete: The partial text the user has typed so far.

    Returns:
        Namespace names matching the *incomplete* prefix, or ``[]``.
    """
    try:
        from kubernetes import client, config  # lazy import
        from kubernetes.config.config_exception import ConfigException

        # Try kubeconfig first, fall back to in-cluster
        try:
            config.load_kube_config()
        except ConfigException:
            try:
                config.load_incluster_config()
            except ConfigException:
                return []

        v1 = client.CoreV1Api()
        namespaces = v1.list_namespace(
            _request_timeout=3,  # hard 3-second timeout
        )
        names: list[str] = [
            ns.metadata.name
            for ns in namespaces.items
            if ns.metadata and ns.metadata.name
        ]

        if incomplete:
            names = [n for n in names if n.startswith(incomplete)]

        return sorted(names)
    except Exception:  # noqa: BLE001  — never crash during completion
        return []

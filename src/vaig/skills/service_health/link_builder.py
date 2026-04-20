"""Pure functions for generating external deep-links from investigation context.

Each builder function accepts a :data:`LinkContext` mapping — a plain
``dict[str, str]`` containing keys discovered during the investigation
(e.g. ``project_id``, ``cluster``, ``namespace``, ``service``,
``datadog_org``, ``argocd_server``, ``argocd_app``).

Links are ONLY generated when ALL required context keys are present.
Missing keys cause the individual link to be silently omitted — no
exception is ever raised.  This guarantees that callers always receive
a valid (possibly empty) list rather than a partially-constructed URL.

Usage::

    from vaig.skills.service_health.link_builder import build_external_links

    ctx = {"project_id": "my-gcp-project", "cluster": "prod-cluster",
           "namespace": "default", "service": "payment-svc",
           "datadog_org": "myorg", "argocd_server": "argocd.example.com",
           "argocd_app": "payment-svc"}

    links = build_external_links(ctx)
    # links.gcp → [ExternalLink(...), ...]
    # links.datadog → [ExternalLink(...), ...]
    # links.argocd → [ExternalLink(...)]
"""

from __future__ import annotations

from urllib.parse import quote

from vaig.skills.service_health.schema import ExternalLink, ExternalLinks

# Type alias — callers pass any str→str mapping
LinkContext = dict[str, str]

# ── SVG icon strings ──────────────────────────────────────────

_GCP_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">'
    '<path fill="#4285F4" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>'
    '<path fill="#fff" d="M7 12h10M12 7v10" stroke="#fff" stroke-width="2"/>'
    '</svg>'
)

_DATADOG_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">'
    '<rect width="24" height="24" rx="4" fill="#632CA6"/>'
    '<path fill="#fff" d="M5 8h14v2H5zm0 4h10v2H5zm0 4h7v2H5z"/>'
    '</svg>'
)

_ARGOCD_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16">'
    '<circle cx="12" cy="12" r="10" fill="#EF7B4D"/>'
    '<path fill="#fff" d="M12 6l-6 10h12z"/>'
    '</svg>'
)


# ── GCP link builders ─────────────────────────────────────────


def build_gcp_links(ctx: LinkContext) -> list[ExternalLink]:
    """Return GCP Console links for the given context.

    Generates up to three links:

    * **GCP Logs Explorer** — requires ``project_id``
    * **GCP Monitoring** — requires ``project_id``
    * **GCP GKE Workloads** — requires ``project_id``, ``cluster``, ``namespace``

    Any link whose required keys are absent is silently omitted.
    """
    links: list[ExternalLink] = []
    project_id = ctx.get("project_id", "")

    if project_id:
        # Logs Explorer
        query = quote(f'resource.labels.project_id="{project_id}"', safe="")
        links.append(ExternalLink(
            label="GCP Logs Explorer",
            url=(
                f"https://console.cloud.google.com/logs/query"
                f";query={query}"
                f";project={project_id}"
            ),
            system="gcp",
            icon=_GCP_ICON,
        ))

        # Cloud Monitoring
        links.append(ExternalLink(
            label="GCP Monitoring",
            url=f"https://console.cloud.google.com/monitoring?project={project_id}",
            system="gcp",
            icon=_GCP_ICON,
        ))

        # GKE Workloads — needs cluster + namespace
        cluster = ctx.get("cluster", "")
        namespace = ctx.get("namespace", "")
        if cluster and namespace:
            ns_enc = quote(namespace, safe="")
            cluster_enc = quote(cluster, safe="")
            links.append(ExternalLink(
                label="GKE Workloads",
                url=(
                    f"https://console.cloud.google.com/kubernetes/workload/overview"
                    f"?project={project_id}"
                    f"&pageState=(%22savedViews%22:(%22i%22:%22%22,"
                    f"%22c%22:[%22gke%2F{cluster_enc}%22],"
                    f"%22n%22:[%22{ns_enc}%22]))"
                ),
                system="gcp",
                icon=_GCP_ICON,
            ))

    return links


# ── Datadog link builders ─────────────────────────────────────


def build_datadog_links(ctx: LinkContext) -> list[ExternalLink]:
    """Return Datadog links for the given context.

    Generates up to two links:

    * **Datadog APM** — requires ``datadog_org``, ``service``
    * **Datadog Dashboards** — requires ``datadog_org``

    Any link whose required keys are absent is silently omitted.
    The ``datadog_org`` value is used as the subdomain when it does not
    contain a period (i.e. it is treated as an org slug, not a FQDN).
    """
    links: list[ExternalLink] = []
    datadog_org = ctx.get("datadog_org", "")

    if not datadog_org:
        return links

    # Determine base host
    if "." in datadog_org:
        # Caller passed a FQDN / full host
        base = f"https://{datadog_org}"
    else:
        base = "https://app.datadoghq.com"

    # APM Service — needs service name
    service = ctx.get("service", "")
    if service:
        service_enc = quote(service, safe="")
        links.append(ExternalLink(
            label=f"Datadog APM — {service}",
            url=f"{base}/apm/services/{service_enc}?env=prod",
            system="datadog",
            icon=_DATADOG_ICON,
        ))

    # Dashboard list — only needs org
    links.append(ExternalLink(
        label="Datadog Dashboards",
        url=f"{base}/dashboard/lists",
        system="datadog",
        icon=_DATADOG_ICON,
    ))

    return links


# ── ArgoCD link builders ──────────────────────────────────────


def build_argocd_links(ctx: LinkContext) -> list[ExternalLink]:
    """Return ArgoCD links for the given context.

    Generates one link:

    * **ArgoCD Application** — requires ``argocd_server``, ``argocd_app``

    The link is omitted when either key is absent.
    """
    links: list[ExternalLink] = []
    argocd_server = ctx.get("argocd_server", "")
    argocd_app = ctx.get("argocd_app", "")

    if argocd_server and argocd_app:
        app_enc = quote(argocd_app, safe="")
        # Normalise server: strip trailing slash; add https if no scheme present
        server = argocd_server.rstrip("/")
        if not server.startswith(("http://", "https://")):
            server = f"https://{server}"
        links.append(ExternalLink(
            label=f"ArgoCD — {argocd_app}",
            url=f"{server}/applications/{app_enc}",
            system="argocd",
            icon=_ARGOCD_ICON,
        ))

    return links


# ── Aggregate builder ─────────────────────────────────────────


def build_external_links(ctx: LinkContext) -> ExternalLinks:
    """Build an :class:`ExternalLinks` object from an investigation context dict.

    Delegates to the three per-system builder functions and assembles the
    results into a single model.  Always returns a valid
    :class:`ExternalLinks` instance (never raises).

    Args:
        ctx: Investigation context mapping — typically contains a subset of:
            ``project_id``, ``cluster``, ``namespace``, ``service``,
            ``datadog_org``, ``argocd_server``, ``argocd_app``.

    Returns:
        :class:`ExternalLinks` with lists populated for each system that
        had sufficient context.  Empty lists for systems with missing keys.
    """
    return ExternalLinks(
        gcp=build_gcp_links(ctx),
        datadog=build_datadog_links(ctx),
        argocd=build_argocd_links(ctx),
    )

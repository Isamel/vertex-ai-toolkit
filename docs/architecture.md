# Architecture

This document describes the system architecture of VAIG using Mermaid diagrams.

## High-Level Architecture

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        chat["vaig chat<br/>(Interactive REPL)"]
        ask["vaig ask<br/>(Single-shot)"]
        live["vaig live<br/>(Infrastructure)"]
        doctor["vaig doctor"]
        stats["vaig stats"]
    end

    subgraph Agents["Agent Layer"]
        orch["Orchestrator"]
        spec["SpecialistAgent<br/>(text-only)"]
        tool["ToolAwareAgent<br/>(tool-use loop)"]
        code["CodingAgent<br/>(file I/O + shell)"]
        infra["InfraAgent<br/>(GKE + GCloud)"]
        chunk["ChunkedProcessor<br/>(Map-Reduce)"]
    end

    subgraph Skills["Skills Layer (29 built-in)"]
        rca["RCA"]
        sh["Service Health<br/>(4-agent pipeline)"]
        anomaly["Anomaly Detection"]
        migration["Code Migration"]
        others["... 25 more"]
    end

    subgraph Tools["Tools Layer"]
        file_tools["File Tools<br/>read, write, edit,<br/>list, search"]
        shell_tools["Shell Tools<br/>run_command"]
        gke_tools["GKE Tools<br/>kubectl, diagnostics,<br/>discovery, mesh,<br/>mutations, security"]
        gcloud["GCloud Tools<br/>Cloud Logging,<br/>Cloud Monitoring"]
        helm_tools["Helm Tools<br/>releases, status,<br/>history, values"]
        argocd_tools["ArgoCD Tools<br/>applications, sync,<br/>diff, history"]
        labels_tool["Labels Tool<br/>kubectl_get_labels"]
        mcp["MCP Bridge<br/>(external tools)"]
        plugins["Plugin Loader<br/>(Python modules)"]
    end

    subgraph Core["Core Layer"]
        client["GeminiClient<br/>(sync + async)"]
        config["Settings<br/>(Pydantic)"]
        auth["Auth<br/>(ADC, SA impersonation,<br/>gcloud token refresh)"]
        session["SessionManager<br/>(SQLite)"]
        cost["CostTracker<br/>(per-session)"]
        telemetry["TelemetryCollector<br/>(SQLite, buffered)"]
        cache["ResponseCache<br/>(LRU + TTL)"]
        lang["Language Detection<br/>(9 languages)"]
    end

    subgraph External["External Services"]
        vertex["Google Vertex AI<br/>(Gemini models)"]
        k8s["Kubernetes API<br/>(GKE clusters)"]
        logging["Cloud Logging"]
        monitoring["Cloud Monitoring"]
        argocd_api["ArgoCD API<br/>(optional)"]
        mcp_servers["MCP Servers<br/>(optional)"]
    end

    chat --> orch
    ask --> orch
    ask --> code
    ask --> infra
    live --> infra
    live --> orch

    orch --> spec
    orch --> tool
    orch --> chunk

    spec --> client
    tool --> client
    code --> client
    infra --> client

    tool --> gke_tools
    tool --> gcloud
    tool --> helm_tools
    tool --> argocd_tools
    tool --> labels_tool
    code --> file_tools
    code --> shell_tools
    infra --> gke_tools
    infra --> gcloud

    client --> vertex
    gke_tools --> k8s
    gcloud --> logging
    gcloud --> monitoring
    argocd_tools --> argocd_api
    argocd_tools --> k8s
    helm_tools --> k8s
    mcp --> mcp_servers

    client --> cache
    client --> auth
    config --> auth
    session --> cost
    orch --> telemetry
```

## Service Health Pipeline (4-Agent Sequential)

```mermaid
sequenceDiagram
    participant User
    participant CLI as vaig live
    participant Orch as Orchestrator
    participant G as health_gatherer<br/>(ToolAwareAgent)
    participant A as health_analyzer<br/>(SpecialistAgent)
    participant V as health_verifier<br/>(ToolAwareAgent)
    participant R as health_reporter<br/>(SpecialistAgent)
    participant K8s as Kubernetes API
    participant CL as Cloud Logging
    participant Gemini as Vertex AI

    User->>CLI: "Why are pods crashing?"
    CLI->>Orch: execute_with_tools(query, skill, registry)

    Note over Orch: Language detection + Autopilot injection

    rect rgb(230, 245, 255)
        Note over G: Pass 1: Data Collection
        Orch->>G: execute(query)
        loop Tool-use loop (max 25 iterations)
            G->>Gemini: generate_with_tools()
            Gemini-->>G: function_call(kubectl_get, ...)
            G->>K8s: kubectl_get("pods")
            K8s-->>G: pod list
            G->>K8s: get_events(event_type="Warning")
            K8s-->>G: events
            G->>K8s: kubectl_describe("replicaset")
            K8s-->>G: RS details
            G->>CL: gcloud_logging_query(severity>=ERROR)
            CL-->>G: log entries
        end
        G-->>Orch: gathered data + Investigation Checklist
    end

    rect rgb(255, 245, 230)
        Note over G: Pass 2: Incremental Deepening
        Orch->>Orch: validate_gatherer_output()
        Orch->>G: deepening_prompt (NO reset, uses history)
        loop Tool-use loop (max_iterations_retry)
            G->>Gemini: generate_with_tools()
            G->>K8s: additional tools only
        end
        G-->>Orch: additional findings
        Note over Orch: Merge Pass 1 + Pass 2
    end

    rect rgb(230, 255, 230)
        Note over A: Analysis Phase
        Orch->>A: gathered data + tools_executed metadata
        A->>Gemini: analyze patterns, correlate events
        Note over A: Causal reasoning (5 Whys)<br/>Management context detection<br/>Verification Gap per finding
        A-->>Orch: structured findings with confidence levels
    end

    rect rgb(255, 230, 255)
        Note over V: Verification Phase
        Orch->>V: findings with Verification Gaps
        loop For each finding with Gap
            V->>K8s: targeted tool call from Gap
            Note over V: Upgrade/downgrade confidence
        end
        V-->>Orch: verified findings
    end

    rect rgb(255, 255, 230)
        Note over R: Report Generation (JSON Schema mode)
        Orch->>R: verified findings
        R->>Gemini: generate (response_schema=HealthReport, mime=application/json)
        Note over R: Gemini returns validated JSON<br/>post_process_report() converts<br/>to Markdown via to_markdown()
        R-->>Orch: final report
    end

    Note over Orch: Validate reporter output
    Orch-->>CLI: OrchestratorResult
    CLI-->>User: Formatted report + cost summary
```

## Tool Layer — GKE Package Structure

```mermaid
graph LR
    subgraph "src/vaig/tools/gke/"
        init["__init__.py<br/>(re-exports)"]
        registry["_registry.py<br/>create_gke_tools()"]
        clients["_clients.py<br/>K8s client cache,<br/>Autopilot detection,<br/>ArgoCD client"]
        resources["_resources.py<br/>Resource maps,<br/>aliases, gap detection"]
        formatters["_formatters.py<br/>Table formatters"]
        cache["_cache.py<br/>TTL cache"]

        kubectl["kubectl.py<br/>get, describe,<br/>logs, top,<br/>get_labels"]
        diag["diagnostics.py<br/>events, rollout,<br/>container status,<br/>node conditions"]
        disc["discovery.py<br/>workloads, mesh,<br/>network topology"]
        mesh["mesh.py<br/>Istio/ASM config,<br/>security, sidecars"]
        mut["mutations.py<br/>scale, restart,<br/>annotate, label"]
        sec["security.py<br/>RBAC check,<br/>exec_command"]
        helm["helm.py<br/>releases, status,<br/>history, values"]
        argocd["argocd.py<br/>applications,<br/>diff, sync history"]
    end

    registry --> kubectl
    registry --> diag
    registry --> disc
    registry --> mesh
    registry --> mut
    registry --> sec
    registry --> helm
    registry --> argocd

    kubectl --> clients
    kubectl --> resources
    kubectl --> formatters
    diag --> clients
    disc --> clients
    mesh --> clients
    helm --> clients
    argocd --> clients

    style helm fill:#ffd,stroke:#aa0
    style argocd fill:#ffd,stroke:#aa0
    style kubectl fill:#dfd,stroke:#0a0
```

## ArgoCD Connection Topologies

```mermaid
graph TB
    subgraph "Topology A: Same Cluster"
        vaig_a["VAIG"] --> cluster_a["Cluster A<br/>(workloads + ArgoCD)"]
    end

    subgraph "Topology B: Management Cluster (same project)"
        vaig_b["VAIG"]
        vaig_b -->|"GKE tools<br/>(kubeconfig default)"| work_b["Cluster A<br/>(workloads)"]
        vaig_b -->|"ArgoCD tools<br/>(argocd.context)"| mgmt_b["Cluster M<br/>(ArgoCD)"]
    end

    subgraph "Topology C: Different Project"
        vaig_c["VAIG"]
        vaig_c -->|"GKE tools"| work_c["Cluster A<br/>(Project Y)"]
        vaig_c -->|"ArgoCD tools<br/>(argocd.context)"| mgmt_c["Cluster M<br/>(Project X)"]
    end

    subgraph "Topology D: API Server (SaaS / any topology)"
        vaig_d["VAIG"]
        vaig_d -->|"GKE tools"| work_d["Any Cluster"]
        vaig_d -->|"ArgoCD REST API<br/>(argocd.server + token)"| api_d["ArgoCD Server<br/>(any location)"]
        api_d -->|manages| work_d
    end
```

## Configuration Layering

```mermaid
graph LR
    env["Environment Variables<br/>VAIG_GCP__PROJECT_ID=..."] -->|highest priority| merged["Merged Settings<br/>(Pydantic)"]
    yaml["YAML Config<br/>config/default.yaml"] -->|medium priority| merged
    defaults["Code Defaults<br/>Pydantic Field defaults"] -->|lowest priority| merged
    merged --> client["GeminiClient"]
    merged --> tools["Tool Registry"]
    merged --> session["SessionManager"]
    merged --> telemetry["TelemetryCollector"]
```

## Cost Tracking Flow

```mermaid
sequenceDiagram
    participant User
    participant REPL
    participant CT as CostTracker
    participant SM as SessionManager
    participant DB as SQLite

    User->>REPL: send message
    REPL->>REPL: _check_budget()
    CT-->>REPL: OK / WARNING / EXCEEDED

    Note over REPL: API call happens...

    REPL->>CT: record(model, tokens_in, tokens_out, thinking)
    CT->>CT: calculate_cost() → accumulate

    User->>REPL: /cost
    REPL->>CT: summary()
    CT-->>REPL: total cost, per-model breakdown

    User->>REPL: /quit
    REPL->>SM: save_cost_data(tracker.to_dict())
    SM->>DB: UPDATE sessions SET metadata = ...

    Note over DB: Cost data persisted per session

    User->>REPL: vaig chat --resume
    REPL->>SM: load_cost_data(session_id)
    SM->>DB: SELECT metadata FROM sessions
    DB-->>SM: cost_data JSON
    SM-->>REPL: restored CostTracker
```

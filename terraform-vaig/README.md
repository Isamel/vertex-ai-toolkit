# terraform-vaig — Terraform Health Gate Module

Gate Terraform deployments on GKE cluster health using `vaig check`.

This module wraps `vaig check` as a Terraform [external data source](https://registry.terraform.io/providers/hashicorp/external/latest/docs/data-sources/external) and uses Terraform 1.5+ [`check` blocks](https://developer.hashicorp.com/terraform/language/checks) to produce a plan-time warning when the cluster is unhealthy.

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Terraform   | 1.5+            | `check` block support |
| Python      | 3.12+           | Required by `vaig` |
| `vaig`      | latest          | `pip install vaig` |
| `jq`        | any             | Parses Terraform stdin JSON |
| `gcloud`    | any             | Authenticated with target project |

## Quick Start

1. **Copy this directory** into your Terraform project:

   ```bash
   cp -r terraform-vaig/ /path/to/your/terraform/project/modules/health-gate/
   ```

2. **Reference as a module** in your root `main.tf`:

   ```hcl
   module "health_gate" {
     source    = "./modules/health-gate"
     namespace = "production"
     project   = "my-gcp-project"
   }
   ```

3. **Run `terraform plan`** — the health check executes during planning:

   ```bash
   terraform plan
   ```

   If the cluster is unhealthy, you'll see:

   ```
   │ Warning: Check block assertion failed
   │
   │   on modules/health-gate/main.tf line 40, in check "cluster_health":
   │   40:     condition     = local.health_status == "HEALTHY"
   │
   │ Cluster health check failed: CRITICAL — 3 critical issues found
   │ (critical: 3, warnings: 2)
   ```

## Variables

| Variable    | Type   | Default | Description |
|-------------|--------|---------|-------------|
| `namespace` | string | `""`    | Kubernetes namespace to check (empty = cluster-wide) |
| `cluster`   | string | `""`    | GKE cluster name (empty = gcloud default) |
| `project`   | string | `""`    | GCP project ID (empty = gcloud default) |
| `timeout`   | number | `120`   | Health check timeout in seconds |

## Outputs

| Output          | Type   | Description |
|-----------------|--------|-------------|
| `health_status` | string | `HEALTHY`, `DEGRADED`, `CRITICAL`, `UNKNOWN`, `ERROR`, or `TIMEOUT` |
| `critical_count`| number | Number of critical findings |
| `warning_count` | number | Number of warning-level findings |
| `summary`       | string | Human-readable summary of the health check |
| `is_cached`     | bool   | Whether the result was served from cache |

## Caching Strategy

The wrapper passes `--cached` to `vaig check` by default, which uses a file-based cache in `~/.cache/vaig/check/`:

- **Cache key**: SHA-256 hash of `(namespace, cluster, project)`
- **TTL**: 300 seconds (5 minutes) by default
- **Behavior**: If a cached result exists and is fresh, it's returned instantly (< 1s). Otherwise, a full health check runs (~30-120s).

This means repeated `terraform plan` calls within the TTL window are fast and don't hit the GKE API.

## How It Works

```
terraform plan
    │
    ▼
vaig-check.sh (reads query JSON from Terraform stdin)
    │ extracts namespace, cluster, project, timeout
    ▼
vaig check --namespace X --cached --timeout 120
    │
    ├─► AI-powered health analysis via Vertex AI
    │
    ▼
JSON result → Terraform external data source
    │
    ▼
check { assert = status == "HEALTHY" }
```

## Exit Codes

| Code | Meaning | Terraform Behavior |
|------|---------|-------------------|
| 0    | Healthy | Data source succeeds |
| 1    | Unhealthy (DEGRADED/CRITICAL) | Data source fails → plan error |
| 2    | Error or timeout | Data source fails → plan error |

The wrapper always outputs valid JSON to stdout, even on errors, to provide meaningful diagnostics in the Terraform output.

## Standalone Usage

You can also use the wrapper outside of Terraform:

```bash
echo '{"namespace": "production"}' | bash vaig-check.sh
```

Or call `vaig check` directly:

```bash
vaig check --namespace production --timeout 60
```

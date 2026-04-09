# GitHub Action: VAIG Health Check

Run GKE service health discovery powered by Vertex AI Gemini on every pull
request.  The action posts a collapsible health report as a PR comment and
can gate merges based on finding severity.

## Quick Start

```yaml
- uses: ./.github/actions/health-check
  with:
    cluster: my-cluster
    project-id: my-gcp-project
    location: us-central1
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `cluster` | **Yes** | — | GKE cluster name |
| `project-id` | **Yes** | — | GCP project ID |
| `location` | **Yes** | — | GKE cluster location (e.g. `us-central1`) |
| `namespace` | No | `default` | Kubernetes namespace to scan |
| `fail-on` | No | `CRITICAL` | Minimum severity to fail: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO` |
| `model` | No | `gemini-2.5-flash` | Gemini model to use |
| `comment` | No | `true` | Post health report as PR comment |
| `timeout` | No | `300` | Pipeline timeout in seconds |

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `status` | string | `pass`, `fail`, or `error` |
| `findings-count` | int | Total number of findings |
| `max-severity` | string | Highest severity: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`, or `NONE` |
| `report` | string | Full health report in markdown |

- `pass` — no findings at or above the `fail-on` threshold
- `fail` — at least one finding meets or exceeds the threshold
- `error` — an exception occurred (timeout, auth failure, etc.)

Use outputs in subsequent steps:

```yaml
- name: Check results
  if: always()
  run: echo "Status is ${{ steps.health.outputs.status }}"
```

## Authentication Setup

The action uses Application Default Credentials (ADC).  You must configure
authentication **before** the action runs using
[`google-github-actions/auth@v2`](https://github.com/google-github-actions/auth).

### Option 1: Workload Identity Federation (Recommended)

WIF is the most secure approach — no long-lived keys.

**Step 1 — Create a Workload Identity Pool:**

```bash
gcloud iam workload-identity-pools create "github-pool" \
  --project="$PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions Pool"
```

**Step 2 — Create a Provider:**

```bash
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="$PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

**Step 3 — Create a Service Account and bind it:**

```bash
gcloud iam service-accounts create vaig-gha \
  --project="$PROJECT_ID" \
  --display-name="VAIG GitHub Actions"

# Grant required roles
for ROLE in container.viewer logging.viewer monitoring.viewer aiplatform.user; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:vaig-gha@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/${ROLE}"
done

# Allow WIF to impersonate the SA
gcloud iam service-accounts add-iam-policy-binding \
  "vaig-gha@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"
```

**Step 4 — Use in your workflow:**

```yaml
- uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: "projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
    service_account: "vaig-gha@PROJECT_ID.iam.gserviceaccount.com"
```

### Option 2: Service Account Key

Simpler but less secure (uses a long-lived JSON key).

1. Create a service account key:
   ```bash
   gcloud iam service-accounts keys create key.json \
     --iam-account="vaig-gha@${PROJECT_ID}.iam.gserviceaccount.com"
   ```

2. Add the key as a GitHub secret named `GCP_SA_KEY`.

3. Use in your workflow:
   ```yaml
   - uses: google-github-actions/auth@v2
     with:
       credentials_json: ${{ secrets.GCP_SA_KEY }}
   ```

## Severity Gating

The `fail-on` input controls the minimum severity that causes the action to
fail (exit code 1).  Findings below the threshold are reported but don't
block the PR.

| `fail-on` | Fails on | Passes on |
|-----------|----------|-----------|
| `CRITICAL` | CRITICAL | HIGH, MEDIUM, LOW, INFO |
| `HIGH` | CRITICAL, HIGH | MEDIUM, LOW, INFO |
| `MEDIUM` | CRITICAL, HIGH, MEDIUM | LOW, INFO |
| `LOW` | CRITICAL, HIGH, MEDIUM, LOW | INFO |
| `INFO` | CRITICAL, HIGH, MEDIUM, LOW, INFO | (nothing passes) |

When no findings are present, the action always passes regardless of threshold.

## Cost Implications

Each action run invokes a full Vertex AI Gemini pipeline (4 sequential
agents).  Typical cost per run:

- **gemini-2.5-flash**: ~$0.005–$0.02
- **gemini-2.5-pro**: ~$0.05–$0.15

Consider using `workflow_dispatch` for manual runs during initial setup,
then switching to `pull_request` once costs are understood.

## Fork PR Limitations

When a workflow runs on a pull request from a **fork**, the `GITHUB_TOKEN`
has read-only permissions by default.  In this case:

- The action **cannot** post PR comments
- The health report is printed to **stdout** as a fallback
- A warning is logged: "Could not post PR comment — insufficient permissions"
- The exit code still follows severity gating normally

To allow comments on fork PRs, configure the workflow with
`pull_request_target` (use with caution — see
[GitHub docs](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#pull_request_target)).

## Idempotent Comments

The action includes a hidden HTML marker (`<!-- vaig-health-check -->`) in
every PR comment.  When the action runs again on the same PR, it finds
the existing comment and **updates** it instead of creating a duplicate.

## Full Example

See [`.github/workflows/health-check-example.yml`](../.github/workflows/health-check-example.yml)
for a complete workflow with WIF authentication and output usage.

# VAIG Web — Cloud Run Deployment Guide

Practical guide for deploying the VAIG web interface to Cloud Run with Identity-Aware Proxy (IAP) authentication.

## 1. Prerequisites

- **gcloud CLI** installed and authenticated (`gcloud auth login`)
- A **GCP project** with billing enabled
- **Firestore** enabled in the project (for session persistence)
- **Docker** installed locally (optional — Cloud Build can build remotely)

```bash
# Set your project
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  firestore.googleapis.com \
  artifactregistry.googleapis.com
```

## 2. Build & Push Docker Image

Use Cloud Build to build and push the container image directly:

```bash
# From the repository root
gcloud builds submit \
  --tag gcr.io/$PROJECT_ID/vaig-web:latest \
  --dockerfile Dockerfile.web \
  .
```

Alternatively, use Artifact Registry (recommended):

```bash
# Create an Artifact Registry repo (one-time)
gcloud artifacts repositories create vaig \
  --repository-format=docker \
  --location=$REGION

# Build and push
gcloud builds submit \
  --tag $REGION-docker.pkg.dev/$PROJECT_ID/vaig/vaig-web:latest \
  --dockerfile Dockerfile.web \
  .
```

## 3. Deploy to Cloud Run

```bash
# Using the service.yaml config
gcloud run services replace deploy/service.yaml \
  --region $REGION

# Or deploy directly with gcloud
gcloud run deploy vaig-web \
  --image gcr.io/$PROJECT_ID/vaig-web:latest \
  --region $REGION \
  --platform managed \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0 \
  --timeout 300 \
  --concurrency 80 \
  --execution-environment gen2 \
  --set-env-vars="PORT=8080" \
  --no-allow-unauthenticated
```

> **Important**: Use `--no-allow-unauthenticated` to enforce IAP auth.

## 4. Enable Identity-Aware Proxy (IAP)

IAP provides Google-account-based access control without writing auth code.

### 4.1 Enable the IAP API

```bash
gcloud services enable iap.googleapis.com
```

### 4.2 Configure OAuth Consent Screen

1. Go to [OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent) in the Google Cloud Console
2. Choose **Internal** (for organization users) or **External** (for any Google account)
3. Fill in:
   - App name: `VAIG Web`
   - User support email: your team email
   - Authorized domains: your domain
4. Save

### 4.3 Create OAuth Credentials

1. Go to [Credentials](https://console.cloud.google.com/apis/credentials)
2. Click **Create Credentials** > **OAuth client ID**
3. Application type: **Web application**
4. Name: `VAIG Web IAP`
5. Authorized redirect URIs: `https://iap.googleapis.com/v1/oauth/clientIds/CLIENT_ID:handleRedirect`
6. Save the **Client ID** and **Client Secret**

### 4.4 Enable IAP for Cloud Run

```bash
# Enable IAP on the Cloud Run service backend
gcloud iap web enable \
  --resource-type=cloud-run \
  --service=vaig-web
```

If prompted, provide the OAuth client ID and secret from step 4.3.

## 5. Configure Allowed Users

Grant IAP access to specific users or groups:

```bash
# Allow a specific user
gcloud iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=vaig-web \
  --member="user:engineer@example.com" \
  --role="roles/iap.httpsResourceAccessor"

# Allow a Google group
gcloud iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=vaig-web \
  --member="group:sre-team@example.com" \
  --role="roles/iap.httpsResourceAccessor"

# Allow an entire domain
gcloud iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=vaig-web \
  --member="domain:example.com" \
  --role="roles/iap.httpsResourceAccessor"
```

## 6. Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `8080` | HTTP port for the web server |
| `VAIG_WEB_DEV_MODE` | No | `false` | Set to `true` for local dev (bypasses IAP auth check) |
| `VAIG_WEB_DEV_USER` | No | `dev@localhost` | Dev-mode fallback user email |
| `GOOGLE_CLOUD_PROJECT` | Yes | — | GCP project ID for Vertex AI and Firestore |
| `VAIG_MODEL` | No | `gemini-2.0-flash` | Default Gemini model |
| `VAIG_REGION` | No | `us-central1` | Vertex AI region |

Set environment variables during deployment:

```bash
gcloud run deploy vaig-web \
  --image gcr.io/$PROJECT_ID/vaig-web:latest \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=$PROJECT_ID,VAIG_REGION=$REGION" \
  --no-allow-unauthenticated
```

## 7. Verify Deployment

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe vaig-web \
  --region $REGION \
  --format='value(status.url)')

echo "Service URL: $SERVICE_URL"

# Health check (bypasses IAP)
curl -s "$SERVICE_URL/health" | python -m json.tool

# Open in browser (IAP will prompt for Google login)
open "$SERVICE_URL"
```

## 8. Troubleshooting

### IAP returns 403

- Verify the user has `roles/iap.httpsResourceAccessor` on the service
- Check OAuth consent screen is configured
- Wait 2-5 minutes after policy changes for propagation

### Cloud Run returns 503

- Check container logs: `gcloud run services logs read vaig-web --region $REGION`
- Verify the Docker image starts correctly locally
- Check memory limits are sufficient

### SSE streaming disconnects

- Ensure `timeoutSeconds: 300` is set (default 60s is too short for long-running queries)
- Check Cloud Run concurrency settings match expected load

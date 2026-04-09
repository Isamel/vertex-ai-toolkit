#!/usr/bin/env bash
# vaig-check.sh — Terraform external data source wrapper for vaig check.
#
# Reads the Terraform query JSON from stdin, extracts parameters, invokes
# `vaig check`, and ALWAYS exits 0 with valid JSON to stdout.
#
# The Terraform `external` data source REQUIRES exit 0 for valid results.
# Health status is communicated via the JSON output — the HCL `check`
# block reads the status field to decide pass/fail.
#
# Usage in HCL:
#   data "external" "vaig_health" {
#     program = ["bash", "${path.module}/vaig-check.sh"]
#     query   = { namespace = var.namespace, ... }
#   }
set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────

error_json() {
  local msg="${1:-unknown error}"
  local escaped_msg
  escaped_msg=$(printf '%s' "$msg" | jq -Rs '.' 2>/dev/null || printf '"%s"' "$msg")
  # Always exit 0 — let Terraform check block decide on health status
  cat <<EOF
{"status":"ERROR","critical_count":"0","warning_count":"0","issues_found":"0","services_checked":"0","summary_text":${escaped_msg},"scope":"","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)","version":"unknown","cached":"false"}
EOF
  exit 0
}

# ── prerequisite checks (pre-stdin) ──────────────────────────────────────────

command -v jq    >/dev/null 2>&1 || { printf '{"status":"ERROR","critical_count":"0","warning_count":"0","issues_found":"0","services_checked":"0","summary_text":"jq not found — install jq to parse Terraform input","scope":"","timestamp":"","version":"unknown","cached":"false"}\n'; exit 0; }
command -v gcloud >/dev/null 2>&1 || error_json "gcloud not found — install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"

GCLOUD_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
[[ -z "$GCLOUD_ACCOUNT" ]] && error_json "no active gcloud account — run: gcloud auth login"

# ── read Terraform stdin JSON ────────────────────────────────────────────────

INPUT=$(cat)

if ! echo "$INPUT" | jq . >/dev/null 2>&1; then
  error_json "invalid or empty JSON on stdin"
fi

NAMESPACE=$(echo "$INPUT" | jq -r '.namespace // empty')
CLUSTER=$(echo "$INPUT"   | jq -r '.cluster   // empty')
PROJECT=$(echo "$INPUT"   | jq -r '.project   // empty')
TIMEOUT=$(echo "$INPUT"   | jq -r '.timeout   // "120"')
VAIG_BIN=$(echo "$INPUT"  | jq -r '.vaig_path // "vaig"')

# ── prerequisite check (post-stdin) ─────────────────────────────────────────

command -v "$VAIG_BIN" >/dev/null 2>&1 || error_json "${VAIG_BIN} not found — install with: pip install vaig"

# ── build vaig check arguments ──────────────────────────────────────────────

ARGS=()
[[ -n "$NAMESPACE" ]] && ARGS+=(--namespace "$NAMESPACE")
[[ -n "$CLUSTER"   ]] && ARGS+=(--cluster   "$CLUSTER")
[[ -n "$PROJECT"   ]] && ARGS+=(--project   "$PROJECT")
ARGS+=(--timeout "$TIMEOUT")
ARGS+=(--cached)

# ── invoke vaig check ───────────────────────────────────────────────────────

OUTPUT=$("$VAIG_BIN" check "${ARGS[@]}" 2>/dev/null) || {
  rc=$?
  # If vaig check wrote valid JSON to stdout, forward it (always exit 0)
  if echo "$OUTPUT" | jq . >/dev/null 2>&1; then
    echo "$OUTPUT"
    exit 0
  fi
  error_json "vaig check failed with exit code ${rc}"
}

# Validate output is JSON before forwarding to Terraform
if echo "$OUTPUT" | jq . >/dev/null 2>&1; then
  echo "$OUTPUT"
  exit 0
else
  error_json "vaig check produced invalid JSON"
fi

#!/usr/bin/env bash
# vaig-check.sh — Terraform external data source wrapper for vaig check.
#
# Reads the Terraform query JSON from stdin, extracts parameters, invokes
# `vaig check`, and always writes valid JSON to stdout — even on error.
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
  cat <<EOF
{"status":"ERROR","critical_count":"0","warning_count":"0","issues_found":"0","services_checked":"0","summary_text":"${msg}","scope":"","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)","version":"unknown","cached":"false"}
EOF
  exit 2
}

# ── prerequisite checks (pre-stdin) ──────────────────────────────────────────

command -v jq   >/dev/null 2>&1 || error_json "jq not found — install jq to parse Terraform input"

GCLOUD_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
[[ -z "$GCLOUD_ACCOUNT" ]] && error_json "no active gcloud account — run: gcloud auth login"

# ── read Terraform stdin JSON ────────────────────────────────────────────────

INPUT=$(cat)

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
  # If vaig check wrote valid JSON to stdout, forward it even on non-zero exit
  if echo "$OUTPUT" | jq . >/dev/null 2>&1; then
    echo "$OUTPUT"
    exit $rc
  fi
  error_json "vaig check failed with exit code ${rc}"
}

# Validate output is JSON before forwarding to Terraform
if echo "$OUTPUT" | jq . >/dev/null 2>&1; then
  echo "$OUTPUT"
else
  error_json "vaig check produced invalid JSON"
fi

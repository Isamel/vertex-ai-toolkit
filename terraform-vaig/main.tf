# main.tf — Terraform health gate using vaig check.
#
# This module invokes `vaig check` via an external data source during
# `terraform plan` and uses a `check` block to gate deployment on
# cluster health.
#
# Requires: Terraform >= 1.5 (for `check` block support).

terraform {
  required_version = ">= 1.5"
}

# ── external data source ────────────────────────────────────────────────────

data "external" "vaig_health" {
  program = ["bash", "${path.module}/vaig-check.sh"]

  query = {
    namespace = var.namespace
    cluster   = var.cluster
    project   = var.project
    timeout   = tostring(var.timeout)
  }
}

# ── locals — parse string values to native types ────────────────────────────

locals {
  health_status    = data.external.vaig_health.result.status
  critical_count   = tonumber(data.external.vaig_health.result.critical_count)
  warning_count    = tonumber(data.external.vaig_health.result.warning_count)
  issues_found     = tonumber(data.external.vaig_health.result.issues_found)
  services_checked = tonumber(data.external.vaig_health.result.services_checked)
  summary_text     = data.external.vaig_health.result.summary_text
  is_cached        = data.external.vaig_health.result.cached == "true"
}

# ── health gate ──────────────────────────────────────────────────────────────

check "cluster_health" {
  assert {
    condition     = local.health_status == "HEALTHY"
    error_message = "Cluster health check failed: ${local.health_status} — ${local.summary_text} (critical: ${local.critical_count}, warnings: ${local.warning_count})"
  }
}

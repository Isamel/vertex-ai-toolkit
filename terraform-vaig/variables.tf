# variables.tf — Input variables for the vaig health-gate module.

variable "namespace" {
  description = "Kubernetes namespace to check. Leave empty for cluster-wide."
  type        = string
  default     = ""
}

variable "cluster" {
  description = "GKE cluster name. Uses gcloud default if not set."
  type        = string
  default     = ""
}

variable "project" {
  description = "GCP project ID. Uses gcloud default if not set."
  type        = string
  default     = ""
}

variable "timeout" {
  description = "Health check timeout in seconds."
  type        = number
  default     = 120
}

variable "vaig_path" {
  description = "Path to the vaig binary. Override if vaig is not on PATH."
  type        = string
  default     = "vaig"
}

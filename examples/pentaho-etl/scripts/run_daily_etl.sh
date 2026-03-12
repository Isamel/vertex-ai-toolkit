#!/usr/bin/env bash
# =============================================================================
# run_daily_etl.sh
# Runner script for the Pentaho daily sales ETL pipeline.
#
# Usage:
#   ./run_daily_etl.sh [--date YYYY-MM-DD] [--archive-inputs YYYY-MM-DD] [--dry-run]
#
# Modes:
#   --date YYYY-MM-DD         Run the full ETL job for the given sales date.
#   --archive-inputs YYYY-MM-DD  Move processed input files to archive.
#   --dry-run                 Print what would happen without executing.
#
# Make executable:  chmod +x scripts/run_daily_etl.sh
# =============================================================================
set -euo pipefail

# ---- Configuration -----------------------------------------------------------
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly KITCHEN="${PENTAHO_HOME:-/opt/pentaho/data-integration}/kitchen.sh"
readonly JOB_FILE="${PROJECT_DIR}/jobs/daily_sales_job.kjb"
readonly LOG_DIR="${PROJECT_DIR}/logs"
readonly INPUT_DIR="${PROJECT_DIR}/data/input"
readonly ARCHIVE_DIR="${PROJECT_DIR}/data/archive"
readonly LOG_LEVEL="${ETL_LOG_LEVEL:-Basic}"

# ---- Globals -----------------------------------------------------------------
DRY_RUN="false"
MODE=""
SALES_DATE=""

# ---- Functions ---------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --date YYYY-MM-DD           Run the Pentaho ETL job for the given date.
  --archive-inputs YYYY-MM-DD Move processed input files to the archive.
  --dry-run                   Show what would be executed without running it.
  -h, --help                  Show this help message.

Examples:
  $(basename "$0") --date 2025-03-15
  $(basename "$0") --archive-inputs 2025-03-15
  $(basename "$0") --date 2025-03-15 --dry-run
EOF
}

log() {
    local level="$1"; shift
    printf '[%s] [%-5s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "$*"
}

log_info()  { log "INFO"  "$@"; }
log_warn()  { log "WARN"  "$@"; }
log_error() { log "ERROR" "$@"; }

validate_date() {
    local d="$1"
    if [[ ! "${d}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        log_error "Invalid date format: '${d}'. Expected YYYY-MM-DD."
        exit 1
    fi
}

ensure_dirs() {
    mkdir -p "${LOG_DIR}" "${ARCHIVE_DIR}"
}

# ---- Run ETL -----------------------------------------------------------------
run_etl() {
    local date="$1"
    local log_file="${LOG_DIR}/etl_${date}.log"

    log_info "Starting ETL pipeline for SALES_DATE=${date}"
    log_info "Job file : ${JOB_FILE}"
    log_info "Log file : ${log_file}"

    if [[ ! -f "${JOB_FILE}" ]]; then
        log_error "Job file not found: ${JOB_FILE}"
        exit 2
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] Would execute:"
        log_info "  ${KITCHEN} \\"
        log_info "    -file=${JOB_FILE} \\"
        log_info "    -param:SALES_DATE=${date} \\"
        log_info "    -level=${LOG_LEVEL} \\"
        log_info "    -log=${log_file}"
        return 0
    fi

    if [[ ! -x "${KITCHEN}" ]]; then
        log_error "Kitchen not found or not executable: ${KITCHEN}"
        log_error "Set PENTAHO_HOME to your PDI installation directory."
        exit 3
    fi

    "${KITCHEN}" \
        -file="${JOB_FILE}" \
        -param:SALES_DATE="${date}" \
        -level="${LOG_LEVEL}" \
        -log="${log_file}"

    local rc=$?

    if [[ ${rc} -eq 0 ]]; then
        log_info "ETL pipeline completed successfully for ${date}."
    else
        log_error "ETL pipeline FAILED for ${date} (exit code ${rc}). Check ${log_file}."
        exit ${rc}
    fi
}

# ---- Archive Inputs ----------------------------------------------------------
archive_inputs() {
    local date="$1"
    local dest="${ARCHIVE_DIR}/${date}"

    log_info "Archiving input files for ${date} → ${dest}/"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] Would move files from ${INPUT_DIR}/ to ${dest}/"
        if [[ -d "${INPUT_DIR}" ]]; then
            log_info "[DRY-RUN] Files that would be archived:"
            find "${INPUT_DIR}" -maxdepth 1 -type f -name "*${date}*" 2>/dev/null \
                | while read -r f; do log_info "  ${f}"; done
            local count
            count=$(find "${INPUT_DIR}" -maxdepth 1 -type f -name "*${date}*" 2>/dev/null | wc -l)
            if [[ "${count}" -eq 0 ]]; then
                log_warn "[DRY-RUN] No files matching *${date}* found in ${INPUT_DIR}/"
            fi
        fi
        return 0
    fi

    mkdir -p "${dest}"

    local moved=0
    if [[ -d "${INPUT_DIR}" ]]; then
        for f in "${INPUT_DIR}"/*"${date}"*; do
            [[ -e "${f}" ]] || continue
            mv "${f}" "${dest}/"
            log_info "  Archived: $(basename "${f}")"
            moved=$((moved + 1))
        done
    fi

    if [[ ${moved} -eq 0 ]]; then
        log_warn "No input files matching *${date}* found in ${INPUT_DIR}/."
    else
        log_info "Archived ${moved} file(s) to ${dest}/."
    fi
}

# ---- Argument parsing --------------------------------------------------------
parse_args() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 0
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --date)
                [[ -z "${2:-}" ]] && { log_error "--date requires a YYYY-MM-DD argument."; exit 1; }
                validate_date "$2"
                SALES_DATE="$2"
                MODE="etl"
                shift 2
                ;;
            --archive-inputs)
                [[ -z "${2:-}" ]] && { log_error "--archive-inputs requires a YYYY-MM-DD argument."; exit 1; }
                validate_date "$2"
                SALES_DATE="$2"
                MODE="archive"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "${MODE}" ]]; then
        log_error "No action specified. Use --date or --archive-inputs."
        usage
        exit 1
    fi
}

# ---- Main --------------------------------------------------------------------
main() {
    parse_args "$@"
    ensure_dirs

    case "${MODE}" in
        etl)
            run_etl "${SALES_DATE}"
            ;;
        archive)
            archive_inputs "${SALES_DATE}"
            ;;
    esac
}

main "$@"

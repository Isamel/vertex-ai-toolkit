/**
 * Shared types for the VAIG VS Code extension.
 *
 * These mirror the server-side contracts defined in
 * `src/vaig/web/routes/health.py` and `src/vaig/web/sse.py`.
 */

// ── Connection ──────────────────────────────────────────────

/** Extension ↔ server connection state machine. */
export type ConnectionState = "disconnected" | "connecting" | "connected";

/** Response shape from `GET /health`. */
export interface HealthResponse {
  readonly status: string;
  readonly version: string;
}

// ── Live Diagnosis ──────────────────────────────────────────

/** Parameters sent as `application/x-www-form-urlencoded` to `POST /live/stream`. */
export interface LiveDiagnosisParams {
  readonly service_name: string;
  readonly question?: string;
  readonly cluster?: string;
  readonly namespace?: string;
  readonly gke_project?: string;
  readonly gke_location?: string;
}

/** All SSE event types emitted by the VAIG live pipeline. */
export type SSEEventType =
  | "agent_start"
  | "agent_end"
  | "tool_call"
  | "chunk"
  | "result"
  | "report_html"
  | "error"
  | "done"
  | "keepalive"
  | "phase";

/** A single parsed SSE event from the live stream. */
export interface SSEEvent {
  readonly event: SSEEventType;
  readonly data: Record<string, unknown>;
}

// ── Configuration ───────────────────────────────────────────

/** Typed representation of the extension's `vaig.*` settings. */
export interface VaigConfig {
  readonly serverUrl: string;
  readonly autoConnect: boolean;
}

// ── Report ──────────────────────────────────────────────────

/** Metadata stored alongside a captured report. */
export interface ReportData {
  readonly html: string;
  readonly serviceName: string;
  readonly capturedAt: Date;
}

/**
 * HTTP client for the VAIG server.
 *
 * - `checkHealth()` — `GET /health` → `HealthResponse`
 * - `streamLiveDiagnosis()` — `POST /live/stream` (form-encoded) → async
 *    iterable of `SSEEvent` via `eventsource-parser`
 *
 * Design decisions (see Design §Arch #1, #5):
 * - Uses native `fetch()` (Node 18+) — no extra HTTP dependency.
 * - SSE parsing via `eventsource-parser` — the only lib that handles
 *   POST-based SSE (native EventSource is GET-only).
 * - Form data via `URLSearchParams` — zero deps, matches server's
 *   `request.form()`.
 */

import { createParser } from "eventsource-parser";
import type { EventSourceMessage } from "eventsource-parser";

import type {
  HealthResponse,
  LiveDiagnosisParams,
  SSEEvent,
  SSEEventType,
} from "../types.js";

/** All known SSE event types — used for type-narrowing parsed events. */
const KNOWN_EVENTS = new Set<string>([
  "agent_start",
  "agent_end",
  "tool_call",
  "chunk",
  "result",
  "report_html",
  "error",
  "done",
  "keepalive",
  "phase",
]);

/** Default timeout for health checks (ms). */
const HEALTH_TIMEOUT_MS = 5_000;

export class VaigClient {
  private controller: AbortController | null = null;

  /**
   * Poll the server's health endpoint.
   *
   * @returns Parsed `HealthResponse` on success.
   * @throws On network error, timeout, or invalid response shape.
   */
  async checkHealth(baseUrl: string): Promise<HealthResponse> {
    const url = `${baseUrl.replace(/\/+$/, "")}/health`;

    const res = await fetch(url, {
      method: "GET",
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });

    if (!res.ok) {
      throw new Error(`Health check failed: HTTP ${res.status}`);
    }

    const body: unknown = await res.json();

    if (!isHealthResponse(body)) {
      throw new Error(
        "Invalid health response — expected { status, version }",
      );
    }

    return body;
  }

  /**
   * Start a live diagnosis and yield SSE events as they arrive.
   *
   * The caller should iterate with `for await` and handle each event type
   * per the spec table (REQ-5).  Call `abort()` to cancel mid-stream.
   */
  async *streamLiveDiagnosis(
    baseUrl: string,
    params: LiveDiagnosisParams,
    devMode = false,
  ): AsyncGenerator<SSEEvent, void, undefined> {
    // Prepare form body — only include defined fields.
    const formData = new URLSearchParams();
    formData.set("service_name", params.service_name);
    if (params.question) formData.set("question", params.question);
    if (params.cluster) formData.set("cluster", params.cluster);
    if (params.namespace) formData.set("namespace", params.namespace);
    if (params.gke_project) formData.set("gke_project", params.gke_project);
    if (params.gke_location) formData.set("gke_location", params.gke_location);

    // Fresh AbortController per stream.
    this.controller = new AbortController();

    const url = `${baseUrl.replace(/\/+$/, "")}/live/stream`;
    const headers: Record<string, string> = {
      "Content-Type": "application/x-www-form-urlencoded",
      Accept: "text/event-stream",
    };
    if (devMode) {
      headers["X-Dev-Mode"] = "true";
    }

    const res = await fetch(url, {
      method: "POST",
      body: formData.toString(),
      headers,
      signal: this.controller.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(
        `Live stream request failed: HTTP ${res.status}${text ? ` — ${text}` : ""}`,
      );
    }

    if (!res.body) {
      throw new Error("Server returned an empty body for live stream");
    }

    // Bridge: ReadableStream → eventsource-parser → AsyncGenerator
    yield* this.parseSSEStream(res.body);
  }

  /** Cancel any in-flight stream request. */
  abort(): void {
    this.controller?.abort();
    this.controller = null;
  }

  // ── Private ─────────────────────────────────────────────────

  /**
   * Read a `ReadableStream<Uint8Array>` and yield parsed SSE events.
   *
   * Uses `eventsource-parser`'s `createParser` to handle multi-line
   * data fields, event types, and reconnection intervals.
   */
  private async *parseSSEStream(
    body: ReadableStream<Uint8Array>,
  ): AsyncGenerator<SSEEvent, void, undefined> {
    // Queue approach: parser pushes events, generator yields them.
    const queue: SSEEvent[] = [];

    const parser = createParser({
      onEvent(event: EventSourceMessage) {
        const eventType = event.event ?? "message";
        if (!KNOWN_EVENTS.has(eventType)) return; // ignore unknown types

        let data: Record<string, unknown>;
        try {
          data = JSON.parse(event.data) as Record<string, unknown>;
        } catch {
          // Non-JSON data (e.g. plain text chunks) — wrap it.
          data = { text: event.data };
        }

        queue.push({
          event: eventType as SSEEventType,
          data,
        });
      },
    });

    const reader = body.getReader();
    const decoder = new TextDecoder();
    let done = false;

    try {
      while (!done) {
        // Yield any queued events first.
        while (queue.length > 0) {
          const event = queue.shift()!;
          yield event;
          if (event.event === "done") return;
        }

        // Read next chunk from the stream.
        const result = await reader.read();
        done = result.done;

        if (result.value) {
          parser.feed(decoder.decode(result.value, { stream: true }));
        }
      }

      // Drain remaining queued events after stream ends.
      while (queue.length > 0) {
        yield queue.shift()!;
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// ── Type guards ───────────────────────────────────────────────

function isHealthResponse(value: unknown): value is HealthResponse {
  if (typeof value !== "object" || value === null) return false;
  const obj = value as Record<string, unknown>;
  return typeof obj["status"] === "string" && typeof obj["version"] === "string";
}

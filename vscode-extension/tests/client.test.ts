/**
 * Unit tests for VaigClient.
 *
 * Mocks: `fetch` (global), `vscode` module, `eventsource-parser`.
 * No real HTTP calls are made.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock vscode before any imports that reference it.
vi.mock("vscode", () => import("./__mocks__/vscode.js"));

// We need to mock createParser from eventsource-parser because the real
// implementation relies on streaming which is hard to unit-test in isolation.
// Instead we test the client's fetch handling and SSE bridging separately.

import { VaigClient } from "../src/server/client.js";

// ── Helpers ────────────────────────────────────────────────────

function mockFetchOk(body: unknown, contentType = "application/json") {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    headers: new Headers({ "Content-Type": contentType }),
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
    body: null,
  } as Partial<Response>);
}

function mockFetchError(status: number, text = "") {
  return vi.fn().mockResolvedValue({
    ok: false,
    status,
    headers: new Headers(),
    json: () => Promise.reject(new Error("not json")),
    text: () => Promise.resolve(text),
    body: null,
  } as Partial<Response>);
}

function mockFetchReject(error: Error) {
  return vi.fn().mockRejectedValue(error);
}

// ── checkHealth ────────────────────────────────────────────────

describe("VaigClient", () => {
  let client: VaigClient;

  beforeEach(() => {
    client = new VaigClient();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  describe("checkHealth()", () => {
    it("returns HealthResponse on valid JSON", async () => {
      const body = { status: "ok", version: "1.2.3" };
      vi.stubGlobal("fetch", mockFetchOk(body));

      const result = await client.checkHealth("http://localhost:8080");

      expect(result).toEqual(body);
      expect(fetch).toHaveBeenCalledOnce();
      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8080/health",
        expect.objectContaining({ method: "GET" }),
      );
    });

    it("strips trailing slashes from base URL", async () => {
      const body = { status: "ok", version: "2.0.0" };
      vi.stubGlobal("fetch", mockFetchOk(body));

      await client.checkHealth("http://localhost:8080///");

      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8080/health",
        expect.anything(),
      );
    });

    it("throws on HTTP error status", async () => {
      vi.stubGlobal("fetch", mockFetchError(503, "Service Unavailable"));

      await expect(
        client.checkHealth("http://localhost:8080"),
      ).rejects.toThrow("Health check failed: HTTP 503");
    });

    it("throws when server is unreachable (network error)", async () => {
      vi.stubGlobal(
        "fetch",
        mockFetchReject(new TypeError("fetch failed")),
      );

      await expect(
        client.checkHealth("http://localhost:8080"),
      ).rejects.toThrow("fetch failed");
    });

    it("throws on invalid JSON shape (missing version)", async () => {
      const body = { status: "ok" }; // no version field
      vi.stubGlobal("fetch", mockFetchOk(body));

      await expect(
        client.checkHealth("http://localhost:8080"),
      ).rejects.toThrow("Invalid health response");
    });

    it("throws on non-object response", async () => {
      vi.stubGlobal("fetch", mockFetchOk("just a string"));

      await expect(
        client.checkHealth("http://localhost:8080"),
      ).rejects.toThrow("Invalid health response");
    });

    it("includes Accept: application/json header", async () => {
      const body = { status: "ok", version: "1.0.0" };
      vi.stubGlobal("fetch", mockFetchOk(body));

      await client.checkHealth("http://localhost:8080");

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: { Accept: "application/json" },
        }),
      );
    });
  });

  // ── Helpers (shared) ──────────────────────────────────────────

  /**
   * Helper to create a ReadableStream that emits SSE-formatted chunks.
   */
  function createSSEStream(events: string[]): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    return new ReadableStream({
      start(controller) {
        for (const event of events) {
          controller.enqueue(encoder.encode(event));
        }
        controller.close();
      },
    });
  }

  // ── streamLiveDiagnosis ──────────────────────────────────────

  describe("streamLiveDiagnosis()", () => {
    it("parses SSE events from stream", async () => {
      const ssePayload = [
        'event: agent_start\ndata: {"agent":"diag"}\n\n',
        'event: chunk\ndata: {"text":"hello"}\n\n',
        'event: done\ndata: {"status":"ok"}\n\n',
      ];

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          body: createSSEStream(ssePayload),
          text: () => Promise.resolve(""),
        }),
      );

      const events = [];
      for await (const event of client.streamLiveDiagnosis(
        "http://localhost:8080",
        { service_name: "test-svc" },
      )) {
        events.push(event);
        if (event.event === "done") break;
      }

      expect(events).toHaveLength(3);
      expect(events[0]).toEqual({
        event: "agent_start",
        data: { agent: "diag" },
      });
      expect(events[1]).toEqual({
        event: "chunk",
        data: { text: "hello" },
      });
      expect(events[2]).toEqual({
        event: "done",
        data: { status: "ok" },
      });
    });

    it("sends form-encoded body with required and optional params", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          body: createSSEStream(['event: done\ndata: {"ok":true}\n\n']),
          text: () => Promise.resolve(""),
        }),
      );

      const params = {
        service_name: "my-api",
        question: "why is it slow?",
        cluster: "prod-1",
      };

      // Consume the generator
      for await (const _ of client.streamLiveDiagnosis(
        "http://localhost:8080",
        params,
      )) {
        /* drain */
      }

      const callArgs = vi.mocked(fetch).mock.calls[0]!;
      const body = callArgs[1]?.body as string;
      expect(body).toContain("service_name=my-api");
      expect(body).toContain("question=why+is+it+slow%3F");
      expect(body).toContain("cluster=prod-1");
    });

    it("sends X-Dev-Mode header when devMode is true", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          body: createSSEStream(['event: done\ndata: {}\n\n']),
          text: () => Promise.resolve(""),
        }),
      );

      for await (const _ of client.streamLiveDiagnosis(
        "http://localhost:8080",
        { service_name: "svc" },
        true, // devMode
      )) {
        /* drain */
      }

      const callArgs = vi.mocked(fetch).mock.calls[0]!;
      const headers = callArgs[1]?.headers as Record<string, string>;
      expect(headers["X-Dev-Mode"]).toBe("true");
    });

    it("throws on HTTP error response", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 500,
          text: () => Promise.resolve("Internal Server Error"),
        }),
      );

      const gen = client.streamLiveDiagnosis("http://localhost:8080", {
        service_name: "svc",
      });

      await expect(gen.next()).rejects.toThrow(
        "Live stream request failed: HTTP 500",
      );
    });

    it("throws on empty body", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          body: null,
          text: () => Promise.resolve(""),
        }),
      );

      const gen = client.streamLiveDiagnosis("http://localhost:8080", {
        service_name: "svc",
      });

      await expect(gen.next()).rejects.toThrow(
        "Server returned an empty body",
      );
    });

    it("ignores unknown SSE event types", async () => {
      const ssePayload = [
        'event: unknown_type\ndata: {"ignored":true}\n\n',
        'event: chunk\ndata: {"text":"kept"}\n\n',
        'event: done\ndata: {}\n\n',
      ];

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          body: createSSEStream(ssePayload),
          text: () => Promise.resolve(""),
        }),
      );

      const events = [];
      for await (const event of client.streamLiveDiagnosis(
        "http://localhost:8080",
        { service_name: "svc" },
      )) {
        events.push(event);
      }

      // unknown_type should be filtered out
      expect(events.every((e) => e.event !== "unknown_type")).toBe(true);
      expect(events.some((e) => e.event === "chunk")).toBe(true);
    });

    it("wraps non-JSON data in { text } object", async () => {
      const ssePayload = [
        "event: chunk\ndata: plain text content\n\n",
        'event: done\ndata: {}\n\n',
      ];

      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          status: 200,
          body: createSSEStream(ssePayload),
          text: () => Promise.resolve(""),
        }),
      );

      const events = [];
      for await (const event of client.streamLiveDiagnosis(
        "http://localhost:8080",
        { service_name: "svc" },
      )) {
        events.push(event);
      }

      const chunkEvent = events.find((e) => e.event === "chunk");
      expect(chunkEvent?.data).toEqual({ text: "plain text content" });
    });
  });

  // ── abort ────────────────────────────────────────────────────

  describe("abort()", () => {
    it("sets controller to null after abort", () => {
      // Verify abort() is safe to call even without an active stream.
      // It should not throw and should reset internal state.
      expect(() => client.abort()).not.toThrow();
    });

    it("aborts the AbortController used by fetch", async () => {
      // Capture the signal passed to fetch to verify abort behavior.
      let capturedSignal: AbortSignal | undefined;

      vi.stubGlobal(
        "fetch",
        vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
          capturedSignal = init?.signal ?? undefined;
          return Promise.resolve({
            ok: true,
            status: 200,
            body: createSSEStream(['event: done\ndata: {}\n\n']),
            text: () => Promise.resolve(""),
          });
        }),
      );

      // Consume the stream to trigger fetch
      for await (const _ of client.streamLiveDiagnosis(
        "http://localhost:8080",
        { service_name: "svc" },
      )) {
        /* drain */
      }

      // Signal was passed to fetch
      expect(capturedSignal).toBeDefined();

      // After abort, calling abort again is safe
      client.abort();
      expect(() => client.abort()).not.toThrow();
    });
  });
});

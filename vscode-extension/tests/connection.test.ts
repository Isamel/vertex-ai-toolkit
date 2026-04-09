/**
 * Unit tests for ConnectionManager.
 *
 * Mocks: vscode module, VaigClient, timers.
 */

import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from "vitest";

// Mock vscode before any imports that reference it.
vi.mock("vscode", () => import("./__mocks__/vscode.js"));

// Mock the config module so constructor doesn't blow up.
vi.mock("../src/config.js", () => ({
  getConfig: vi.fn(() => ({
    serverUrl: "http://localhost:8080",
    autoConnect: true,
    devMode: true,
  })),
}));

// Mock VaigClient so we control checkHealth results.
vi.mock("../src/server/client.js", () => {
  const MockVaigClient = vi.fn();
  MockVaigClient.prototype.checkHealth = vi.fn();
  MockVaigClient.prototype.abort = vi.fn();
  return { VaigClient: MockVaigClient };
});

import { ConnectionManager } from "../src/server/connection.js";
import { VaigClient } from "../src/server/client.js";
import { getConfig } from "../src/config.js";

// ── Tests ──────────────────────────────────────────────────────

describe("ConnectionManager", () => {
  let manager: ConnectionManager;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(getConfig).mockReturnValue({
      serverUrl: "http://localhost:8080",
      autoConnect: true,
      devMode: true,
    });
    manager = new ConnectionManager();
  });

  afterEach(() => {
    manager.dispose();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  // ── State transitions ────────────────────────────────────────

  describe("state transitions", () => {
    it("starts in disconnected state", () => {
      expect(manager.state).toBe("disconnected");
    });

    it("transitions disconnected → connecting → connected on successful health check", async () => {
      const states: string[] = [];
      manager.onDidChangeState((s) => states.push(s));

      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
      });

      await manager.connect();

      expect(states).toEqual(["connecting", "connected"]);
      expect(manager.state).toBe("connected");
    });

    it("transitions connecting → disconnected on health check failure", async () => {
      const states: string[] = [];
      manager.onDidChangeState((s) => states.push(s));

      vi.mocked(VaigClient.prototype.checkHealth).mockRejectedValueOnce(
        new Error("unreachable"),
      );

      await manager.connect();

      expect(states).toEqual(["connecting", "disconnected"]);
      expect(manager.state).toBe("disconnected");
    });

    it("stores server version when connected", async () => {
      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "2.5.0",
      });

      await manager.connect();

      expect(manager.serverVersion).toBe("2.5.0");
    });

    it("clears server version on disconnect", async () => {
      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
      });

      await manager.connect();
      expect(manager.serverVersion).toBe("1.0.0");

      manager.disconnect();
      expect(manager.serverVersion).toBeUndefined();
    });
  });

  // ── Backoff ──────────────────────────────────────────────────

  describe("exponential backoff", () => {
    it("uses increasing backoff intervals: 10 → 20 → 40 → 60 → 60 (capped)", async () => {
      // Every health check fails
      vi.mocked(VaigClient.prototype.checkHealth).mockRejectedValue(
        new Error("down"),
      );

      await manager.connect();

      // After first failure, backoff is 10s (initial), next should be 20s
      // The timer was scheduled with 10_000ms (initial backoff).
      // After that fires, next backoff = 20_000, then 40_000, then 60_000 (cap)

      // Advance 10s — triggers retry #1
      await vi.advanceTimersByTimeAsync(10_000);
      // Now backoff becomes 20s

      // Advance 20s — triggers retry #2
      await vi.advanceTimersByTimeAsync(20_000);
      // Now backoff becomes 40s

      // Advance 40s — triggers retry #3
      await vi.advanceTimersByTimeAsync(40_000);
      // Now backoff becomes 60s (capped at MAX_BACKOFF_MS)

      // Advance 60s — triggers retry #4
      await vi.advanceTimersByTimeAsync(60_000);
      // Backoff should still be 60s (capped)

      // Advance another 60s — should still be 60s
      await vi.advanceTimersByTimeAsync(60_000);

      // Total calls: initial + 5 retries = 6
      expect(VaigClient.prototype.checkHealth).toHaveBeenCalledTimes(6);
    });

    it("resets backoff after successful connection", async () => {
      // First call fails, second succeeds
      vi.mocked(VaigClient.prototype.checkHealth)
        .mockRejectedValueOnce(new Error("down"))
        .mockResolvedValueOnce({ status: "ok", version: "1.0.0" });

      await manager.connect();
      // First failure → schedules retry at 10s

      await vi.advanceTimersByTimeAsync(10_000);
      // Retry succeeds → backoff resets, schedules at normal 10s interval

      expect(manager.state).toBe("connected");
    });
  });

  // ── connect() ────────────────────────────────────────────────

  describe("connect()", () => {
    it("uses provided URL when given", async () => {
      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
      });

      await manager.connect("http://custom:9090");

      expect(manager.serverUrl).toBe("http://custom:9090");
      expect(VaigClient.prototype.checkHealth).toHaveBeenCalledWith(
        "http://custom:9090",
      );
    });

    it("reads URL from config when none provided", async () => {
      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
      });

      await manager.connect();

      expect(manager.serverUrl).toBe("http://localhost:8080");
    });
  });

  // ── disconnect() ─────────────────────────────────────────────

  describe("disconnect()", () => {
    it("clears timers and resets state", async () => {
      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
      });

      await manager.connect();
      expect(manager.state).toBe("connected");

      manager.disconnect();

      expect(manager.state).toBe("disconnected");
      expect(manager.serverVersion).toBeUndefined();
    });

    it("does not fire state change if already disconnected", () => {
      const states: string[] = [];
      manager.onDidChangeState((s) => states.push(s));

      manager.disconnect();

      // Already disconnected — no state change event
      expect(states).toEqual([]);
    });
  });

  // ── dispose() ────────────────────────────────────────────────

  describe("dispose()", () => {
    it("stops polling and aborts in-flight requests", async () => {
      vi.mocked(VaigClient.prototype.checkHealth).mockResolvedValueOnce({
        status: "ok",
        version: "1.0.0",
      });

      await manager.connect();
      manager.dispose();

      // Advancing timers should NOT trigger more health checks
      const callCount = vi.mocked(
        VaigClient.prototype.checkHealth,
      ).mock.calls.length;

      await vi.advanceTimersByTimeAsync(30_000);

      expect(
        VaigClient.prototype.checkHealth,
      ).toHaveBeenCalledTimes(callCount);
    });
  });

  // ── Event emitter ────────────────────────────────────────────

  describe("onDidChangeState", () => {
    it("fires event on each state change", async () => {
      const states: string[] = [];
      manager.onDidChangeState((s) => states.push(s));

      vi.mocked(VaigClient.prototype.checkHealth)
        .mockResolvedValueOnce({ status: "ok", version: "1.0.0" })
        .mockRejectedValueOnce(new Error("down"));

      await manager.connect(); // disconnected → connecting → connected

      // Trigger next poll (fails)
      await vi.advanceTimersByTimeAsync(10_000); // connected → disconnected

      expect(states).toEqual([
        "connecting",
        "connected",
        "disconnected",
      ]);
    });
  });
});

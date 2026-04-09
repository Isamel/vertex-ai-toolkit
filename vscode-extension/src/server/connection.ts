/**
 * Connection manager — health polling, state machine, exponential backoff.
 *
 * Lifecycle:
 *   disconnected → connecting → connected → disconnected (on failure)
 *
 * Health polling runs on a `setTimeout` at 10s when connected.
 * On failure, retries with exponential backoff (10 → 20 → 40 → 60s cap).
 * Emits `onDidChangeState` for the status bar and commands to react.
 *
 * Implements `vscode.Disposable` — push to `context.subscriptions`.
 */

import * as vscode from "vscode";

import { getConfig } from "../config.js";
import type { ConnectionState } from "../types.js";
import { VaigClient } from "./client.js";

/** Normal polling interval when connected (ms). */
const POLL_INTERVAL_MS = 10_000;

/** Initial backoff delay on health failure (ms). */
const INITIAL_BACKOFF_MS = 10_000;

/** Maximum backoff delay (ms). */
const MAX_BACKOFF_MS = 60_000;

export class ConnectionManager implements vscode.Disposable {
  private readonly client = new VaigClient();
  private readonly stateEmitter = new vscode.EventEmitter<ConnectionState>();

  /** Fired whenever the connection state changes. */
  readonly onDidChangeState: vscode.Event<ConnectionState> =
    this.stateEmitter.event;

  private _state: ConnectionState = "disconnected";
  private _serverVersion: string | undefined;
  private _serverUrl: string;

  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private currentBackoff = INITIAL_BACKOFF_MS;
  private disposed = false;
  private _isPolling = false;
  private _abortController: AbortController | null = null;

  constructor() {
    this._serverUrl = getConfig().serverUrl;
  }

  // ── Public API ────────────────────────────────────────────

  /** Current connection state. */
  get state(): ConnectionState {
    return this._state;
  }

  /** Server version string (only defined when connected). */
  get serverVersion(): string | undefined {
    return this._serverVersion;
  }

  /** The URL currently being polled. */
  get serverUrl(): string {
    return this._serverUrl;
  }

  /** The underlying HTTP client (shared with commands). */
  get httpClient(): VaigClient {
    return this.client;
  }

  /**
   * Start connecting to the server.
   *
   * If a URL is provided it overrides the saved setting. Otherwise
   * reads the current `vaig.serverUrl` configuration value.
   */
  async connect(url?: string): Promise<void> {
    if (this._isPolling) return;

    if (url) {
      this._serverUrl = url;
    } else {
      this._serverUrl = getConfig().serverUrl;
    }

    this._abortController = new AbortController();
    this.setState("connecting");
    await this.poll();
  }

  /** Stop polling, cancel in-flight health checks, and transition to disconnected. */
  disconnect(): void {
    this.clearTimer();
    this._abortController?.abort();
    this._abortController = null;
    this._serverVersion = undefined;
    this.currentBackoff = INITIAL_BACKOFF_MS;
    this.setState("disconnected");
  }

  /** Clean up timers and event emitter. */
  dispose(): void {
    this.disposed = true;
    this.clearTimer();
    this.client.abort();
    this.stateEmitter.dispose();
  }

  // ── Private ─────────────────────────────────────────────────

  private setState(next: ConnectionState): void {
    if (next === this._state) return;
    this._state = next;
    this.stateEmitter.fire(next);
  }

  /**
   * Execute one health check and schedule the next poll.
   *
   * On success → connected, reset backoff, schedule at normal interval.
   * On failure → disconnected, double backoff (capped), schedule retry.
   */
  private async poll(): Promise<void> {
    if (this.disposed || this._isPolling) return;

    this._isPolling = true;
    try {
      const health = await this.client.checkHealth(this._serverUrl);
      this._serverVersion = health.version;
      this.currentBackoff = INITIAL_BACKOFF_MS;
      this.setState("connected");
      this.scheduleNext(POLL_INTERVAL_MS);
    } catch {
      this._serverVersion = undefined;
      this.setState("disconnected");
      this.scheduleNext(this.currentBackoff);
      this.currentBackoff = Math.min(
        this.currentBackoff * 2,
        MAX_BACKOFF_MS,
      );
    } finally {
      this._isPolling = false;
    }
  }

  private scheduleNext(delayMs: number): void {
    this.clearTimer();
    if (this.disposed) return;
    this.pollTimer = setTimeout(() => {
      void this.poll();
    }, delayMs);
  }

  private clearTimer(): void {
    if (this.pollTimer !== null) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  }
}

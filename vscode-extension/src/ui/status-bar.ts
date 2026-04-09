/**
 * Status bar manager — displays VAIG connection state.
 *
 * Shows one of three states in the VS Code status bar (left, priority 100):
 *   - Connected:    "$(check) VAIG: vX.Y.Z"       (default background)
 *   - Disconnected: "$(error) VAIG: Disconnected"  (error background)
 *   - Connecting:   "$(sync~spin) VAIG: Connecting..." (no special background)
 *
 * Clicking the item runs `vaig.connect`.
 *
 * Implements `vscode.Disposable` — push to `context.subscriptions`.
 */

import * as vscode from "vscode";

import type { ConnectionState } from "../types.js";
import type { ConnectionManager } from "../server/connection.js";

export class StatusBarManager implements vscode.Disposable {
  private readonly item: vscode.StatusBarItem;
  private readonly stateListener: vscode.Disposable;

  constructor(connectionManager: ConnectionManager) {
    // Left-aligned, priority 100 (appears early in the status bar).
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100,
    );

    this.item.command = "vaig.connect";
    this.item.name = "VAIG Connection Status";

    // Set initial state.
    this.update(connectionManager.state, connectionManager.serverVersion);

    // Subscribe to state changes.
    this.stateListener = connectionManager.onDidChangeState((state) => {
      this.update(state, connectionManager.serverVersion);
    });

    this.item.show();
  }

  dispose(): void {
    this.stateListener.dispose();
    this.item.dispose();
  }

  // ── Private ─────────────────────────────────────────────────

  private update(state: ConnectionState, version: string | undefined): void {
    switch (state) {
      case "connected":
        this.item.text = `$(check) VAIG: v${version ?? "?"}`;
        this.item.tooltip = `Connected to VAIG server v${version ?? "unknown"}`;
        this.item.backgroundColor = undefined;
        break;

      case "disconnected":
        this.item.text = "$(error) VAIG: Disconnected";
        this.item.tooltip = "Click to connect to VAIG server";
        this.item.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.errorBackground",
        );
        break;

      case "connecting":
        this.item.text = "$(sync~spin) VAIG: Connecting...";
        this.item.tooltip = "Connecting to VAIG server...";
        this.item.backgroundColor = undefined;
        break;
    }
  }
}

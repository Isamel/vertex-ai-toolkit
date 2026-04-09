/**
 * VAIG VS Code Extension — entry point.
 *
 * Stub for Phase 1-2. Full implementation in Phase 4 (task 4.1).
 */

import * as vscode from "vscode";

import { getConfig } from "./config.js";
import { ConnectionManager } from "./server/connection.js";

export function activate(context: vscode.ExtensionContext): void {
  const outputChannel = vscode.window.createOutputChannel("VAIG");
  outputChannel.appendLine("VAIG extension activating…");

  const connectionManager = new ConnectionManager();
  context.subscriptions.push(connectionManager);

  // Auto-connect if configured.
  const config = getConfig();
  if (config.autoConnect) {
    void connectionManager.connect();
  }

  // Placeholder command registrations — full implementation in Phase 3-4.
  context.subscriptions.push(
    vscode.commands.registerCommand("vaig.connect", () => {
      void connectionManager.connect();
    }),
    vscode.commands.registerCommand("vaig.liveDiagnosis", () => {
      void vscode.window.showInformationMessage(
        "Live diagnosis — coming in Phase 3",
      );
    }),
    vscode.commands.registerCommand("vaig.openReport", () => {
      void vscode.window.showInformationMessage(
        "Report viewer — coming in Phase 3",
      );
    }),
  );

  outputChannel.appendLine(`VAIG extension activated (server: ${config.serverUrl})`);
}

export function deactivate(): void {
  // Disposables registered via context.subscriptions are cleaned up by VS Code.
}

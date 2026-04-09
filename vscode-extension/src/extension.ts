/**
 * VAIG VS Code Extension — entry point.
 *
 * Wires together:
 *   - ConnectionManager (health polling, state machine)
 *   - StatusBarManager (visual connection indicator)
 *   - ReportPanelManager (WebView for diagnosis reports)
 *   - Commands: connect, liveDiagnosis, openReport
 *
 * All disposables are pushed to `context.subscriptions` so VS Code
 * cleans them up automatically on deactivation.
 */

import * as vscode from "vscode";

import { getConfig } from "./config.js";
import { createConnectCommand } from "./commands/connect.js";
import { createLiveDiagnosisCommand } from "./commands/run-live.js";
import type { ReportStore } from "./commands/run-live.js";
import { createOpenReportCommand } from "./commands/open-report.js";
import { ConnectionManager } from "./server/connection.js";
import { StatusBarManager } from "./ui/status-bar.js";
import { ReportPanelManager } from "./ui/report-panel.js";

export function activate(context: vscode.ExtensionContext): void {
  const outputChannel = vscode.window.createOutputChannel("VAIG Diagnosis");
  outputChannel.appendLine("VAIG extension activating...");

  // ── Core services ──────────────────────────────────────────

  const connectionManager = new ConnectionManager();
  const statusBarManager = new StatusBarManager(connectionManager);
  const reportPanelManager = new ReportPanelManager();

  // Mutable store for the last captured report (shared between commands).
  const reportStore: ReportStore = { lastReport: null };

  // ── Commands ───────────────────────────────────────────────

  const connectCmd = vscode.commands.registerCommand(
    "vaig.connect",
    createConnectCommand(connectionManager),
  );

  const liveDiagnosisCmd = vscode.commands.registerCommand(
    "vaig.liveDiagnosis",
    createLiveDiagnosisCommand(
      connectionManager,
      reportPanelManager,
      reportStore,
      outputChannel,
    ),
  );

  const openReportCmd = vscode.commands.registerCommand(
    "vaig.openReport",
    createOpenReportCommand(reportPanelManager, reportStore),
  );

  // ── Register all disposables ───────────────────────────────

  context.subscriptions.push(
    connectionManager,
    statusBarManager,
    reportPanelManager,
    outputChannel,
    connectCmd,
    liveDiagnosisCmd,
    openReportCmd,
  );

  // ── Auto-connect ───────────────────────────────────────────

  const config = getConfig();
  if (config.autoConnect) {
    outputChannel.appendLine(
      `Auto-connecting to ${config.serverUrl}...`,
    );
    void connectionManager.connect();
  }

  outputChannel.appendLine(
    `VAIG extension activated (server: ${config.serverUrl})`,
  );
}

export function deactivate(): void {
  // All disposables registered via context.subscriptions are
  // cleaned up automatically by VS Code.
}

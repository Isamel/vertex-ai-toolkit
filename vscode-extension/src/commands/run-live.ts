/**
 * `vaig.liveDiagnosis` command — stream a live infrastructure diagnosis.
 *
 * Behavior (REQ-5):
 *   - Guards on connected state — shows error if disconnected.
 *   - Prompts for `service_name` (required).
 *   - POSTs to `/live/stream` and consumes SSE events.
 *   - Logs all events to "VAIG Diagnosis" OutputChannel.
 *   - Caches `report_html` for the report command.
 *   - Shows notification on completion with "View Report" button.
 *
 * SSE event handling follows the contract in the spec table (REQ-5).
 */

import * as vscode from "vscode";

import { getConfig } from "../config.js";
import type { ConnectionManager } from "../server/connection.js";
import type { ReportPanelManager } from "../ui/report-panel.js";
import type { ReportData, SSEEvent } from "../types.js";

/** Mutable container for the last captured report. */
export interface ReportStore {
  lastReport: ReportData | null;
}

/**
 * Create and return the live diagnosis command handler.
 *
 * @param connectionManager - Shared connection manager instance.
 * @param reportPanelManager - Panel manager for showing reports.
 * @param reportStore - Mutable store for the last captured report.
 * @param outputChannel - Dedicated output channel for diagnosis logs.
 */
export function createLiveDiagnosisCommand(
  connectionManager: ConnectionManager,
  reportPanelManager: ReportPanelManager,
  reportStore: ReportStore,
  outputChannel: vscode.OutputChannel,
): () => Promise<void> {
  return async () => {
    // Guard: must be connected.
    if (connectionManager.state !== "connected") {
      void vscode.window.showErrorMessage(
        "Not connected to VAIG server. Run 'VAIG: Connect to Server' first.",
      );
      return;
    }

    // Prompt for service name (required).
    const serviceName = await vscode.window.showInputBox({
      title: "Service Name",
      prompt: "Enter the service name to diagnose",
      placeHolder: "e.g., my-service",
      validateInput: (value) => {
        if (!value.trim()) {
          return "Service name is required";
        }
        return undefined;
      },
    });

    // User pressed Escape or empty.
    if (!serviceName) return;

    const config = getConfig();
    const client = connectionManager.httpClient;

    // Clear and show the output channel.
    outputChannel.clear();
    outputChannel.show(true); // preserveFocus = true

    outputChannel.appendLine(
      `━━━ VAIG Live Diagnosis: ${serviceName} ━━━`,
    );
    outputChannel.appendLine(`Server: ${connectionManager.serverUrl}`);
    outputChannel.appendLine(`Started: ${new Date().toLocaleTimeString()}`);
    outputChannel.appendLine("");

    try {
      const stream = client.streamLiveDiagnosis(
        connectionManager.serverUrl,
        { service_name: serviceName },
        config.devMode,
      );

      for await (const event of stream) {
        handleSSEEvent(event, outputChannel, reportStore, serviceName);
      }

      // Stream ended — show completion notification.
      const action = await vscode.window.showInformationMessage(
        `Diagnosis complete for "${serviceName}"`,
        "View Report",
      );

      if (action === "View Report" && reportStore.lastReport) {
        reportPanelManager.show(reportStore.lastReport.html);
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unknown error during diagnosis";

      outputChannel.appendLine(`\n❌ Error: ${message}`);

      void vscode.window.showErrorMessage(
        `Diagnosis failed: ${message}`,
      );
    }
  };
}

// ── SSE Event Handler ────────────────────────────────────────

/**
 * Handle a single SSE event by logging to the output channel
 * and updating state as needed.
 */
function handleSSEEvent(
  event: SSEEvent,
  outputChannel: vscode.OutputChannel,
  reportStore: ReportStore,
  serviceName: string,
): void {
  switch (event.event) {
    case "agent_start": {
      const name = String(event.data["name"] ?? "unknown");
      const index = Number(event.data["index"] ?? 0);
      const total = Number(event.data["total"] ?? "?");
      outputChannel.appendLine(`▶ Agent ${name} (${index + 1}/${total})`);
      break;
    }

    case "agent_end": {
      const name = String(event.data["name"] ?? "unknown");
      outputChannel.appendLine(`✓ Agent ${name} completed`);
      break;
    }

    case "tool_call": {
      const tool = String(event.data["tool"] ?? "unknown");
      const durationMs = event.data["duration_ms"];
      const duration = durationMs !== undefined ? ` (${durationMs}ms)` : "";
      outputChannel.appendLine(`  🔧 ${tool}${duration}`);
      break;
    }

    case "phase": {
      const phase = String(event.data["phase"] ?? "unknown");
      const strategy = event.data["strategy"]
        ? ` (${String(event.data["strategy"])})`
        : "";
      outputChannel.appendLine(`📋 Phase: ${phase}${strategy}`);
      break;
    }

    case "chunk": {
      const text = String(event.data["text"] ?? "");
      if (text) {
        outputChannel.append(text);
      }
      break;
    }

    case "report_html": {
      const html = String(event.data["html"] ?? "");
      if (html) {
        reportStore.lastReport = {
          html,
          serviceName,
          capturedAt: new Date(),
        };
        outputChannel.appendLine("\n📊 Report captured.");
      }
      break;
    }

    case "error": {
      const message = String(event.data["message"] ?? "Unknown error");
      outputChannel.appendLine(`\n❌ Error: ${message}`);
      void vscode.window.showErrorMessage(`VAIG: ${message}`);
      break;
    }

    case "result": {
      // Structured result — log summary, store for future use.
      outputChannel.appendLine("\n📦 Structured result received.");
      break;
    }

    case "done": {
      outputChannel.appendLine(
        `\n━━━ Diagnosis complete (${new Date().toLocaleTimeString()}) ━━━`,
      );
      break;
    }

    case "keepalive":
      // Silently ignore — per spec.
      break;
  }
}

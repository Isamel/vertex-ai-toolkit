/**
 * `vaig.openReport` command — open the last captured diagnosis report.
 *
 * Behavior (REQ-6):
 *   - If no report has been captured: show info message.
 *   - If a report exists: open/reveal the ReportPanel with the stored HTML.
 */

import * as vscode from "vscode";

import type { ReportPanelManager } from "../ui/report-panel.js";
import type { ReportStore } from "./run-live.js";

/**
 * Create and return the open-report command handler.
 *
 * @param reportPanelManager - Panel manager for showing reports.
 * @param reportStore - Store containing the last captured report.
 */
export function createOpenReportCommand(
  reportPanelManager: ReportPanelManager,
  reportStore: ReportStore,
): () => void {
  return () => {
    if (!reportStore.lastReport) {
      void vscode.window.showInformationMessage(
        "No report available. Run a live diagnosis first.",
      );
      return;
    }

    reportPanelManager.show(reportStore.lastReport.html);
  };
}

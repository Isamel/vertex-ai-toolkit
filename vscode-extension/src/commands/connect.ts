/**
 * `vaig.connect` command — prompt for URL and connect to the server.
 *
 * Behavior (REQ-4):
 *   - Shows InputBox pre-filled with current `vaig.serverUrl`.
 *   - Validates URL format before attempting connection.
 *   - On success: updates the `vaig.serverUrl` setting globally.
 *   - On failure: shows error message with option to change the URL.
 *   - If already connected to the SAME URL: shows info message.
 */

import * as vscode from "vscode";

import { getConfig, updateConfig } from "../config.js";
import type { ConnectionManager } from "../server/connection.js";

/**
 * Create and return the connect command handler.
 *
 * @param connectionManager - Shared connection manager instance.
 */
export function createConnectCommand(
  connectionManager: ConnectionManager,
): () => Promise<void> {
  return async () => {
    const currentUrl = getConfig().serverUrl;

    const url = await vscode.window.showInputBox({
      title: "VAIG Server URL",
      prompt: "Enter the VAIG server URL",
      value: currentUrl,
      placeHolder: "http://localhost:8080",
      validateInput: (value) => {
        try {
          const parsed = new URL(value);
          if (!["http:", "https:"].includes(parsed.protocol)) {
            return "URL must use http:// or https://";
          }
          return undefined;
        } catch {
          return "Invalid URL format";
        }
      },
    });

    // User pressed Escape.
    if (url === undefined) return;

    try {
      await connectionManager.connect(url);

      // Persist the new URL if it differs from the current setting.
      if (url !== currentUrl) {
        await updateConfig("serverUrl", url);
      }

      void vscode.window.showInformationMessage(
        `Connected to VAIG server at ${url}`,
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unknown connection error";

      const action = await vscode.window.showErrorMessage(
        `Failed to connect to VAIG: ${message}`,
        "Change URL",
      );

      if (action === "Change URL") {
        // Re-run the command — recursive via command palette.
        await vscode.commands.executeCommand("vaig.connect");
      }
    }
  };
}

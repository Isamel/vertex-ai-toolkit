/**
 * Typed wrapper around `vscode.workspace.getConfiguration('vaig')`.
 *
 * Every call reads the latest user/workspace settings — no caching,
 * so runtime changes via Settings UI take effect immediately.
 */

import * as vscode from "vscode";

import type { VaigConfig } from "./types.js";

const SECTION = "vaig";

/** Read the current `vaig.*` configuration. */
export function getConfig(): VaigConfig {
  const cfg = vscode.workspace.getConfiguration(SECTION);

  return {
    serverUrl: cfg.get<string>("serverUrl", "http://localhost:8080"),
    autoConnect: cfg.get<boolean>("autoConnect", true),
  };
}

/**
 * Update a single `vaig.*` setting at the global (user) level.
 *
 * Useful after a successful manual connect — persist the new URL so
 * subsequent sessions pick it up automatically.
 */
export async function updateConfig<K extends keyof VaigConfig>(
  key: K,
  value: VaigConfig[K],
): Promise<void> {
  const cfg = vscode.workspace.getConfiguration(SECTION);
  await cfg.update(key, value, vscode.ConfigurationTarget.Global);
}

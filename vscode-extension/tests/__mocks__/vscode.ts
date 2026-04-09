/**
 * Minimal vscode API mock for vitest.
 *
 * Only stubs the surface area actually used by the modules under test:
 * - workspace.getConfiguration
 * - window.createOutputChannel, showInformationMessage, showErrorMessage, showInputBox
 * - window.createStatusBarItem
 * - StatusBarAlignment
 * - EventEmitter / Event
 * - Disposable
 * - Uri.parse
 * - ConfigurationTarget
 */

import { vi } from "vitest";

// ── EventEmitter ──────────────────────────────────────────────

type Listener<T> = (e: T) => void;

export class EventEmitter<T> {
  private listeners: Listener<T>[] = [];

  readonly event = (listener: Listener<T>): { dispose: () => void } => {
    this.listeners.push(listener);
    return {
      dispose: () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      },
    };
  };

  fire(data: T): void {
    for (const listener of this.listeners) {
      listener(data);
    }
  }

  dispose(): void {
    this.listeners = [];
  }
}

// ── Disposable ────────────────────────────────────────────────

export class Disposable {
  constructor(private readonly callOnDispose: () => void) {}
  dispose(): void {
    this.callOnDispose();
  }
}

// ── StatusBarAlignment ────────────────────────────────────────

export const StatusBarAlignment = { Left: 1, Right: 2 } as const;

// ── ConfigurationTarget ───────────────────────────────────────

export const ConfigurationTarget = {
  Global: 1,
  Workspace: 2,
  WorkspaceFolder: 3,
} as const;

// ── Uri ───────────────────────────────────────────────────────

export const Uri = {
  parse: (value: string) => ({ toString: () => value, fsPath: value }),
  file: (path: string) => ({ toString: () => path, fsPath: path }),
};

// ── window ────────────────────────────────────────────────────

export const window = {
  createOutputChannel: vi.fn(() => ({
    appendLine: vi.fn(),
    append: vi.fn(),
    clear: vi.fn(),
    show: vi.fn(),
    hide: vi.fn(),
    dispose: vi.fn(),
    name: "mock-channel",
  })),
  showInformationMessage: vi.fn(),
  showErrorMessage: vi.fn(),
  showWarningMessage: vi.fn(),
  showInputBox: vi.fn(),
  createStatusBarItem: vi.fn(() => ({
    text: "",
    tooltip: "",
    color: undefined,
    command: undefined,
    show: vi.fn(),
    hide: vi.fn(),
    dispose: vi.fn(),
  })),
};

// ── workspace ─────────────────────────────────────────────────

const defaultConfigValues: Record<string, Record<string, unknown>> = {
  vaig: {
    serverUrl: "http://localhost:8080",
    autoConnect: true,
  },
};

export const workspace = {
  getConfiguration: vi.fn((section?: string) => {
    const sectionDefaults = section
      ? defaultConfigValues[section] ?? {}
      : {};
    return {
      get: vi.fn(
        <T>(key: string, defaultValue?: T): T =>
          (sectionDefaults[key] as T) ?? (defaultValue as T),
      ),
      update: vi.fn(),
      has: vi.fn((key: string) => key in sectionDefaults),
      inspect: vi.fn(),
    };
  }),
  onDidChangeConfiguration: vi.fn(() => ({ dispose: vi.fn() })),
};

// ── commands ──────────────────────────────────────────────────

export const commands = {
  registerCommand: vi.fn(),
  executeCommand: vi.fn(),
};

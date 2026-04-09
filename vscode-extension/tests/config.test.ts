/**
 * Unit tests for getConfig() and updateConfig().
 *
 * Mocks the vscode.workspace.getConfiguration API.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock vscode before any imports that reference it.
vi.mock("vscode", () => import("./__mocks__/vscode.js"));

import { getConfig, updateConfig } from "../src/config.js";
import { workspace, ConfigurationTarget } from "vscode";

describe("config", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("getConfig()", () => {
    it("returns default values when no user overrides exist", () => {
      const cfg = getConfig();

      expect(cfg).toEqual({
        serverUrl: "http://localhost:8080",
        autoConnect: true,
        devMode: true,
      });
    });

    it("calls workspace.getConfiguration with 'vaig' section", () => {
      getConfig();

      expect(workspace.getConfiguration).toHaveBeenCalledWith("vaig");
    });

    it("returns custom serverUrl when overridden", () => {
      vi.mocked(workspace.getConfiguration).mockReturnValueOnce({
        get: vi.fn((key: string, defaultValue?: unknown) => {
          if (key === "serverUrl") return "http://custom:9090";
          return defaultValue;
        }),
        update: vi.fn(),
        has: vi.fn(),
        inspect: vi.fn(),
      } as never);

      const cfg = getConfig();

      expect(cfg.serverUrl).toBe("http://custom:9090");
    });

    it("returns custom autoConnect when overridden to false", () => {
      vi.mocked(workspace.getConfiguration).mockReturnValueOnce({
        get: vi.fn((key: string, defaultValue?: unknown) => {
          if (key === "autoConnect") return false;
          return defaultValue;
        }),
        update: vi.fn(),
        has: vi.fn(),
        inspect: vi.fn(),
      } as never);

      const cfg = getConfig();

      expect(cfg.autoConnect).toBe(false);
    });

    it("returns all custom values when fully overridden", () => {
      vi.mocked(workspace.getConfiguration).mockReturnValueOnce({
        get: vi.fn((key: string) => {
          const overrides: Record<string, unknown> = {
            serverUrl: "https://prod.example.com",
            autoConnect: false,
            devMode: false,
          };
          return overrides[key];
        }),
        update: vi.fn(),
        has: vi.fn(),
        inspect: vi.fn(),
      } as never);

      const cfg = getConfig();

      expect(cfg).toEqual({
        serverUrl: "https://prod.example.com",
        autoConnect: false,
        devMode: false,
      });
    });

    it("reads fresh config on each call (no caching)", () => {
      getConfig();
      getConfig();
      getConfig();

      // Each call should invoke getConfiguration
      expect(workspace.getConfiguration).toHaveBeenCalledTimes(3);
    });
  });

  describe("updateConfig()", () => {
    it("updates a setting at Global level", async () => {
      const mockUpdate = vi.fn().mockResolvedValue(undefined);
      vi.mocked(workspace.getConfiguration).mockReturnValueOnce({
        get: vi.fn(),
        update: mockUpdate,
        has: vi.fn(),
        inspect: vi.fn(),
      } as never);

      await updateConfig("serverUrl", "http://new-server:8080");

      expect(mockUpdate).toHaveBeenCalledWith(
        "serverUrl",
        "http://new-server:8080",
        ConfigurationTarget.Global,
      );
    });
  });
});

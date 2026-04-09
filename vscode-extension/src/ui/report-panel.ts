/**
 * Report panel — displays VAIG diagnosis reports in a WebView.
 *
 * Singleton-ish: if the panel is already open, `show()` reveals it and
 * updates the content rather than creating a new panel.
 *
 * CSP policy (Design §Arch #2):
 *   - `default-src 'none'`
 *   - `script-src 'unsafe-inline' https://cdn.jsdelivr.net`
 *   - `style-src 'unsafe-inline'`
 *   - `img-src https: data:`
 *   - `font-src https:`
 *
 * Implements `vscode.Disposable` — push to `context.subscriptions`.
 */

import * as vscode from "vscode";

const VIEW_TYPE = "vaigReport";
const PANEL_TITLE = "VAIG Health Report";

export class ReportPanelManager implements vscode.Disposable {
  private panel: vscode.WebviewPanel | null = null;

  /**
   * Open (or reveal) the report panel and inject the given HTML.
   *
   * @param html - The full report HTML string received via SSE `report_html`.
   */
  show(html: string): void {
    if (this.panel) {
      // Reuse existing panel — update content and bring to front.
      this.panel.webview.html = this.wrapHtml(html, this.panel.webview);
      this.panel.reveal(vscode.ViewColumn.One);
      return;
    }

    // Create a new panel.
    this.panel = vscode.window.createWebviewPanel(
      VIEW_TYPE,
      PANEL_TITLE,
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
      },
    );

    this.panel.webview.html = this.wrapHtml(html, this.panel.webview);

    // Track lifecycle — null out reference when user closes the panel.
    this.panel.onDidDispose(() => {
      this.panel = null;
    });
  }

  /** Whether a report is currently being displayed. */
  get isVisible(): boolean {
    return this.panel !== null;
  }

  dispose(): void {
    this.panel?.dispose();
    this.panel = null;
  }

  // ── Private ─────────────────────────────────────────────────

  /**
   * Wrap the raw report HTML with a proper CSP meta tag.
   *
   * The report SPA uses inline styles and scripts, so we allow
   * `unsafe-inline`. External scripts are limited to jsdelivr CDN
   * (for Mermaid rendering).
   */
  private wrapHtml(rawHtml: string, webview: vscode.Webview): string {
    // Generate a nonce for additional CSP flexibility (not used for
    // inline scripts, but available if we add extension-owned scripts).
    const _webview = webview; // retain reference for future nonce use

    void _webview;

    const csp = [
      "default-src 'none'",
      "script-src 'unsafe-inline' https://cdn.jsdelivr.net",
      "style-src 'unsafe-inline'",
      "img-src https: data:",
      "font-src https:",
    ].join("; ");

    // If the HTML already has a <head>, inject CSP there.
    // Otherwise, wrap with a minimal document.
    if (rawHtml.includes("<head>")) {
      return rawHtml.replace(
        "<head>",
        `<head>\n<meta http-equiv="Content-Security-Policy" content="${csp}">`,
      );
    }

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${PANEL_TITLE}</title>
</head>
<body>
${rawHtml}
</body>
</html>`;
  }
}

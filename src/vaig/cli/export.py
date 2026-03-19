"""Export formatting — convert results to JSON, Markdown, and HTML."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from markdown_it import MarkdownIt

# ── Markdown renderer (module-level singleton) ───────────────────────────────
# commonmark preset + table + strikethrough; html=False for XSS protection.
_md_renderer: MarkdownIt = MarkdownIt("commonmark", {"html": False}).enable(
    ["table", "strikethrough"]
)

# ── Tokyo Night dark-theme HTML template ─────────────────────────────────────
# Uses the same color palette as src/vaig/ui/spa_template.html.
# Single placeholder: {content} — the rendered HTML fragment.
TOKYO_NIGHT_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>VAIG Analysis Report</title>
<style>
  /* ── Reset & Base ─────────────────────────────────────────── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg-base:    #1a1b26;
    --bg-surface: #24283b;
    --bg-muted:   #414868;
    --bg-hover:   #2f3549;
    --text-main:  #c0caf5;
    --text-dim:   #787c99;
    --text-bright:#e1e3f4;
    --border:     #3b3f5c;
    --accent:     #7aa2f7;
    --green:      #9ece6a;
    --yellow:     #e0af68;
    --red:        #f7768e;
    --cyan:       #7dcfff;
    --font-mono:  "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace;
    --font-sans:  system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --radius:     6px;
  }}

  html {{ scroll-behavior: smooth; }}

  body {{
    font-family: var(--font-sans);
    background: var(--bg-base);
    color: var(--text-main);
    line-height: 1.7;
    min-height: 100vh;
    padding: 2rem 1rem;
  }}

  /* ── Scrollbar ──────────────────────────────────────────── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg-base); }}
  ::-webkit-scrollbar-thumb {{ background: var(--bg-muted); border-radius: 3px; }}

  /* ── Content wrapper ────────────────────────────────────── */
  .content {{
    max-width: 900px;
    margin: 0 auto;
  }}

  /* ── Headings ───────────────────────────────────────────── */
  h1 {{
    font-size: 1.8rem;
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    margin-top: 0;
  }}
  h2 {{
    font-size: 1.3rem;
    color: var(--text-bright);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
  }}
  h3 {{
    font-size: 1.1rem;
    color: var(--cyan);
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
  }}
  h4, h5, h6 {{
    color: var(--text-bright);
    margin-top: 1rem;
    margin-bottom: 0.5rem;
  }}

  /* ── Paragraph & inline ─────────────────────────────────── */
  p {{ margin-bottom: 1rem; }}
  strong {{ color: var(--text-bright); font-weight: 700; }}
  em {{ color: var(--yellow); font-style: italic; }}
  del {{ color: var(--text-dim); text-decoration: line-through; }}
  a {{ color: var(--accent); text-decoration: underline; }}
  a:hover {{ color: var(--cyan); }}

  /* ── Code ───────────────────────────────────────────────── */
  code {{
    font-family: var(--font-mono);
    font-size: 0.875em;
    background: var(--bg-surface);
    color: var(--green);
    padding: 2px 6px;
    border-radius: 3px;
    border: 1px solid var(--border);
  }}
  pre {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    overflow-x: auto;
    margin-bottom: 1rem;
  }}
  pre code {{
    background: none;
    border: none;
    padding: 0;
    font-size: 0.85em;
    color: var(--text-main);
  }}

  /* ── Tables ─────────────────────────────────────────────── */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1.5rem;
    font-size: 0.9em;
  }}
  thead {{
    background: var(--bg-surface);
  }}
  th {{
    padding: 0.6rem 1rem;
    text-align: left;
    color: var(--accent);
    border-bottom: 2px solid var(--border);
    font-weight: 700;
  }}
  td {{
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-main);
  }}
  tr:hover td {{
    background: var(--bg-hover);
  }}

  /* ── Blockquote ─────────────────────────────────────────── */
  blockquote {{
    border-left: 4px solid var(--accent);
    padding: 0.5rem 1rem;
    margin: 1rem 0;
    background: var(--bg-surface);
    border-radius: 0 var(--radius) var(--radius) 0;
    color: var(--text-dim);
  }}
  blockquote p {{ margin-bottom: 0; }}

  /* ── Lists ──────────────────────────────────────────────── */
  ul, ol {{
    padding-left: 1.5rem;
    margin-bottom: 1rem;
  }}
  li {{ margin-bottom: 0.3rem; }}

  /* ── HR ─────────────────────────────────────────────────── */
  hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
  }}

  /* ── Print ──────────────────────────────────────────────── */
  @media print {{
    body {{ background: #fff; color: #000; }}
    h1, h2, h3 {{ color: #000; border-color: #ccc; }}
    pre, code {{ background: #f6f8fa; color: #000; border-color: #ccc; }}
    table {{ border: 1px solid #ccc; }}
    th {{ background: #f0f0f0; color: #000; }}
    td {{ border-color: #ccc; }}
    a {{ color: #0066cc; }}
  }}
</style>
</head>
<body>
<div class="content">
{content}
</div>
</body>
</html>"""


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    """Metadata included in every export."""

    model: str
    skill: str | None
    timestamp: str
    tokens: dict[str, int]
    cost: str | None
    vaig_version: str


@dataclass
class ExportPayload:
    """Structured export data that can be serialized to any format."""

    question: str
    response: str
    metadata: ExportMetadata
    context_files: list[str] = field(default_factory=list)
    agent_results: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Render as a Markdown report."""
        lines = [
            "# VAIG Analysis Report",
            "",
            f"**Model**: {self.metadata.model}  ",
        ]
        if self.metadata.skill:
            lines.append(f"**Skill**: {self.metadata.skill}  ")
        lines.append(f"**Date**: {self.metadata.timestamp}  ")
        if self.metadata.cost:
            lines.append(f"**Estimated Cost**: {self.metadata.cost}  ")
        tokens = self.metadata.tokens
        if tokens:
            lines.append(
                f"**Tokens**: {tokens.get('total_tokens', 0):,} total "
                f"({tokens.get('prompt_tokens', 0):,} prompt + "
                f"{tokens.get('completion_tokens', 0):,} completion)  "
            )
        lines.append("")

        # Question
        lines.extend(["## Question", "", self.question, ""])

        # Context files
        if self.context_files:
            lines.extend(["## Context Files", ""])
            for f in self.context_files:
                lines.append(f"- `{f}`")
            lines.append("")

        # Response
        lines.extend(["## Response", "", self.response, ""])

        # Agent results (for multi-agent)
        if self.agent_results:
            lines.extend(["## Agent Details", ""])
            for ar in self.agent_results:
                name = ar.get("agent_name", "unknown")
                content = ar.get("content", "")
                lines.extend([f"### {name}", "", content, ""])

        # Cost & Usage Summary section (detailed table at the end)
        cost_section = self._build_cost_summary_section()
        if cost_section:
            lines.append(cost_section)

        lines.extend(["---", f"*Generated by VAIG v{self.metadata.vaig_version}*"])
        return "\n".join(lines)

    def _build_cost_summary_section(self) -> str | None:
        """Build a ``## Cost & Usage Summary`` markdown section.

        Returns ``None`` if no meaningful token data is available.
        """
        tokens = self.metadata.tokens
        if not tokens:
            return None

        prompt_tokens = tokens.get("prompt_tokens", 0)
        completion_tokens = tokens.get("completion_tokens", 0)
        thinking_tokens = tokens.get("thinking_tokens", 0)
        total_tokens = tokens.get("total_tokens", 0)

        # Skip if everything is zero
        if total_tokens == 0 and prompt_tokens == 0 and completion_tokens == 0:
            return None

        lines = [
            "## Cost & Usage Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Input tokens | {prompt_tokens:,} |",
            f"| Output tokens | {completion_tokens:,} |",
        ]

        if thinking_tokens:
            lines.append(f"| Thinking tokens | {thinking_tokens:,} |")

        lines.append(f"| Total tokens | {total_tokens:,} |")
        lines.append(f"| Model | {self.metadata.model} |")

        if self.metadata.cost:
            lines.append(f"| Estimated cost | {self.metadata.cost} |")

        lines.append("")
        return "\n".join(lines)

    def to_html(self) -> str:
        """Render as a self-contained HTML report using Tokyo Night dark theme.

        Converts the full ``to_markdown()`` output through ``markdown-it-py``
        (commonmark + table + strikethrough; HTML injection disabled) and
        wraps the fragment in ``TOKYO_NIGHT_HTML_TEMPLATE``.
        """
        md_content = self.to_markdown()
        html_fragment = _md_renderer.render(md_content)
        return TOKYO_NIGHT_HTML_TEMPLATE.format(content=html_fragment)


def format_export(
    payload: ExportPayload,
    fmt: str,
) -> str:
    """Format an export payload in the requested format.

    Args:
        payload: The structured export data.
        fmt: Format string — "json", "md"/"markdown", or "html".

    Returns:
        Formatted string content.

    Raises:
        ValueError: If the format is not recognized.
    """
    fmt = fmt.lower().strip()
    if fmt == "json":
        return payload.to_json()
    if fmt in ("md", "markdown"):
        return payload.to_markdown()
    if fmt == "html":
        return payload.to_html()
    msg = f"Unsupported export format: {fmt!r}. Use 'json', 'md', or 'html'."
    raise ValueError(msg)

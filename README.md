# Claude Code Session Exporter

A zero-dependency Python CLI that exports [Claude Code](https://docs.anthropic.com/en/docs/claude-code) sessions into standalone, self-contained HTML files.

Sessions are rendered as editorial-style technical transcripts with full markdown support, syntax-highlighted code blocks, and collapsible thinking/tool-use sections.

## Features

- **Interactive TUI browser** — curses-based split-pane interface with vim-style navigation, real-time filtering, and live session preview
- **Multiple export modes** — browse interactively, list sessions, or export directly by UUID or file path
- **Rich HTML output** — distinct visual treatment for user messages, assistant responses, thinking blocks, tool calls, and tool results
- **Truly self-contained files** — marked.js, highlight.js, and the syntax theme are inlined into every export; no CDN, no web fonts, no network access required. Files render fully offline and reference nothing on disk
- **Native system fonts** — uses the fonts that ship with macOS (San Francisco, New York, SF Mono), so text looks right out of the box with no font downloads
- **Zero Python dependencies** — uses only the standard library

## Requirements

- Python 3.7+
- Claude Code sessions in `~/.claude/projects/`

## Usage

**Browse sessions interactively** (recommended):

```bash
python claude_export.py
python claude_export.py --browse -p "project-filter"
```

**List available sessions:**

```bash
python claude_export.py --list
python claude_export.py --list -p "myproject"
```

**Export a specific session:**

```bash
python claude_export.py <session-uuid>
python claude_export.py <session-uuid> -o output.html
python claude_export.py path/to/session.jsonl
```

Output files open in any modern browser and are fully self-contained: the markdown renderer (marked.js), syntax highlighter (highlight.js), and its theme are inlined at export time from the vendored copies in [`assets/`](assets/), and the UI relies on native macOS system fonts. No network connection is needed to view an export. If a vendored asset is missing at export time, the exporter falls back to loading that one resource from a CDN.

## Running Tests

```bash
python3 -m unittest discover -s tests
```

If your `python` already points to Python 3, you can also use:

```bash
python -m unittest discover -s tests
```

## License

[MIT](LICENSE)

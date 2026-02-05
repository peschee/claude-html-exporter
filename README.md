# Claude Code Session Exporter

A zero-dependency Python CLI that exports [Claude Code](https://docs.anthropic.com/en/docs/claude-code) sessions into standalone, self-contained HTML files.

Sessions are rendered as editorial-style technical transcripts with full markdown support, syntax-highlighted code blocks, and collapsible thinking/tool-use sections.

## Features

- **Interactive TUI browser** — curses-based split-pane interface with vim-style navigation, real-time filtering, and live session preview
- **Multiple export modes** — browse interactively, list sessions, or export directly by UUID or file path
- **Rich HTML output** — distinct visual treatment for user messages, assistant responses, thinking blocks, tool calls, and tool results
- **Standalone files** — generated HTML works in any modern browser with no local dependencies
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

Output files open in any modern browser. CDN resources (Tailwind CSS v4, marked.js, highlight.js, IBM Plex fonts) are loaded at view time.

## License

[MIT](LICENSE)

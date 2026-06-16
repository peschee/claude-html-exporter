# CLAUDE.md

Zero-dependency Python CLI that exports Claude Code sessions to standalone HTML.

## Commands
- Run TUI browser: `python claude_export.py`
- List sessions: `python claude_export.py --list [-p PROJECT_FILTER]`
- Export one: `python claude_export.py <session-uuid|path.jsonl> [-o out.html]`
- Tests: `python3 -m unittest discover -s tests`
- `--verbose` enables `_debug` stderr logging.

## Architecture
Single file `claude_export.py` (~1800 lines), split by `# ---` banner sections:
discovery → `resolve_session` → `parse_jsonl` → `build_conversation` →
assets/HTML generation → `SessionBrowser` (curses TUI) → `cmd_*` handlers → `main`.

- Reads sessions from `~/.claude/projects/` (`CLAUDE_DIR`).
- Prefers `sessions-index.json` per project; falls back to scanning `.jsonl` files.
- `assets/` (marked.js, highlight.js, theme) are inlined into each export at
  export time; falls back to a CDN URL per asset if a vendored file is missing.

## Constraints & Gotchas
- **Stdlib only** — do not add third-party dependencies.
- Tool results are truncated at `TRUNCATE_LIMIT` (50,000 chars).
- `build_conversation` skips sidechain messages.
- Exported `*.html` files are gitignored build artifacts.
- macOS-oriented: uses native system fonts (SF, New York, SF Mono) in output.

## Tests
`tests/test_claude_export.py` appends the repo root to `sys.path` and imports
`claude_export` directly. Run with `unittest discover -s tests`.

# CLAUDE.md

Zero-dependency Python CLI that exports Claude Code sessions to standalone HTML.

## Commands
- Run TUI browser: `python claude_export.py`
- List sessions: `python claude_export.py --list [-p PROJECT_FILTER]`
- Export one: `python claude_export.py <session-uuid|path.jsonl> [-o out.html]`
- Tests: `python3 -m unittest discover -s tests`
- `--verbose` enables `_debug` stderr logging.

## Architecture
Single file `claude_export.py` (~2700 lines), split by `# ---` banner sections:
discovery → `resolve_session` → `parse_jsonl` → `build_conversation` →
assets/HTML generation → `SessionBrowser` (curses TUI) → `cmd_*` handlers → `main`.

- Reads sessions from `~/.claude/projects/` (`CLAUDE_DIR`).
- Prefers `sessions-index.json` per project; falls back to scanning `.jsonl` files.
  In practice no project has an index, so `modified` comes from the file mtime.
- Session label: the last `ai-title` record in the file (read from the last
  64 KB), falling back to the first real human prompt. `_clean_prompt` unwraps
  `<command-name>` slash commands and strips caveat/system-reminder blocks.
- `assets/` (marked.js, highlight.js, theme) are inlined into each export at
  export time; falls back to a CDN URL per asset if a vendored file is missing.

## Constraints & Gotchas
- **Stdlib only** — do not add third-party dependencies.
- Tool results are truncated at `TRUNCATE_LIMIT` (50,000 chars). Image blocks
  (tool results and pasted prompt images) are embedded as base64 data URIs,
  capped at `IMAGE_LIMIT` (5 MB) each, so exports with screenshots get large.
- The TUI renders UTF-8; `_tui_safe_text` only replaces wide/emoji/control
  chars that would break the one-char-one-cell column math.
- Export HTML: sidebar of human prompts (`buildNav`), day dividers, Edit
  diffs, `@media print` opens all `details`, dark palette via
  `prefers-color-scheme`. All DOM is built with createElement/textContent;
  only markdown prose goes through innerHTML.
- `build_conversation` skips sidechain messages.
- Compaction leaves two records back to back: a `system`/`compact_boundary`
  line with `compactMetadata`, then a `user` line with `isCompactSummary`
  whose content is the summary Claude wrote. The pre-compaction history stays
  in the file. The exporter folds both into one `role: compaction` message
  (divider + collapsible summary) and the TUI stub/preview skip the summary
  so it never shows up as a human prompt.
- Exported `*.html` files are gitignored build artifacts.
- macOS-oriented: uses native system fonts (SF, New York, SF Mono) in output.
- marked v5+ removed the inline `highlight` option; highlighting runs as a
  post-render `hljs.highlightElement` pass over `pre code`. The hljs theme must
  be **dark** (github-dark) to match the `#282c34` code background, and
  `.prose pre` sets a base color so hljs token spans win on CSS specificity.

## Tests
`tests/test_claude_export.py` appends the repo root to `sys.path` and imports
`claude_export` directly. Run with `unittest discover -s tests`.

## Verifying / upgrading exports
- Verify self-containment: render `file:///path/export.html` (offline by
  definition) and `grep -coE '(src|href)="https?://'` should be 0.
- Re-vendor `assets/` from cdnjs (highlight.js 11.9.0 `highlight.min.js` +
  `styles/github-dark.min.css`) and jsdelivr (`marked@12/marked.min.js`).

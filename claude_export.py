#!/usr/bin/env python3
"""Claude Code Conversation Exporter

Reads Claude Code session JSONL files and generates standalone HTML files
with a clean, modern chat UI.

Usage:
    python claude_export.py --list [-p PROJECT_FILTER]
    python claude_export.py SESSION_ID [-o output.html]
    python claude_export.py path/to/session.jsonl [-o output.html]
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


CLAUDE_DIR = Path.home() / ".claude" / "projects"
TRUNCATE_LIMIT = 50_000


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def find_sessions(project_filter=None):
    """Scan ~/.claude/projects/ for sessions.

    Returns list of dicts with keys:
        session_id, project, path, first_prompt, created, modified, git_branch
    """
    sessions = []
    if not CLAUDE_DIR.exists():
        return sessions

    for project_dir in sorted(CLAUDE_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        if project_filter and project_filter.lower() not in project_name.lower():
            continue

        # Try sessions-index.json first
        index_path = project_dir / "sessions-index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
                for entry in index.get("entries", []):
                    sessions.append({
                        "session_id": entry.get("sessionId", ""),
                        "project": project_name,
                        "project_path": entry.get("projectPath", ""),
                        "path": entry.get("fullPath", ""),
                        "first_prompt": entry.get("firstPrompt", ""),
                        "created": entry.get("created", ""),
                        "modified": entry.get("modified", ""),
                        "git_branch": entry.get("gitBranch", ""),
                    })
                continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: scan .jsonl files directly
        for jsonl_path in sorted(project_dir.glob("*.jsonl")):
            session_id = jsonl_path.stem
            info = _read_session_stub(jsonl_path)
            sessions.append({
                "session_id": session_id,
                "project": project_name,
                "project_path": "",
                "path": str(jsonl_path),
                "first_prompt": info.get("first_prompt", ""),
                "created": info.get("created", ""),
                "modified": "",
                "git_branch": info.get("git_branch", ""),
            })

    return sessions


def _read_session_stub(path):
    """Read first user line from a JSONL to extract basic info."""
    try:
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "user":
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    prompt = content if isinstance(content, str) else ""
                    return {
                        "first_prompt": prompt[:200],
                        "created": obj.get("timestamp", ""),
                        "git_branch": obj.get("gitBranch", ""),
                    }
    except Exception:
        pass
    return {}


def resolve_session(arg):
    """Resolve a session argument to a full .jsonl file path.

    Accepts:
        - A direct file path (absolute or relative)
        - A session UUID (searches all projects)
    """
    # Direct path
    p = Path(arg)
    if p.exists() and p.suffix == ".jsonl":
        return str(p.resolve())

    # UUID lookup
    for project_dir in CLAUDE_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        candidate = project_dir / f"{arg}.jsonl"
        if candidate.exists():
            return str(candidate)

    print(f"Error: Could not find session '{arg}'", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def parse_jsonl(path):
    """Read and parse all lines from a JSONL file."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return lines


def extract_metadata(lines):
    """Extract session metadata from parsed JSONL lines."""
    meta = {
        "session_id": "",
        "project": "",
        "model": "",
        "date": "",
        "git_branch": "",
        "cwd": "",
    }
    for obj in lines:
        if not meta["session_id"] and obj.get("sessionId"):
            meta["session_id"] = obj["sessionId"]
        if not meta["git_branch"] and obj.get("gitBranch"):
            meta["git_branch"] = obj["gitBranch"]
        if not meta["cwd"] and obj.get("cwd"):
            meta["cwd"] = obj["cwd"]
        if not meta["date"] and obj.get("timestamp"):
            meta["date"] = obj["timestamp"]
        if obj.get("type") == "assistant":
            msg = obj.get("message", {})
            if not meta["model"] and msg.get("model"):
                meta["model"] = msg["model"]

    return meta


# ---------------------------------------------------------------------------
# Conversation building
# ---------------------------------------------------------------------------

def build_conversation(lines):
    """Build a clean conversation from JSONL lines.

    Returns list of message dicts:
        {role, timestamp, blocks: [{type, text, ...}]}
    """
    # Pass 1: collect assistant messages, merging by message id
    assistant_msgs = {}  # msg_id -> {obj with merged content}
    tool_map = {}  # tool_use_id -> tool_name

    relevant = [
        obj for obj in lines
        if obj.get("type") in ("user", "assistant") and not obj.get("isSidechain")
    ]

    # Pass 1: merge assistant messages by id and build tool map
    for obj in relevant:
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message", {})
        msg_id = msg.get("id", "")
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        if msg_id not in assistant_msgs:
            assistant_msgs[msg_id] = {
                "blocks": [],
                "seen_types": set(),
                "timestamp": obj.get("timestamp", ""),
                "stop_reason": msg.get("stop_reason"),
            }

        entry = assistant_msgs[msg_id]
        if msg.get("stop_reason"):
            entry["stop_reason"] = msg["stop_reason"]
        entry["timestamp"] = obj.get("timestamp", "") or entry["timestamp"]

        for block in content:
            btype = block.get("type", "")
            # Deduplicate: text and thinking blocks may repeat across streamed chunks
            if btype == "text":
                text = block.get("text", "")
                if text.strip():
                    # Replace any existing text block (later one is more complete)
                    entry["blocks"] = [
                        b for b in entry["blocks"] if b.get("type") != "text"
                    ]
                    entry["blocks"].append({"type": "text", "text": text})
            elif btype == "thinking":
                thinking = block.get("thinking", "")
                if thinking.strip():
                    entry["blocks"] = [
                        b for b in entry["blocks"] if b.get("type") != "thinking"
                    ]
                    entry["blocks"].insert(0, {"type": "thinking", "text": thinking})
            elif btype == "tool_use":
                tool_id = block.get("id", "")
                tool_name = block.get("name", "unknown")
                tool_map[tool_id] = tool_name
                # Only add if not already present (by tool id)
                existing_ids = {
                    b.get("tool_id") for b in entry["blocks"] if b.get("type") == "tool_use"
                }
                if tool_id not in existing_ids:
                    entry["blocks"].append({
                        "type": "tool_use",
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "input": block.get("input", {}),
                    })

    # Pass 2: build ordered conversation
    conversation = []
    seen_assistant_ids = set()

    for obj in relevant:
        ts = obj.get("timestamp", "")

        if obj.get("type") == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, str):
                if content.strip():
                    conversation.append({
                        "role": "user",
                        "timestamp": ts,
                        "blocks": [{"type": "text", "text": content}],
                    })
            elif isinstance(content, list):
                # Check if this is a tool_result response or a user prompt
                has_tool_result = any(b.get("type") == "tool_result" for b in content)
                if has_tool_result:
                    # Attach tool results to the previous assistant message
                    tool_results = []
                    for block in content:
                        if block.get("type") == "tool_result":
                            tool_results.append(_normalize_tool_result(block, tool_map))
                    if tool_results and conversation and conversation[-1]["role"] == "assistant":
                        conversation[-1]["blocks"].extend(tool_results)
                    elif tool_results:
                        # No preceding assistant msg; add standalone
                        conversation.append({
                            "role": "tool",
                            "timestamp": ts,
                            "blocks": tool_results,
                        })
                else:
                    # User prompt with text blocks
                    texts = []
                    for block in content:
                        if block.get("type") == "text":
                            t = block.get("text", "")
                            if t.strip():
                                texts.append(t)
                    if texts:
                        conversation.append({
                            "role": "user",
                            "timestamp": ts,
                            "blocks": [{"type": "text", "text": "\n\n".join(texts)}],
                        })

        elif obj.get("type") == "assistant":
            msg = obj.get("message", {})
            msg_id = msg.get("id", "")
            if msg_id in seen_assistant_ids:
                continue
            seen_assistant_ids.add(msg_id)
            if msg_id in assistant_msgs:
                entry = assistant_msgs[msg_id]
                if entry["blocks"]:
                    conversation.append({
                        "role": "assistant",
                        "timestamp": entry["timestamp"],
                        "blocks": entry["blocks"],
                    })

    return conversation


def _normalize_tool_result(block, tool_map):
    """Normalize a tool_result block into a clean dict."""
    tool_use_id = block.get("tool_use_id", "")
    tool_name = tool_map.get(tool_use_id, "unknown")
    is_error = block.get("is_error", False)
    content = block.get("content", "")

    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "\n".join(parts)
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    if len(text) > TRUNCATE_LIMIT:
        text = text[:TRUNCATE_LIMIT] + f"\n\n... [truncated, {len(text)} chars total]"

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "is_error": is_error,
        "text": text,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(messages, metadata):
    """Produce a complete standalone HTML string."""
    # Escape </script> in JSON payload
    json_data = json.dumps({"messages": messages, "metadata": metadata}, ensure_ascii=False)
    json_data = json_data.replace("</", "<\\/")

    date_str = metadata.get("date", "")
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            date_display = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            date_display = date_str
    else:
        date_display = "Unknown"

    title = f"Claude Code Session — {date_display}"

    return HTML_TEMPLATE.replace("{{JSON_DATA}}", json_data).replace("{{TITLE}}", title)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}}</title>
<script src="https://cdn.tailwindcss.com/4"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body { font-family: 'Inter', system-ui, sans-serif; }

/* Markdown prose styles */
.prose { line-height: 1.7; }
.prose p { margin: 0.5em 0; }
.prose p:first-child { margin-top: 0; }
.prose p:last-child { margin-bottom: 0; }
.prose ul, .prose ol { margin: 0.5em 0; padding-left: 1.5em; }
.prose ul { list-style-type: disc; }
.prose ol { list-style-type: decimal; }
.prose li { margin: 0.25em 0; }
.prose h1, .prose h2, .prose h3, .prose h4 { font-weight: 600; margin: 1em 0 0.5em; }
.prose h1 { font-size: 1.5em; }
.prose h2 { font-size: 1.25em; }
.prose h3 { font-size: 1.1em; }
.prose pre { margin: 0.75em 0; border-radius: 0.5rem; overflow-x: auto; }
.prose code { font-size: 0.9em; }
.prose :not(pre) > code {
    background: rgba(0,0,0,0.06); padding: 0.15em 0.35em; border-radius: 0.25rem;
}
.prose pre > code { background: none; padding: 0; }
.prose blockquote {
    border-left: 3px solid #d1d5db; padding-left: 1em; margin: 0.75em 0;
    color: #6b7280; font-style: italic;
}
.prose table { border-collapse: collapse; margin: 0.75em 0; width: 100%; }
.prose th, .prose td { border: 1px solid #e5e7eb; padding: 0.4em 0.75em; text-align: left; }
.prose th { background: #f9fafb; font-weight: 600; }
.prose a { color: #2563eb; text-decoration: underline; }
.prose hr { border: none; border-top: 1px solid #e5e7eb; margin: 1em 0; }
.prose img { max-width: 100%; border-radius: 0.5rem; }

/* User bubble prose overrides */
.user-prose :not(pre) > code { background: rgba(255,255,255,0.2); }
.user-prose a { color: #bfdbfe; }
.user-prose blockquote { border-left-color: rgba(255,255,255,0.4); color: #bfdbfe; }
.user-prose th, .user-prose td { border-color: rgba(255,255,255,0.2); }
.user-prose th { background: rgba(255,255,255,0.1); }

/* Details/summary styles */
details summary { cursor: pointer; user-select: none; }
details summary::-webkit-details-marker { display: none; }
details summary::before {
    content: '\25B6'; display: inline-block; margin-right: 0.5em;
    transition: transform 0.2s; font-size: 0.7em; vertical-align: middle;
}
details[open] summary::before { transform: rotate(90deg); }

/* Tool content */
.tool-content { max-height: 400px; overflow-y: auto; }
.tool-content pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body class="bg-gray-50 min-h-screen">

<script type="application/json" id="conversation-data">{{JSON_DATA}}</script>

<div id="app"></div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const raw = document.getElementById('conversation-data').textContent;
    const data = JSON.parse(raw);
    const messages = data.messages;
    const metadata = data.metadata;

    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try { return hljs.highlight(code, {language: lang}).value; } catch(e) {}
            }
            try { return hljs.highlightAuto(code).value; } catch(e) {}
            return code;
        },
        breaks: false,
        gfm: true,
    });

    const app = document.getElementById('app');

    // Header
    const header = document.createElement('div');
    header.className = 'sticky top-0 z-10 bg-white/80 backdrop-blur-sm border-b border-gray-200 shadow-sm';

    const headerInner = document.createElement('div');
    headerInner.className = 'max-w-4xl mx-auto px-4 py-3';

    const headerRow = document.createElement('div');
    headerRow.className = 'flex items-center justify-between flex-wrap gap-2';

    // Left side: logo + title
    const headerLeft = document.createElement('div');
    headerLeft.className = 'flex items-center gap-3';

    const logo = document.createElement('div');
    logo.className = 'w-8 h-8 bg-amber-500 rounded-lg flex items-center justify-center';
    const logoText = document.createElement('span');
    logoText.className = 'text-white font-bold text-sm';
    logoText.textContent = 'CC';
    logo.appendChild(logoText);

    const titleBlock = document.createElement('div');
    const h1 = document.createElement('h1');
    h1.className = 'text-sm font-semibold text-gray-900';
    h1.textContent = 'Claude Code Session';
    titleBlock.appendChild(h1);
    if (metadata.cwd) {
        const sub = document.createElement('p');
        sub.className = 'text-xs text-gray-500';
        sub.textContent = metadata.cwd;
        titleBlock.appendChild(sub);
    }

    headerLeft.appendChild(logo);
    headerLeft.appendChild(titleBlock);

    // Right side: metadata chips
    const headerRight = document.createElement('div');
    headerRight.className = 'flex items-center gap-4 text-xs text-gray-500';

    if (metadata.model) {
        const chip = document.createElement('span');
        chip.className = 'bg-gray-100 px-2 py-1 rounded font-mono';
        chip.textContent = metadata.model;
        headerRight.appendChild(chip);
    }
    if (metadata.git_branch) {
        const chip = document.createElement('span');
        chip.className = 'bg-gray-100 px-2 py-1 rounded';
        chip.textContent = 'branch: ' + metadata.git_branch;
        headerRight.appendChild(chip);
    }
    if (metadata.date) {
        const chip = document.createElement('span');
        chip.textContent = formatDate(metadata.date);
        headerRight.appendChild(chip);
    }

    headerRow.appendChild(headerLeft);
    headerRow.appendChild(headerRight);
    headerInner.appendChild(headerRow);
    header.appendChild(headerInner);
    app.appendChild(header);

    // Chat container
    const chat = document.createElement('div');
    chat.className = 'max-w-4xl mx-auto px-4 py-6 space-y-4';
    app.appendChild(chat);

    // Render messages
    messages.forEach(function(msg) {
        if (msg.role === 'user') {
            renderUserMessage(chat, msg);
        } else if (msg.role === 'assistant') {
            renderAssistantMessage(chat, msg);
        }
    });

    // Scroll padding at bottom
    const spacer = document.createElement('div');
    spacer.className = 'h-16';
    chat.appendChild(spacer);
});

function renderUserMessage(container, msg) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-end';

    const bubble = document.createElement('div');
    bubble.className = 'max-w-[85%] bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm';

    msg.blocks.forEach(function(block) {
        if (block.type === 'text') {
            const div = document.createElement('div');
            div.className = 'prose user-prose text-sm text-white';
            div.innerHTML = renderMarkdown(block.text);
            bubble.appendChild(div);
        }
    });

    const ts = document.createElement('div');
    ts.className = 'text-[10px] text-blue-200 mt-1 text-right';
    ts.textContent = formatTime(msg.timestamp);
    bubble.appendChild(ts);

    wrapper.appendChild(bubble);
    container.appendChild(wrapper);
}

function renderAssistantMessage(container, msg) {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-start';

    const card = document.createElement('div');
    card.className = 'max-w-[85%] space-y-2';

    msg.blocks.forEach(function(block) {
        if (block.type === 'text') {
            const div = document.createElement('div');
            div.className = 'bg-white rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-gray-100';
            const prose = document.createElement('div');
            prose.className = 'prose text-sm text-gray-800';
            prose.innerHTML = renderMarkdown(block.text);
            div.appendChild(prose);
            card.appendChild(div);
        } else if (block.type === 'thinking') {
            card.appendChild(createCollapsible('Thinking', block.text, 'purple'));
        } else if (block.type === 'tool_use') {
            var label = block.tool_name || 'Tool';
            var body = '';
            if (block.input) {
                try { body = JSON.stringify(block.input, null, 2); }
                catch(e) { body = String(block.input); }
            }
            card.appendChild(createCollapsible(label, body, 'amber'));
        } else if (block.type === 'tool_result') {
            var resultLabel = (block.tool_name || 'Result');
            var isError = block.is_error;
            var color = isError ? 'red' : 'green';
            var prefix = isError ? 'Error' : 'Result';
            card.appendChild(createCollapsible(prefix + ': ' + resultLabel, block.text || '(empty)', color));
        }
    });

    const ts = document.createElement('div');
    ts.className = 'text-[10px] text-gray-400 mt-1';
    ts.textContent = formatTime(msg.timestamp);
    card.appendChild(ts);

    wrapper.appendChild(card);
    container.appendChild(wrapper);
}

function createCollapsible(title, content, color) {
    var colors = {
        purple: { bg: 'bg-purple-50', border: 'border-purple-200', title: 'text-purple-700', dot: 'bg-purple-400' },
        amber:  { bg: 'bg-amber-50',  border: 'border-amber-200',  title: 'text-amber-700',  dot: 'bg-amber-400'  },
        green:  { bg: 'bg-green-50',  border: 'border-green-200',  title: 'text-green-700',  dot: 'bg-green-400'  },
        red:    { bg: 'bg-red-50',    border: 'border-red-200',    title: 'text-red-700',    dot: 'bg-red-400'    },
    };
    var c = colors[color] || colors.purple;

    var details = document.createElement('details');
    details.className = c.bg + ' ' + c.border + ' border rounded-xl overflow-hidden';

    var summary = document.createElement('summary');
    summary.className = 'px-3 py-2 text-xs font-medium ' + c.title + ' flex items-center gap-2';

    var dot = document.createElement('span');
    dot.className = 'w-2 h-2 rounded-full ' + c.dot + ' inline-block';
    summary.appendChild(dot);
    summary.appendChild(document.createTextNode(title));
    details.appendChild(summary);

    var body = document.createElement('div');
    body.className = 'tool-content px-3 pb-3';
    var pre = document.createElement('pre');
    pre.className = 'text-xs text-gray-700 whitespace-pre-wrap break-words';
    pre.textContent = content;
    body.appendChild(pre);
    details.appendChild(body);

    return details;
}

function renderMarkdown(text) {
    // Use marked to render markdown, then return the HTML string.
    // Note: This renders user-owned local data, not untrusted web content.
    return marked.parse(text);
}

function formatDate(isoStr) {
    try {
        var d = new Date(isoStr);
        return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
    } catch(e) { return isoStr; }
}

function formatTime(isoStr) {
    try {
        var d = new Date(isoStr);
        return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch(e) { return ''; }
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_list(args):
    """List available sessions."""
    sessions = find_sessions(project_filter=args.project)
    if not sessions:
        print("No sessions found.")
        return

    # Group by project
    by_project = {}
    for s in sessions:
        by_project.setdefault(s["project"], []).append(s)

    for project, sess_list in sorted(by_project.items()):
        print(f"\n  {project}")
        print(f"  {'─' * 60}")
        for s in sess_list:
            date = s.get("created", "")[:10] or "???"
            prompt = s.get("first_prompt", "")[:80]
            sid = s["session_id"][:12]
            branch = s.get("git_branch", "")
            branch_str = f" [{branch}]" if branch else ""
            print(f"  {sid}  {date}{branch_str}  {prompt}")

    print()


def cmd_export(args):
    """Export a session to HTML."""
    path = resolve_session(args.session)
    lines = parse_jsonl(path)
    metadata = extract_metadata(lines)
    messages = build_conversation(lines)

    if not messages:
        print("No messages found in session.", file=sys.stderr)
        sys.exit(1)

    html = generate_html(messages, metadata)

    if args.output:
        output_path = args.output
    else:
        sid = metadata.get("session_id", "session")[:12]
        output_path = f"claude-session-{sid}.html"

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Exported {len(messages)} messages to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Claude Code sessions to standalone HTML files."
    )

    parser.add_argument("--list", action="store_true", help="List available sessions")
    parser.add_argument("-p", "--project", help="Filter projects by name substring")
    parser.add_argument("session", nargs="?", help="Session UUID or path to .jsonl file")
    parser.add_argument("-o", "--output", help="Output HTML file path")

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    elif args.session:
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

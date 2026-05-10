#!/usr/bin/env python3
"""Build the read-only data bundle for the Academic vault home dashboard."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote


VAULT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = VAULT / "98_attachment" / "vault-home"
OUTPUT_HTML = OUTPUT_DIR / "index.html"
OUTPUT_JS = OUTPUT_DIR / "dashboard-data.js"
CSS_FILE = OUTPUT_DIR / "dashboard.css"
JS_FILE = OUTPUT_DIR / "dashboard.js"
VAULT_NAME = "Academic"

SKIP_DIRS = {
    ".git",
    ".obsidian",
    ".codex",
    ".claude",
    ".trash",
    "node_modules",
    "__pycache__",
}

COURSE_ROOTS = {
    "01_Math": "Math",
    "02_Economy": "Economy",
    "03_Computer_Science": "Computer Science",
}

QUICK_LINK_CANDIDATES = [
    ("Study Record", "Overview & Study Record.md", "global"),
    ("Inbox", "00_inbox", "capture"),
    ("Daily Notes", "99_学习情况记录", "log"),
    ("Time Series Hub", "00_factor/00_hub/Time Series Analysis-hub.md", "hub"),
    ("Time Series Atlas", "01_Math/06_时间序列分析/Time Series Course Atlas.canvas", "canvas"),
    ("Linear Algebra Hub", "00_factor/00_hub/Linear Algebra-hub.md", "hub"),
    ("Econometrics Hub", "00_factor/00_hub/Econometrics-hub.md", "hub"),
]


def rel(path: Path) -> str:
    return path.relative_to(VAULT).as_posix()


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(VAULT).parts[:-1])


def iter_files(*suffixes: str) -> list[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    files: list[Path] = []
    for path in VAULT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        if path.suffix.lower() in suffix_set:
            files.append(path)
    return files


def obsidian_uri(vault_path: str, action: str = "open") -> str:
    if action == "daily":
        return f"obsidian://daily?vault={quote(VAULT_NAME)}"
    if action == "search":
        return f"obsidian://search?vault={quote(VAULT_NAME)}&query={quote('path:' + vault_path)}"
    return f"obsidian://open?vault={quote(VAULT_NAME)}&file={quote(vault_path, safe='')}"


def file_entry(path: Path) -> dict:
    relative = rel(path)
    return {
        "title": path.stem,
        "path": relative,
        "uri": obsidian_uri(relative),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def folder_entry(path: Path, title: str | None = None) -> dict:
    relative = rel(path)
    return {
        "title": title or path.name,
        "path": relative,
        "uri": obsidian_uri(relative, "search"),
    }


def run_git_status() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=VAULT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def today_data() -> dict:
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    now = datetime.now()
    filename = f"{now:%Y-%m-%d}——{weekdays[now.weekday()]}.md"
    daily_path = VAULT / "99_学习情况记录" / filename
    inbox = VAULT / "00_inbox"
    inbox_items = [p for p in inbox.iterdir() if p.is_file()] if inbox.exists() else []
    if daily_path.exists():
        daily_note = file_entry(daily_path)
    else:
        daily_note = {
            "title": daily_path.stem,
            "path": rel(daily_path),
            "uri": obsidian_uri("", "daily"),
            "mtime": None,
        }
    daily_note.update({"exists": daily_path.exists(), "dailyUri": obsidian_uri("", "daily")})
    return {
        "date": now.strftime("%Y-%m-%d"),
        "weekday": weekdays[now.weekday()],
        "dailyNote": daily_note,
        "inbox": {**folder_entry(inbox, "Inbox"), "count": len(inbox_items)},
    }


def course_data() -> list[dict]:
    courses: list[dict] = []
    for root_name, label in COURSE_ROOTS.items():
        root = VAULT / root_name
        if not root.exists():
            continue
        course_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        for course_dir in course_dirs:
            files = [p for p in course_dir.rglob("*") if p.is_file() and not should_skip(p)]
            notes = [p for p in files if p.suffix.lower() == ".md"]
            canvases = [p for p in files if p.suffix.lower() == ".canvas"]
            pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
            recent = sorted(notes + canvases, key=lambda p: p.stat().st_mtime, reverse=True)[:4]
            newest = max((p.stat().st_mtime for p in notes + canvases), default=None)
            courses.append(
                {
                    "title": course_dir.name,
                    "root": label,
                    "path": rel(course_dir),
                    "uri": folder_entry(course_dir)["uri"],
                    "noteCount": len(notes),
                    "canvasCount": len(canvases),
                    "pdfCount": len(pdfs),
                    "hasBigPicture": len(canvases) > 0,
                    "lastTouched": datetime.fromtimestamp(newest).isoformat(timespec="seconds") if newest else None,
                    "recent": [file_entry(p) for p in recent],
                }
            )
    return sorted(courses, key=lambda item: (item["root"], item["title"].lower()))


def factase_data() -> dict:
    root = VAULT / "00_factor"
    categories = []
    if root.exists():
        folders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        for folder in folders:
            notes = sorted(folder.glob("*.md"))
            recent = sorted(notes, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
            categories.append(
                {
                    "title": folder.name,
                    "path": rel(folder),
                    "uri": folder_entry(folder)["uri"],
                    "count": len(notes),
                    "recent": [file_entry(p) for p in recent],
                }
            )
    hubs = [item for item in categories if item["title"] == "00_hub"]
    hub_notes = []
    hub_dir = root / "00_hub"
    if hub_dir.exists():
        hub_notes = [file_entry(p) for p in sorted(hub_dir.glob("*.md"), key=lambda p: p.name.lower())]
    return {
        "total": sum(item["count"] for item in categories),
        "categories": categories,
        "hubCount": hubs[0]["count"] if hubs else 0,
        "hubs": hub_notes,
    }


def canvas_data() -> list[dict]:
    canvases = []
    for path in sorted(iter_files(".canvas"), key=lambda p: rel(p).lower()):
        node_count = None
        edge_count = None
        valid = True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            node_count = len(payload.get("nodes", []))
            edge_count = len(payload.get("edges", []))
        except Exception:
            valid = False
        canvases.append(
            {
                **file_entry(path),
                "nodeCount": node_count,
                "edgeCount": edge_count,
                "valid": valid,
            }
        )
    return canvases


def recent_notes(limit: int = 14) -> list[dict]:
    notes = sorted(iter_files(".md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [file_entry(path) for path in notes[:limit]]


def quick_links() -> list[dict]:
    links = []
    for title, relative, kind in QUICK_LINK_CANDIDATES:
        path = VAULT / relative
        if path.exists():
            base = folder_entry(path, title) if path.is_dir() else file_entry(path)
            base.update({"title": title, "kind": kind})
            links.append(base)
    return links


def health(markdown_files: list[Path], canvas_files: list[Path]) -> dict:
    git_lines = run_git_status()
    suffix_counts = Counter(
        p.suffix.lower() or "[none]"
        for p in VAULT.rglob("*")
        if p.is_file() and not should_skip(p)
    )
    return {
        "gitDirty": len(git_lines),
        "gitLines": git_lines[:16],
        "topFileTypes": [{"suffix": suffix, "count": count} for suffix, count in suffix_counts.most_common(8)],
        "notes": len(markdown_files),
        "canvases": len(canvas_files),
    }


def build_payload() -> dict:
    markdown_files = iter_files(".md")
    canvas_files = iter_files(".canvas")
    courses = course_data()
    canvases = canvas_data()
    facts = factase_data()
    generated = datetime.now().isoformat(timespec="seconds")
    return {
        "generatedAt": generated,
        "counts": {
            "notes": len(markdown_files),
            "canvases": len(canvas_files),
            "factaseCards": facts["total"],
            "courseFolders": len(courses),
            "coursesWithCanvas": sum(1 for course in courses if course["hasBigPicture"]),
        },
        "today": today_data(),
        "courses": courses,
        "factase": facts,
        "canvases": canvases,
        "recentNotes": recent_notes(),
        "quickLinks": quick_links(),
        "health": health(markdown_files, canvas_files),
    }


def h(value: object) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def short_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%d %b, %H:%M")
    except ValueError:
        return value


def render_meta(items: list[str]) -> str:
    return "".join(f'<span class="pill">{h(item)}</span>' for item in items if item)


def render_link_card(item: dict, class_name: str = "note-link", extra_meta: list[str] | None = None) -> str:
    meta = list(extra_meta or [])
    if item.get("path"):
        meta.append(item["path"])
    if item.get("mtime"):
        meta.append(short_time(item["mtime"]))
    return f"""
      <a class="{class_name}" href="{h(item.get("uri", "#"))}">
        <strong>{h(item.get("title", "Untitled"))}</strong>
        <div class="meta">{render_meta(meta)}</div>
      </a>
    """


def render_actions(payload: dict) -> str:
    return "\n".join(
        f'<a class="action-link" href="{h(item["uri"])}">{h(item["title"])}</a>'
        for item in payload["quickLinks"][:5]
    )


def render_metrics(payload: dict) -> str:
    metrics = [
        (payload["counts"]["notes"], "Markdown notes"),
        (payload["counts"]["factaseCards"], "Factase cards"),
        (payload["counts"]["canvases"], "Canvas maps"),
        (
            f'{payload["counts"]["coursesWithCanvas"]}/{payload["counts"]["courseFolders"]}',
            "Courses with maps",
        ),
    ]
    return "\n".join(
        f'<div class="metric"><strong>{h(value)}</strong><span>{h(label)}</span></div>'
        for value, label in metrics
    )


def render_today(payload: dict) -> str:
    today = payload["today"]["dailyNote"]
    today_title = "Open today's daily note" if today["exists"] else "Create today's daily note"
    return "\n".join(
        [
            render_link_card(
                {
                    "title": today_title,
                    "path": today["path"],
                    "uri": today["dailyUri"],
                    "mtime": today.get("mtime"),
                }
            ),
            render_link_card(
                {
                    "title": f'Inbox · {payload["today"]["inbox"]["count"]} items',
                    "path": payload["today"]["inbox"]["path"],
                    "uri": payload["today"]["inbox"]["uri"],
                }
            ),
        ]
    )


def render_courses(payload: dict) -> str:
    roots = ["Math", "Economy", "Computer Science"]
    lanes = []
    for root in roots:
        courses = [course for course in payload["courses"] if course["root"] == root]
        courses.sort(key=lambda item: (not item["hasBigPicture"], -item["noteCount"], item["title"].lower()))
        cards = []
        for course in courses[:7]:
            meta = [
                f'{course["noteCount"]} notes',
                f'{course["canvasCount"]} canvas',
            ]
            if course["pdfCount"]:
                meta.append(f'{course["pdfCount"]} pdf')
            recent = "\n".join(
                f'<a href="{h(item["uri"])}">{h(item["title"])}</a>'
                for item in course["recent"][:2]
            )
            canvas_class = " has-canvas" if course["hasBigPicture"] else ""
            cards.append(
                f"""
                <div class="course-card{canvas_class}">
                  <a class="course-title" href="{h(course["uri"])}"><strong>{h(course["title"])}</strong></a>
                  <div class="meta">{render_meta(meta)}</div>
                  <div class="recent-mini">{recent}</div>
                </div>
                """
            )
        lanes.append(
            f"""
            <section class="course-lane">
              <div class="lane-title"><span>{h(root)}</span><span>{len(courses)}</span></div>
              {"".join(cards)}
            </section>
            """
        )
    return "\n".join(lanes)


def render_factase(payload: dict) -> str:
    return "\n".join(
        f"""
        <a class="fact-card" href="{h(category["uri"])}">
          <strong>{h(category["count"])}</strong>
          <span>{h(category["title"])}</span>
        </a>
        """
        for category in payload["factase"]["categories"]
    )


def render_canvases(payload: dict) -> str:
    return "\n".join(
        render_link_card(
            canvas,
            "canvas-item",
            [f'{canvas.get("nodeCount", "?")} nodes', f'{canvas.get("edgeCount", "?")} edges'],
        )
        for canvas in payload["canvases"]
    )


def render_recent(payload: dict) -> str:
    return "\n".join(render_link_card(item) for item in payload["recentNotes"][:10])


def render_health(payload: dict) -> str:
    rows = [
        ("Git changed files", payload["health"]["gitDirty"]),
        ("Tracked notes", payload["health"]["notes"]),
        ("Tracked canvases", payload["health"]["canvases"]),
        ("Generated", short_time(payload["generatedAt"])),
    ]
    rendered = [
        f'<div class="health-row"><span>{h(label)}</span><strong>{h(value)}</strong></div>'
        for label, value in rows
    ]
    if payload["health"]["gitLines"]:
        rendered.append(
            f"""
            <div class="note-link">
              <strong>Current git surface</strong>
              <div class="meta">{render_meta(payload["health"]["gitLines"][:8])}</div>
            </div>
            """
        )
    return "\n".join(rendered)


def render_index_html(payload: dict) -> str:
    css = CSS_FILE.read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Academic Vault Home</title>
    <style>
{css}
    </style>
  </head>
  <body>
    <main class="vault-shell">
      <header class="topbar">
        <div>
          <p class="eyebrow">Academic Vault</p>
          <h1>Vault Home</h1>
        </div>
        <nav class="top-actions" aria-label="Quick links">
          {render_actions(payload)}
        </nav>
      </header>

      <section class="hero-board" aria-label="Vault overview">
        <div class="hero-copy">
          <p class="eyebrow">Today</p>
          <h2>{h(payload["today"]["date"])} {h(payload["today"]["weekday"])}</h2>
          <p class="muted">Generated {h(short_time(payload["generatedAt"]))}</p>
        </div>
        <div class="metric-strip">{render_metrics(payload)}</div>
      </section>

      <section class="atlas-grid">
        <article class="panel panel-today">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Today entry</h2>
          </div>
          <div class="stack">{render_today(payload)}</div>
        </article>

        <article class="panel panel-map">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Learning map</h2>
          </div>
          <div class="course-map">{render_courses(payload)}</div>
        </article>

        <article class="panel panel-factase">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Factase</h2>
          </div>
          <div class="factase-grid">{render_factase(payload)}</div>
        </article>

        <article class="panel panel-canvas">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Big Picture canvases</h2>
          </div>
          <div class="canvas-list">{render_canvases(payload)}</div>
        </article>

        <article class="panel panel-recent">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Recent notes</h2>
          </div>
          <div class="link-list">{render_recent(payload)}</div>
        </article>

        <article class="panel panel-health">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Vault health</h2>
          </div>
          <div class="stack">{render_health(payload)}</div>
        </article>
      </section>
    </main>
  </body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    OUTPUT_JS.write_text(f"window.__VAULT_HOME_DATA__ = {data};\n", encoding="utf-8")
    OUTPUT_HTML.write_text(render_index_html(payload), encoding="utf-8")
    print(f"Wrote {rel(OUTPUT_JS)}")
    print(f"Wrote {rel(OUTPUT_HTML)}")


if __name__ == "__main__":
    main()

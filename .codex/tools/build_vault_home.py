#!/usr/bin/env python3
"""Build the read-only data bundle for the Academic vault home dashboard."""

from __future__ import annotations

import json
import re
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

LESSON_RE = re.compile(r"^(?P<num>\d{1,2})[_-](?P<title>.+)\.md$")
SECTION_RE = re.compile(r"^(?P<num>\d+)[_-]")
LESSON_EXCLUDE_KEYWORDS = (
    "作业",
    "考试",
    "划重点",
    "补充",
    "course map",
    "exam",
    "review",
    "roadmap",
    "index",
    "main",
    "零散",
)

STATUS_ORDER = {
    "not-started": 0,
    "material-ready": 1,
    "attended": 2,
    "notes-readable": 3,
    "mapped": 4,
    "reviewable": 5,
}

STATUS_META = {
    "not-started": {"mark": "○", "label": "未开始"},
    "material-ready": {"mark": "◌", "label": "材料已入库"},
    "attended": {"mark": "◔", "label": "听过但未消化"},
    "notes-readable": {"mark": "◕", "label": "课程笔记可读"},
    "mapped": {"mark": "●", "label": "已接入 Big Picture"},
    "reviewable": {"mark": "⬤", "label": "可复习 / 可输出"},
}

FLAG_META = {
    "blocked": {"mark": "!", "label": "卡住 / 需要回看"},
    "unclear": {"mark": "?", "label": "概念不清楚"},
    "needs-map": {"mark": "↗", "label": "需要补 Big Picture 连接"},
    "has-factase": {"mark": "◆", "label": "已有 Factase 支撑"},
    "needs-review": {"mark": "⟳", "label": "需要复习"},
}

HUB_HINTS = (
    ("时间序列", "Time Series Analysis-hub.md"),
    ("linear algebra", "Linear Algebra-hub.md"),
    ("多元统计", "Multivariate Statistics-hub.md"),
    ("计量", "Econometrics-hub.md"),
    ("game theory", "Game Theory-hub.md"),
    ("财务管理", "Financial Management-hub.md"),
    ("金融机构", "Interest Rate Risk Management-hub.md"),
    ("风险管理", "Capital and Risk Management-hub.md"),
    ("证券投资", "VaR-hub.md"),
)


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


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def is_lesson_note(path: Path) -> bool:
    name = path.name.lower()
    if not LESSON_RE.match(path.name):
        return False
    return not any(keyword in name for keyword in LESSON_EXCLUDE_KEYWORDS)


def lesson_sort_label(path: Path, course_dir: Path) -> tuple[float, str]:
    match = LESSON_RE.match(path.name)
    number = int(match.group("num")) if match else 999
    parent = path.parent.relative_to(course_dir)
    if parent.parts:
        section_match = SECTION_RE.match(parent.parts[0])
        if section_match:
            return (int(section_match.group("num")) + number / 100, path.name)
    return (number, path.name)


def lesson_display_label(path: Path, course_dir: Path) -> str:
    match = LESSON_RE.match(path.name)
    number = match.group("num") if match else "?"
    parent = path.parent.relative_to(course_dir)
    if parent.parts:
        section_match = SECTION_RE.match(parent.parts[0])
        if section_match:
            return f"{int(section_match.group('num'))}.{number.zfill(2)}"
    return number.zfill(2)


def lesson_display_title(path: Path) -> str:
    match = LESSON_RE.match(path.name)
    if not match:
        return path.stem
    return match.group("title").replace("-", " ").replace("_", " ").strip()


def course_canvas_text(course_dir: Path) -> str:
    chunks: list[str] = []
    for canvas in course_dir.rglob("*.canvas"):
        chunks.append(read_text(canvas))
    return "\n".join(chunks)


def course_hub(course_title: str) -> dict | None:
    lower_title = course_title.lower()
    hub_dir = VAULT / "00_factor" / "00_hub"
    for hint, filename in HUB_HINTS:
        if hint.lower() in lower_title:
            hub_path = hub_dir / filename
            if hub_path.exists():
                return file_entry(hub_path)
    return None


def infer_lesson_status(path: Path, course_dir: Path, canvas_text: str) -> tuple[str, list[str]]:
    content = read_text(path)
    relative = rel(path)
    status = "attended" if len(content.strip()) < 800 else "notes-readable"
    haystacks = [relative, path.name, path.stem, lesson_display_title(path)]
    if any(needle and needle in canvas_text for needle in haystacks):
        status = "mapped"

    flags: list[str] = []
    if status == "notes-readable":
        flags.append("needs-map")
    if any(keyword in content[:1200] for keyword in ("TODO", "不清楚", "待补", "???")):
        flags.append("unclear")
    if status in {"mapped", "reviewable"}:
        days_since_edit = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
        if days_since_edit > 21:
            flags.append("needs-review")
    return status, flags


def make_missing_lesson(course_dir: Path, number: int) -> dict:
    label = str(number).zfill(2)
    return {
        "label": label,
        "title": f"Lecture {label}",
        "path": "",
        "uri": "",
        "status": "not-started",
        "statusMark": STATUS_META["not-started"]["mark"],
        "statusLabel": STATUS_META["not-started"]["label"],
        "flags": [],
        "flagMarks": [],
        "missing": True,
        "sortKey": number,
    }


def lesson_entry(path: Path, course_dir: Path, canvas_text: str, hub: dict | None) -> dict:
    status, flags = infer_lesson_status(path, course_dir, canvas_text)
    if hub and status in {"notes-readable", "mapped", "reviewable"}:
        flags.append("has-factase")
    label = lesson_display_label(path, course_dir)
    sort_key = lesson_sort_label(path, course_dir)[0]
    unique_flags = list(dict.fromkeys(flags))
    return {
        **file_entry(path),
        "label": label,
        "title": lesson_display_title(path),
        "status": status,
        "statusMark": STATUS_META[status]["mark"],
        "statusLabel": STATUS_META[status]["label"],
        "flags": unique_flags,
        "flagMarks": [FLAG_META[flag]["mark"] for flag in unique_flags],
        "missing": False,
        "sortKey": sort_key,
    }


def course_rail_data(course_dir: Path, root_label: str) -> dict | None:
    lesson_paths = sorted(
        [path for path in course_dir.rglob("*.md") if is_lesson_note(path)],
        key=lambda path: lesson_sort_label(path, course_dir),
    )
    if len(lesson_paths) < 2:
        return None

    canvas_text = course_canvas_text(course_dir)
    hub = course_hub(course_dir.name)
    lessons = [lesson_entry(path, course_dir, canvas_text, hub) for path in lesson_paths]

    top_level_numbers = {
        int(LESSON_RE.match(path.name).group("num")): path
        for path in lesson_paths
        if path.parent == course_dir and LESSON_RE.match(path.name)
    }
    if len(top_level_numbers) >= 3:
        low, high = min(top_level_numbers), max(top_level_numbers)
        if 0 <= high - low <= 40:
            by_number = {int(lesson["label"]): lesson for lesson in lessons if lesson["label"].isdigit()}
            merged = []
            for number in range(low, high + 1):
                merged.append(by_number.get(number) or make_missing_lesson(course_dir, number))
            nested = [lesson for lesson in lessons if not lesson["label"].isdigit()]
            lessons = sorted(merged + nested, key=lambda item: item["sortKey"])

    status_counts = Counter(lesson["status"] for lesson in lessons)
    flag_counts = Counter(flag for lesson in lessons for flag in lesson["flags"])
    processed = sum(
        1
        for lesson in lessons
        if STATUS_ORDER[lesson["status"]] >= STATUS_ORDER["notes-readable"]
    )
    mapped = sum(
        1
        for lesson in lessons
        if STATUS_ORDER[lesson["status"]] >= STATUS_ORDER["mapped"]
    )
    return {
        "title": course_dir.name,
        "root": root_label,
        "path": rel(course_dir),
        "uri": folder_entry(course_dir)["uri"],
        "hub": hub,
        "lessons": lessons,
        "lessonCount": len(lessons),
        "processedCount": processed,
        "mappedCount": mapped,
        "statusCounts": dict(status_counts),
        "flagCounts": dict(flag_counts),
        "needsMapCount": flag_counts.get("needs-map", 0),
        "needsReviewCount": flag_counts.get("needs-review", 0),
    }


def progress_data() -> dict:
    rails: list[dict] = []
    for root_name, label in COURSE_ROOTS.items():
        root = VAULT / root_name
        if not root.exists():
            continue
        for course_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            rail = course_rail_data(course_dir, label)
            if rail:
                rails.append(rail)

    totals = Counter()
    focus_nodes = []
    for rail in rails:
        for lesson in rail["lessons"]:
            totals[lesson["status"]] += 1
            for flag in lesson["flags"]:
                totals[flag] += 1
            if "needs-map" in lesson["flags"] or "unclear" in lesson["flags"] or "needs-review" in lesson["flags"]:
                focus_nodes.append(
                    {
                        "course": rail["title"],
                        "root": rail["root"],
                        "label": lesson["label"],
                        "title": lesson["title"],
                        "statusLabel": lesson["statusLabel"],
                        "flags": lesson["flags"],
                        "flagMarks": lesson["flagMarks"],
                        "uri": lesson.get("uri", rail["uri"]),
                    }
                )

    focus_nodes.sort(
        key=lambda item: (
            "needs-map" not in item["flags"],
            "unclear" not in item["flags"],
            item["root"],
            item["course"],
            item["label"],
        )
    )
    return {
        "rails": rails,
        "totals": {
            "courses": len(rails),
            "lessons": sum(rail["lessonCount"] for rail in rails),
            "processed": sum(rail["processedCount"] for rail in rails),
            "mapped": sum(rail["mappedCount"] for rail in rails),
            "needsMap": totals["needs-map"],
            "needsReview": totals["needs-review"],
            "unclear": totals["unclear"],
            "notStarted": totals["not-started"],
        },
        "focus": focus_nodes[:8],
        "statusMeta": STATUS_META,
        "flagMeta": FLAG_META,
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
        "progress": progress_data(),
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
    progress_links = [
        item
        for item in payload["quickLinks"]
        if item.get("kind") in {"global", "hub", "canvas"}
    ]
    pinned = "\n".join(
        f'<a class="action-link" href="{h(item["uri"])}">{h(item["title"])}</a>'
        for item in progress_links[:3]
    )
    return "\n".join(
        [
            '<button class="action-link button-link" type="button" id="open-course-manager">管理课程</button>',
            pinned,
        ]
    )


def render_metrics(payload: dict) -> str:
    progress = payload["progress"]["totals"]
    metrics = [
        (progress["courses"], "course rails"),
        (progress["lessons"], "lesson points"),
        (progress["processed"], "notes readable+"),
        (progress["needsMap"], "need map links"),
    ]
    return "\n".join(
        f'<div class="metric"><strong>{h(value)}</strong><span>{h(label)}</span></div>'
        for value, label in metrics
    )


def render_compact_stats(payload: dict) -> str:
    progress = payload["progress"]["totals"]
    stats = [
        (progress["courses"], "courses"),
        (progress["lessons"], "lessons"),
        (progress["processed"], "readable+"),
        (progress["mapped"], "mapped"),
        (short_time(payload["generatedAt"]), "generated"),
    ]
    return "\n".join(
        f'<span class="summary-pill"><strong>{h(value)}</strong>{h(label)}</span>'
        for value, label in stats
    )


def render_status_legend(payload: dict) -> str:
    rows = []
    for key in ("not-started", "material-ready", "attended", "notes-readable", "mapped", "reviewable"):
        item = payload["progress"]["statusMeta"][key]
        rows.append(
            f"""
            <div class="legend-row">
              <span class="legend-node status-{h(key)}">{h(item["mark"])}</span>
              <span>{h(item["label"])}</span>
            </div>
            """
        )
    return "\n".join(rows)


def render_flag_legend(payload: dict) -> str:
    rows = []
    for key in ("blocked", "unclear", "needs-map", "has-factase", "needs-review"):
        item = payload["progress"]["flagMeta"][key]
        rows.append(
            f"""
            <div class="legend-row">
              <span class="flag-sample flag-{h(key)}">{h(item["mark"])}</span>
              <span>{h(item["label"])}</span>
            </div>
            """
        )
    return "\n".join(rows)


def render_rail_node(lesson: dict) -> str:
    if lesson["missing"]:
        return f"""
        <span class="rail-node status-not-started is-missing" title="{h(lesson["statusLabel"])}">
          <span class="node-mark">{h(lesson["statusMark"])}</span>
          <span class="node-label">{h(lesson["label"])}</span>
        </span>
        """
    flags = "".join(
        f'<span class="node-flag flag-{h(flag)}">{h(FLAG_META[flag]["mark"])}</span>'
        for flag in lesson["flags"]
    )
    flag_labels = " · ".join(FLAG_META[flag]["label"] for flag in lesson["flags"])
    title = f'{lesson["label"]} {lesson["title"]}｜{lesson["statusLabel"]}'
    if flag_labels:
        title += f"｜{flag_labels}"
    return f"""
    <a class="rail-node status-{h(lesson["status"])}" href="{h(lesson["uri"])}" title="{h(title)}">
      <span class="node-mark">{h(lesson["statusMark"])}</span>
      <span class="node-label">{h(lesson["label"])}</span>
      <span class="node-flags">{flags}</span>
    </a>
    """


def render_progress_rails(payload: dict) -> str:
    rails = []
    for index, rail in enumerate(payload["progress"]["rails"], start=1):
        nodes = "\n".join(render_rail_node(lesson) for lesson in rail["lessons"])
        meta = [
            f'{rail["processedCount"]}/{rail["lessonCount"]} processed',
            f'{rail["mappedCount"]} mapped',
        ]
        if rail["needsMapCount"]:
            meta.append(f'{rail["needsMapCount"]} need map')
        if rail["needsReviewCount"]:
            meta.append(f'{rail["needsReviewCount"]} review')
        hub_link = ""
        if rail.get("hub"):
            hub_link = f'<a class="rail-hub" href="{h(rail["hub"]["uri"])}">Hub</a>'
        rails.append(
            f"""
            <div class="rail-row" data-course-path="{h(rail["path"])}" data-course-title="{h(rail["title"])}" data-course-root="{h(rail["root"])}" style="order: {index}">
              <div class="rail-course">
                <a class="rail-title" href="{h(rail["uri"])}">{h(rail["title"])}</a>
                <div class="rail-sub">
                  <span class="rail-root-label">{h(rail["root"])}</span>
                  {hub_link}
                </div>
              </div>
              <div class="rail-track">{nodes}</div>
              <div class="rail-meta">{render_meta(meta)}</div>
            </div>
            """
        )
    return "\n".join(rails)


def render_course_manager() -> str:
    return """
      <div class="course-manager" id="course-manager" hidden>
        <div class="manager-card" role="dialog" aria-modal="true" aria-labelledby="course-manager-title">
          <div class="manager-head">
            <div>
              <p class="eyebrow">Manual layer</p>
              <h2 id="course-manager-title">管理课程轨道</h2>
            </div>
            <button class="icon-button" type="button" id="close-course-manager" aria-label="Close">×</button>
          </div>

          <div class="manager-grid">
            <section class="manager-section">
              <h3>调整已有课程</h3>
              <p class="manager-hint">显示、改名和排序只影响这个看板，不改原笔记。</p>
              <div class="manager-list" id="course-manager-list"></div>
            </section>

            <section class="manager-section">
              <h3>手动添加课程</h3>
              <p class="manager-hint">用于暂时还没有被自动扫描到的课程，先占一个轨道。</p>
              <label class="manager-field">
                <span>课程名</span>
                <input id="manual-course-title" type="text" placeholder="例如 Advanced Macroeconomics" />
              </label>
              <label class="manager-field">
                <span>分类</span>
                <input id="manual-course-root" type="text" placeholder="Math / Economy / CS / Custom" />
              </label>
              <label class="manager-field">
                <span>节数</span>
                <input id="manual-course-lessons" type="number" min="1" max="60" value="12" />
              </label>
              <label class="manager-field">
                <span>可选路径</span>
                <input id="manual-course-path" type="text" placeholder="01_Math/..." />
              </label>
              <button class="primary-button" type="button" id="add-manual-course">添加到看板</button>
              <div class="manual-course-list" id="manual-course-list"></div>
            </section>
          </div>

          <div class="manager-actions">
            <button class="primary-button" type="button" id="save-course-manager">保存看板调整</button>
            <button class="ghost-button" type="button" id="reset-course-manager">恢复自动扫描</button>
          </div>
        </div>
      </div>
    """


def render_interaction_script() -> str:
    return r"""
    <script>
      (() => {
        const storageKey = "vault-home-course-board-v1";
        const manager = document.getElementById("course-manager");
        const list = document.getElementById("course-manager-list");
        const manualList = document.getElementById("manual-course-list");
        const railMap = document.getElementById("rail-map");
        const baseRows = Array.from(document.querySelectorAll(".rail-row[data-course-path]"));

        const escapeHtml = (value) =>
          String(value ?? "").replace(/[&<>"']/g, (char) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
          })[char]);

        const obsidianOpen = (path) => {
          if (!path) return "#";
          return `obsidian://open?vault=Academic&file=${encodeURIComponent(path)}`;
        };

        const readConfig = () => {
          try {
            return JSON.parse(localStorage.getItem(storageKey) || "{}");
          } catch {
            return {};
          }
        };

        const writeConfig = (config) => {
          localStorage.setItem(storageKey, JSON.stringify(config));
        };

        const defaults = () => baseRows.map((row, index) => ({
          path: row.dataset.coursePath,
          title: row.dataset.courseTitle,
          root: row.dataset.courseRoot,
          visible: true,
          order: index + 1,
        }));

        const mergedCourses = (config) => {
          const saved = new Map((config.courses || []).map((course) => [course.path, course]));
          return defaults().map((course) => ({ ...course, ...(saved.get(course.path) || {}) }));
        };

        const renderManualRows = (manualCourses) => {
          railMap.querySelectorAll(".rail-row.is-manual").forEach((row) => row.remove());
          manualCourses.forEach((course, index) => {
            const count = Math.max(1, Math.min(Number(course.lessonCount) || 1, 60));
            const nodes = Array.from({ length: count }, (_, i) => {
              const label = String(i + 1).padStart(2, "0");
              return `
                <span class="rail-node status-not-started is-missing" title="手动占位">
                  <span class="node-mark">○</span>
                  <span class="node-label">${label}</span>
                </span>
              `;
            }).join("");
            const row = document.createElement("div");
            row.className = "rail-row is-manual";
            row.style.order = Number(course.order) || 9000 + index;
            row.innerHTML = `
              <div class="rail-course">
                <a class="rail-title" href="${escapeHtml(obsidianOpen(course.path))}">${escapeHtml(course.title || "Untitled course")}</a>
                <div class="rail-sub">
                  <span class="rail-root-label">${escapeHtml(course.root || "Manual")}</span>
                  <span class="rail-hub manual-badge">Manual</span>
                </div>
              </div>
              <div class="rail-track">${nodes}</div>
              <div class="rail-meta"><span class="pill">${count} planned</span></div>
            `;
            railMap.appendChild(row);
          });
        };

        const applyConfig = () => {
          const config = readConfig();
          const courses = mergedCourses(config);
          const byPath = new Map(courses.map((course) => [course.path, course]));
          baseRows.forEach((row, index) => {
            const course = byPath.get(row.dataset.coursePath);
            row.hidden = course?.visible === false;
            row.style.order = Number(course?.order) || index + 1;
            const title = row.querySelector(".rail-title");
            const root = row.querySelector(".rail-root-label");
            if (title) title.textContent = course?.title || row.dataset.courseTitle;
            if (root) root.textContent = course?.root || row.dataset.courseRoot;
          });
          renderManualRows(config.manualCourses || []);
        };

        const renderManager = () => {
          const config = readConfig();
          const courses = mergedCourses(config);
          list.innerHTML = courses.map((course) => `
            <div class="manager-course" data-path="${escapeHtml(course.path)}">
              <label class="manager-check">
                <input type="checkbox" data-field="visible" ${course.visible === false ? "" : "checked"} />
                <span>显示</span>
              </label>
              <input type="number" data-field="order" value="${escapeHtml(course.order)}" min="1" />
              <input type="text" data-field="title" value="${escapeHtml(course.title)}" />
              <span class="manager-path">${escapeHtml(course.path)}</span>
            </div>
          `).join("");
          const manualCourses = config.manualCourses || [];
          manualList.innerHTML = manualCourses.length ? manualCourses.map((course, index) => `
            <div class="manual-course-editor" data-index="${index}">
              <button class="delete-manual-course" type="button" data-delete-manual="${index}">删除</button>
              <input type="number" data-manual-field="order" value="${escapeHtml(course.order ?? (9000 + index))}" min="1" />
              <input type="text" data-manual-field="title" value="${escapeHtml(course.title || "")}" />
              <input type="text" data-manual-field="root" value="${escapeHtml(course.root || "Manual")}" />
              <input type="number" data-manual-field="lessonCount" value="${escapeHtml(course.lessonCount || 12)}" min="1" max="60" />
              <input type="text" data-manual-field="path" value="${escapeHtml(course.path || "")}" placeholder="optional path" />
            </div>
          `).join("") : '<div class="manager-empty">还没有手动课程。</div>';
        };

        const openManager = () => {
          renderManager();
          manager.hidden = false;
        };

        const collectManagerConfig = () => ({
          ...readConfig(),
          courses: Array.from(list.querySelectorAll(".manager-course")).map((row) => ({
            path: row.dataset.path,
            visible: row.querySelector('[data-field="visible"]').checked,
            order: Number(row.querySelector('[data-field="order"]').value) || 1,
            title: row.querySelector('[data-field="title"]').value.trim() || row.dataset.path,
          })),
          manualCourses: Array.from(manualList.querySelectorAll(".manual-course-editor")).map((row, index) => ({
            id: Number(row.dataset.index) || Date.now() + index,
            order: Number(row.querySelector('[data-manual-field="order"]').value) || 9000 + index,
            title: row.querySelector('[data-manual-field="title"]').value.trim() || "Untitled course",
            root: row.querySelector('[data-manual-field="root"]').value.trim() || "Manual",
            lessonCount: Number(row.querySelector('[data-manual-field="lessonCount"]').value) || 12,
            path: row.querySelector('[data-manual-field="path"]').value.trim(),
          })),
        });

        const closeManager = () => {
          manager.hidden = true;
        };

        document.getElementById("open-course-manager")?.addEventListener("click", openManager);
        document.getElementById("close-course-manager")?.addEventListener("click", closeManager);
        manager?.addEventListener("click", (event) => {
          if (event.target === manager) closeManager();
        });

        document.getElementById("save-course-manager")?.addEventListener("click", () => {
          const config = collectManagerConfig();
          writeConfig(config);
          applyConfig();
          closeManager();
        });

        document.getElementById("reset-course-manager")?.addEventListener("click", () => {
          localStorage.removeItem(storageKey);
          applyConfig();
          closeManager();
        });

        document.getElementById("add-manual-course")?.addEventListener("click", () => {
          const title = document.getElementById("manual-course-title").value.trim();
          if (!title) return;
          const config = manager.hidden ? readConfig() : collectManagerConfig();
          const manualCourses = config.manualCourses || [];
          manualCourses.push({
            id: Date.now(),
            title,
            root: document.getElementById("manual-course-root").value.trim() || "Manual",
            lessonCount: Number(document.getElementById("manual-course-lessons").value) || 12,
            path: document.getElementById("manual-course-path").value.trim(),
            order: 9000 + manualCourses.length,
          });
          config.manualCourses = manualCourses;
          writeConfig(config);
          applyConfig();
          renderManager();
          ["manual-course-title", "manual-course-root", "manual-course-path"].forEach((id) => {
            document.getElementById(id).value = "";
          });
        });

        manualList?.addEventListener("click", (event) => {
          const button = event.target.closest("[data-delete-manual]");
          if (!button) return;
          const index = Number(button.dataset.deleteManual);
          const config = readConfig();
          config.manualCourses = (config.manualCourses || []).filter((_, itemIndex) => itemIndex !== index);
          writeConfig(config);
          renderManager();
          applyConfig();
        });

        applyConfig();
      })();
    </script>
    """


def render_focus(payload: dict) -> str:
    items = []
    for item in payload["progress"]["focus"]:
        flags = "".join(
            f'<span class="node-flag flag-{h(flag)}">{h(FLAG_META[flag]["mark"])}</span>'
            for flag in item["flags"]
        )
        items.append(
            f"""
            <a class="focus-item" href="{h(item["uri"])}">
              <strong>{h(item["course"])} · {h(item["label"])}</strong>
              <span>{h(item["title"])}</span>
              <div class="meta">
                <span class="pill">{h(item["statusLabel"])}</span>
                {flags}
              </div>
            </a>
            """
        )
    if not items:
        return '<div class="empty-state">No immediate processing flags detected.</div>'
    return "\n".join(items)


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
          <h1>Course Board</h1>
        </div>
        <nav class="top-actions" aria-label="Quick links">
          {render_actions(payload)}
        </nav>
      </header>

      <section class="board-toolbar" aria-label="Board controls">
        <div>
          <p class="eyebrow">Course Rail Map</p>
          <h2>一门课一条轨道，一节课一个点。</h2>
        </div>
        <div class="summary-strip">{render_compact_stats(payload)}</div>
      </section>

      <section class="progress-layout">
        <article class="panel panel-legend board-legend">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>State grammar</h2>
          </div>
          <div class="legend-grid">
            <div>{render_status_legend(payload)}</div>
            <div>{render_flag_legend(payload)}</div>
          </div>
        </article>

        <article class="panel rail-panel">
          <div class="panel-head">
            <span class="dot"></span>
            <h2>Course Rail Map</h2>
          </div>
          <div class="rail-map" id="rail-map">{render_progress_rails(payload)}</div>
        </article>
      </section>

      {render_course_manager()}
    </main>
    {render_interaction_script()}
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

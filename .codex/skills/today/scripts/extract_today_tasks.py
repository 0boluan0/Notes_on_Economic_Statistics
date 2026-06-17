#!/usr/bin/env python3
"""
Extract task blocks for the `today` skill.

Outputs markdown by default so it can be pasted directly into a note.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
WEEKDAY_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_WEEK1_START = dt.date(2026, 2, 2)
DAILY_NOTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}——[A-Za-z]{3}\.md$")
PLACEHOLDER_TASKS = {
    "暂无历史未完成任务",
    "未识别到今日任务，请手动检查 `Overview & Study Record.md`",
    "从 `99_学习情况记录/workbench.md` 的 `## 当前焦点` 添加今天要做的事",
}


def parse_date(raw: Optional[str]) -> dt.date:
    if not raw:
        return dt.date.today()
    return dt.datetime.strptime(raw, "%Y-%m-%d").date()


def sanitize_cell(text: str) -> str:
    cleaned = re.sub(r"[*`]+", "", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def is_moved_task_line(line: str) -> bool:
    return "转移至" in line or "~~" in line


def is_placeholder_task(text: str) -> bool:
    cleaned = text.strip()
    if cleaned in PLACEHOLDER_TASKS:
        return True
    return "未识别到今日任务" in cleaned or "暂无历史未完成任务" in cleaned


def find_unfinished_tasks(diary_dir: Path, exclude_filename: Optional[str]) -> List[str]:
    task_pattern = re.compile(r"^\s*-\s\[\s\]\s+(.*\S)\s*$")
    tasks: List[str] = []
    if not diary_dir.exists():
        return tasks

    for note in sorted(diary_dir.glob("*.md")):
        if not DAILY_NOTE_RE.match(note.name):
            continue
        if exclude_filename and note.name == exclude_filename:
            continue
        text = note.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            match = task_pattern.match(line)
            if not match or is_moved_task_line(line):
                continue
            task_text = match.group(1).strip()
            if is_placeholder_task(task_text):
                continue
            tasks.append(task_text)
    return dedupe_keep_order(tasks)


def find_tasks_moved_to_date(
    diary_dir: Path, exclude_filename: Optional[str], target_date: dt.date
) -> List[str]:
    moved_pattern = re.compile(
        r"^\s*-\s*~~(.*?)~~\s*[（(]\s*转移至\s*(\d{4}-\d{2}-\d{2})\s*[)）]\s*$"
    )
    tasks: List[str] = []
    if not diary_dir.exists():
        return tasks

    for note in sorted(diary_dir.glob("*.md")):
        if not DAILY_NOTE_RE.match(note.name):
            continue
        if exclude_filename and note.name == exclude_filename:
            continue
        text = note.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            match = moved_pattern.match(line)
            if not match:
                continue
            task_text = match.group(1).strip()
            moved_to = match.group(2).strip()
            if moved_to != target_date.isoformat():
                continue
            if is_placeholder_task(task_text):
                continue
            tasks.append(task_text)

    return dedupe_keep_order(tasks)


def mark_tasks_moved(
    diary_dir: Path,
    exclude_filename: Optional[str],
    tasks_to_move: List[str],
    target_date: dt.date,
) -> Dict[str, int]:
    if not tasks_to_move or not diary_dir.exists():
        return {}

    task_set = {task.strip() for task in tasks_to_move if task.strip()}
    task_pattern = re.compile(r"^(\s*-\s\[\s\]\s+)(.*\S)\s*$")
    moved_count: Dict[str, int] = {}

    for note in sorted(diary_dir.glob("*.md")):
        if not DAILY_NOTE_RE.match(note.name):
            continue
        if exclude_filename and note.name == exclude_filename:
            continue
        text = note.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        changed = False

        for idx, line in enumerate(lines):
            match = task_pattern.match(line)
            if not match or is_moved_task_line(line):
                continue
            task_text = match.group(2).strip()
            if is_placeholder_task(task_text):
                continue
            if task_text not in task_set:
                continue
            lines[idx] = f"- ~~{task_text}~~（转移至 {target_date:%Y-%m-%d}）"
            moved_count[note.name] = moved_count.get(note.name, 0) + 1
            changed = True

        if changed:
            note.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return moved_count


def parse_week1_start(text: str) -> dt.date:
    match = re.search(r"第一周第一天是\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if not match:
        return DEFAULT_WEEK1_START
    y, m, d = map(int, match.groups())
    return dt.date(y, m, d)


def find_week_section(text: str, week_index: int) -> Optional[str]:
    matches = list(re.finditer(r"第\s*(\d+)\s*周", text))
    if not matches:
        return None

    candidate_indices = [i for i, m in enumerate(matches) if int(m.group(1)) == week_index]
    if not candidate_indices:
        return None

    # Prefer the later match because files may mention "第1周" in prose before the actual table.
    for idx in reversed(candidate_indices):
        start = matches[idx].start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end]
        if "|" in section:
            return section

    idx = candidate_indices[-1]
    start = matches[idx].start()
    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
    return text[start:end]


def is_table_separator(cells: List[str]) -> bool:
    for cell in cells:
        compact = cell.replace(":", "").replace("-", "").replace(" ", "")
        if compact:
            return False
    return True


def extract_today_overview_tasks(
    overview_text: str, target_date: dt.date
) -> Tuple[List[str], Optional[int], Optional[str]]:
    week1_start = parse_week1_start(overview_text)
    delta_days = (target_date - week1_start).days
    if delta_days < 0:
        return [], None, None

    week_index = delta_days // 7 + 1
    weekday_zh = WEEKDAY_ZH[target_date.weekday()]

    section = find_week_section(overview_text, week_index)
    if not section:
        return [], week_index, weekday_zh

    headers: Optional[List[str]] = None
    row_cells: Optional[List[str]] = None

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        cells = [sanitize_cell(x) for x in line.strip("|").split("|")]
        if len(cells) < 2 or is_table_separator(cells):
            continue

        if "日程" in cells[0]:
            headers = cells
            continue

        if cells[0].startswith(weekday_zh):
            row_cells = cells
            break

    if not row_cells:
        return [], week_index, weekday_zh

    default_headers = ["日程", "时长/时间块", "听课任务", "练习任务", "从零实现", "笔记/验收"]
    use_headers = headers if headers and len(headers) >= len(row_cells) else default_headers

    tasks: List[str] = []
    for idx in range(2, len(row_cells)):
        value = row_cells[idx].strip()
        if not value or value in {"-", "—"}:
            continue
        label = use_headers[idx] if idx < len(use_headers) else f"任务{idx - 1}"
        label = sanitize_cell(label)
        tasks.append(f"{label}：{value}")

    return tasks, week_index, weekday_zh


def extract_workbench_focus_tasks(workbench_path: Path) -> List[str]:
    if not workbench_path.exists():
        return []

    text = workbench_path.read_text(encoding="utf-8", errors="ignore")
    task_pattern = re.compile(r"^\s*-\s\[\s\]\s+(.*\S)\s*$")
    tasks: List[str] = []
    in_focus = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if level <= 2:
                in_focus = title == "当前焦点"
                continue
        if not in_focus:
            continue

        match = task_pattern.match(line)
        if not match:
            continue
        task_text = match.group(1).strip()
        if is_placeholder_task(task_text):
            continue
        tasks.append(task_text)

    return dedupe_keep_order(tasks)


def build_markdown(unfinished_tasks: List[str], today_tasks: List[str]) -> str:
    out: List[str] = []
    out.append("## 历史未完成任务")
    if unfinished_tasks:
        for task in unfinished_tasks:
            out.append(f"- [ ] {task}")
    else:
        out.append("- [ ] 暂无历史未完成任务")

    out.append("")
    out.append("## 今日任务")
    if today_tasks:
        for task in today_tasks:
            out.append(f"- [ ] {task}")
    else:
        out.append("- [ ] 从 `99_学习情况记录/workbench.md` 的 `## 当前焦点` 添加今天要做的事")

    out.append("")
    out.append("## 今日完成内容")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract task sections for today note.")
    parser.add_argument("--date", help="Date in YYYY-MM-DD. Defaults to local today.")
    parser.add_argument(
        "--diary-dir",
        default="99_学习情况记录",
        help="Directory containing previous diary notes.",
    )
    parser.add_argument(
        "--workbench",
        default="99_学习情况记录/workbench.md",
        help="Workbench file path.",
    )
    parser.add_argument(
        "--mark-moved",
        action="store_true",
        help="Mark old unfinished tasks as transferred to the target date.",
    )
    parser.add_argument(
        "--include-open-history",
        action="store_true",
        help="Include every unchecked task from old daily notes, not only tasks transferred to the target date.",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    target_date = parse_date(args.date)
    day_abbrev = WEEKDAY_EN[target_date.weekday()]
    target_filename = f"{target_date:%Y-%m-%d}——{day_abbrev}.md"

    diary_dir = Path(args.diary_dir)
    workbench_path = Path(args.workbench)

    moved_to_today_tasks = find_tasks_moved_to_date(
        diary_dir, exclude_filename=target_filename, target_date=target_date
    )
    open_history_tasks = (
        find_unfinished_tasks(diary_dir, exclude_filename=target_filename)
        if args.include_open_history or args.mark_moved
        else []
    )
    if args.mark_moved:
        mark_tasks_moved(
            diary_dir,
            exclude_filename=target_filename,
            tasks_to_move=open_history_tasks,
            target_date=target_date,
        )
    unfinished_tasks = dedupe_keep_order(moved_to_today_tasks + open_history_tasks)
    today_tasks = extract_workbench_focus_tasks(workbench_path)

    if args.output_format == "json":
        payload = {
            "date": target_date.isoformat(),
            "weekday_zh": WEEKDAY_ZH[target_date.weekday()],
            "weekday_en": day_abbrev,
            "target_filename": target_filename,
            "unfinished_tasks": unfinished_tasks,
            "today_tasks": today_tasks,
            "workbench_found": workbench_path.exists(),
            "workbench": str(workbench_path),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(build_markdown(unfinished_tasks, today_tasks))


if __name__ == "__main__":
    main()

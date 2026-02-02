#!/usr/bin/env python3
"""
Extract module 2 and module 3 task blocks for the `today` skill.

Outputs markdown by default so it can be pasted directly into a note.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

WEEKDAY_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
WEEKDAY_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_WEEK1_START = dt.date(2026, 2, 2)


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


def find_unfinished_tasks(diary_dir: Path, exclude_filename: Optional[str]) -> List[str]:
    task_pattern = re.compile(r"^\s*-\s\[\s\]\s+(.*\S)\s*$")
    tasks: List[str] = []
    if not diary_dir.exists():
        return tasks

    for note in sorted(diary_dir.glob("*.md")):
        if exclude_filename and note.name == exclude_filename:
            continue
        text = note.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            match = task_pattern.match(line)
            if not match:
                continue
            tasks.append(match.group(1).strip())
    return dedupe_keep_order(tasks)


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

        if cells[0] == weekday_zh:
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


def build_markdown(unfinished_tasks: List[str], today_tasks: List[str]) -> str:
    out: List[str] = []
    out.append("## 模块二｜历史未完成任务")
    if unfinished_tasks:
        for task in unfinished_tasks:
            out.append(f"- [ ] {task}")
    else:
        out.append("- [ ] 暂无历史未完成任务")

    out.append("")
    out.append("## 模块三｜今日任务（来自 Overview & Study Record）")
    if today_tasks:
        for task in today_tasks:
            out.append(f"- [ ] {task}")
    else:
        out.append("- [ ] 未识别到今日任务，请手动检查 `Overview & Study Record.md`")

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
        "--overview",
        default="Overview & Study Record.md",
        help="Overview file path.",
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
    overview_path = Path(args.overview)

    unfinished_tasks = find_unfinished_tasks(diary_dir, exclude_filename=target_filename)
    if overview_path.exists():
        overview_text = overview_path.read_text(encoding="utf-8", errors="ignore")
        today_tasks, week_index, weekday_zh = extract_today_overview_tasks(overview_text, target_date)
    else:
        today_tasks, week_index, weekday_zh = [], None, WEEKDAY_ZH[target_date.weekday()]

    if args.output_format == "json":
        payload = {
            "date": target_date.isoformat(),
            "weekday_zh": weekday_zh,
            "weekday_en": day_abbrev,
            "target_filename": target_filename,
            "unfinished_tasks": unfinished_tasks,
            "overview_today_tasks": today_tasks,
            "overview_week_index": week_index,
            "overview_found": overview_path.exists(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(build_markdown(unfinished_tasks, today_tasks))


if __name__ == "__main__":
    main()

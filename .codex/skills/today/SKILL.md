---
name: today
description: Generate or update today's task-only journal note when the user says /today or asks for a daily task log. Use template `00_inbox/日记模版.md` (fallback `00_inbox/日记模板.md`), create the note in `99_学习情况记录`, list historical unfinished tasks above today's tasks, read today's focus tasks from `99_学习情况记录/workbench.md`, keep `今日完成内容`, and preserve the user's manual transfer convention `- ~~task~~（转移至 YYYY-MM-DD）`. Do not include news or depend on `Overview & Study Record.md`.
---

# Today

## Overview

Create one task-only daily note and save it to the study-log folder.

Use this workflow every time the user asks for `/today`.

## Workflow

1. Resolve required paths.
   - Prefer template `00_inbox/日记模版.md`.
   - If missing, fallback to `00_inbox/日记模板.md`.
   - Use output folder `99_学习情况记录`.
   - Use workbench file `99_学习情况记录/workbench.md`.

2. Determine today's target filename.
   - Format: `YYYY-MM-DD——<DayAbbrev>.md` (for example `2026-02-02——Mon.md`).
   - If the file already exists, update it in place instead of creating a duplicate.

3. Build the task sections with the helper script.
   - Run:
     - `python3 .codex/skills/today/scripts/extract_today_tasks.py --date YYYY-MM-DD`
   - Use the generated markdown block directly.
   - Default behavior is read-only: do not modify old diaries.
   - By default, historical unfinished tasks include only items explicitly marked as transferred to today with `转移至 YYYY-MM-DD`.
   - Only add `--include-open-history` if the user explicitly asks to collect all unchecked tasks from old daily notes.
   - Only add `--mark-moved` if the user explicitly asks to mark old unfinished tasks as transferred to today.

4. Compose the final note.
   - Start from the resolved diary template content.
   - Place sections in this order:
     - `## 历史未完成任务`
     - `## 今日任务`
     - `## 今日完成内容`
   - Use markdown checkboxes (`- [ ]`) for all task bullets.
   - Do not include news, maps, finance summaries, tech summaries, or global situation sections.

5. Save and verify.
   - Confirm file is under `99_学习情况记录`.
   - Confirm all three task sections exist.
   - Confirm historical unfinished tasks appear above today's tasks.
   - Confirm today's tasks come from `## 当前焦点` in `99_学习情况记录/workbench.md` when available.

## Notes

- Do not edit `.obsidian/` files.
- Preserve manual task transfers in the form `- ~~task~~（转移至 YYYY-MM-DD）`.
- Include transferred tasks only on the matching target date.
- Do not pull every unchecked task from old diaries unless the user explicitly asks for a full backlog sweep.
- Ignore placeholder tasks such as `暂无历史未完成任务` and `未识别到今日任务`.
- Keep wording concise and execution-focused.

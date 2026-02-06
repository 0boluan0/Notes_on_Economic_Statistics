---
name: today
description: Generate today's journal note when the user says /today or asks for a daily log. Use template `00_inbox/日记模版.md` (fallback `00_inbox/日记模板.md`), create the note in `99_学习情况记录`, add a bilingual global situation dashboard (map + a concise table summary under the map) for the latest 24 hours, list unfinished tasks from existing notes in `99_学习情况记录`, and list today's tasks extracted from `Overview & Study Record.md` as markdown checkboxes.
---

# Today

## Overview

Create one completed daily note with three ordered modules and save it to the study-log folder.

Use this workflow every time the user asks for `/today`.

## Workflow

1. Resolve required paths.
   - Prefer template `00_inbox/日记模版.md`.
   - If missing, fallback to `00_inbox/日记模板.md`.
   - Use output folder `99_学习情况记录`.
   - Use overview file `Overview & Study Record.md`.

2. Determine today's target filename.
   - Format: `YYYY-MM-DD——<DayAbbrev>.md` (for example `2026-02-02——Mon.md`).
   - If the file already exists, update it in place instead of creating a duplicate.

3. Build module 1 (global dashboard, concise table).
   - Run:
     - `python3 .codex/skills/today/scripts/build_news_dashboard.py --date YYYY-MM-DD`
   - The script creates a static SVG map in `98_attachment/dashboards` and prints the full module 1 markdown to stdout.
   - Use the printed markdown block directly (it already includes map embed + a heat-weighted summary table).
   - Optional: add `--translate` and set `LIBRETRANSLATE_URL` to enable machine translation for the Chinese lines.
   - Follow `references/news-format.md`.

4. Build module 2 and module 3 (tasks) with the helper script.
   - Run:
     - `python3 .codex/skills/today/scripts/extract_today_tasks.py --date YYYY-MM-DD`
   - Use the generated markdown block directly.
   - If the script reports missing data, insert explicit placeholders instead of guessing.

5. Compose the final note.
   - Start from the resolved diary template content.
   - Place module order exactly:
     - Module 1: Global Situation Dashboard (bilingual)
     - Module 2: Unfinished Tasks (from old diaries)
     - Module 3: Today's Tasks (from overview plan)
   - Use markdown checkboxes (`- [ ]`) for all task bullets.

6. Save and verify.
   - Confirm file is under `99_学习情况记录`.
   - Confirm all three modules exist.
   - Confirm module 1 includes the map embed and the summary table within latest 24 hours.

## Notes

- Do not edit `.obsidian/` files.
- Do not fabricate news. If data is insufficient, state that clearly in the note.
- Keep wording concise and execution-focused.

---
name: weekly-review
description: Run an interactive weekly learning review for the Academic Obsidian vault, write `99_学习情况记录/week-review/YYYY-Www.md`, and archive the reviewed week's closed daily notes. Use when the user asks for weekly review, week review, 周复盘, 本周复盘, or wants to close and archive a week of daily notes.
---

# Weekly Review

## Overview

Review one ISO week of daily notes, ask the user reflective questions, write one weekly review note, then archive that reviewed week's closed daily notes. Do not automatically modify Overview, Workbench, or Deadlines; list suggested sync changes for the user to approve separately.

## Workflow

1. Resolve week and paths.
   - Default to the current ISO week unless the user names a week or date.
   - Active daily notes live in `99_学习情况记录` and use `YYYY-MM-DD——<DayAbbrev>.md`.
   - Archived daily notes live in `99_学习情况记录/archive/daily/YYYY-Www/` with the same filenames.
   - Output folder is `99_学习情况记录/week-review`.
   - Output filename is `YYYY-Www.md`, for example `2026-W28.md`.
   - Read `99_学习情况记录/Overview & Study Record.md`, `workbench.md`, and `deadlines.md` for context.

2. Read the seven daily notes for the target week.
   - Search both the active root and that week's archive folder. If the same filename exists in both, stop and report the duplicate instead of choosing one silently.
   - Summarize planned work, completed work, plan-unplanned work, transfers, and closeout notes.
   - If some days are missing, continue and mention the missing dates.
   - Ignore old news/dashboard sections if present in legacy notes.

3. Ask the user before writing the final review.
   - Ask 4-6 concise questions about current state, wins, blockers, energy, priority shifts, and next week.
   - Ground questions in the week’s actual daily notes.
   - Wait for answers before writing the file.

4. Write the weekly review note.
   - Use this structure:
     - `# YYYY-Www 周复盘`
     - `## 本周概览`
     - `## 实际推进`
     - `## 卡点与拖延`
     - `## 状态反思`
     - `## 下周调整`
     - `## 建议同步`
       - `### Overview`
       - `### Workbench`
       - `### Deadline`
   - Keep the note concrete. Prefer short bullets tied to real daily notes.
   - Include a `## 缺失记录` section only when one or more daily notes are missing.

5. Close and archive the reviewed week.
   - Archive only after the final weekly review exists, includes the user's answers, and has passed the review checks below.
   - For the current ISO week, do not archive before Sunday close. On Sunday or later, every daily note carrying a `student-os:shutdown` marker must be reconciled first; if the newest note is still open, read the local `today` skill and run its manual shutdown reconciliation before moving anything.
   - Create `99_学习情况记录/archive/daily/YYYY-Www/` lazily, then move only that ISO week's root-level daily notes into it. Preserve filenames and contents; never delete, overwrite, or archive the weekly review note.
   - If an archive target already exists, stop on any mismatch. If all target daily notes are already archived and none remain at root, make no archival changes.
   - Leave `99_学习情况记录/week-review/YYYY-Www.md` outside the archive as the visible record of the week.

6. Verify.
   - Confirm the file is under `99_学习情况记录/week-review`.
   - Confirm the review includes the user’s answers, not only automatic summary.
   - Confirm no Overview/Workbench/Deadline edits were made unless explicitly requested after the review.
   - Confirm no target-week daily note remains at the root, every moved filename exists once in the week archive, and no archived daily note has an open shutdown marker.
   - Re-run the archive check and confirm it makes no additional changes.

## Notes

- Treat daily notes as execution evidence, not perfect truth.
- Treat `Learning Progress Dashboard` data as prototype-only unless the user explicitly says it has been redesigned and calibrated.
- Use direct reading and ordinary file moves; do not create a bespoke archival utility unless the simple workflow demonstrably fails.

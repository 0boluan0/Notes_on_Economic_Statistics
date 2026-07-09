---
name: weekly-review
description: Run an interactive weekly learning review for the Academic Obsidian vault. Use when the user asks for weekly review, week review, 周复盘, 本周复盘, or wants to review a week of daily notes and produce `99_学习情况记录/week-review/YYYY-Www.md`.
---

# Weekly Review

## Overview

Review one ISO week of daily notes, ask the user reflective questions, then write one weekly review note. Do not automatically modify Overview, Workbench, or Deadlines; list suggested sync changes for the user to approve separately.

## Workflow

1. Resolve week and paths.
   - Default to the current ISO week unless the user names a week or date.
   - Daily notes live in `99_学习情况记录` and use `YYYY-MM-DD——<DayAbbrev>.md`.
   - Output folder is `99_学习情况记录/week-review`.
   - Output filename is `YYYY-Www.md`, for example `2026-W28.md`.
   - Read `99_学习情况记录/Overview & Study Record.md`, `workbench.md`, and `deadlines.md` for context.

2. Read the seven daily notes for the target week.
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

5. Verify.
   - Confirm the file is under `99_学习情况记录/week-review`.
   - Confirm the review includes the user’s answers, not only automatic summary.
   - Confirm no Overview/Workbench/Deadline edits were made unless explicitly requested after the review.

## Notes

- Treat daily notes as execution evidence, not perfect truth.
- Treat `Learning Progress Dashboard` data as prototype-only unless the user explicitly says it has been redesigned and calibrated.
- Do not run scripts. Use direct reading and LLM judgment.

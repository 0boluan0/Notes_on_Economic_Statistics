---
name: today
description: Generate or update today's task-only journal note when the user says /today or asks for a daily task log. Use template `00_inbox/日记模版.md` (fallback `00_inbox/日记模板.md`), create the note in `99_学习情况记录`, read the current learning snapshot from `99_学习情况记录/Overview & Study Record.md`, show upcoming deadlines from `99_学习情况记录/deadlines.md`, list historical unfinished tasks above today's workbench view, embed `99_学习情况记录/workbench.md#当前项目` as the live workbench, keep today-completed, plan-unplanned work, and closeout review sections, and preserve the user's manual transfer convention `- ~~task~~（转移至 YYYY-MM-DD）`. Do not include news. Do not use Learning Progress Dashboard data as a factual source unless the user explicitly says it has been redesigned and calibrated.
---

# Today

## Overview

Create one task-only daily note and save it to the study-log folder. Do the reading and judgment directly as an LLM; do not call helper scripts for task extraction.

Use the four-layer system:

- `Overview & Study Record.md`: current learning map and time-slice context. Read it for background, not as a daily task list.
- `99_学习情况记录/workbench.md`: high-frequency project desktop and current task source. Embed it in the daily note instead of copying tasks.
- `99_学习情况记录/deadlines.md`: one-file deadline register. Read upcoming incomplete deadlines and compute D-days.
- Daily note: today's recall, task execution, plan-unplanned work, and closeout review.

Treat `Learning Progress Dashboard` and `98_attachment/vault-home/learning-board.md` as prototype data for now. Do not read them when generating `/today` unless the user explicitly asks.

Use this workflow every time the user asks for `/today`.

## Workflow

1. Resolve required paths.
   - Prefer template `00_inbox/日记模版.md`.
   - If missing, fallback to `00_inbox/日记模板.md`.
   - Use output folder `99_学习情况记录`.
   - Use overview file `99_学习情况记录/Overview & Study Record.md`.
   - Use workbench file `99_学习情况记录/workbench.md`.
   - Use deadline file `99_学习情况记录/deadlines.md`.

2. Determine today's target filename.
   - Format: `YYYY-MM-DD——<DayAbbrev>.md` (for example `2026-02-02——Mon.md`).
   - If the file already exists, update it in place instead of creating a duplicate.

3. Read source notes directly.
   - Read `Overview & Study Record.md` first to understand the current learning map, active/queued subjects, and abandoned old plans.
   - Use Overview only as context. Do not turn every listed course into a daily task.
   - Read `deadlines.md` and list incomplete deadlines due within 14 days. Compute `D-N` yourself from the target date.
   - Read previous daily notes only for tasks explicitly marked as transferred to the target date with `转移至 YYYY-MM-DD`.
   - Read `workbench.md` to understand the current desk. Do not duplicate its tasks into the daily note.
   - Do not collect every unchecked task from old daily notes unless the user explicitly asks for a backlog sweep.
   - Do not mark old notes as moved unless the user explicitly asks.

4. Compose the final note.
   - Start from the resolved diary template content.
   - If the template already contains the generated section headings, replace the matching task block instead of duplicating headings.
   - Place sections in this order:
     - `## 即将到期`
     - `## 历史未完成任务`
     - `## 工作台`
     - `## 今日完成`
     - `## 计划外完成`
     - `## 收尾复盘`
   - Under `## 工作台`, use `![[workbench#当前项目]]`.
   - Use markdown checkboxes (`- [ ]`) for transferred historical tasks only.
   - Do not include news, maps, finance summaries, tech summaries, or global situation sections.

5. Save and verify.
   - Confirm file is under `99_学习情况记录`.
   - Confirm all six sections exist.
   - Confirm upcoming deadlines include computed `D-N` labels.
   - Confirm historical unfinished tasks appear above today's tasks.
   - Confirm `## 工作台` embeds `![[workbench#当前项目]]`.
   - Confirm no helper script was required.

## Notes

- Do not edit `.obsidian/` files.
- Preserve manual task transfers in the form `- ~~task~~（转移至 YYYY-MM-DD）`.
- Include transferred tasks only on the matching target date.
- Do not pull every unchecked task from old diaries unless the user explicitly asks for a full backlog sweep.
- Ignore placeholder tasks such as `暂无历史未完成任务` and `未识别到今日任务`.
- Do not invent tasks from project names. Workbench remains the only current task source.
- Deadline format is `- [ ] YYYY-MM-DD｜[[项目]]｜事项｜硬截止` or similar. Completed deadlines use `[x]` and are ignored.
- Ignore `Learning Progress Dashboard` data until the user explicitly says it is redesigned and calibrated.
- Keep wording concise and execution-focused.

---
name: today
description: Generate or update today's task-only journal note when the user says /today or asks for a daily task log. Use template `00_inbox/日记模版.md` (fallback `00_inbox/日记模板.md`), create the note in `99_学习情况记录`, read the current learning snapshot from `99_学习情况记录/Overview & Study Record.md`, show upcoming deadlines from `99_学习情况记录/deadlines.md`, list historical unfinished tasks above today's workbench view, create paired LN905 Listening into Writing and Reading into Writing practice on the current weekly topic, embed `99_学习情况记录/workbench.md#当前项目` as the live workbench, keep today-completed, plan-unplanned work, and closeout review sections, and preserve the user's manual transfer convention `- ~~task~~（转移至 YYYY-MM-DD）`. Do not include news. Do not use Learning Progress Dashboard data as a factual source unless the user explicitly says it has been redesigned and calibrated.
---

# Today

## Overview

Create one task-only daily note and save it to the study-log folder. Do the reading and judgment directly as an LLM; do not call helper scripts for task extraction.

Use the four-layer system:

- `Overview & Study Record.md`: current learning map and time-slice context. Read it for background, not as a daily task list.
- `99_学习情况记录/workbench.md`: high-frequency project desktop and current task source. Embed it in the daily note instead of copying tasks.
- `99_学习情况记录/deadlines.md`: one-file deadline register. Read upcoming incomplete deadlines and compute D-days.
- `99_学习情况记录/teach/listening-into-writing/`: persistent LN905 Listening into Writing mission, resources, teaching notes, and daily lessons.
- `99_学习情况记录/teach/reading-into-writing/`: persistent LN905 Reading into Writing mission, resources, teaching notes, printable source packs, and digital responses. Its `NOTES.md` frontmatter holds the shared `current_week_topic`.
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
   - Use teaching workspace `99_学习情况记录/teach/listening-into-writing`.
   - Use teaching workspace `99_学习情况记录/teach/reading-into-writing`.

2. Determine today's target filename.
   - Format: `YYYY-MM-DD——<DayAbbrev>.md` (for example `2026-02-02——Mon.md`).
   - If the file already exists, update it in place instead of creating a duplicate.

3. Read source notes directly.
   - Read `Overview & Study Record.md` first to understand the current learning map, active/queued subjects, and abandoned old plans.
   - Use Overview only as context. Do not turn every listed course into a daily task.
   - Read `deadlines.md` and list incomplete deadlines due within 14 days. Compute `D-N` yourself from the target date.
   - Read previous daily notes only for tasks explicitly marked as transferred to the target date with `转移至 YYYY-MM-DD`.
   - Read `workbench.md` to understand the current desk. Do not duplicate its tasks into the daily note.
   - Read the Listening into Writing `MISSION.md`, `NOTES.md`, existing lessons, and any demonstrated `learning-records/` before choosing the next exercise.
   - Read the Reading into Writing `MISSION.md`, `NOTES.md`, existing practice files, and any demonstrated `learning-records/`.
   - Do not collect every unchecked task from old daily notes unless the user explicitly asks for a backlog sweep.
   - Do not mark old notes as moved unless the user explicitly asks.

4. Resolve the shared weekly topic.
   - Read `current_week_topic` from `99_学习情况记录/teach/reading-into-writing/NOTES.md`. Both daily lessons must use this topic.
   - Keep using that topic until the user explicitly changes the course week or topic. Never infer a change merely because a new source mentions another subject.

5. Prepare today's Listening into Writing practice.
   - If today's note already references an existing lesson on the current weekly topic, preserve it. If it points to another topic, replace the note entry with a new matching lesson but keep the older lesson file.
   - Find one currently reachable video from the verified official TED YouTube channel. Prefer 15–22 minutes, a clear academic argument with distinguishable claims and evidence, an official TED transcript, and the closest possible match to the weekly topic. A small duration exception is acceptable when the talk is especially suitable.
   - Verify the title, channel, link, duration, and transcript before writing the lesson. Do not invent or guess a URL. Avoid talks already used in existing lessons.
   - Choose one focus from the smallest current gap: lecture structure, selective note-taking, paraphrasing, supported criticality, or full timed integration. If there is no demonstrated performance yet, begin with lecture structure and selective note-taking. Do not combine every focus into one daily exercise.
   - Save a self-contained HTML lesson under `99_学习情况记录/teach/listening-into-writing/lessons/` using the next `000N-<dash-case-name>.html` filename required by the `teach` skill.
   - Include the YouTube source link, a transcript link revealed only for checking, exact timings, a note scaffold, a writing prompt and word target, an interactive feedback loop, a short answer framework, and a final L/S/N/W/C bottleneck reflection. Keep ordinary daily practice to roughly 35–55 minutes; reserve a full 40-minute writing simulation for at most twice a week unless the user requests more.

6. Prepare today's Reading into Writing practice.
   - If today's note already references an existing topic-matching source-pack PDF and digital response file, preserve them. Otherwise use the next shared `000N-<dash-case-topic>` basename in `99_学习情况记录/teach/reading-into-writing/practice/`.
   - Create the printable source pack in LaTeX and retain both `.tex` and compiled `.pdf` files. The PDF is for paper-and-pen reading and annotation: include the question and exactly three attributed papers or adapted extracts, but no answer lines or writing worksheet.
   - Prefer course-provided extracts on the weekly topic. Otherwise use three currently reachable authoritative or peer-reviewed sources with claims that can genuinely be compared. Adapt and cite material; do not reproduce long copyrighted passages.
   - Match Paper B: require selection, cross-text synthesis, paraphrasing, a direct answer to the question, and source attribution. Organise tasks by themes rather than one source at a time.
   - Create a separate Markdown response file for computer work. It should link the PDF and lead the learner through question judgment, selective source compression, explicit cross-text relationships, paragraph reasoning chains, a 600-word budget, a blank response section, a criteria-based self-check, and a bottleneck reflection. Give process scaffolding without prewriting an answer the learner could submit. Keep ordinary daily drills shorter when appropriate; reserve a full two-hour, 600-word simulation for at most twice a week unless the user requests more.
   - Compile the PDF with the `latex-compile` workflow and visually inspect the rendered pages with the `pdf` workflow before linking it.

7. Compose the final note.
   - Start from the resolved diary template content.
   - If the template already contains the generated section headings, replace the matching task block instead of duplicating headings.
   - Place sections in this order:
     - `## 即将到期`
     - `## 历史未完成任务`
     - `## Listening into Writing`
     - `## Reading into Writing`
     - `## 工作台`
     - `## 今日完成`
     - `## 计划外完成`
     - `## 收尾复盘`
   - Under `## 工作台`, use `![[workbench#当前项目]]`.
   - Under `## Listening into Writing`, add one checkbox linking the HTML lesson, followed by the verified YouTube link, video duration, today's single focus, and estimated total practice time.
   - Under `## Reading into Writing`, add one checkbox linking both the printable PDF source pack and the Markdown response file, followed by its question, today's single focus, and estimated total practice time.
   - Use markdown checkboxes (`- [ ]`) for transferred historical tasks only.
   - Do not include news, maps, finance summaries, tech summaries, or global situation sections.

8. Save and verify.
   - Confirm file is under `99_学习情况记录`.
   - Confirm all eight sections exist.
   - Confirm upcoming deadlines include computed `D-N` labels.
   - Confirm historical unfinished tasks appear above today's tasks.
   - Confirm `## 工作台` embeds `![[workbench#当前项目]]`.
   - Confirm the Listening HTML, Reading `.tex`, Reading PDF, and Reading response file exist; today's note links to the Listening lesson, Reading PDF, and Reading response exactly once; and both skills match the same current weekly topic. Confirm the Listening YouTube and transcript links are reachable and the Reading PDF renders correctly.
   - Confirm no helper script was required.

## Notes

- Do not edit `.obsidian/` files.
- Preserve manual task transfers in the form `- ~~task~~（转移至 YYYY-MM-DD）`.
- Include transferred tasks only on the matching target date.
- Do not pull every unchecked task from old diaries unless the user explicitly asks for a full backlog sweep.
- Ignore placeholder tasks such as `暂无历史未完成任务` and `未识别到今日任务`.
- Do not invent tasks from project names. Workbench remains the only current task source.
- The user explicitly requested both daily LN905 skills lessons, so they are the only exceptions to the workbench-only task rule. Keep them out of `workbench.md` and do not create other inferred daily tasks.
- Re-running `/today` must be idempotent: reuse today's existing topic-matching lesson entries.
- Deadline format is `- [ ] YYYY-MM-DD｜[[项目]]｜事项｜硬截止` or similar. Completed deadlines use `[x]` and are ignored.
- Ignore `Learning Progress Dashboard` data until the user explicitly says it is redesigned and calibrated.
- Keep wording concise and execution-focused.

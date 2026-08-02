---
name: today
description: Generate or update the Student OS daily home in the Academic Obsidian vault. Use when the user says /today, asks for today's plan, or the Student OS heartbeat wakes. Reconcile the most recent unclosed daily note, audit deadlines and calendar, schedule canonical tasks without duplicating them, update the fixed-date Today views, and preserve all user-written text.
---

# Today

Build one calm, omission-resistant daily home. Make routine scheduling decisions yourself; ask only when a choice would materially change the result.

## Canonical system

- Overall: `99_学习情况记录/Overview & Study Record.md`; confirmed learning map, never a copied daily task list.
- Plans: `99_学习情况记录/学习计划/`; long sequential courses and their canonical checkboxes.
- Workbench: `99_学习情况记录/workbench.md`; standalone project tasks plus a dynamic next-action view.
- Deadlines: `99_学习情况记录/deadlines.md`; hard obligations tagged `#student-os/deadline`.
- Today: `99_学习情况记录/YYYY-MM-DD——ddd.md`; scheduled-task view, read-only deadline radar, completed-task view, and free writing.
- Canonical work units are the only checkboxes tagged `#student-os/task`. Never copy one into Today or Workbench.

Use `00_inbox/日记模版.md` (fallback `00_inbox/日记模板.md`). Re-running for the same date updates the existing note and must not create duplicate sections, tasks, lessons, or calendar events.

## First-awake reconciliation

1. Find the newest earlier daily note.
2. Ignore legacy notes that have no `student-os:shutdown` marker.
3. If the newest marked note is still `open`, reconcile its unfinished scheduled tasks before planning today:
   - preserve every user-written line;
   - do not invent a reflection or claim the user completed anything;
   - move only still-open scheduling metadata and matching generated `Study Plan` events;
   - mark the note `student-os:shutdown: reconciled YYYY-MM-DD` only after the writes succeed.
4. Do not create empty notes for missed days.

A manual shutdown request may run the same reconciliation early. There is no fixed evening time and no “end today” button requirement.

## Sources and deadline audit

1. Read Overall, Workbench, Deadlines, all `student_os: learning-plan` files, and today's relevant course/project source files.
2. Read Apple Calendar for fixed commitments. Treat every calendar except exact name `Study Plan` as read-only. If `LSE` exists, use it as the read-only timetable.
   - If `LSE` is missing or the timetable cannot be read, do not place movable study blocks. Keep the next actions visible in Workbench, sync only independently verified fixed windows, and surface the source-check failure.
3. Show every incomplete hard deadline from D-0 through D-14, inclusive, as plain non-checkbox lines in Today's managed deadline block. Also show a deadline beyond D-14 when work for it is already scheduled.
4. Update Workbench's managed risk block from the same facts.
5. Do a bounded daily Student Hub/Moodle comparison when the logged-in source is reachable. On the first Sunday run, inspect assignments, files and announcements in full. Never silently treat an unreachable source as checked, never expose or store an ICS token, and never close a submission without Moodle/Turnitin status or a receipt.
6. Normal runs are quiet. Escalate only for a new D-14 deadline, an earlier deadline change, impossible capacity, repeated source-check failure, or an unresolved near-deadline obligation.

## Scheduling

- Schedule only canonical `#student-os/task` lines by adding or changing their Tasks `⏳ YYYY-MM-DD` metadata. Scheduled automation may edit only that metadata in plan files and Workbench task sources; do not rewrite their content.
- Hard deadline rows remain separate verified obligations. “Content done” and “submitted” are distinct.
- All generated calendar blocks go to `Study Plan`, for all future dates, never another calendar. Put a stable marker derived from source path plus task description in the event notes; update/delete only events carrying that marker.
- Plan inside 09:00–19:30. Reserve 19:30–20:00 for packing/buffer and assume departure at 20:00. Never auto-schedule after 19:30.
- Use at most 70% of genuinely free weekday time and 50% on weekends.
- Preserve travel and buffers. A free gap of at least 60 minutes may hold a 45-minute study block; 30–59 minutes may hold light administration; under 30 minutes stays empty.
- A small assignment should have a 24-hour personal buffer and a major assignment 48 hours.
- D-14 work always remains visible. When near-deadline workload is heavy, suppress important-but-not-urgent fillers. When it is light, steadily advance confirmed self-directed courses by selecting their first incomplete lecture.
- For video courses, one lecture is one task. It is complete only when the user watched it and personally judges it understood. Do not invent required notes, exercises, tests, or mastery gates.
- Do not generate LN905 Listening into Writing or Reading into Writing drills until a detailed plan has been agreed and recorded in Overall or a linked plan.

## Compose Today

- Filename: `YYYY-MM-DD——ddd.md`.
- Start from the template if missing.
- Keep these sections once: `今日安排`, `临近 Deadline`, `今日已完成`, `随手写`, `收尾`.
- Set the note's `date` property to the actual `YYYY-MM-DD` date. Do not replace the native Tasks query blocks; they read that fixed property so historical pages stay fixed without enabling Tasks JavaScript queries.
- Replace only content between `student-os:deadline-radar:start/end`.
- Preserve all text in `随手写`, `收尾`, and outside managed blocks.
- Set `student-os:today-generated: YYYY-MM-DD` only after source tasks, Today, risk radar and calendar sync all succeed.

## Verify

- Each actionable item has one canonical checkbox.
- Today and Workbench queries filter exact Student OS tags and do not include course-note self-checks.
- Scheduled tasks and generated `Study Plan` events agree.
- D-0 through D-14 obligations are all visible.
- No generated block extends past 19:30.
- No user-written text changed.
- A second run makes no additional changes.

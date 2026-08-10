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
- Today: active notes live at `99_学习情况记录/YYYY-MM-DD——ddd.md`; after a completed weekly review, closed notes move unchanged to `99_学习情况记录/archive/daily/YYYY-Www/`.
- Canonical work units are the only checkboxes tagged `#student-os/task`. Never copy one into Today or Workbench.

Use `00_inbox/日记模版.md` (fallback `00_inbox/日记模板.md`). Re-running for the same date updates the existing note and must not create duplicate sections, tasks, lessons, or calendar events.

## First-awake reconciliation

1. Find the newest earlier daily note across the active root and `99_学习情况记录/archive/daily/`.
   - Archived notes are closed historical evidence. Never reschedule from, rewrite, or move an archived note during a Today run.
   - An archived note with an open shutdown marker is an integrity failure: report it instead of silently treating it as closed.
2. Ignore legacy notes that have no `student-os:shutdown` marker.
3. If the newest marked note is still `open`, reconcile its unfinished scheduled tasks before planning today:
   - preserve every user-written line;
   - do not invent a reflection or claim the user completed anything;
   - before moving metadata, collect that date's canonical completions, still-open scheduled tasks, and existing user-written notes;
   - create or update the previous note's `student-os:shutdown-summary:start/end` block with a factual recap: distinguish planned from additional completions, identify unfinished work and its carry date, and include only reminders the user actually wrote; use plain lines, never new checkboxes;
   - move only still-open scheduling metadata and matching generated `Study Plan` events;
   - mark the note `student-os:shutdown: reconciled YYYY-MM-DD` only after the summary, task metadata and calendar writes all succeed.
4. Do not create empty notes for missed days.

A manual shutdown request may run the same factual-summary and reconciliation flow early. There is no fixed evening time and no “end today” button requirement.

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
- All generated calendar blocks go to `Study Plan`, for all future dates, never another calendar. Put a stable marker derived from source path plus task description in the event notes; update/delete only events carrying that marker. Give every generated block a display alert at the event start (`trigger interval: 0`) and preserve any other user-created alerts.
- On weekdays, place movable study blocks only from 15:30 through 19:30. On weekends, movable blocks may start at 09:00. Reserve 19:30–20:00 for packing/buffer and assume departure at 20:00. Fixed classes, travel, appointments and timed assessments may occur earlier; a preparation block tied directly to a fixed assessment may immediately precede it. Never auto-schedule movable work before the applicable start time or after 19:30.
- Use at most 70% of genuinely free weekday time and 50% on weekends.
- Preserve travel and buffers. A free gap of at least 60 minutes may hold a 45-minute study block; 30–59 minutes may hold light administration; under 30 minutes stays empty.
- A small assignment should have a 24-hour personal buffer and a major assignment 48 hours.
- D-14 work always remains visible. When near-deadline workload is heavy, suppress important-but-not-urgent fillers. When it is light, steadily advance confirmed self-directed courses by selecting their first incomplete lecture.
- For video courses, one lecture is one task. It is complete only when the user watched it and personally judges it understood. Do not invent required notes, exercises, tests, or mastery gates.
- Generate LN905 materials only for canonical sessions already defined in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`; never invent an extra daily checkbox or change a session's meaning.
- Schedule only the standalone daily `词灵` tasks already present in the confirmed LN905 plan, using the exact word-count or duration dose written on the canonical task. Do not auto-create another vocabulary-prep block before Listening or Reading. For a timed submission, follow the plan's topic-vocabulary blocks and never expose the test material beforehand.

## LN905 planned material preparation

Run this content-preparation pass before composing Today whenever an LN905 practice task is scheduled for today.

1. Read the linked session path, focus, weekly topic, source ledger, latest completed output and any teacher feedback. Sync Reading `NOTES.md`'s `current_week_topic` to the plan's current weekly topic. Use evidence to tune the scaffold without changing the task's meaning. The task line and its link must already exist; material generation never creates a second task.
2. If the linked artifact is already marked ready and its companion files pass basic checks, reuse the artifact itself. Do not regenerate it and do not reuse any of its sources in another session.
3. For Listening, select one previously unused video from the verified official TED YouTube channel, preferably 15–20 minutes, and verify the matching TED transcript. Create a standalone Markdown teaching record at the task's fixed path. Include only the agreed training focus, verified material links, relevant starting evidence, an empty `student-os:learning-log` block and a session-specific init prompt naming this exact task and record path. The prompt launches a new Codex thread that may teach only this part and must stop at its completion; never create a master thread prompt that chooses or advances several daily parts. Do not prebuild a worksheet, timer, field set or full sequence. Do not turn ordinary teaching practice into an unsupported full attempt. If the session needs suggested topic vocabulary, record it as reference material without creating another task or calendar block.
4. For Reading, use the number and type of sources required by the canonical task: one source for narrow comprehension/summary work, two for explicit synthesis practice, and three for a full simulation unless the task states otherwise. Use previously unused peer-reviewed papers that genuinely support the task's comparison or debate, and create accurate exam-style adapted extracts rather than copying long passages. At the task's fixed basename, preserve the `.tex`, compile the printable `.pdf`, and create a standalone Markdown teaching-and-review record with its own task-specific init prompt. The PDF contains only the question, required extracts and source attributions; the Markdown contains the agreed focus, verified starting evidence, material links and durable learning log, not a full matrix or simultaneous response fields. Its dedicated thread handles only this part and stops on completion. Render every PDF page and inspect it before marking the session ready.
5. For other AI-guided LN905 parts such as English production, readiness checks or post-test review, create the same one-part Markdown record and part-specific init prompt from the named existing outputs, criteria and feedback. Do not make that thread handle Listening, Reading or any other adjacent part, and do not introduce new sources unless the canonical task requires them.
6. Treat every existing lesson, source pack, `RESOURCES.md` entry and ledger identifier as used. Compare YouTube video IDs and paper DOI/canonical URLs. Append new identifiers to the plan ledger only after all session files verify successfully.
7. This pass may create the fixed session artifacts and append verified source identifiers. It may not add, duplicate, complete or rewrite a canonical checkbox; the scheduling pass remains limited to Tasks scheduling metadata.
8. If source verification, compilation or rendering fails, keep the task canonical but report the broken preparation explicitly and do not set `student-os:today-generated`.
9. Classify the session from its canonical task text. A task explicitly containing `完整模拟` is an independent simulation; every other LN905 practice task is guided AI-in-the-loop practice.
10. In guided practice, the part-specific thread's first reply begins with a compact orientation card: `今天在学`, `为什么`, `学会的样子`, and `你已经会`. Name one transferable capability, connect it to the learner's actual output or teacher feedback, define observable success and preserve an existing strength. This is a stable skill map, not a preview of every future task; every later reply briefly says where the current action sits on that map.
11. Give one small but meaningful learning move per turn, normally an explanation plus one application taking about 5–10 minutes. Accept Chinese, keywords, arrows or incomplete English when appropriate. Never reduce the lesson to serial fill-in blanks, transcription, user-facing gates or copying a full sentence the AI just wrote. Quote every source detail required for the action unless memory is the explicit target. A large independent batch followed by one checkpoint also does not count as AI-in-the-loop practice. If the user says they do not know what they are learning, pause, explain the capability and why the abandoned action failed, and issue no new exercise until the orientation is understood.
12. Give every guided LN905 part's Markdown record an append-only block between `student-os:learning-log:start/end`. Before every teaching reply, append a timestamped record containing the learner's exact input, AI teaching/diagnosis, current skill-map location, the single next meaningful action, and the internal continuation decision. Preserve earlier entries, never remove the log when reusing a ready artifact, and do not append an exact duplicate of the latest turn. Chat is the teaching interface; the Markdown block is the canonical review archive. If the write fails, report that the learning record was not saved.
13. In an independent simulation, do not insert AI help before the timer ends. Add only a post-attempt debrief and targeted revision step.

## Compose Today

- Before creating or updating, search both the active root and daily archive for the same date. If that date is already archived, stop without changes; never recreate an archived date at the root.
- Active filename: `99_学习情况记录/YYYY-MM-DD——ddd.md`.
- Start from the template if missing.
- Keep these sections once: `今日安排`, `临近 Deadline`, `今日已完成`, `随手写`, `收尾`.
- Set the note's `date` property to the actual `YYYY-MM-DD` date. Do not replace the native Tasks query blocks; they read that fixed property so historical pages stay fixed without enabling Tasks JavaScript queries.
- Replace only content between `student-os:deadline-radar:start/end`.
- Keep one `student-os:shutdown-summary:start/end` block under `收尾`; replace only that block during shutdown reconciliation.
- Preserve all text in `随手写`, `收尾`, and outside managed blocks.
- Set `student-os:today-generated: YYYY-MM-DD` only after source tasks, Today, risk radar and calendar sync all succeed.

## Verify

- Each actionable item has one canonical checkbox.
- Today and Workbench queries filter exact Student OS tags and do not include course-note self-checks.
- Scheduled tasks and generated `Study Plan` events agree.
- Every generated `Study Plan` event has a display alert at its start time, with no duplicate start alert.
- Every scheduled guided LN905 part has one linked Markdown record containing exactly one part-specific init prompt and one `student-os:learning-log` block; its prompt names only that canonical task, requires a dedicated thread, begins with the four-field orientation card, prohibits recall-dependent or copy-back exercises and stops at that part's completion.
- D-0 through D-14 obligations are all visible.
- No movable weekday block starts before 15:30, no movable weekend block starts before 09:00, and no generated block extends past 19:30. Earlier fixed commitments and their directly attached preparation blocks are the only exceptions.
- A reconciled prior note contains one factual shutdown-summary block and no invented completion or reflection.
- No daily filename exists in both the active root and archive, and an archived same-date note is never recreated.
- No user-written text changed.
- A second run makes no additional changes.

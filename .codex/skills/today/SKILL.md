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
2. Read the applicable downloaded Moodle timetable for the user's LN905 Group 2 before scheduling: Week 1 is `07_Programme/01_LN905_LSE-language-class/PDF/00_Course-Info/Week-1/Wk1 Group 2.pdf`; Weeks 2–4 are `07_Programme/01_LN905_LSE-language-class/PDF/00_Course-Info/Weeks-2-4/week2-4 group2.pdf`. Treat its class periods as fixed commitments. Then read Apple Calendar; every calendar except exact name `Study Plan` is read-only, and `LSE` is a supplemental source for deadlines or changes, not proof that an empty day has no class.
   - If neither the applicable Group 2 PDF nor another verified timetable source can be read, do not place movable study blocks. Keep the next actions visible in Workbench, sync only independently verified fixed windows, and surface the source-check failure.
3. Show every incomplete hard deadline from D-0 through D-14, inclusive, as plain non-checkbox lines in Today's managed deadline block. Also show a deadline beyond D-14 when work for it is already scheduled.
4. Update Workbench's managed risk block from the same facts.
5. Do a bounded daily Student Hub/Moodle comparison when the logged-in source is reachable. On the first Sunday run, inspect assignments, files and announcements in full. Never silently treat an unreachable source as checked, never expose or store an ICS token, and never close a submission without Moodle/Turnitin status or a receipt.
6. Normal runs are quiet. Escalate only for a new D-14 deadline, an earlier deadline change, impossible capacity, repeated source-check failure, or an unresolved near-deadline obligation.

## Scheduling

- Schedule only canonical `#student-os/task` lines by adding or changing their Tasks `⏳ YYYY-MM-DD` metadata. Scheduled automation may edit only that metadata in plan files and Workbench task sources; do not rewrite their content.
- Hard deadline rows remain separate verified obligations. “Content done” and “submitted” are distinct.
- All generated calendar blocks go to `Study Plan`, for all future dates, never another calendar. Put a stable marker derived from source path plus task description in the event notes; update/delete only events carrying that marker. Give every generated block a display alert at the event start (`trigger interval: 0`) and preserve any other user-created alerts.
- On weekdays, place movable study blocks only from 15:30 through 19:30. On Saturdays, movable blocks may start at 09:00. Sunday is a full rest day: never schedule movable study, vocabulary, administration or catch-up work. A fixed Sunday obligation may still appear; when a submission deadline falls on Sunday and the portal permits early submission, schedule completion and submission by Saturday. Reserve 19:30–20:00 for packing/buffer and assume departure at 20:00. Fixed classes, travel, appointments and timed assessments may occur earlier; a preparation block tied directly to a fixed assessment may immediately precede it. Never auto-schedule movable work before the applicable start time or after 19:30.
- Use at most 70% of genuinely free weekday time and 50% on Saturdays.
- Preserve travel and buffers. A free gap of at least 60 minutes may hold a 45-minute study block; 30–59 minutes may hold light administration; under 30 minutes stays empty.
- A small assignment should have a 24-hour personal buffer and a major assignment 48 hours.
- D-14 work always remains visible. When near-deadline workload is heavy, suppress important-but-not-urgent fillers. When it is light, steadily advance confirmed self-directed courses by selecting their first incomplete lecture.
- For video courses, one lecture is one task. It is complete only when the user watched it and personally judges it understood. Do not invent required notes, exercises, tests, or mastery gates.
- Generate LN905 materials only for canonical sessions already defined in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`; never invent an extra daily checkbox or change a session's meaning.
- Schedule only the standalone Monday–Saturday `词灵` tasks already present in the confirmed LN905 plan, using the exact word-count or duration dose written on the canonical task. Never create or schedule a Sunday vocabulary task. Do not auto-create another vocabulary-prep block before Listening or Reading. For a timed submission, follow the plan's topic-vocabulary blocks and never expose the test material beforehand.
- Schedule LN905 ordinary practice only as one end-to-end Paper A or Paper B unit. One canonical checkbox and one Markdown/teaching task cover the complete chain and may receive several `Study Plan` blocks across days; only the finished full output and prescribed whole-output revision close it. Continue the newest unfinished unit before opening another. `词灵` remains separately checkable.
- Lock all new learner-facing LN905 materials, examples, callbacks and simulations to Social Media, Gender, Demographics or Climate Change. Reject any fifth-topic source. Keep an assigned Essay on its official topic as a separate obligation, never as generic practice material.
- Treat the Friday course Listening and Reading tests as the week's default full-performance samples. Do not schedule a redundant artificial full simulation in the same week unless a course sample is missing, AI-assisted, unusable, or the plan explicitly calls for a retest.

## LN905 planned material preparation

Run this content-preparation pass before composing Today whenever an LN905 practice task is scheduled for today.

1. Act as the LN905 mentor. Read `99_学习情况记录/teach/LN905 Exam Playbook.md`, the sole cross-session state source `99_学习情况记录/teach/LN905 Skill Bank.md`, the canonical task plan, the linked end-to-end record, weekly exam topic, source ledger, latest complete output, newest Friday samples and teacher feedback. Reuse the newest unfinished unit before selecting another. The task line and link must already exist; material generation never creates another checkbox.
   - Select current high-risk skill IDs and every naturally due callback, but keep their teaching and evidence inside the complete Paper A/B chain. Never create a skill clinic, standalone input drill, Shared Writing drill, readiness check or sentence clinic.
   - One Paper A unit runs `complete listening → notes/transcript repair/map → top-down plan → 200–400-word critical summary → whole-output feedback/revision`. One Paper B unit runs `full three-extract pack → map → thesis/paragraph jobs → about 600-word essay → whole-output feedback/revision`.
2. If the linked artifact is already marked ready and its companion files pass basic checks, reuse the artifact itself. Do not regenerate it and do not reuse any of its sources in another session.
3. For Paper A, select one previously unused complete video on the unit's one allowed exam topic from a verified official source and verify its transcript. Create one Markdown record and init prompt for the entire end-to-end unit. The dedicated teaching task resumes from that record across days and stops only after the full critical summary and revision.
4. For Paper B, use three previously unused credible scholarly sources on the unit's one allowed exam topic and create three accurate exam-style adapted extracts totaling at least the longer observed Paper B packet (currently about 1,650 body words), with each extract in the observed 425–610-word range. Preserve `.tex`, compile and visually verify `.pdf`, then create one Markdown record and init prompt for the entire end-to-end unit. It stops only after the full essay and revision.
5. Do not generate standalone English-production, readiness, post-test repair or callback tasks. Diagnose these from the complete output and teach them inside the next/current end-to-end record; Friday calibration remains read-only evidence work rather than a learner exercise.
6. Treat every existing lesson, source pack, `RESOURCES.md` entry and ledger identifier as used. Compare YouTube video IDs and paper DOI/canonical URLs. Append new identifiers to the plan ledger only after all session files verify successfully.
7. This pass may create the fixed session artifacts and append verified source identifiers. It may not add, duplicate, complete or rewrite a canonical checkbox; the scheduling pass remains limited to Tasks scheduling metadata.
8. If source verification, compilation or rendering fails, keep the task canonical but report the broken preparation explicitly and do not set `student-os:today-generated`.
9. Classify the session from its canonical task text. A task explicitly containing `完整模拟` is an independent simulation; every other LN905 practice task is guided AI-in-the-loop practice.
10. In every guided record, write exactly one hidden mentor-owned block between `student-os:mentor-brief:start/end` before the init prompt. Fix `今日 principal`, `重点技能 IDs`, `起点证据`, `完整产出`, `端到端教学链`, `允许支架`, silent callbacks/opportunities, hint ladder, feedback priority, completion evidence, writeback rule and stop boundary. The assistant may choose only the next 5–10 minute move inside this chain. Intermediate map, plan or sentence work writes progress/evidence but never closes the task.
11. When introducing a sentence pattern or procedure, prescribe same-day deliberate practice as `explain function and cue → supported discrimination or imitation → varied production → integration into an exam-style paragraph/summary`. Write the evidence only to the matching `99_学习情况记录/teach/LN905 Skill Bank.md` record. A `new` skill reaches `guided` only when all four steps have evidence; same-day prompted success never counts as independent. On later suitable material, observe any number of due skills only when each has a natural opportunity, without announcing them before the first complete output or manufacturing extra blanks. Record each callback as `independent`, `guided`, `incorrect` or `not observable`; no opportunity is not failure. If missed, use `function cue → structure cue → short contrast/model` and return to the principal. One unprompted use reaches `independent-1`; `stable` requires two unprompted uses on different material including one timed or Friday sample. Callback retrieval remains inside the scheduled output and never becomes another checkbox.
   - If `neath_sync: pending`, run `.codex/skills/today/scripts/sync_ln905_vocabulary.py` once after local planning. Failure is non-blocking and must not unset an otherwise valid Today marker. The script may read/create/update only; never add a deletion path or expose the Keychain credential.
12. In guided practice, use the official assessment introductions, marking criteria and Academic Writing slides as authority. The first reply or resumed reply gives `这次完整产出`, `为什么走完整链`, `考试流程`, `完成标准`, `你已经会` and `当前节点`. Every later reply locates the learner on that same chain and gives one meaningful next move. Do not reveal silent callbacks before the full first output.
13. Before asking the learner to act, teach the purpose and information transformation in plain Chinese and use one worked contrast or example when the mechanism is not already obvious. Then give one small but meaningful learning move, normally one application taking about 5–10 minutes. Accept Chinese, keywords, arrows or incomplete English when appropriate. Optimize for fast transferable skill gain, not local textual perfection: correct immediately only errors that invalidate the named capability's meaning, relationship, evidence or scope; defer recurring language patterns to a sentence clinic and let isolated wording, spelling, style and minor grammar errors pass. Never keep the learner on the same sentence or micro-distinction for serial turns. After one unsuccessful revision of the same issue, show a concise contrast or model and move to an integrated application; unless sentence production is the named target, require at most one whole-output revision. Never reduce the lesson to serial fill-in blanks, transcription, user-facing gates or copying a full sentence the AI just wrote. Quote every source detail required for the action unless memory is the explicit target. A large independent batch followed by one checkpoint also does not count as AI-in-the-loop practice. If the user says they do not know what they are learning, pause, explain the capability and why the abandoned action failed, and issue no new exercise until the orientation is understood.
   - In a top-down Writing drill, receive the whole planned or written unit before judging paraphrase. Accept different vocabulary, syntax, voice and information order when the proposition remains usable. Do not interrupt for local word-strength or style preferences; intervene only for a material change to source ownership, certainty, scope, causality or the evidence relationship serving the writer's answer, and batch lesser precision feedback afterward.
14. Give every guided LN905 unit's Markdown record an append-only block between `student-os:learning-log:start/end`. Before every teaching reply, append the learner's exact input, AI teaching/diagnosis, current end-to-end stage, the single next meaningful action and continuation decision, plus silent callback evidence after the first complete output. Preserve earlier entries and resume from them across days. If the write fails, report that the learning record was not saved.
15. In an independent simulation, do not insert AI help before the timer ends. Add only a post-attempt debrief and targeted revision step.

## LN905 Friday calibration

On the first Student OS run after a Friday course test appears in `07_Programme/01_LN905_LSE-language-class/00_inbox/`, read the newest paired Listening and Reading raw outputs and compare them with the previous Friday and any later teacher feedback. Diagnose acquisition separately and the shared Writing nodes jointly. Update skill evidence and use persistent weaknesses as teaching emphasis inside the next end-to-end unit; never generate a separate repair task. Do not remind callbacks before the timed attempt. One unprompted success reaches at most `independent-1`; only a second use on different material with a Friday/timed sample can reach `stable`. Calibration is read-only toward the Inbox.

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
- Every scheduled guided LN905 unit has one linked Markdown record containing exactly one unit init prompt, one mentor brief and one `student-os:learning-log` block. The prompt covers one complete Paper A/B chain, resumes across days, and cannot close on an intermediate map, plan or sentence.
- LN905 Skill Bank IDs are unique, every status is one of `new/guided/independent-1/stable/repair`, every callback result is one of `independent/guided/incorrect/not observable`, and no `new → guided` transition lacks all four explicit-loop evidence steps. If vocabulary sync runs, a repeated run creates no duplicate managed word and never changes or deletes non-LN905 collections; `pending` is non-blocking.
- An ordinary day continues at most one unfinished Paper A/B guided unit; several calendar blocks may reference the same canonical task. `词灵` does not count. A Paper B pack has three academic extracts in the observed 425–610-word range and at least 1,650 body words total. All learner-facing material uses exactly one of the four allowed topics. A Friday sample used for calibration remains byte-for-byte unchanged in the course Inbox.
- D-0 through D-14 obligations are all visible.
- No movable weekday block starts before 15:30, no movable Saturday block starts before 09:00, no movable block exists on Sunday, and no generated block extends past 19:30. Earlier fixed commitments and their directly attached preparation blocks are the only exceptions.
- A reconciled prior note contains one factual shutdown-summary block and no invented completion or reflection.
- No daily filename exists in both the active root and archive, and an archived same-date note is never recreated.
- No user-written text changed.
- A second run makes no additional changes.

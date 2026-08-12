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
- On weekdays, place movable study blocks only from 15:30 through 19:30. On weekends, movable blocks may start at 09:00. Reserve 19:30–20:00 for packing/buffer and assume departure at 20:00. Fixed classes, travel, appointments and timed assessments may occur earlier; a preparation block tied directly to a fixed assessment may immediately precede it. Never auto-schedule movable work before the applicable start time or after 19:30.
- Use at most 70% of genuinely free weekday time and 50% on weekends.
- Preserve travel and buffers. A free gap of at least 60 minutes may hold a 45-minute study block; 30–59 minutes may hold light administration; under 30 minutes stays empty.
- A small assignment should have a 24-hour personal buffer and a major assignment 48 hours.
- D-14 work always remains visible. When near-deadline workload is heavy, suppress important-but-not-urgent fillers. When it is light, steadily advance confirmed self-directed courses by selecting their first incomplete lecture.
- For video courses, one lecture is one task. It is complete only when the user watched it and personally judges it understood. Do not invent required notes, exercises, tests, or mastery gates.
- Generate LN905 materials only for canonical sessions already defined in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`; never invent an extra daily checkbox or change a session's meaning.
- Schedule only the standalone daily `词灵` tasks already present in the confirmed LN905 plan, using the exact word-count or duration dose written on the canonical task. Do not auto-create another vocabulary-prep block before Listening or Reading. For a timed submission, follow the plan's topic-vocabulary blocks and never expose the test material beforehand.
- Treat Listening and Reading as input routes into one shared Writing tree. Select one training lane rather than pairing both modalities by habit: either one standalone Shared Writing part, or a same-day input → Shared Writing pair using the same material. Place the Writing block immediately after the input block, allowing only a short break, so the new map is still fresh. Keep the pair as two canonical tasks with separate Markdown records and init prompts. If capacity cannot hold both, schedule standalone Writing with a compact supplied fact bank or short academic extract; never label that short support as Reading practice. If one shared Writing node remains weak in two comparable independent outputs, prioritize that node in the next two guided parts.
- Treat the Friday course Listening and Reading tests as the week's default full-performance samples. Do not schedule a redundant artificial full simulation in the same week unless a course sample is missing, AI-assisted, unusable, or the plan explicitly calls for a retest.

## LN905 planned material preparation

Run this content-preparation pass before composing Today whenever an LN905 practice task is scheduled for today.

1. Act as the LN905 mentor. Read `99_学习情况记录/teach/LN905 Exam Playbook.md`, the cross-session skill ledger in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`, the linked session path, focus, weekly topic, source ledger, latest completed output, the newest Friday course samples when present, and any teacher feedback. Sync Reading `NOTES.md`'s `current_week_topic` to the plan's current weekly topic. Use evidence to select one primary capability, its deliberate-practice dose and at most two due skills that fit a natural silent interleaved check. The task line and its link must already exist; material generation never creates a second task.
   - Classify the principal as either an input acquisition drill or a shared Writing drill. An input drill stops at a usable meaning/evidence map and does not require prose. Its paired Writing record must take that same map as its starting evidence without retesting retrieval. A standalone Writing drill receives a compact fact bank or short academic extract sufficient for the writing decision, then proceeds top-down from question and writer answer through paragraph functions and evidence roles before prose; never disguise short support material as Reading practice or exhaustive extraction as Writing practice.
2. If the linked artifact is already marked ready and its companion files pass basic checks, reuse the artifact itself. Do not regenerate it and do not reuse any of its sources in another session.
3. For Listening, select one previously unused video from the verified official TED YouTube channel, preferably 15–20 minutes, and verify the matching TED transcript. Create a standalone Markdown teaching record at the task's fixed path. Include only the agreed training focus, verified material links, relevant starting evidence, an empty `student-os:learning-log` block and a session-specific init prompt naming this exact task and record path. The prompt launches a new Codex thread that may teach only this part and must stop at its completion; never create a master thread prompt that chooses or advances several daily parts. Do not prebuild a worksheet, timer, field set or full sequence. Do not turn ordinary teaching practice into an unsupported full attempt. If the session needs suggested topic vocabulary, record it as reference material without creating another task or calendar block.
4. For a genuine Reading input drill, use three previously unused credible scholarly sources and create three accurate exam-style adapted extracts whose total body length is at least the longer observed Paper B packet (currently about 1,650 words); keep individual extracts in the observed 425–610-word academic-test range. Match or exceed the academic register, argument density and cross-source relationships in the user's Gender and Social Media packets. For a standalone Shared Writing drill, use only a compact fact bank or short academic extract sufficient for the named Writing decision; do not call it Reading practice. At the task's fixed basename, preserve the `.tex`, compile the printable `.pdf`, and create a standalone Markdown teaching-and-review record with its own task-specific init prompt. The PDF contains only the question, required extracts and source attributions; the Markdown contains the agreed focus, verified starting evidence, material links and durable learning log, not a full matrix or simultaneous response fields. Its dedicated thread handles only this part and stops on completion. Render every PDF page and inspect it before marking the session ready.
5. For other AI-guided LN905 parts such as English production, readiness checks or post-test review, create the same one-part Markdown record and part-specific init prompt from the named existing outputs, criteria and feedback. Do not make that thread handle Listening, Reading or any other adjacent part, and do not introduce new sources unless the canonical task requires them.
6. Treat every existing lesson, source pack, `RESOURCES.md` entry and ledger identifier as used. Compare YouTube video IDs and paper DOI/canonical URLs. Append new identifiers to the plan ledger only after all session files verify successfully.
7. This pass may create the fixed session artifacts and append verified source identifiers. It may not add, duplicate, complete or rewrite a canonical checkbox; the scheduling pass remains limited to Tasks scheduling metadata.
8. If source verification, compilation or rendering fails, keep the task canonical but report the broken preparation explicitly and do not set `student-os:today-generated`.
9. Classify the session from its canonical task text. A task explicitly containing `完整模拟` is an independent simulation; every other LN905 practice task is guided AI-in-the-loop practice.
10. In every guided record, write exactly one hidden mentor-owned block between `student-os:mentor-brief:start/end` before the init prompt. Fix these fields: `今日 principal`, `起点证据`, `本次产出`, `刻意练习链`, `允许支架`, `静默交织观察`, `反馈优先级`, `完成证据`, and `停止边界`. The brief is the teaching assistant's complete authority: it may grade each response and choose only the next 5–10 minute move inside the fixed chain. It may not edit the brief or skill ledger, select a new capability, create future homework, reschedule work, expose a silent observation before the first output, or declare long-term mastery. It may close the one canonical session task after the prescribed output meets the brief.
11. When introducing a sentence pattern or procedure, prescribe same-day deliberate practice as `explain function and cue → supported discrimination or imitation → varied production → integration into an exam-style paragraph/summary`. Add or update one ledger row with the skill, cue, prompted evidence, next suitable unprompted observation and status. On a later suitable part, embed at most two due ledger skills into natural source work without announcing them first; record whether they appeared independently. Same-day prompted success remains `guided`; use `stable` only after two unprompted uses on different material, including a timed or Friday sample when available. A retrieval check is part of the scheduled output, never another checkbox.
12. In guided practice, treat the official LN905 assessment introductions, marking criteria and Academic Writing slides cited by `99_学习情况记录/teach/LN905 Exam Playbook.md` as the authority for capabilities and success criteria; use teacher feedback to set priority. If uncertain, inspect those course files instead of inventing a rule, and label any symbols or time split not prescribed by the course as a training recommendation. Label the part as either a Listening/Reading input-route capability or a shared Writing node. The part-specific thread's first reply begins with a compact orientation card: `今天在学`, `为什么`, `考试位置`, `学会的样子`, and `你已经会`. Name one transferable capability; explain the real human writing problem it solves, how the action changes the information in front of the learner, and which next writing decision it unlocks; locate exactly where it is used in the playbook; define observable success; and preserve an existing strength. Do not reveal a silent interleaved observation in this card. “The teacher or criteria requires it” is not a sufficient `为什么`. The record and prompt also state one concrete output and one integration action. This is a stable skill map, not a preview of every future task; every later reply briefly says where the current action sits on that map, and completion records one exam cue in the form `遇到 X → 做 Y`.
13. Before asking the learner to act, teach the purpose and information transformation in plain Chinese and use one worked contrast or example when the mechanism is not already obvious. Then give one small but meaningful learning move, normally one application taking about 5–10 minutes. Accept Chinese, keywords, arrows or incomplete English when appropriate. Optimize for fast transferable skill gain, not local textual perfection: correct immediately only errors that invalidate the named capability's meaning, relationship, evidence or scope; defer recurring language patterns to a sentence clinic and let isolated wording, spelling, style and minor grammar errors pass. Never keep the learner on the same sentence or micro-distinction for serial turns. After one unsuccessful revision of the same issue, show a concise contrast or model and move to an integrated application; unless sentence production is the named target, require at most one whole-output revision. Never reduce the lesson to serial fill-in blanks, transcription, user-facing gates or copying a full sentence the AI just wrote. Quote every source detail required for the action unless memory is the explicit target. A large independent batch followed by one checkpoint also does not count as AI-in-the-loop practice. If the user says they do not know what they are learning, pause, explain the capability and why the abandoned action failed, and issue no new exercise until the orientation is understood.
   - In a top-down Writing drill, receive the whole planned or written unit before judging paraphrase. Accept different vocabulary, syntax, voice and information order when the proposition remains usable. Do not interrupt for local word-strength or style preferences; intervene only for a material change to source ownership, certainty, scope, causality or the evidence relationship serving the writer's answer, and batch lesser precision feedback afterward.
14. Give every guided LN905 part's Markdown record an append-only block between `student-os:learning-log:start/end`. Before every teaching reply, append a timestamped record containing the learner's exact input, AI teaching/diagnosis, current skill-map location, the single next meaningful action and the internal continuation decision, plus any silent interleaved-skill evidence after the first output. Preserve earlier entries, never remove the log when reusing a ready artifact, and do not append an exact duplicate of the latest turn. Chat is the teaching interface; the Markdown block is the canonical review archive. If the write fails, report that the learning record was not saved.
15. In an independent simulation, do not insert AI help before the timer ends. Add only a post-attempt debrief and targeted revision step.

## LN905 Friday calibration

On the first Student OS run after a Friday course test appears in `07_Programme/01_LN905_LSE-language-class/00_inbox/`, read the newest paired Listening and Reading raw outputs and compare them with the previous Friday and any later teacher feedback. Diagnose Listening and Reading acquisition separately up to the meaning/evidence map, then assess the shared Writing nodes jointly. Update only materially changed starting evidence and training priority in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`; do not manufacture a progress percentage. If the same shared Writing node is weak across two comparable Friday samples, make it the focus of the next two guided parts. If a sample is missing, AI-assisted or incomplete, record the limitation and do not treat it as an independent baseline. Calibration is read-only: never move, delete, rename or rewrite anything in the course Inbox.

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
- Every scheduled guided LN905 part has one linked Markdown record containing exactly one part-specific init prompt, one mentor brief and one `student-os:learning-log` block. Its prompt names only that canonical task, defines the thread as a teaching assistant under the immutable mentor brief, begins with the five-field orientation card, states a concrete output and integration action, prohibits recall-dependent or copy-back exercises and stops at that part's completion.
- An ordinary day has either one standalone guided LN905 part or one explicit input → Shared Writing pair using the same material; it never pairs Listening and Reading modalities by habit. Each paired part has its own checkbox, Markdown record and prompt, and the calendar places Writing directly after input. `词灵` does not count toward this lane limit. A genuine Reading input pack has three academic extracts in the observed 425–610-word range and at least 1,650 body words total. A Friday sample used for calibration remains byte-for-byte unchanged in the course Inbox, and any missing or limited calibration source is stated explicitly.
- D-0 through D-14 obligations are all visible.
- No movable weekday block starts before 15:30, no movable weekend block starts before 09:00, and no generated block extends past 19:30. Earlier fixed commitments and their directly attached preparation blocks are the only exceptions.
- A reconciled prior note contains one factual shutdown-summary block and no invented completion or reflection.
- No daily filename exists in both the active root and archive, and an archived same-date note is never recreated.
- No user-written text changed.
- A second run makes no additional changes.

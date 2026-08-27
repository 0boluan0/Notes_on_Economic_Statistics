# CLAUDE.md

This file provides guidance to AI agents working in this Obsidian vault.

## Repository Overview

This is an all-in-one academic and learning workspace. It contains course notes, reconstructed knowledge, papers and research notes, learning methods, and daily/weekly planning. Original paper files remain in Zotero; Obsidian stores notes, relationships, navigation, and learning state.

## Vault Structure

Numbered prefixes define folder ordering:

- **00_Knowledge/** - Shared knowledge files in one flat area; `_hubs/` contains only a small number of genuine cross-course navigation Hubs
- **00_inbox/** - Durable user-owned general capture and working area; never bulk-empty, move, or delete it during knowledge or course processing. Course directories have separate ephemeral `00_inbox/` folders for raw class inputs
- **01_Math/** - Mathematics courses
- **02_Economy/** - Economics and finance courses
- **03_Computer_Science/** - Computer science courses
- **04_Fragments/** - Captured external material awaiting or undergoing integration
- **05_tools/** - Tool and workflow notes
- **06_paper/** - Paper notes; original papers remain in Zotero
- **07_Programme/** - Programme-related material
- **98_attachment/** - Media and PDF attachments
- **99_学习情况记录/** - Daily/weekly study logs
- **毕业论文/** - Graduation thesis workspace
- **Excalidraw/** - Excalidraw diagrams

Do not hand-edit `.obsidian/`.

## Student operating system (active decisions)

### Design intent

- “All in one” means one coherent workflow and entry system, not one monolithic Markdown file.
- Optimize first for not missing school obligations and for reducing the user's micro-decisions. Ask only when a choice would materially change the result; decide small scheduling and presentation details autonomously.
- Keep operational pressure separate from the progress views that provide motivation. Do not turn a workload-driven pause into a false failure signal.

### Canonical roles

- **Today** is the daily home page: the day's scheduled actions, check-off surface, `今日已完成`, and an optional free-writing/closing area. It is the primary place where the user completes tasks.
- **Workbench** is the operational and risk queue. It aggregates current next actions across projects. A standalone one-off task may live there; a course or project task remains in its own source plan and is only shown in Workbench as a view.
- **Overall** is the human-readable source of truth for what the user has committed to learn. It lists both currently active and confirmed-to-learn tracks. Descriptions may stay inline, but any actionable checkbox lives in a linked plan or project file. Current position is derived from the first incomplete real unit, not maintained as a second manual status.
- **Dashboard** is a read-only learning and school-growth board that provides progress visibility and emotional value. It is not a second Workbench and does not carry deadline warnings, administrative chores, task-completion controls, or urgent decisions.
- **Deadlines** stores dated constraints and verified submission state. **Calendar** stores fixed commitments and generated study blocks. Together they prevent omissions and drive scheduling without polluting the growth Dashboard.
- The Course Atlas remains a knowledge map. It may show a learning position and next entry point, but operational tasks stay in the student operating system above.

### One task, one checkbox

- Every actionable item has exactly one canonical checkbox. Never copy the same task into Overall, a plan, Workbench, Today, and Dashboard as independently editable checkboxes.
- Today and Workbench show dynamic views of canonical tasks. Completing a task in Today updates its source, records the actual completion date, removes it from `今日安排`, and makes it appear in that date's `今日已完成` section.
- `⌘+Enter` is an alternative interface for completing or capturing against the same canonical task; it must not create another completion state.
- `今日已完成` includes every task actually completed that day, including useful work that was not originally scheduled.
- For formal submissions, “content complete” and “submitted” are separate states. Only a verified Moodle/Turnitin status or submission receipt closes the hard obligation.
- Canonical work units use the exact tag `#student-os/task`; verified hard obligations use `#student-os/deadline`. Operational queries must filter these exact tags so course-note self-checks never enter the system.
- Long sequential plans live under `99_学习情况记录/学习计划/`. Scheduled runs may change only Tasks scheduling metadata on their canonical task lines, not rewrite the task meaning.

### Learning tracks and progress

- Every user-confirmed must-learn track remains visible: detailed when active, compact in the confirmed queue when not yet started, and retained in a completed-results area after completion.
- Finite courses use real units and can display completed/total progress. Ongoing capabilities such as listening or reading-to-writing use the current practice plan and recent completed sessions; never invent a fake percentage.
- Multi-stage school projects may appear on the Dashboard as stage progress. One-off school tasks appear as recent accomplishments after completion rather than becoming permanent Dashboard cards.
- Show school responsibilities and self-directed growth as distinct kinds of accomplishment.
- For video courses, one lecture is one task. It is complete when the user has watched it and judges that they understood it. Do not add mandatory notes, exercises, tests, or mastery gates unless the agreed course plan explicitly includes them.

### Daily planning and automation

- Today's deadline radar shows every incomplete obligation from D-0 through D-14 inclusive. Also show a later deadline once work for it has been scheduled.
- Near-deadline work takes capacity first. When that workload is heavy, omit important-but-not-urgent fillers; when it is light, advance confirmed self-directed courses from their first incomplete unit. Make ordinary scheduling choices autonomously.
- On weekdays, movable study blocks start no earlier than 15:30 and end by 19:30. On Saturdays, movable blocks may start at 09:00 and end by 19:30. Sunday is a full rest day: never schedule movable study, vocabulary, administration or catch-up work. A fixed Sunday obligation may still appear; when a submission deadline falls on Sunday and the portal permits early submission, finish and submit by Saturday. Reserve 19:30–20:00 for packing/buffer. Fixed classes, travel, appointments and timed assessments may occur earlier; a required preparation block tied directly to a fixed assessment may immediately precede it. Use at most 70% of genuinely free weekday time and 50% on Saturdays.
- Temporary explicit exception for the confirmed LN905 Paper A/B exam sprint from 2026-08-25 through 2026-08-31: movable LN905 sprint work may start in the morning on weekdays and Monday 2026-08-31 because the user explicitly committed the full week and requested a morning-start pipeline. Fixed obligations and the 70% weekday / 50% Saturday capacity limits still take precedence; Sunday 2026-08-30 remains a full rest day.
- A gap of at least 60 minutes may hold a 45-minute study block; 30–59 minutes may hold light administration; shorter gaps remain empty. Preserve travel and buffers. Target a 24-hour personal buffer for small assignments and 48 hours for major assignments.
- Generated blocks live only in the writable iCloud Apple calendar `Study Plan`; never use the local `On My Mac` calendar with the same name. All other calendars, including `LSE`, are read-only inputs. Sync all future generated blocks, not only today's, and give every generated block a display alert at its start time so the planned start is an active cue rather than a passive calendar entry.
- The user is in LN905 **Group 2**. During this course, the matching downloaded Moodle Group 2 timetable PDF is the baseline fixed-class input. Before treating its room or timing as current, check recent official LSE course email; a dated official course email overrides the static PDF only for the dates it names. The Apple `LSE` subscription is supplemental and may contain deadlines without class meetings. Never infer “no class” from an empty `LSE` day.
- If neither the applicable Group 2 timetable nor another verified timetable source is readable, do not guess around an empty calendar: schedule only independently verified fixed windows and keep movable work unscheduled until a timetable source returns.
- Morning generation is an idempotent heartbeat in the same local Codex task, never a worktree or a new recurring task. The first awake run reconciles the newest marked but unclosed daily note, then creates or updates Today. If the computer or app is off, the next heartbeat catches up; do not fabricate empty missed-day notes.
- Every reconciliation writes a managed factual shutdown summary into the previous daily note before closing it. Derive it only from canonical tasks actually completed that date, unfinished tasks scheduled for that date, their verified carry date, and reminders the user already wrote. Distinguish planned from additional completions, use no new checkboxes, and never invent mood, causes, or reflection.
- There is no fixed evening shutdown trigger. A manual shutdown request may reconcile early; otherwise the next morning does it. Preserve all user writing in Today.
- Active current-week daily notes stay at the root of `99_学习情况记录/`. A completed weekly review stays visible at `99_学习情况记录/week-review/YYYY-Www.md`; after that review is complete and every marked daily note is shutdown-reconciled, move that ISO week's daily notes unchanged to `99_学习情况记录/archive/daily/YYYY-Www/`.
- Never archive the current ISO week before Sunday close, never archive before the review includes the user's answers, and never delete or overwrite a daily note during archival. If a review runs earlier, defer the move. A repeated archive run must be idempotent.
- Today and weekly review searches cover both active and archived daily notes. Archived notes are closed historical evidence: do not rewrite them, and never recreate an archived date at the root. After archival, only the weekly review remains in the visible weekly-history layer.
- Daily checks compare Student Hub/Moodle and recent official LSE course email when each logged-in source is reachable; the first Sunday run performs the fuller assignments/files/announcements audit. Never claim a source was checked when unavailable or expose a private calendar token.
- The confirmed replacement LN905 capability sequence lives in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`. Its active `#student-os/task` lines may enter Today; the cancelled legacy design remains `#student-os/paused` and must never be scheduled or regenerated. Formal timed submissions and hard deadlines remain active in their separate sources.
- The complete LN905 final-exam content scope is exactly four topics: **Social Media, Gender, Demographics and Climate Change**. This is an exhaustive list, not a set of examples. All new learner-facing practice material, card examples, recalls, callbacks and simulations must stay within those topics; no fifth topic may be introduced for teaching convenience. A formally assigned course essay on another topic remains a mandatory obligation, but its content is not reused as general LN905 practice.
- Keep lexical access, material comprehension, argument organization and English sentence production diagnostically distinct, but make each ordinary guided practice unit end-to-end. One Paper A unit runs from complete listening through notes/map, top-down plan, full critical summary and whole-output feedback/revision; one Paper B unit runs from the full three-extract pack through map, top-down plan, full essay and whole-output feedback/revision. The unit may span days, but only its finished end product closes the single canonical checkbox.
- Ordinary skill-building practice is teaching, not a mock assessment. AI may use Chinese explanation, multiple-choice discrimination, worked examples, sentence frames, collaborative drafting, replay and transcripts, then fade support only after demonstrated success. The AI-as-mirror restriction applies to assessed-answer feedback, not to separate teaching examples.
- Do not use giveaway multiple-choice questions whose answer can be inferred from wording, option tone or absurd distractors without understanding the material. Diagnostic options must all be genuinely plausible and require the target distinction; otherwise use a short Chinese explanation, relationship arrow or minimal constructed response. Never record success on an invalid item as evidence of mastery.
- A replacement LN905 sequence or timetable becomes active only after the diagnostic discussion and user confirmation. Once confirmed, keep one canonical checkbox per real practice unit; a standalone Monday–Saturday `词灵` session may have its own checkbox, while any preparation derived only from another task remains inside that task rather than becoming a duplicate checkbox.
- Explicitly label LN905 work as one of three phases: diagnostic/course design, teaching practice, or complete simulation. A diagnostic prompt measures one layer only; do not turn its answer into an immediate revision exercise or daily task. Announce and confirm the transition before teaching begins.
- Vocabulary practice through the website `词灵` is a parallel Monday–Saturday baseline, not a stage that must be finished before other LN905 work. Every scheduled vocabulary block or task must state a concrete dose as either a word count or a duration; choose that dose autonomously from the day's workload and the material rather than asking the user to make a micro-decision.
- Temporary explicit exception for the confirmed LN905 Paper A/B sprint from 2026-08-25 through 2026-08-31: the three-tier Anki sprint set is the only separately checkable vocabulary baseline on the six study days and replaces ordinary `词灵` blocks; never schedule both on the same date. Sunday 2026-08-30 still has no vocabulary task.
- The Neath (`匿词`) API is the approved Student OS interface for vocabulary-library management. Retrieve its credential only from macOS Keychain service `com.openai.codex.neath-api`, account `student-os`; never write or echo the key in vault files, prompts, logs, calendar notes, shell output or Git. `99_学习情况记录/teach/LN905 Skill Bank.md` is the local vocabulary source. Sync its managed entries idempotently to the five LN905 collections with read/create/update only; never expose an automatic delete path. A remote failure sets `neath_sync: pending` and never blocks Today or local practice.
- LN905 vocabulary selection is exam-use only. Keep high-probability topic words needed to understand Listening or Reading even when the learner may not produce them, plus reporting, causal, comparative and evaluative language the learner can directly use in notes or Writing. Exclude labels used only to teach writing methods, classroom procedure or marking, such as `paraphrase`, `thesis statement`, `nominalisation`, `hedging` and `passive voice`; the underlying skills remain teachable but their labels do not enter Neath unless later evidence shows that the word itself is likely to occur in exam input or the learner's answer.
- Anki is a selective production-practice subset, not a mirror of the Skill Bank. Sync only exam-use vocabulary and items directly usable in Writing or Speaking: words, collocations, sentence/discussion frames, compact editing/grammar/style rules, and pronunciation/delivery targets. Exclude classroom roles, workflow/process cards, Listening/Reading/note-taking strategies, evaluation/check questions, mapping/planning rules and exam-procedure cards. During a user-confirmed short exam sprint, one deliberately small recognition-only vocabulary tier extracted from the official topic materials may also enter Anki under the same five source decks; give it a separate sprint tag, test only English-to-meaning recognition, exclude it from every production query and Daily Recall, and never treat it as spelling, active use or mastery evidence. Removing an Anki note never deletes its canonical Skill Bank card.
- Keep Anki vocabulary under exactly five `LN905::Vocabulary` subdecks matching the canonical collections: `Academic Core`, `Social Media`, `Gender`, `Demographics` and `Climate Change`. Upsert each word into its source collection without creating a duplicate in the parent deck.
- Vocabulary remains a separately checkable Monday–Saturday baseline; Sunday has no vocabulary task. Inside an end-to-end unit, AI may teach and scaffold the current stage dynamically, but map, planning, sentence work and evaluation are saved only as progress in the same record; they are never separate drills, checkboxes or completion states. Continue the newest unfinished unit before opening another ordinary practice unit.
- Treat Listening and Reading as different input routes into one shared Writing capability tree. Modality-specific training ends when the learner has a usable meaning/evidence map; task interpretation, controlling answer, selection, accurate transformation, whole-text organization, paragraph reasoning, synthesis/evaluation, English production and timed completion are shared Writing capabilities, not two separate curricula.
- Keep input acquisition and top-down Writing conceptually distinct inside the same end-to-end record so the bottleneck remains diagnosable. A usable map immediately feeds `question/task → provisional answer/thesis → paragraph functions → evidence roles/selection → prose`; it is a checkpoint, not a task completion boundary.
- Calibrate guided Paper B units to the user's real packets. Until newer independent tests justify recalibration, use three academically written extracts adapted from credible scholarly sources, with total body length at least the longer observed packet (currently about 1,780 words); keep individual extracts in the observed academic-test range (currently about 425–653 body words), and make the language and source relationships at least as demanding as the Gender, Social Media and Demographics tests. Do not substitute a short fact bank or partial pack.
- Paraphrasing means preserving the material proposition while changing its wording and syntax, not matching the source lexically. In a top-down Writing drill, judge source use only after seeing the whole planned or written unit. Do not interrupt for ordinary synonym choice, sentence restructuring or a local difference in tone; intervene only when the output materially reverses or invents source ownership, certainty, scope, causal relationship or the evidence needed by the writer's claim. Batch lesser precision issues after the main Writing decision.
- Select the next Paper A or Paper B end-to-end unit adaptively from the latest independent output. Weak nodes determine where the tutor gives more explanation and feedback inside the whole chain; they never generate a standalone micro-drill. Do not run Paper A and Paper B ordinary units concurrently unless a formal timetable requires both.
- Use each Friday's course Listening into Writing and Reading into Writing tests as the primary weekly calibration samples. Read the raw work placed in `07_Programme/01_LN905_LSE-language-class/00_inbox/` on the next Student OS run after it appears, and incorporate teacher feedback when available. Diagnose Listening and Reading input separately, diagnose the shared Writing tree jointly, and let that evidence update training priority. The Friday tests normally replace extra artificial full simulations that week unless a test is missing, AI-assisted, unusable, or an explicit retest is required. Calibration is read-only: never move, delete or rewrite Inbox files; course-Inbox cleanup remains a separate course-processing workflow.
- Guided Listening material must preserve the complete argument. Do not shorten a complete talk by starting mid-argument or ending before its conclusion. When capacity requires a shorter exercise, select a shorter self-contained talk or clip whose boundaries retain the framing, central claim, supporting structure, and qualifications.
- Every AI-guided LN905 end-to-end unit uses one standalone Markdown file, one init prompt and one dedicated Codex teaching task for the entire chain; do not use HTML or split stages across separate tasks. The same task may resume across days from the append-only learning log and stops only after the complete output and prescribed whole-output revision.
- The Student OS task is the **mentor**: it selects Paper A or Paper B, the exam topic, current high-risk skill IDs, support level, evidence standard and end point. The dedicated Codex task is the **teaching assistant**: it teaches the current stage and assigns the next 5–10 minute move inside that one end-to-end chain. It may not create a standalone remedial drill, switch material, schedule future work or declare long-term mastery.
- Every guided record contains exactly one mentor-owned block between `student-os:mentor-brief:start/end`. It fixes `今日 principal`, `重点技能 IDs`, starting evidence, complete output, end-to-end teaching chain, allowed support, just-in-time prompted skills/language, independent-only observation opportunities, hint ladder, feedback priority, completion evidence, writeback rule and stop boundary. Intermediate stages append progress and skill evidence but never close the task.
- `99_学习情况记录/teach/LN905 Skill Bank.md` and its linked `LN905 Skill Bank/Cards/` are the only cross-session state source for all learned academic/assessment-transferable LN905 capabilities and vocabulary. The catalog keeps the capability map and protocols; every reusable sentence frame, collocation or small writing decision gets one atomic card whose front is the item and whose collapsed back holds its function, use conditions, boundaries, examples and append-only learning/recall history. Upsert rather than create synonym duplicates. Course notes retain full explanations; the practice plan retains tasks only. Use IDs in `AW`, `PA`, `PB`, `PC`, `AI`, `DISC`, `PRON` and `LEX`. Exclude pure Everyday English, classroom administration and game rules. Backfill skills only from taught material; vocabulary for a user-confirmed exam topic may be preloaded from official course materials before that class without activating its other skill states.
- Every important LN905 skill still needs `explanation → supported application → varied application → integration into a real exam-style output`, but all four steps occur naturally inside end-to-end units; never create a skill clinic or checkbox for one phrase, sentence form or isolated node. Legal states remain `new`, `guided`, `independent-1`, `stable` and `repair`.
- In guided teaching practice, useful academic sentence patterns, vocabulary/collocations and writing principles are not silent by default. When the learner is about to write a sentence or make a paragraph decision where an already taught or currently targeted item naturally fits, prompt it just in time: state the sentence's job, explain why the item fits, offer a small set of reusable frames or word choices, then let the learner construct the sentence. Do not supply a finished answer. Prompted use is recorded as `guided`, never `independent`. Silent observation is reserved for an explicit independent/complete simulation, a Friday/timed output, or a later mentor-designated unaided check after supported use; no reminder is given before those attempts. Do not manufacture callback blanks, interrupt for irrelevant items or turn the prose into a checklist; no natural opportunity is never failure.
- Whole-output feedback must make a reusable sentence problem understandable at the exact point where it is corrected. In the same correction row or comment, show the sentence job, the learner's concrete failure, the canonical Skill Bank frame link, the frame itself, when to use it and its meaning boundary; a separate frame index must never substitute for this inline explanation. Then upsert the same card with the independent evidence and, if needed, the smallest reusable variant and a source-faithful four-topic example. Do not prompt it before another independent/timed attempt, and do not create a separate drill or checkbox.
- Add one embedded active-recall block at each daily opening and shutdown. Daily recall uses the same production-language subset as Anki: exam-use words, phrases, collocations, sentence/discussion/cohesion frames, compact grammar/editing/style rules, and pronunciation/delivery targets. It must exclude task analysis, source mapping, evidence selection, paragraph planning, note-taking, reading/listening strategies and other procedural capability cards; those are taught and observed only inside end-to-end practice or timed calibration. Daily Recall tests retrieval rather than recognition or translation: hide the target English item, give its Chinese function, a genuine use condition and any meaning boundary needed to distinguish it, then ask the learner to retrieve the English word, collocation or sentence-frame skeleton. Do not require a complete self-made sentence or exact word-for-word reconstruction; reveal the canonical item and card back only after the answer. Immediately after the learner answers, preserve that answer and write a compact item-by-item answer key directly below it in the daily note: show the canonical English, the essential function or boundary, and whether the attempt was correct, partial or incorrect. Never leave a completed Recall attempt without its correction. A prompt that displays the target English and asks for a Chinese translation is invalid and creates no recall evidence. Actual production is observed in end-to-end writing. Each block includes every eligible card due or overdue at that checkpoint; shutdown also includes every eligible card newly learned or used since opening whose first recall is due that day. Never sample or cap. After correct recall, advance `+1, +3, +7, +14, +30` days; partial/incorrect returns next day; no answer is not failure and remains due. This remains one managed note block, including on Sunday, never a task, checkbox, deadline or calendar block, and never changes capability mastery automatically.
- Friday Paper A/B, Paper C and teacher feedback are primary calibration evidence. Timed attempts receive no callback reminder beforehand; update the Skill Bank only afterward. The Friday calibration also writes material callback evidence and the next appropriate silent observation, without creating another completion state or checkbox.
- The canonical top-down procedure is `99_学习情况记录/teach/LN905 Exam Playbook.md`. Every guided unit traverses that whole Paper A or Paper B procedure; current high-risk nodes affect teaching emphasis, not the unit boundary.
- Official LN905 assessment introductions, marking criteria and Academic Writing slides define the capability tree and success criteria; teacher feedback personalizes support inside the end-to-end unit. When uncertain, inspect those course files instead of inventing a skill, structure or marking rule.
- All learner-facing explanations, feedback and instructions use natural, conversational Chinese with short sentences. Explain every necessary English term, abbreviation or technical label immediately in plain Chinese and, when useful, one concrete example. This rule applies regardless of which skill, plugin or workflow is active. Skills may organize internal work, but their jargon, templates and process ceremony must not leak into the conversation or override a more specific course teaching contract and its existing record.
- End-to-end guided practice means the learner personally performs every exam-relevant decision in the order required in the exam. AI may explain and scaffold the current decision, but must not silently complete or hide an intermediate step—such as source routing, source mapping, thesis choice, paragraph jobs or evidence selection—and then resume at a later step.
- The first reply in every guided unit gives a compact orientation card: `这次完整产出`, `为什么走完整链`, `考试流程`, `完成标准`, `你已经会`, plus the current starting stage. Every later reply locates the learner on the same end-to-end map and gives one meaningful next move; no intermediate map, plan or sentence earns a separate completion.
- Give one small but meaningful learning move per turn, normally an explanation plus one application that takes about 5–10 minutes. Do not atomize teaching into serial blanks, transcription, ceremonial gates or retyping a sentence the AI has already supplied. Quote or display all source material needed for the action unless memory itself is the explicit target; ordinary guided practice must not accidentally test recall. Do not expose a full worksheet or require a large independent batch before feedback.
- Optimize ordinary guided practice for the fastest transferable skill gain, not local textual perfection. Correct immediately only an error that invalidates the named capability's meaning, relationship, evidence or scope; batch recurring language patterns for a later sentence clinic, and let isolated wording, spelling, style and minor grammar errors pass. Never keep the learner on the same sentence or micro-distinction for serial turns. After one unsuccessful revision of the same issue, teach it with a concise contrast or model and move to an integrated application. Unless sentence production is the named capability, require at most one whole-output revision and accept completion with minor language errors once the target capability is demonstrated.
- If the user says they do not know what they are learning, pause the exercise immediately. Explain the capability, why the abandoned action failed and what evidence would demonstrate learning; do not issue another exercise until the orientation is understood. Avoid user-facing `gate`, `pass/fail` or similar bureaucratic language.
- Keep an append-only block between `student-os:learning-log:start/end` in that Markdown record. Before every teaching reply, append the timestamped learner input, AI teaching/feedback, the current skill-map location, the single next meaningful action, and the internal continuation decision. Preserve all earlier entries and do not append an exact duplicate of the latest turn. If the write fails, say that the learning record was not saved.
- For a task explicitly marked `完整模拟` or an official timed submission, preserve exam validity: the user receives no AI help during the timed attempt. AI joins only after time expires for diagnosis, feedback and one targeted revision.
- The confirmed **Academic Writing one-week exam sprint** is pass-oriented and follows an ordered `hacking` method without assigning its phases to dates until the learner separately confirms the schedule: first build a minimal exam-use scaffold bank and learn it through retrieval; next improve Listening/Reading speed and accuracy on complete material, ending each input with a usable evidence map but without creating standalone micro-drill checkboxes; finally run complete timed Paper A/B simulations, diagnose the whole output and revise at most once. Do not spend scarce time on slow discovery-led reading instruction or extended foundations unless an error would materially break task or source accuracy. First establish a compact, function-labelled bank of sentence-frame skeletons indexed by question type: judgement, definition-plus-cause, recommendation, cause-plus-solution and extent/effectiveness questions require different thesis jobs, so never teach one universal thesis frame. In use, the learner must analyse the task parts and sources into content slots before selecting and filling a matching frame, never reverse-fit source meaning or a predetermined conclusion into memorised language. For the first guided essay, the tutor may provide paragraph jobs, source roles and reusable sentence-frame skeletons, while the learner still chooses the degree of the thesis, transforms evidence and writes all final prose. Use the official conditions as the target: approximately 600 words in 90 minutes, about 30 minutes for reading/planning and 60 for writing, with a direct answer, accurate use of all three texts, cross-text synthesis, logical paragraphs and a conclusion. Once full simulations begin, prioritize complete timed practice and whole-output correction over isolated exercises. Keep the Climate simulation and mandatory credit-scoring assignment in their separate existing records.

### Maintenance contract

- This section records durable, confirmed workflow decisions, not mutable daily progress or speculative ideas. Current progress belongs in Overall or the relevant source plan; deadlines belong in Deadlines.
- Whenever the user later confirms a material change to this workflow, update this section in both `AGENTS.md` and `CLAUDE.md` in the same change. Keep the two copies semantically aligned.
- The latest explicit user decision wins over an older rule. Update the files when that happens rather than preserving contradictory compatibility text.

## Active Knowledge Architecture

### Goal and organizing principle

- The system is designed for fast knowledge recovery: even after forgetting a course, the user should be able to see its overall direction, locate the relevant block, and relearn it quickly.
- A lecture is necessarily linear; knowledge is a network. Preserve lecture order as source context, but reorganize final knowledge according to its logical relationships.
- The knowledge network must still have a visible backbone and hierarchy. It must not become a flat graph of unexplained links.

### Canonical terms

#### Course note

A note that preserves course context: what was covered, the lecturer's sequence and emphasis, classroom examples, exercises, and course-specific requirements. Many existing course notes were generated by AI from course materials, so they are inputs to reconstruction rather than automatically trusted explanations.

#### Knowledge file

One **retrieval-atomic recovery unit**: opening it should answer one stable question or support one coherent use deeply enough to relearn and apply it. A knowledge file may combine definitions, intuition, assumptions, derivations, procedures, diagnostics, examples, counterexamples, and writing guidance when they belong to that same retrieval intent.

Do not split a file merely because it contains definitions, explanations, procedures, proofs, diagnostics, or writing guidance.

#### Atomic knowledge point

A precisely addressable idea, usually a heading or block inside a knowledge file. “Atomic” means independently locatable, not necessarily a separate Markdown file. Prefer links such as `[[双重差分法（DID）#平行趋势与辅助假设]]` or block links before extracting another file.

#### Hub

A navigation page for a real, dense cluster of independent knowledge files. A Hub is not a knowledge-content type. Keep Hub count limited and create one only after the cluster actually exists; do not pre-create a Hub for every important term.

#### Course Atlas

A Canvas overview for an entire course. It shows the course backbone, hierarchy, meaningful cross-links, the learner's current position, and the next entry point. It is a map, not a duplicate textbook.

### Merge and split rule

A broad lecture chapter or syllabus block is not a knowledge-file boundary. Split peer concepts when they have independent search intent, assumptions, use cases, or cross-course reuse. Do not split one method into definition/proof/procedure/diagnostic fragments that are normally used together: atomicity is measured by retrieval intent, not by heading count or length.

Keep sections in the same knowledge file when they:

- are normally retrieved and used together;
- share the same purpose, assumptions, and scope;
- form one continuous explanation;
- would become contextless fragments if separated.

For example, one DID knowledge file may contain the problem DID solves, 2×2 intuition, regression form, parallel trends, estimation, diagnostics, interpretation, and practical workflow.

A file is not complete merely because every syllabus term appears in one short paragraph. Canonical knowledge explanations must be deep enough to reconstruct the mechanism, conditions, examples, failure modes, distinctions, and application workflow relevant to the topic. Thin surveys belong in the Course Atlas or course overview.

Final knowledge files contain knowledge, not migration commentary. Never leave old-topic title stubs, `original ... retained for continuity`, `原主题名称`, `从原主题保留`, compatibility headings, or prose explaining where migrated text came from. Keep useful alternative names only in semantically reviewed YAML aliases, integrate substantive material into the current knowledge logic, and delete migration-only residue.

Create a separate knowledge file only when at least one real boundary appears:

- a materially different use case or audience;
- a materially different scope or set of assumptions;
- enough independent complexity to answer a separate substantial question;
- repeated independent reuse across multiple topics;
- keeping it inside the parent would cause conceptual confusion.

DDD or staggered-adoption DID may eventually become separate files because their designs and assumptions differ. A short explanation of parallel trends should normally remain a section until it develops a genuine independent role.

### Knowledge-file reading structure

- Begin every knowledge file with a one-screen `> [!summary] 快速恢复` callout before the detailed body.
- In plain language, the callout states: the problem this topic addresses, one concrete example or intuitive anchor, the central idea or difficulty, why it matters, and specific places to continue.
- Continuation links point to a relevant heading, related knowledge file, or Course Atlas. Do not fill the callout with a generic list of “related notes.”
- Standardize only this recovery entry. The detailed body follows the topic's own knowledge logic rather than a universal template.
- Write the callout for a reader who has mostly forgotten the subject; formal definitions and derivations come after the basic mental model.
- Before `## 来源与核验`, add `## 最小自检` with 3–5 substantive questions and collapsed `> [!answer]-` answers.
- Include at least one explain-in-your-own-words question and one application, diagnosis, or distinction question. Avoid trivia and prompts answerable by merely copying the preceding sentence.
- The self-check is a two-minute recovery check, not a mandatory spaced-repetition system. Do not create review schedules or recurring tasks from it unless the user later requests them.

### Course notes versus knowledge files

- Course notes preserve course-specific context; knowledge files provide the clearest current explanation.
- Course notes are permanent records. Never delete, replace, collapse, or treat them as disposable because a Course Atlas, overview, or knowledge file covers similar material. Deleting a course note requires explicit, file-specific user authorization.
- Treat each course note as a continuous reading view assembled in the course's own sequence. Keep the canonical explanation in the knowledge file and transclude required sections inline with `![[Knowledge file#Heading]]` or block embeds.
- Use ordinary `[[...]]` links only for optional extensions that the reader does not need to open immediately. Material required to understand the current course note must appear inline.
- Write course-specific bridge prose before and between embeds to explain why the course moves from one idea to the next. Do not produce a navigation list or an unexplained stack of embeds.
- Design transcluded knowledge sections to remain understandable when read independently; do not make them depend on unembedded source-file context.
- Even when a course note contains no unique substantive explanation, preserve it as a coherent assembled reading path rather than reducing it to a list of links.
- Avoid maintaining two editable copies of the same explanation: the knowledge file is the canonical content, while the course note supplies sequence, transitions, lecturer emphasis, examples, exercises, notation, and assessment context.
- Each course directory uses `00_课程总览.md` as its assembled continuous reading path. Original classroom notes remain separate, permanent course records; the overview may transclude or bridge them but never replaces them. Assessment, exercise, presentation, and source-context files likewise remain separate when they serve their distinct course role.
- Do not auto-link every occurrence of a term. Link only where opening the target helps at that reading point; repeated terms, mathematical symbols, formula fragments, code identifiers, and OCR noise are not navigation. When an old card becomes a heading in a broader knowledge file, use `[[Knowledge file#Heading|existing display text]]`, and remove meaningless generated link markup without changing the visible source text.
- Large course blocks may open the original course note or a useful child Canvas. A precise concept reference may open a validated knowledge file or one of its headings.

### Bilingual preservation

- Course notes and knowledge files are Chinese-first bilingual notes. Preserve every existing Chinese passage verbatim, including English terms embedded in it; never delete, paraphrase, reorder, or translate away the Chinese layer when adding English.
- Add an idiomatic English translation immediately after each Chinese semantic block. Keep Chinese headings unchanged and place an italic or plain-text English subtitle below them; never create a second Markdown heading, so the outline and heading links remain stable.
- Pure navigation controls are the exception: do not duplicate quick-jump lists, tables of contents, local-material lists, index-only sections, or link-only blocks in English. Preserve the original Chinese or mixed-language navigation once; these blocks do not carry explanatory knowledge.
- When pure navigation and explanatory prose occur inside the same callout or bilingual block, omit only the English copy of the navigation controls and retain the English explanation. A mixed block is not permission to duplicate `Local entry`, `Local materials`, PDF, exercise, or answer-link lists.
- Require English-only visible prose in Obsidian reading view. Preserve Chinese wikilink targets and source paths, but give them idiomatic English display labels such as `[[中文目标|English label]]`; do not leave visibly Chinese link text inside an English block.
- Keep translated callout content inside the original callout. Every marker, English line, and blank line belonging to the translation must carry the same `>` nesting prefix. Do not add a second bare English callout title such as `Self-test answer`; the existing Chinese callout title remains the single control label.
- Do not let an English ordered list continue the numbering of the Chinese list above it in Obsidian. Render English numbers explicitly as bold text with line breaks, for example `**1.** ...<br>`, rather than as a second Markdown ordered list. Before that first English number, keep a blank line and a standalone `&nbsp;` inside the English block so Obsidian closes the Chinese list; inside callouts, both lines retain the same `>` prefix. Prefix nested levels with one visible `↳` per level, for example `↳ **1.** ...<br>`; indentation entities alone are not reliable in Obsidian reading view.
- Keep YAML, code, standalone formulas, block IDs, link targets, and embeds structurally unchanged. Translate visible prose around them while preserving notation.
- Wrap every inserted English block between `<!-- bilingual-en:start -->` and `<!-- bilingual-en:end -->` so the English layer can be removed mechanically and the original Chinese layer verified byte-for-byte.
- Treat bilingual preservation and link migration as separate auditable operations. Adding English must be byte-for-byte reversible. A later link migration may change only wikilink targets, headings, aliases, or meaningless generated link markup when necessary, while preserving the rendered Chinese display text; it does not authorise rewriting course prose.

### Physical organization

- Store shared knowledge files directly in `00_Knowledge/`. Do not divide them into subject folders such as Mathematics, Economics, or Computer Science.
- A knowledge file may serve multiple courses and fields. Express this many-to-many membership through Course Atlases, Hubs, heading/block links, and backlinks instead of assigning one folder as its intellectual owner.
- Keep each Course Atlas in its corresponding course directory beside that course's notes, for example `01_Math/06_时间序列分析/时间序列分析 Course Atlas.canvas`.
- Keep cross-course Hubs in `00_Knowledge/_hubs/`, and create them only for already dense clusters.
- Give every course directory its own `00_inbox/` for recordings, transcripts, handwriting exports, screenshots, and other raw inputs awaiting processing.
- Course-local directories named `00_inbox/` are ephemeral and ignored by Git. Never use a course-local Inbox as durable storage or cite it as the permanent location of a source.
- Root `00_inbox/` is the explicit exception: it is a durable, user-owned working area. Reading it may inform a task, but migration or course processing must not clear, reorganize, move, or delete its contents. Deleting any file there requires explicit, file-specific user authorization.

### Ongoing course-processing workflow

- The user places raw material in the corresponding course's `00_inbox/` and asks Codex to process that course or class session.
- Codex reads the Course Atlas and existing course/knowledge files, processes every Inbox item, updates or creates the minimum necessary knowledge files, runs the source-based verification pass, assembles the continuous course note with transclusions, updates the Course Atlas, and verifies links and rendered continuity.
- Keep durable authoritative material such as a syllabus, slide deck, assigned reading, or problem set only when it is needed for future traceability. Move it out of `00_inbox/` into the course directory before cleanup.
- Treat recordings, generated transcripts, temporary exports, and processing intermediates as disposable. Delete them promptly only after confirming that their usable content has been incorporated, the resulting notes were written successfully, and any source needed for future verification has been preserved outside the Inbox.
- After processing, that course-local `00_inbox/` must be empty. This cleanup rule never applies to root `00_inbox/`. Briefly report what was retained, moved, and deleted, including whether deletion is recoverable.

### Sources, quality, and `source-checked`

- Start course reconstruction from the syllabus, slides, and other original course materials. A textbook may not exist.
- Course materials define the course's center but not the limit of knowledge. When materials are incomplete or poor, consult authoritative external sources because the objective is mastery, not merely exam reproduction.
- Clearly distinguish sourced claims, explanations synthesized by AI, course-specific opinions, and unresolved uncertainty.
- A knowledge file must remain readable to someone who is half-familiar or has forgotten the topic. Naming a condition or formula is not a substitute for explaining it.

- Use the course syllabus and slides to establish course scope and notation; use textbooks or authoritative courses to improve explanations; use primary papers for new methods, disputed claims, and exact identification conditions.
- Blogs and informal explainers may supply intuition or examples, but they cannot independently support a core definition, assumption, formula, or conclusion.
- Keep prose readable instead of citing every sentence. Add a nearby `> [!source] 本节依据` callout when a section contains important definitions, assumptions, formulas, method-dependent claims, or conclusions that may change with the literature.
- End every knowledge file with `## 来源与核验`. Each entry states which part of the file the source supports; never append an unexplained bibliography dump.
- Assign `status: source-checked` only after the core claims, terminology, formulas, assumptions, and method-sensitive conclusions are traceable to suitable sources and have actually been checked against them.

### Two-pass AI quality workflow

- Do not require the user to approve knowledge files one by one. AI performs both construction and a separate verification pass; the user reviews Course Atlases and reports problems encountered during real use.
- **Construction pass:** read course materials and suitable external sources; consolidate duplicates; write the knowledge file, recovery callout, aliases, links, and source notes. Keep `status: needs-review` throughout this pass.
- **Verification pass:** reopen and compare against the cited sources. Audit definitions, terminology, formulas, conditions, examples, assumptions, scope, aliases, merge/split decisions, heading links, relationship explanations, and whether the recovery callout works for a reader who has forgotten the topic.
- Prose polishing or a second unaided reread is not verification. The verification pass must actually use the sources.
- Change the status to `source-checked` only after verification succeeds. If a later edit materially changes core claims or formulas, return the file to `needs-review` until it is checked again.
- User feedback such as “this section is unclear” or “this relationship is wrong” triggers revision of the relevant knowledge file and affected Atlas nodes, followed by another verification pass.

### Course Atlas and relationship language

- Use Canvas as the primary whole-course map. Markdown is the main format for detailed knowledge. Add a child Canvas only when spatial structure adds real value.
- A major node must explain:
  1. what the block is, in plain language;
  2. a concrete example or intuitive anchor;
  3. why it matters;
  4. how it follows from or changes prior knowledge;
  5. what will be learned inside it;
  6. brief clickable sources.
- Do not use generic edge labels such as `前置`, `相关`, or `扩展` as if they explained the relationship. Put the actual relationship in a small bridge node in one or two sentences. If the relationship cannot be explained, do not draw it.
- The Course Atlas may display the learner's current position and next entry point. Detailed tasks remain in Workbench or daily notes.

## Markdown Conventions

### Frontmatter

```yaml
---
aliases:
  - Common alternative name
status: source-checked
---
```

- Knowledge files in `00_Knowledge/` have exactly two required properties: `aliases` and `status`.
- `status` has only two valid values:
  - `needs-review`: the file is not reliable for citation or automatic linking.
  - `source-checked`: the file satisfies the source-and-verification rules above.
- Do not add `subject`, `course`, `type`, created/updated dates, or similar classification metadata by default. Course and field membership come from links, Course Atlases, Hubs, and backlinks.
- In `00_Knowledge/`, choose the filename the user is most likely to recognize and search. Chinese, English, acronyms, and mixed names are all valid; never impose one language mechanically.
- Add common Chinese names, English full names, established abbreviations, and genuinely used variants to `aliases`.
- Review every alias semantically against the note content and accepted terminology. Python may validate YAML syntax, missing fields, and exact duplicates, but it must not invent, translate, approve, or remove aliases.

### Content

- Primary language is Chinese; use English where it is the natural technical language.
- Use hierarchical headings beginning at `#`.
- Use `$...$` and `$$...$$` for LaTeX.
- Use `[[note]]`, `[[note#heading]]`, and block links for internal navigation.
- Store linked media in `98_attachment/`.
- Use standard fenced code blocks with a language label. Code blocks contain source code, not REPL transcripts; print results explicitly.

### Executable Python blocks

Code Emitter runs Python through Pyodide rather than the local Python installation. Standard-library imports normally work directly. Third-party packages require asynchronous installation with `micropip`.

```python
import micropip
await micropip.install("numpy")
import numpy as np

a = np.random.rand(3, 2)
b = np.random.rand(2, 5)
print(a @ b)
```

Only Python, TypeScript, and JavaScript run in the local sandbox. Other languages may be sent to third-party execution services; never include secrets or sensitive source code.

## Templates and Learning Records

Durable Obsidian templates and QuickAdd scripts are stored in `05_tools/Obsidian/`. Course-local `00_inbox/` directories contain raw inputs and are emptied after successful processing; root `00_inbox/` is the durable user-owned exception and is never cleared by this workflow.

Daily notes live in `99_学习情况记录` and use the `YYYY-MM-DD——ddd` format.

## Git and Safety

- The Obsidian Git plugin creates automated daily commits in the form `自动: YYYY-MM-DD HH:MM`.
- The vault's `.gitignore` ignores course-local `00_inbox/` contents while explicitly keeping root `00_inbox/` trackable; inspect the file before assuming any additional path is ignored.
- Do not commit secrets or personal tokens.
- Preserve unrelated user changes. Review `git diff` before committing.
- When reorganizing notes, update links or use Obsidian rename behavior so references are not broken.

## Working with This Vault

When modifying notes:

1. Determine whether the target is a course note, knowledge file, atomic heading/block, Hub, or Course Atlas.
2. Read the relevant original course material before relying on unvalidated AI-generated notes.
3. Prefer improving one coherent file over creating several fragments.
4. Add only meaningful links and explain non-obvious relationships.
5. Verify frontmatter, backlinks, embeds, formulas, and referenced assets after editing.

When working with tasks:

1. Use the Tasks plugin's existing custom statuses.
2. Use `==text==` for Templater variable substitution.
3. Keep detailed task management in daily notes or Workbench, not in course knowledge maps.

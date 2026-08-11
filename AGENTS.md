# Repository Guidelines

## Project Structure & Module Organization
- Root is an Obsidian vault for academic notes. Key folders:
  - Root `00_inbox/`: the user's general capture and working area. Its existing files are durable and tracked; never bulk-empty, move, or delete this directory during knowledge or course processing. Course directories have separate ephemeral `00_inbox/` folders for raw class inputs.
  - `00_Knowledge/`: shared knowledge files in one flat knowledge area; `_hubs/` contains only a small number of genuine cross-course navigation Hubs.
  - `01_Math/`, `02_Economy/`, `03_Computer_Science/`: course and subject notebooks.
  - `04_Fragments/`: captured external material awaiting or undergoing integration.
  - `05_tools/`: tool and workflow notes.
  - `06_paper/`: paper notes; original papers remain in Zotero.
  - `07_Programme/`: programme-related material.
  - `98_attachment/`: images and assets referenced by notes.
  - `Excalidraw/`: diagrams created with the Excalidraw plugin.
  - `.obsidian/`: workspace config and plugins (do not hand-edit).

## Build, Test, and Development Commands
- No build step is required; open the folder in Obsidian.
- Content search: `rg "pattern" 02_Economy` (fast grep across notes).
- Change review: `git status && git diff` before committing.

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
- On weekdays, movable study blocks start no earlier than 15:30 and end by 19:30. On weekends, movable blocks may start at 09:00 and end by 19:30. Reserve 19:30–20:00 for packing/buffer. Fixed classes, travel, appointments and timed assessments may occur earlier; a required preparation block tied directly to a fixed assessment may immediately precede it. Use at most 70% of genuinely free weekday time and 50% on weekends.
- A gap of at least 60 minutes may hold a 45-minute study block; 30–59 minutes may hold light administration; shorter gaps remain empty. Preserve travel and buffers. Target a 24-hour personal buffer for small assignments and 48 hours for major assignments.
- Generated blocks live only in the writable Apple calendar `Study Plan`; all other calendars, including `LSE`, are read-only inputs. Sync all future generated blocks, not only today's, and give every generated block a display alert at its start time so the planned start is an active cue rather than a passive calendar entry.
- If `LSE` is missing or unreadable, do not guess around an empty timetable: schedule only independently verified fixed windows and keep movable work unscheduled until the timetable source returns.
- Morning generation is an idempotent heartbeat in the same local Codex task, never a worktree or a new recurring task. The first awake run reconciles the newest marked but unclosed daily note, then creates or updates Today. If the computer or app is off, the next heartbeat catches up; do not fabricate empty missed-day notes.
- Every reconciliation writes a managed factual shutdown summary into the previous daily note before closing it. Derive it only from canonical tasks actually completed that date, unfinished tasks scheduled for that date, their verified carry date, and reminders the user already wrote. Distinguish planned from additional completions, use no new checkboxes, and never invent mood, causes, or reflection.
- There is no fixed evening shutdown trigger. A manual shutdown request may reconcile early; otherwise the next morning does it. Preserve all user writing in Today.
- Active current-week daily notes stay at the root of `99_学习情况记录/`. A completed weekly review stays visible at `99_学习情况记录/week-review/YYYY-Www.md`; after that review is complete and every marked daily note is shutdown-reconciled, move that ISO week's daily notes unchanged to `99_学习情况记录/archive/daily/YYYY-Www/`.
- Never archive the current ISO week before Sunday close, never archive before the review includes the user's answers, and never delete or overwrite a daily note during archival. If a review runs earlier, defer the move. A repeated archive run must be idempotent.
- Today and weekly review searches cover both active and archived daily notes. Archived notes are closed historical evidence: do not rewrite them, and never recreate an archived date at the root. After archival, only the weekly review remains in the visible weekly-history layer.
- Daily Student Hub/Moodle checks compare the visible timetable/deadlines when the logged-in source is reachable; the first Sunday run performs the fuller assignments/files/announcements audit. Never claim a source was checked when unavailable or expose a private calendar token.
- The confirmed replacement LN905 capability sequence lives in `99_学习情况记录/学习计划/LN905 Listening and Reading Practice.md`. Its active `#student-os/task` lines may enter Today; the cancelled legacy design remains `#student-os/paused` and must never be scheduled or regenerated. Formal timed submissions and hard deadlines remain active in their separate sources.
- Keep lexical access, material comprehension, argument organization and English sentence production diagnostically distinct, but combine adjacent layers dynamically when the learner's evidence and daily capacity support it. Do not impose one-layer-per-day, and do not make a full lecture or three-paper pack, structure recovery, evaluation and a long blank-page response the entry requirement for ordinary practice.
- Ordinary skill-building practice is teaching, not a mock assessment. AI may use Chinese explanation, multiple-choice discrimination, worked examples, sentence frames, collaborative drafting, replay and transcripts, then fade support only after demonstrated success. The AI-as-mirror restriction applies to assessed-answer feedback, not to separate teaching examples.
- Do not use giveaway multiple-choice questions whose answer can be inferred from wording, option tone or absurd distractors without understanding the material. Diagnostic options must all be genuinely plausible and require the target distinction; otherwise use a short Chinese explanation, relationship arrow or minimal constructed response. Never record success on an invalid item as evidence of mastery.
- A replacement LN905 sequence or timetable becomes active only after the diagnostic discussion and user confirmation. Once confirmed, keep one canonical checkbox per real practice unit; a standalone daily `词灵` session may have its own checkbox, while any preparation derived only from another task remains inside that task rather than becoming a duplicate checkbox.
- Explicitly label LN905 work as one of three phases: diagnostic/course design, teaching practice, or complete simulation. A diagnostic prompt measures one layer only; do not turn its answer into an immediate revision exercise or daily task. Announce and confirm the transition before teaching begins.
- Vocabulary practice through the website `词灵` is a parallel daily baseline, not a stage that must be finished before other LN905 work. Every scheduled vocabulary block or task must state a concrete dose as either a word count or a duration; choose that dose autonomously from the day's workload and the material rather than asking the user to make a micro-decision.
- The Neath (`匿词`) API is the approved Student OS interface for vocabulary-library management. Retrieve its credential only from macOS Keychain service `com.openai.codex.neath-api`, account `student-os`; never write or echo the key in vault files, prompts, logs, calendar notes, shell output or Git. Read operations may be used when relevant; create or edit only on a user request or an agreed practice workflow, and require explicit target confirmation before any deletion.
- Schedule comprehension, accurate summary, and evidence-grounded evaluation dynamically from recent evidence, material difficulty, deadline pressure, and available capacity. A day may focus on one component or combine adjacent components; never impose a fixed serial timetable or a one-layer-per-day rule, and do not require the whole chain on a day when that would overload the user.
- Every AI-guided LN905 practice part—including Listening, Reading, English production, readiness checks and post-test review—uses its own standalone Markdown file as its durable review record and contains its own session-specific init prompt; do not create a guided-practice HTML worksheet or one master prompt for several parts. One part launches one dedicated Codex teaching thread scoped only to that canonical task. It must stop after that part and never select, expose, advance or complete another daily part.
- The first reply in every guided part gives a compact orientation card before the exercise: `今天在学` names one transferable capability in plain Chinese, `为什么` ties it to the user's real output or teacher feedback, `学会的样子` states an observable finish condition, and `你已经会` identifies the relevant existing strength. This is a skill map, not a list of all future tasks. Every later reply briefly locates the learner on that same map so the purpose of the current action remains visible.
- Give one small but meaningful learning move per turn, normally an explanation plus one application that takes about 5–10 minutes. Do not atomize teaching into serial blanks, transcription, ceremonial gates or retyping a sentence the AI has already supplied. Quote or display all source material needed for the action unless memory itself is the explicit target; ordinary guided practice must not accidentally test recall. Do not expose a full worksheet or require a large independent batch before feedback.
- If the user says they do not know what they are learning, pause the exercise immediately. Explain the capability, why the abandoned action failed and what evidence would demonstrate learning; do not issue another exercise until the orientation is understood. Avoid user-facing `gate`, `pass/fail` or similar bureaucratic language.
- Keep an append-only block between `student-os:learning-log:start/end` in that Markdown record. Before every teaching reply, append the timestamped learner input, AI teaching/feedback, the current skill-map location, the single next meaningful action, and the internal continuation decision. Preserve all earlier entries and do not append an exact duplicate of the latest turn. If the write fails, say that the learning record was not saved.
- For a task explicitly marked `完整模拟` or an official timed submission, preserve exam validity: the user receives no AI help during the timed attempt. AI joins only after time expires for diagnosis, feedback and one targeted revision.

### Maintenance contract

- This section records durable, confirmed workflow decisions, not mutable daily progress or speculative ideas. Current progress belongs in Overall or the relevant source plan; deadlines belong in Deadlines.
- Whenever the user later confirms a material change to this workflow, update this section in both `AGENTS.md` and `CLAUDE.md` in the same change. Keep the two copies semantically aligned.
- The latest explicit user decision wins over an older rule. Update the files when that happens rather than preserving contradictory compatibility text.

## Knowledge Architecture (active decisions)

### Purpose

- The vault is an all-in-one learning workspace: academic knowledge, course learning, papers, and day-to-day study planning belong here. Original papers remain in Zotero; Obsidian stores the knowledge structure, notes, links, and learning state.
- The goal is not to preserve lecture order. A course is delivered linearly, but knowledge is a network. Lecture order is useful source context, never the final organization.
- Optimize for rapid recovery: after forgetting a topic, the user should be able to see the whole-course direction, enter the relevant knowledge file, and relearn it without feeling lost.

### Canonical objects

- **Course note**: records the course context—what the course covered, the lecturer's emphasis, examples, exercises, and sequence. It is source material, not automatically the best final explanation.
- **Knowledge file**: one retrieval-atomic recovery unit that answers one stable question or supports one coherent use. It must explain that unit deeply enough to relearn and use it; definitions, intuition, assumptions, derivations, procedures, diagnostics, examples, counterexamples, and reporting guidance stay together when they serve that same retrieval intent.
- **Atomic knowledge point**: a precisely addressable heading or block inside a knowledge file. Atomicity does not require a separate file; link directly with `[[File#Heading]]` or block links.
- **Hub**: a scarce navigation page created only after several independent knowledge files form a genuinely dense cluster. A Hub is not a content type and must not be pre-created for every important term.
- **Course Atlas**: a Canvas overview of a whole course. It shows the backbone, hierarchy, meaningful cross-links, current position, and next entry point. Detailed explanations live in linked knowledge files or course notes.

### Merge and split rule

- A broad course block is not a knowledge-file boundary. Split peer concepts when they have independent search intent, assumptions, use cases, or cross-course reuse, even if a lecturer taught them in one chapter.
- Do not split a single method or concept into definition/proof/procedure/diagnostic fragments when those parts are normally needed together. Atomicity is measured by retrieval intent, not by heading count or file length.
- Keep material together when it is normally retrieved together, shares the same assumptions and use context, and forms one continuous explanation. For example, a DID file may contain its intuition, 2×2 setup, regression form, parallel trends, diagnostics, interpretation, and practical workflow.
- Split only when a part has a materially different use case or scope, different assumptions that would be confused with the parent topic, substantial independent complexity, or repeated independent reuse elsewhere.
- Prefer a heading/block link before extracting a new file. Promote a section to its own file only after an actual independent role appears.
- A knowledge file is not complete merely because every syllabus term appears once. Each major claim must be explained sufficiently to reconstruct the mental model, including the mechanism, conditions, at least one worked or concrete example, important failure modes or distinctions, and the route to application where relevant. Thin survey paragraphs belong in an Atlas or overview, not in the canonical knowledge explanation.
- Final knowledge files contain knowledge, not migration commentary. Do not leave old-topic title stubs, `original ... retained for continuity`, `原主题名称`, `从原主题保留`, compatibility headings, or explanations of where text came from. Put genuinely useful alternative names in reviewed YAML aliases, integrate substantive material into the current logic, and delete migration-only residue.

### Knowledge-file reading structure

- Every knowledge file begins with a one-screen `> [!summary] 快速恢复` callout before the detailed body.
- The quick-recovery callout answers, in plain language: what problem this topic addresses, one concrete example or anchor, the central idea or difficulty, why it matters, and where to continue.
- Make the continuation links specific: link to a relevant heading, related knowledge file, or Course Atlas rather than listing generic related notes.
- Only the quick-recovery entry is standardized. Organize the detailed body according to the topic's actual knowledge logic; do not force every file into one universal section template.
- A reader who has mostly forgotten the topic should gain a basic mental model from the callout before encountering formal definitions or derivations.
- Before `## 来源与核验`, add `## 最小自检` with 3–5 substantive questions and collapsed `> [!answer]-` answers.
- Include at least one explain-in-your-own-words question and one application, diagnosis, or distinction question. Avoid trivia and prompts answerable by merely copying the preceding sentence.
- The self-check is a two-minute recovery check, not a mandatory spaced-repetition system. Do not create review schedules or recurring tasks from it unless the user later requests them.

### Course notes and knowledge files

- Course notes preserve course-specific context. Knowledge files provide the clearest current explanation and may synthesize slides, syllabi, textbooks, and authoritative external sources.
- Course notes are permanent records. Never delete, replace, collapse, or treat them as disposable because an Atlas, overview, or knowledge file covers similar material. Moving the canonical explanation into Knowledge does not remove the course note's role as the record of sequence, teaching context, notation, examples, exercises, and source provenance. Deleting any course note requires explicit, file-specific user authorization.
- Treat each course note as a continuous reading view assembled in the course's own sequence. Keep the canonical explanation in the knowledge file and transclude required sections inline with `![[Knowledge file#Heading]]` or block embeds.
- Use ordinary `[[...]]` links only for optional extensions that the reader does not need to open immediately. Material required to understand the current course note must appear inline.
- Write course-specific bridge prose before and between embeds to explain why the course moves from one idea to the next. Do not produce a navigation list or an unexplained stack of embeds.
- Design transcluded knowledge sections to remain understandable when read independently; do not make them depend on unembedded source-file context.
- Even when a course note contains no unique substantive explanation, preserve it as a coherent assembled reading path rather than reducing it to a list of links.
- Avoid maintaining two editable copies of the same explanation: the knowledge file is the canonical content, while the course note supplies sequence, transitions, lecturer emphasis, examples, exercises, notation, and assessment context.
- Each course directory uses `00_课程总览.md` as its assembled continuous reading path. The original classroom notes remain separate, permanent course records; the overview may transclude or bridge them but never replaces them. Assessment, exercise, presentation, and source-context files likewise remain separate when they serve their distinct course role.
- Do not auto-link every occurrence of a term. Add a wikilink only where opening the target helps at that exact reading point; repeated terms, mathematical symbols, formula fragments, code identifiers, and OCR noise are not navigation. When a former card becomes a heading inside a broader knowledge file, retarget the useful occurrence to `[[Knowledge file#Heading|existing display text]]` and remove meaningless generated link markup without changing the visible source text.
- Course materials are the center of course reconstruction. If they are incomplete or poor, authoritative external sources may be used because the objective is mastery, not merely reproducing one lecturer's material.

### Bilingual preservation

- Course notes and knowledge files are Chinese-first bilingual notes. Preserve every existing Chinese passage verbatim, including every English term already embedded in it; do not delete, paraphrase, reorder, or translate away any part of the Chinese source text while adding English.
- Immediately after each Chinese semantic block, add an idiomatic English translation that communicates the same meaning rather than following Chinese word order mechanically. Translate headings by adding an italic or plain-text English subtitle below the unchanged Chinese heading; never create a second Markdown heading, so the outline and existing heading links remain stable.
- Pure navigation controls are the exception: do not duplicate quick-jump lists, tables of contents, local-material lists, index-only sections, or link-only blocks in English. Preserve the original Chinese or mixed-language navigation once; these blocks do not carry explanatory knowledge.
- When pure navigation and explanatory prose occur inside the same callout or bilingual block, omit only the English copy of the navigation controls and retain the English explanation. A mixed block is not permission to duplicate `Local entry`, `Local materials`, PDF, exercise, or answer-link lists.
- The English block must render as English-only prose in Obsidian reading view. When a wikilink target or local source filename is Chinese, retain the target but add an idiomatic English display label, for example `[[中文目标|English label]]`; a grammatically English sentence that visibly exposes a Chinese target name is not complete.
- Keep translated callout content inside the original callout. Every marker, English line, and blank line belonging to the translation must carry the same `>` nesting prefix. Do not add a second bare English callout title such as `Self-test answer`; the existing Chinese callout title remains the single control label.
- Do not let an English ordered list continue the numbering of the Chinese list above it in Obsidian. Render English numbers explicitly as bold text with line breaks, for example `**1.** ...<br>`, rather than as a second Markdown ordered list. Before that first English number, keep a blank line and a standalone `&nbsp;` inside the English block so Obsidian closes the Chinese list; inside callouts, both lines retain the same `>` prefix. Prefix nested levels with one visible `↳` per level, for example `↳ **1.** ...<br>`; indentation entities alone are not reliable in Obsidian reading view.
- Leave YAML, code, standalone formulas, block IDs, link targets, and embeds structurally unchanged. Translate visible prose around them and preserve technical notation in the English version.
- Delimit every inserted English block with `<!-- bilingual-en:start -->` and `<!-- bilingual-en:end -->`. These comments are hidden in reading view and make it possible to remove all English additions mechanically and verify that the original Chinese layer remains byte-for-byte intact.
- Keep bilingual preservation and link migration as two auditable operations. Adding English must be byte-for-byte reversible. A later link migration may change only wikilink targets, headings, aliases, or meaningless generated link markup when necessary, but it must preserve the rendered Chinese display text and be verified separately; it is not permission to rewrite course prose.

### Sources and `source-checked`

- Use the course syllabus and slides to establish course scope and notation; use textbooks or authoritative courses to improve explanations; use primary papers for new methods, disputed claims, and exact identification conditions.
- Blogs and informal explainers may supply intuition or examples, but they cannot independently support a core definition, assumption, formula, or conclusion.
- Keep prose readable: do not cite every sentence. Add a nearby `> [!source] 本节依据` callout when a section contains important definitions, assumptions, formulas, method-dependent claims, or conclusions that may change with the literature.
- End each knowledge file with `## 来源与核验`. For every listed source, state which part of the file it supports; do not append an unexplained bibliography dump.
- Assign `status: source-checked` only after the core claims, terminology, formulas, assumptions, and method-sensitive conclusions are traceable to suitable sources and have actually been checked against them.

### Two-pass AI quality workflow

- The user does not approve knowledge files one by one. AI performs both construction and a separate verification pass; the user reviews Course Atlases and reports confusion found during real use.
- **Construction pass:** read course materials and suitable external sources; consolidate duplicates; write the knowledge file, recovery callout, aliases, links, and source notes. Keep `status: needs-review` throughout this pass.
- **Verification pass:** reopen and compare against the cited sources. Audit definitions, terminology, formulas, conditions, examples, assumptions, scope, aliases, merge/split decisions, heading links, relationship explanations, and whether the recovery callout works for a reader who has forgotten the topic.
- Do not treat prose polishing or a second unaided reread as verification. The verification pass must actually use the sources.
- Change the status to `source-checked` only after the verification pass succeeds. When later corrections materially affect core claims or formulas, return the file to `needs-review` until it is checked again.
- User feedback such as “this section is unclear” or “this relationship is wrong” triggers revision of the relevant knowledge file and any affected Atlas nodes, followed by another verification pass.

### Physical organization

- Store shared knowledge files directly in `00_Knowledge/`; do not divide them into subject folders such as Mathematics, Economics, or Computer Science.
- A knowledge file may be used by multiple courses and fields. Express these many-to-many relationships through Course Atlases, Hubs, heading/block links, and backlinks rather than a single folder location.
- Keep each Course Atlas in its own course directory beside the corresponding course notes, for example `01_Math/05_随机过程/随机过程 Course Atlas.canvas`.
- Keep cross-course Hubs in `00_Knowledge/_hubs/`. Create only the few Hubs justified by an already dense cluster.
- Give every course directory its own `00_inbox/` for recordings, transcripts, handwriting exports, screenshots, and other raw inputs awaiting processing.
- Course-local directories named `00_inbox/` are ephemeral and ignored by Git. Never use a course-local Inbox as durable storage or cite it as the permanent location of a source.
- Root `00_inbox/` is the explicit exception: it is a durable, user-owned working area. Reading it may inform a task, but migration or course processing must not clear, reorganize, move, or delete its contents. Deleting any file there requires explicit, file-specific user authorization.

### Ongoing course-processing workflow

- The user places raw material in the corresponding course's `00_inbox/` and asks Codex to process that course or class session.
- Codex reads the Course Atlas and existing course/knowledge files, processes every Inbox item, updates or creates the minimum necessary knowledge files, runs the source-based verification pass, assembles the continuous course note with transclusions, updates the Course Atlas, and verifies links and rendered continuity.
- Keep durable authoritative material such as a syllabus, slide deck, assigned reading, or problem set only when it is needed for future traceability. Move it out of `00_inbox/` into the course directory before cleanup.
- Treat recordings, generated transcripts, temporary exports, and processing intermediates as disposable. Delete them promptly only after confirming that their usable content has been incorporated, the resulting notes were written successfully, and any source needed for future verification has been preserved outside the Inbox.
- After processing, that course-local `00_inbox/` must be empty. This cleanup rule never applies to root `00_inbox/`. Briefly report what was retained, moved, and deleted, including whether deletion is recoverable.

### Visual knowledge maps

- Use Canvas as the primary whole-course map; Markdown remains the detailed explanation format. Complex blocks may open a child Canvas only when the extra spatial layer is genuinely useful.
- A major map node must be understandable to a learner who has mostly forgotten the material. Include: what it is in plain language, a concrete example or anchor, why it matters, how it connects to prior knowledge, what the block contains, and brief sources.
- Do not label connections only as `前置`, `相关`, or similar generic words. Use a small explanatory bridge node stating the actual relation in one or two sentences. If the relation cannot be explained, do not draw it.
- Tasks remain in Workbench/daily notes. The Course Atlas shows only learning position and the next entry point.

## Coding Style & Naming Conventions
- Markdown notes:
  - Begin with YAML frontmatter bounded by `---`.
  - Every knowledge file in `00_Knowledge/` has exactly two required properties:
    - `aliases: []`: semantically reviewed names and abbreviations.
    - `status: needs-review | source-checked`: `needs-review` means the file is not reliable for citation or automatic linking; `source-checked` means it satisfies the source-and-verification rules above.
  - Do not add `subject`, `course`, `type`, created/updated dates, or similar classification metadata by default. Course and field membership come from links, Course Atlases, Hubs, and backlinks.
  - Filenames:
    - In `00_Knowledge/`, use the name the user is most likely to recognize and search. Chinese, English, acronyms, and mixed names are all allowed; do not impose one language mechanically.
    - Put common Chinese names, English full names, established abbreviations, and genuinely used variants in `aliases`.
    - Every alias must be reviewed semantically by an AI against the note content and accepted terminology. Python may check YAML syntax, missing fields, and exact duplicates, but must not invent, translate, approve, or remove aliases by itself.
    - Course notes and Hub notes keep their existing naming style; a Hub may use a `-hub` suffix such as `增长理论-hub.md`.
  - Headings start at `#` with sentence‑case titles; prefer short sections.
- Python utilities: `snake_case` filenames, 4‑space indentation, Black-compatible style.

## 可执行代码块（Code Emitter）规范
- 代码块仍然使用标准 fenced code block：三反引号包裹，并写语言标签（如 `python` / `javascript` / `typescript` / `html`）。
- 只写源码，不要粘贴 REPL 记录（例如 `>>>` 不是合法 Python 源码）。
- 想展示结果就显式输出：Python 用 `print(...)`（或该语言的标准输出方式），不要把运行结果文本混在代码里。
- 语言安全边界：只有 Python/TypeScript/JavaScript 在本地沙盒执行；其他语言会发送到第三方网站执行，禁止放敏感源码/密钥。

### Python import 要点（Pyodide）
- Code Emitter 的 Python 是 WebAssembly 的 Pyodide，不是本机 Python。
- 标准库（`math`/`os` 等）通常可直接 import。
- 第三方库需按 README 用 `micropip` 安装，并且是异步 `await`。

### 可复用模板
```python
print(type(5))
print(type(3.0))
```

```python
import micropip
await micropip.install("numpy")
import numpy as np

a = np.random.rand(3, 2)
b = np.random.rand(2, 5)
print(a @ b)
```

```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

## Testing Guidelines
- No unit tests. Validate content changes manually:
  - After running the script, inspect with `git diff`.
  - Verify backlinks and embeds render in Obsidian; check missing assets in `98_attachment/`.
  - Spot-check tags and aliases on a few updated notes.

## Commit & Pull Request Guidelines
- Commit messages: concise and descriptive. Examples:
  - `notes: 重建DID知识文件`
  - `notes(02_Economy): 新增索罗模型卡片`
  - `notes: 审核知识文件别名`
- Group related edits; avoid mixing content and config changes.
- PRs (if used): include summary, affected folders, and screenshots for visual changes.

## Security & Configuration Tips
- Do not commit secrets or personal tokens. Avoid manual edits in `.obsidian/`.
- Keep large media in `98_attachment/`; link via relative paths.
- When reorganizing files, update links or use Obsidian’s rename to preserve references.

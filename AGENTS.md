# Academic Vault Instructions

## Instruction budget

- This is the canonical repository-instruction file. `CLAUDE.md` only points here; never duplicate the rules in both files.
- Add a rule only when it is a confirmed, durable, vault-specific decision that materially changes repeated work or prevents likely data loss, and cannot be inferred from the current vault or delegated to a task-specific skill or workflow note.
- Do not record current progress, current deadlines, one-off task context, speculative preferences, directory tours, command or syntax tutorials, examples, implementation commentary, or rules already supplied by the platform or an applicable skill.
- When the user changes a decision, replace the old rule in place and remove superseded wording. Do not preserve compatibility prose or append another near-duplicate.
- Keep the shortest wording that preserves the decision. If operational detail grows, move it to the relevant skill or source-of-truth note and leave only the invariant or pointer here.

## Safety and ownership

- Root `00_inbox/` is durable, private, user-owned workspace and stays outside Git. Never bulk-empty, move, reorganize, or delete it during knowledge or course processing; deleting an item there requires explicit, file-specific authorization.
- A course-local `00_inbox/` is ephemeral. Empty it only after usable content has been incorporated successfully and any authoritative source needed later has been moved into durable course storage; report what was retained and deleted and whether deletion is recoverable.
- Course notes are permanent source records even when atoms or Canvases cover the same material. Deleting one requires explicit, file-specific authorization.
- Course-material updates keep only the latest teacher-issued version; move superseded files to Trash instead of archiving previous versions.
- Preserve unrelated edits and all user-written daily-note text. Do not hand-edit `.obsidian/`, commit secrets, or break links when moving notes.
- Keep disposable AI scratch files outside the vault; retain necessary resume state and verification evidence, and remove superseded intermediates after acceptance.
- Store actual learning progress, mastery, review schedules, and personal evidence only in ignored private records; shared atoms and public course notes/Canvases contain general learning content rather than personal state.
- Original papers remain in Zotero; the vault stores notes, relationships, navigation, and learning state.
- Keep private quant notes, strategies, code, data, and outputs in root `08_quant/`, excluded from Git and public exports. General reusable atoms may live in `00_Knowledge/`, without private strategy details, results, or links into `08_quant/`.

## Student operating system

- Before personal planning, teaching, learning-record work, or LSE programme/support decisions, read `99_学习情况记录/Student OS 规则.md` when present and follow its relevant resource routing. Its private operating rules stay outside Git.
- Course-material updates include the linked Hanvon learning books and computer code/data entries. Before updates or any Hanvon transfer, classification, cleanup, or handwriting return, read `99_学习情况记录/手写接收/汉王/传输与归档规则.md`; delivery requires the intended native entry point and classification, not file copying alone.

## Knowledge architecture

- Do not use Dataview or DataviewJS in vault notes or templates; use ordinary Markdown links and tables for navigation and summaries.
- A **knowledge atom** is the canonical Markdown object for one independently recallable or applicable definition, proposition, distinction, decision rule, or procedural judgement. A same-named **Topic Canvas** integrates its atoms; a course-local **Course Atlas** shows the whole-course backbone; the global **Base** is query-only; `_hubs/` contains only genuinely dense cross-topic navigation.
- Store shared atoms and Topic Canvases directly in `00_Knowledge/`, Course Atlases beside their course notes, and scarce Hubs in `00_Knowledge/_hubs/`. Express many-to-many membership through links and projections, not subject subfolders or duplicated content.
- Give each atom one semantic responsibility. A reusable theorem, condition, boundary, counterexample, distinction, or decision step gets its own atom even if a lecture grouped it with others. Atomicity limits the topic, not explanatory depth: explain that item fully enough for a reader encountering it for the first time.
- Every atom requires `student_os: knowledge-atom`, a stable unique `atom_id`, semantically reviewed `aliases`, and `status: needs-review | source-checked`. Do not add subject/course/type/date classification by default. If `atom_type` exists, it must match the atom's actual responsibility; use `definition` for reviewed concept entries.
- Use a short recognizable filename and a natural H1 naming the concept, theorem, or question. Keep `atom_id` stable and use genuine alternative names as aliases. A concept entry opens by explaining what the concept means in ordinary language; introduce formal notation and conditions after establishing their meaning.
- Keep learner-facing Properties hidden while retaining YAML. Write a coherent explanation whose headings and sequence fit the concept. Use callouts selectively to distinguish examples, key results, and pitfalls while keeping the explanation continuous. Explain new symbols and reasoning with concrete examples. Introduce needed prerequisites in the prose with a brief orientation and an inline wikilink to the exact definition atom, so a newcomer can follow the explanation and open any missing background. End with `## 来源与核验`; add a recall/application prompt only when it helps understanding.
- Source reliability, learner mastery, and review scheduling are independent. `status` records source verification only; Canvases, Atlases, overviews, and Bases may project these states but never own another editable copy.
- Author aliases, atom boundaries, relationship meanings, and link destinations semantically. Scripts may inventory references and validate syntax, IDs, and targets, but must not make those judgements. Link only where opening the exact target helps the reader.
- Course `00_课程总览.md` is a continuous course-order reading path: preserve bridge prose and embed the exact required atoms inline. It complements rather than replaces permanent classroom, assessment, exercise, and source-context notes.
- Use course materials for scope and notation, authoritative textbooks/courses for explanation, and primary papers for new or disputed methods and exact conditions. Blogs may support intuition, not core claims. Each source entry states what it supports.
- Construction keeps `status: needs-review`. Verification must reopen the cited sources and check claims, terminology, formulas, assumptions, boundaries, examples, aliases, links, and Canvas relationships; only then set `source-checked`. Material claim or formula changes return the atom to `needs-review` until rechecked.
- Migrate one topic transactionally: inventory every inbound Markdown and Canvas reference, map each one in context, update all affected projections, validate the result, and only then delete the legacy composite. Any ambiguity or failed validation keeps or restores the pre-migration state; leave no stub, dual source, or half-migration.
- Topic Canvas nodes are compact link cards. Generic concept labels target definition atoms; narrow claims get their own visible nodes. State non-obvious relationships precisely or in a short bridge node, never as unexplained `前置` or `相关` edges.

## Bilingual preservation

- Course notes and atoms are bilingual, with English as the primary reading language and Chinese as support. Preserve existing content in both languages verbatim, including embedded terms.
- Wrap English semantic blocks with `<!-- bilingual-en:start -->` and `<!-- bilingual-en:end -->`. New notes and reading layouts present English first (left of Chinese in parallel layouts), with English headings and Chinese subtitles. Carry existing section numbering into the English headings. Changes to existing notes must preserve heading anchors and link targets.
- Do not duplicate pure navigation, contents, local-material, or link-only blocks. English prose must render English-only: retain Chinese link targets but add English display labels.
- Keep translated callout lines and blank lines inside the original callout prefix. To prevent list-number continuation, precede English numbering with a blank line and `&nbsp;`, render numbers as `**1.** ...<br>`, and mark nesting with visible `↳`.
- Keep YAML, code, standalone formulas, block IDs, link targets, and embeds structurally unchanged. Treat bilingual insertion and later link migration as separate, independently verifiable operations.

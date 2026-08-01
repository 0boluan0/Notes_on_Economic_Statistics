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
- **Overall** is the human-readable source of truth for what the user has committed to learn. It lists both currently active and confirmed-to-learn tracks. Short plans may live inline; long sequential plans live in a linked or embedded plan file. Current position is derived from the first incomplete real unit, not maintained as a second manual status.
- **Dashboard** is a read-only learning and school-growth board that provides progress visibility and emotional value. It is not a second Workbench and does not carry deadline warnings, administrative chores, task-completion controls, or urgent decisions.
- **Deadlines** stores dated constraints and verified submission state. **Calendar** stores fixed commitments and generated study blocks. Together they prevent omissions and drive scheduling without polluting the growth Dashboard.
- The Course Atlas remains a knowledge map. It may show a learning position and next entry point, but operational tasks stay in the student operating system above.

### One task, one checkbox

- Every actionable item has exactly one canonical checkbox. Never copy the same task into Overall, a plan, Workbench, Today, and Dashboard as independently editable checkboxes.
- Today and Workbench show dynamic views of canonical tasks. Completing a task in Today updates its source, records the actual completion date, removes it from `今日安排`, and makes it appear in that date's `今日已完成` section.
- `⌘+Enter` is an alternative interface for completing or capturing against the same canonical task; it must not create another completion state.
- `今日已完成` includes every task actually completed that day, including useful work that was not originally scheduled.
- For formal submissions, “content complete” and “submitted” are separate states. Only a verified Moodle/Turnitin status or submission receipt closes the hard obligation.

### Learning tracks and progress

- Every user-confirmed must-learn track remains visible: detailed when active, compact in the confirmed queue when not yet started, and retained in a completed-results area after completion.
- Finite courses use real units and can display completed/total progress. Ongoing capabilities such as listening or reading-to-writing use the current practice plan and recent completed sessions; never invent a fake percentage.
- Multi-stage school projects may appear on the Dashboard as stage progress. One-off school tasks appear as recent accomplishments after completion rather than becoming permanent Dashboard cards.
- Show school responsibilities and self-directed growth as distinct kinds of accomplishment.
- For video courses, one lecture is one task. It is complete when the user has watched it and judges that they understood it. Do not add mandatory notes, exercises, tests, or mastery gates unless the agreed course plan explicitly includes them.

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

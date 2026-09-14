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
- Preserve unrelated edits and all user-written daily-note text. Do not hand-edit `.obsidian/`, commit secrets, or break links when moving notes.
- Store actual learning progress, mastery, review schedules, and personal evidence only in ignored private records; shared atoms and public course notes/Canvases contain general learning content rather than personal state.
- Original papers remain in Zotero; the vault stores notes, relationships, navigation, and learning state.

## Knowledge architecture

- A **knowledge atom** is the canonical Markdown object for one independently recallable or applicable definition, proposition, distinction, decision rule, or procedural judgement. A same-named **Topic Canvas** integrates its atoms; a course-local **Course Atlas** shows the whole-course backbone; the global **Base** is query-only; `_hubs/` contains only genuinely dense cross-topic navigation.
- Store shared atoms and Topic Canvases directly in `00_Knowledge/`, Course Atlases beside their course notes, and scarce Hubs in `00_Knowledge/_hubs/`. Express many-to-many membership through links and projections, not subject subfolders or duplicated content.
- Give each atom one semantic responsibility. A reusable theorem, condition, boundary, counterexample, distinction, or decision step gets its own atom even if a lecture grouped it with others. Keep only the explanation, conditions, boundary, minimal rationale, and smallest useful example or contrast needed to use that item.
- Every atom requires `student_os: knowledge-atom`, a stable unique `atom_id`, semantically reviewed `aliases`, and `status: needs-review | source-checked`. Do not add subject/course/type/date classification by default. If `atom_type` exists, it must match the atom's actual responsibility; use `definition` for reviewed concept entries.
- Use three identity layers: a short recognizable filename; a complete proposition as H1 and `aliases[0]`; and stable `atom_id`. A concept-entry filename may be short, but its H1 and first body block must directly define the concept rather than open on a narrow property.
- Keep learner-facing Properties hidden while retaining YAML. Start the visible body with the core statement and use condition; end with `## 来源与核验`. Add a recall/application prompt only when it genuinely tests the atom.
- Source reliability, learner mastery, and review scheduling are independent. `status` records source verification only; Canvases, Atlases, overviews, and Bases may project these states but never own another editable copy.
- Author aliases, atom boundaries, relationship meanings, and link destinations semantically. Scripts may inventory references and validate syntax, IDs, and targets, but must not make those judgements. Link only where opening the exact target helps the reader.
- Course `00_课程总览.md` is a continuous course-order reading path: preserve bridge prose and embed the exact required atoms inline. It complements rather than replaces permanent classroom, assessment, exercise, and source-context notes.
- Use course materials for scope and notation, authoritative textbooks/courses for explanation, and primary papers for new or disputed methods and exact conditions. Blogs may support intuition, not core claims. Each source entry states what it supports.
- Construction keeps `status: needs-review`. Verification must reopen the cited sources and check claims, terminology, formulas, assumptions, boundaries, examples, aliases, links, and Canvas relationships; only then set `source-checked`. Material claim or formula changes return the atom to `needs-review` until rechecked.
- Migrate one topic transactionally: inventory every inbound Markdown and Canvas reference, map each one in context, update all affected projections, validate the result, and only then delete the legacy composite. Any ambiguity or failed validation keeps or restores the pre-migration state; leave no stub, dual source, or half-migration.
- Topic Canvas nodes are compact link cards. Generic concept labels target definition atoms; narrow claims get their own visible nodes. State non-obvious relationships precisely or in a short bridge node, never as unexplained `前置` or `相关` edges.

## Bilingual preservation

- Course notes and atoms are Chinese-first bilingual. When adding English, preserve the existing Chinese layer byte-for-byte, including embedded English terms; do not delete, paraphrase, reorder, or translate it away.
- Add idiomatic English immediately after each Chinese semantic block and wrap every inserted block with `<!-- bilingual-en:start -->` and `<!-- bilingual-en:end -->`. Keep Chinese headings unchanged and add English subtitles without creating another heading.
- Do not duplicate pure navigation, contents, local-material, or link-only blocks. English prose must render English-only: retain Chinese link targets but add English display labels.
- Keep translated callout lines and blank lines inside the original callout prefix. To prevent list-number continuation, precede English numbering with a blank line and `&nbsp;`, render numbers as `**1.** ...<br>`, and mark nesting with visible `↳`.
- Keep YAML, code, standalone formulas, block IDs, link targets, and embeds structurally unchanged. Treat bilingual insertion and later link migration as separate, independently verifiable operations.

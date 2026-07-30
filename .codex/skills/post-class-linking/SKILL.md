---
name: post-class-linking
description: Process a course Inbox after class and update the course overview, reviewed knowledge files, transclusions, and Course Atlas. Use when the user asks for 课后整理、课后建链、双链更新 or 增量建链.
---

# Post-class linking

## Purpose

Turn new class material into one coherent learning system without producing duplicate explanations or automatic keyword-link noise.

## Workflow

1. Identify the course and read its `Course Atlas.canvas`, `00_课程总览.md`, permanent classroom notes, retained assessment/exercise notes, and every item in that course's local `00_inbox/`. Root `00_inbox/` is a separate durable user workspace and is outside this skill's cleanup scope.
2. Separate durable course material from disposable recordings, transcripts, screenshots, and processing intermediates. Move only durable sources into the course directory.
3. Decide whether each new idea belongs in an existing knowledge-file heading or requires a new retrieval-atomic knowledge file under the repository merge/split rule. Keep one concept or method deep and coherent; do not package peer concepts together merely because the course presented them in one chapter.
4. Construct or revise the minimum necessary knowledge files. Use course materials for scope and notation, add authoritative sources when needed, and keep `status: needs-review` during construction.
5. Reopen the cited sources and verify definitions, formulas, assumptions, examples, aliases, scope, and links. Set `status: source-checked` only after this pass succeeds. Alias decisions are reviewed semantically by the agent; scripts may only validate syntax and exact duplicates.
6. Update `00_课程总览.md` as a continuous reading path. Required explanations appear inline through `![[Knowledge file]]`, heading, or block transclusions, with course-specific bridge prose before and between embeds. Classroom notes remain permanent and must never be deleted, replaced, or collapsed into the overview.
7. Preserve the existing Chinese layer of every edited course or knowledge note verbatim and add idiomatic English immediately after each Chinese semantic block, wrapped in the repository's bilingual markers.
8. Check the English layer in Obsidian reading view. Chinese wikilink targets and source paths stay unchanged but require English display labels, so every English block renders as English-only prose.
9. Update the Course Atlas only where the course backbone, learning position, next entry, or a meaningful relationship changed. Express each non-obvious relation in a bridge node; never add a generic `前置` or `相关` edge.
10. Validate all edited links, embeds, Canvas references, heading/block targets, aliases, sources, bilingual coverage, Chinese-layer preservation, English-only rendering, and reading continuity. Check that no circular embed was introduced.
11. After successful incorporation, empty only that course-local Inbox. Never empty, move, reorganize, or delete anything in root `00_inbox/`; deleting a root-Inbox file requires explicit, file-specific user authorization. Report which durable sources were retained and which disposable course-Inbox artifacts were deleted, including whether deletion is recoverable. Never delete a classroom note as part of Inbox cleanup.

## Link rules

- Use ordinary `[[...]]` links only for optional extensions.
- Use `![[...]]` for content required to read the course note continuously.
- Link an atomic point to `[[File#Heading]]` or a block before extracting another file.
- Add a link only when its role can be stated precisely in the surrounding prose or Atlas bridge.
- Never link repeated terms, mathematical symbols, formula fragments, code identifiers, or OCR noise merely because their text matches a knowledge name.
- Do not maintain backlink panels or generated related-note lists; Obsidian backlinks already provide reverse visibility.

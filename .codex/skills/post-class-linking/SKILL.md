---
name: post-class-linking
description: Process a course Inbox after class and update knowledge atoms, topic Canvases, the course overview, and Course Atlas. Use when the user asks for 课后整理、课后建链、双链更新 or 增量建链.
---

# Post-class linking

## Purpose

Turn new class material into one coherent learning system of atomic Markdown knowledge, semantic topic Canvases, course-order reading views, and query-only projections without duplicate explanations or automatic keyword-link noise.

## Workflow

1. Identify the course and read its `Course Atlas.canvas`, relevant same-named topic Canvases, `00_课程总览.md`, permanent classroom notes, existing knowledge atoms, retained assessment/exercise notes, and every item in that course's local `00_inbox/`. Root `00_inbox/` is a separate durable user workspace and is outside this skill's cleanup scope.
2. Separate durable course material from disposable recordings, transcripts, screenshots, and processing intermediates. Move only durable sources into the course directory.
3. Manually decide which independently recallable or applicable definitions, propositions, distinctions, decision rules, or procedural judgements map to existing atoms and which require new atoms. Every independent item is one knowledge-atom Markdown file with a stable `atom_id`; never hide it as only a heading or block in a composite file.
4. Construct or revise the minimum necessary atoms. Keep each atom limited to the explanation, conditions, boundary, minimal rationale, example, and recall/application prompt needed for that one item. Use course materials for scope and notation, add authoritative sources when needed, and keep source `status: needs-review` during construction.
5. Reopen the cited sources and verify each atom's claim, formula, assumptions, boundary, example, aliases, scope, and links. Set source `status: source-checked` only after this pass succeeds. Keep source status, learner mastery evidence, and review scheduling as three independent state dimensions.
6. Create or update the same-named topic Canvas as the topic's integrated semantic entry. Arrange atom file nodes and express each meaningful directed relation with a defensible label or bridge; do not duplicate atom explanations or state in the Canvas.
7. Update `00_课程总览.md` as a continuous reading path in the course's own order. Embed each required atom directly with `![[Knowledge atom]]` and write course-specific bridge prose before and between embeds. Do not recreate a composite topic Markdown. Classroom notes remain permanent and must never be deleted, replaced, or collapsed into the overview.
8. Preserve the existing Chinese layer of every edited course note or atom verbatim and add idiomatic English immediately after each explanatory Chinese semantic block, wrapped in the repository's bilingual markers. Render a translated heading as an italic or plain-text subtitle, never as a second Markdown heading. Do not duplicate pure navigation controls.
9. Check the English layer in Obsidian reading view. Chinese wikilink targets and source paths stay unchanged but require English display labels, so every English block renders as English-only prose. In a callout, retain the single original Chinese callout title and keep every English marker, line, and blank line at the same `>` nesting level.
10. Update the Course Atlas only where the course backbone, learning position, next entry, or a meaningful relationship changed. It is the course-level projection; do not duplicate the topic Canvas. Express each non-obvious relation precisely, never with an unexplained generic `前置` or `相关` edge.
11. Treat any replacement of a legacy composite topic Markdown as one topic-level transaction: inventory every inbound wikilink, embed, heading/block target, and Canvas file node; manually map each reference in context to the exact atom, topic Canvas, course overview, or Course Atlas; update all affected views; and validate the result. Delete the legacy composite Markdown only after no inbound reference remains unmapped and every check succeeds. Otherwise keep or restore the pre-migration topic; never leave a compatibility stub, dual source, or half-migrated state.
12. Validate all edited links, embeds, Canvas JSON and file nodes, unique IDs, aliases, sources, state separation, bilingual coverage, Chinese-layer preservation, English-only rendering, reading continuity, and absence of circular embeds. Confirm that the global Base remains a query-only projection rather than a second authoring surface.
13. After successful incorporation, empty only that course-local Inbox. Never empty, move, reorganize, or delete anything in root `00_inbox/`; deleting a root-Inbox file requires explicit, file-specific user authorization. Report which durable sources were retained and which disposable course-Inbox artifacts were deleted, including whether deletion is recoverable. Never delete a classroom note as part of Inbox cleanup.

## Link rules

- Use ordinary `[[...]]` links only for optional extensions.
- Use direct atom embeds `![[Knowledge atom]]` for content required to read the course overview continuously.
- Use a heading or block link only for supporting detail that is not independently recallable, applicable, or reusable. An independent proposition, distinction, or procedural judgement must be its own atom.
- Add a link only when its role can be stated precisely in the surrounding prose or Atlas bridge.
- Never link repeated terms, mathematical symbols, formula fragments, code identifiers, or OCR noise merely because their text matches a knowledge name.
- Do not maintain backlink panels or generated related-note lists; Obsidian backlinks already provide reverse visibility.
- Scripts may enumerate candidate sections and inbound references, validate YAML/JSON, detect duplicate IDs, and check targets. They must never decide atom boundaries, allocate content, approve aliases, infer relationship meaning, or choose migration link targets.

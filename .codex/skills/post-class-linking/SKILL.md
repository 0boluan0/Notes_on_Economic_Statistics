---
name: post-class-linking
description: Run initial full linking and ongoing incremental linking between course notes (01_Math/02_Economy/03_Computer_Science) and 00_factor. Use when user asks to execute 课后建链 / 双链更新 / 增量建链.
---

# Post Class Linking

## Purpose

Maintain bidirectional visibility between course notes and `00_factor`:
- Course notes get wikilinks to factor cards.
- Factor cards show a Dataview backlink panel for course-note references.

## Entry commands

Use the pipeline script:

```bash
.codex/skills/post-class-linking/scripts/run_link_pipeline.sh preview
.codex/skills/post-class-linking/scripts/run_link_pipeline.sh apply
```

Optional flags:

```bash
.codex/skills/post-class-linking/scripts/run_link_pipeline.sh preview --full
.codex/skills/post-class-linking/scripts/run_link_pipeline.sh preview --incremental
.codex/skills/post-class-linking/scripts/run_link_pipeline.sh preview --no-inject-backlink-panel
```

## Standard workflow

1. Run `preview` first and read the report under `reports/linking/`.
2. Summarize expected file/link changes to the user.
3. After confirmation, run `apply`.
4. Share `git diff --stat` and key sample diffs.

## State and incremental baseline

- State file: `reports/linking/.state.json`
- Baseline key: `last_processed_commit`
- Incremental scope: files changed in git range
  `last_processed_commit..HEAD` plus current staged/unstaged changes, restricted to:
  - `01_Math`
  - `02_Economy`
  - `03_Computer_Science`

If baseline commit is invalid, do not force incremental; ask to run `--full`.

## Main implementation file

- `link_knowledge.py`

## Reporting

Each run writes a JSON report to `reports/linking/` with run stats, ambiguity samples, and error list.

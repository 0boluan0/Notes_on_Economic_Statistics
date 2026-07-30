---
name: tool
description: Compatibility entry for old `tool` requests. Prefer the unified `fragments` skill, which routes project/tool URLs to `05_tools` and general source digests to `04_Fragments`.
---

# Tool

## Overview

Compatibility wrapper for the unified `fragments` skill.

Use this only when the user explicitly invokes `tool`. Otherwise use `fragments` and let it route the input.

## Workflow

1. If the user sends only `tool` without URL, ask for one URL in a single follow-up.
2. For `tool <url>` or an explicitly tool-oriented request, run the same tool-note path used by `fragments`:

```bash
python3 .codex/skills/tool/scripts/build_tool_note.py \
  --url "<PROJECT_URL>" \
  --vault-root "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic"
```

3. Read script stdout and report the created/updated absolute file path.
4. If generation fails, return the error and suggest providing a more direct project URL.
5. After a successful write, the script regenerates `05_tools/00_content.md`.

## Output Contract

- Output folder: `05_tools/`
- Output filename: official project name sanitized for filesystem safety.
- Directory page: `05_tools/00_content.md`, regenerated after every successful note write.
- Existing file with same name: overwrite by default.
- Required sections:
  - `基本信息`
  - `项目介绍（What it does）`
  - `安装方法（Installation）`
  - `首次使用（First run）`
  - `后续使用（Daily usage）`
  - `常见问题与排错（Troubleshooting）`
  - `参考来源（References）`
- If source data is missing, keep section and write explicit `信息不足` + `TODO`.

## Source Priority

- GitHub URL: repo API -> release API -> README API -> homepage metadata.
- Non-GitHub URL: page metadata -> page links -> enrich with detected GitHub repo if present.
- For details, see `references/extraction-rules.md`.

## Notes

- Keep language Chinese-first and preserve key English technical terms.
- Keep the directory page as a detailed navigation table: tool title, language/platform, source, install status, first-run status, cleanup gaps, and one-sentence summary.
- Do not edit `.obsidian/`.

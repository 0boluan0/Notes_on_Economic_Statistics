---
name: tool
description: Generate or update a detailed markdown note in 05_tools from a project URL. Use when user says `tool` with a URL in the same message, or sends a standalone GitHub/website URL and wants intro + installation + first-run + daily usage + troubleshooting + references in Chinese-first style.
---

# Tool

## Overview

Convert one project URL into one structured note under `05_tools/`, using GitHub metadata/README/releases and optional website metadata.

## Workflow

1. Parse the first valid URL from the user message.
2. If the user sends only `tool` without URL, ask for one URL in a single follow-up.
3. Run:

```bash
python3 .codex/skills/tool/scripts/build_tool_note.py \
  --url "<PROJECT_URL>" \
  --vault-root "/Users/fengyihang/Library/Mobile Documents/iCloud~md~obsidian/Documents/Academic"
```

4. Read script stdout and report the created/updated absolute file path.
5. If generation fails, return the error and suggest providing a more direct project URL.

## Output Contract

- Output folder: `05_tools/`
- Output filename: official project name sanitized for filesystem safety.
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
- Do not edit `.obsidian/`.

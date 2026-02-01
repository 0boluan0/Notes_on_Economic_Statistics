# Repository Guidelines

## Project Structure & Module Organization
- Root is an Obsidian vault for academic notes. Key folders:
  - `00_inbox/`: quick captures to triage.
  - `00_factor/`: structured “cards” by type (`00_hub/`, `framework/`, `concept/`, `system/`, `procedure/`, `proof/`, `writing/`, `undefined/`).
  - `01_Math/`, `02_Economy/`, `03_Computer_Science/`, `04_method/`: topic notebooks.
  - `98_attachment/`: images and assets referenced by notes.
  - `Excalidraw/`: diagrams created with the Excalidraw plugin.
  - `.obsidian/`: workspace config and plugins (do not hand-edit).
- Utilities: `optimize_frontmatter.py`.

## Build, Test, and Development Commands
- No build step is required; open the folder in Obsidian.
- Frontmatter maintenance: `python3 optimize_frontmatter.py 00_factor`
  - Updates `aliases` and `tags` for notes in `00_factor/*`.
- Content search: `rg "pattern" 02_Economy` (fast grep across notes).
- Change review: `git status && git diff` before committing.

## Knowledge Architecture (00_factor)
- Card types map to strict roles (see CLAUDE.md):
  - `concept` 只回答“它是什么？”最小可检索定义、符号、最小例子；不含流程/证明。
  - `framework` 只回答“为什么/何时用？”强调直觉、假设、边界与失败模式；不写步骤。
  - `procedure` 可执行步骤清单；按 Step 1/2/3 输出到可交付物。
  - `system` 可信度与落地：诊断、稳健性、排错、复现规范与风险点。
  - `proof` 数学链条：假设→推导→结论；以严密推导为中心。
  - `writing` 表达与呈现：段落模板、解读话术与常用句式。
  - `00_hub` 主题导航页，聚合并链接相关卡片与课程笔记。
- 边界自检（新增/重构时快速判断）：
  - 去掉步骤还能成立 → framework；去掉解释仍可执行 → procedure。
  - 关键词是诊断/稳健/排错 → system；必须逐步推导才成立 → proof。
  - 能直接粘进报告的表达资产 → writing；仅定义级信息 → concept。
- 命名与位置：`增长理论-hub.md`、`VaR.md`、`OLS估计步骤.md` 等放入对应子目录。
- Frontmatter 建议：
  - 所有卡片含 `aliases: []`、`tags: []`；一般笔记可含 `date`, `科目`。

## Coding Style & Naming Conventions
- Markdown notes:
  - Begin with YAML frontmatter bounded by `---`.
  - Required fields in `00_factor`: `aliases: []`, `tags: []` (script helps keep consistent).
  - Filenames: concise Chinese titles; use hyphens for suffix types, e.g., `增长理论-hub.md`.
  - Headings start at `#` with sentence‑case titles; prefer short sections.
- Python utilities: `snake_case` filenames, 4‑space indentation, Black-compatible style.

## Testing Guidelines
- No unit tests. Validate content changes manually:
  - After running the script, inspect with `git diff`.
  - Verify backlinks and embeds render in Obsidian; check missing assets in `98_attachment/`.
  - Spot-check tags and aliases on a few updated notes.

## Commit & Pull Request Guidelines
- Commit messages: concise and descriptive. Examples:
  - `自动: 更新00_factor frontmatter`
  - `notes(02_Economy): 新增索罗模型卡片`
  - `util: refine optimize_frontmatter alias logic`
- Group related edits; avoid mixing content and config changes.
- PRs (if used): include summary, affected folders, and screenshots for visual changes.

## Security & Configuration Tips
- Do not commit secrets or personal tokens. Avoid manual edits in `.obsidian/`.
- Keep large media in `98_attachment/`; link via relative paths.
- When reorganizing files, update links or use Obsidian’s rename to preserve references.

# Repository Guidelines

## Project Structure & Module Organization
- Root is an Obsidian vault for academic notes. Key folders:
  - `00_inbox/`: quick captures to triage.
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

## Knowledge Architecture (active decisions)

### Purpose

- The vault is an all-in-one learning workspace: academic knowledge, course learning, papers, and day-to-day study planning belong here. Original papers remain in Zotero; Obsidian stores the knowledge structure, notes, links, and learning state.
- The goal is not to preserve lecture order. A course is delivered linearly, but knowledge is a network. Lecture order is useful source context, never the final organization.
- Optimize for rapid recovery: after forgetting a topic, the user should be able to see the whole-course direction, enter the relevant knowledge file, and relearn it without feeling lost.

### Canonical objects

- **Course note**: records the course context—what the course covered, the lecturer's emphasis, examples, exercises, and sequence. It is source material, not automatically the best final explanation.
- **Knowledge file**: one coherent recovery unit that explains a topic well enough to relearn it. It may contain definitions, intuition, assumptions, derivations, procedures, diagnostics, examples, and writing guidance when those parts serve the same learning/use context.
- **Atomic knowledge point**: a precisely addressable heading or block inside a knowledge file. Atomicity does not require a separate file; link directly with `[[File#Heading]]` or block links.
- **Hub**: a scarce navigation page created only after several independent knowledge files form a genuinely dense cluster. A Hub is not a content type and must not be pre-created for every important term.
- **Course Atlas**: a Canvas overview of a whole course. It shows the backbone, hierarchy, meaningful cross-links, current position, and next entry point. Detailed explanations live in linked knowledge files or course notes.

### Merge and split rule

- Default to one file when one file can explain the subject clearly. Do not split merely because content can be labelled definition/framework/procedure/proof/system/writing.
- Keep material together when it is normally retrieved together, shares the same assumptions and use context, and forms one continuous explanation. For example, a DID file may contain its intuition, 2×2 setup, regression form, parallel trends, diagnostics, interpretation, and practical workflow.
- Split only when a part has a materially different use case or scope, different assumptions that would be confused with the parent topic, substantial independent complexity, or repeated independent reuse elsewhere.
- Prefer a heading/block link before extracting a new file. Promote a section to its own file only after an actual independent role appears.

### Knowledge-file reading structure

- Every knowledge file begins with a one-screen `> [!summary] 快速恢复` callout before the detailed body.
- The quick-recovery callout answers, in plain language: what problem this topic addresses, one concrete example or anchor, the central idea or difficulty, why it matters, and where to continue.
- Make the continuation links specific: link to a relevant heading, related knowledge file, or Course Atlas rather than listing generic related notes.
- Only the quick-recovery entry is standardized. Organize the detailed body according to the topic's actual knowledge logic; do not force every file into one universal section template.
- A reader who has mostly forgotten the topic should gain a basic mental model from the callout before encountering formal definitions or derivations.

### Course notes and knowledge files

- Course notes preserve course-specific context. Knowledge files provide the clearest current explanation and may synthesize slides, syllabi, textbooks, and authoritative external sources.
- Treat each course note as a continuous reading view assembled in the course's own sequence. Keep the canonical explanation in the knowledge file and transclude required sections inline with `![[Knowledge file#Heading]]` or block embeds.
- Use ordinary `[[...]]` links only for optional extensions that the reader does not need to open immediately. Material required to understand the current course note must appear inline.
- Write course-specific bridge prose before and between embeds to explain why the course moves from one idea to the next. Do not produce a navigation list or an unexplained stack of embeds.
- Design transcluded knowledge sections to remain understandable when read independently; do not make them depend on unembedded source-file context.
- Even when a course note contains no unique substantive explanation, preserve it as a coherent assembled reading path rather than reducing it to a list of links.
- Avoid maintaining two editable copies of the same explanation: the knowledge file is the canonical content, while the course note supplies sequence, transitions, lecturer emphasis, examples, exercises, notation, and assessment context.
- Existing AI-generated knowledge notes have not been validated. Do not use them as authoritative sources or as automatic link targets until rewritten and checked.
- Course materials are the center of course reconstruction. If they are incomplete or poor, authoritative external sources may be used because the objective is mastery, not merely reproducing one lecturer's material.

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

### Visual knowledge maps

- Use Canvas as the primary whole-course map; Markdown remains the detailed explanation format. Complex blocks may open a child Canvas only when the extra spatial layer is genuinely useful.
- A major map node must be understandable to a learner who has mostly forgotten the material. Include: what it is in plain language, a concrete example or anchor, why it matters, how it connects to prior knowledge, what the block contains, and brief sources.
- Do not label connections only as `前置`, `相关`, or similar generic words. Use a small explanatory bridge node stating the actual relation in one or two sentences. If the relation cannot be explained, do not draw it.
- Tasks remain in Workbench/daily notes. The Course Atlas shows only learning position and the next entry point.

### Decisions still open

- Complete the physical migration once, after the remaining conventions are confirmed; do not perform piecemeal moves meanwhile.

### One-time migration workflow

- Inventory the defined migration scope before rewriting individual knowledge files. Do not begin from the existing note boundaries, because they may encode duplicates, artificial fragments, or missing cross-course relationships.
- Primary scope: `01_Math/`, `02_Economy/`, and `03_Computer_Science/`, including course notes and course materials stored inside those directories.
- Supporting scope: `04_Fragments/`, `05_tools/`, `06_paper/`, and `00_inbox/`. Preserve their existing roles and extract only genuinely reusable academic knowledge when relevant.
- Inspect the current AI-generated knowledge area only to discover existing names, links, duplication, and candidate content. It is not an evidentiary source.
- Explicitly exclude `07_Programme/`, `98_attachment/`, and `99_学习情况记录/` from this migration. Do not scan, reorganize, mine, or infer knowledge gaps from them.
- The idea of detecting repeated learning difficulties from future learning records is outside the current migration; current records do not support it.
- The inventory is a temporary migration artifact, not a permanent Obsidian note layer. Remove it after migration verification succeeds.
- For each candidate knowledge point, record where it appears, its actual use contexts, synonymous names, source quality, and the proposed destination: an existing-file heading, a new knowledge file, or a rare Hub.
- Use the inventory to deduplicate synonyms, decide merge/split boundaries, identify shared cross-course knowledge, and derive each course's logical backbone.
- Then execute in order: finalize file boundaries and names → construct knowledge files → run source-based verification → build Course Atlases → replace links and verify backlinks/embeds → switch to the new physical structure once.
- Do not preserve temporary inventory tables or parallel permanent versions after the cutover.
- Verify every course-note transclusion in Obsidian reading view, including heading/block targets, continuity around embeds, and absence of circular embeds.

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

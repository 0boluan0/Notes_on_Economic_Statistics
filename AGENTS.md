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
- 命名规范（知识点卡片）：`00_factor` 的 concept/framework/procedure/system/proof/writing 优先使用**英文文件名**；英文名过长则用**常见缩写**。在 `aliases` 中补**英文全称**与**中文译名**（必要时含常用缩写）。
- 命名与位置：`增长理论-hub.md`、`VaR.md`、`OLS估计步骤.md` 等放入对应子目录。
- Frontmatter 建议：
  - 所有卡片含 `aliases: []`、`tags: []`；一般笔记可含 `date`, `科目`。

## Coding Style & Naming Conventions
- Markdown notes:
  - Begin with YAML frontmatter bounded by `---`.
  - Required fields in `00_factor`: `aliases: []`, `tags: []` (script helps keep consistent).
  - Filenames:
    - `00_factor` 知识点卡片（concept/framework/procedure/system/proof/writing）：优先英文命名；过长则用缩写，并在 `aliases` 写全称与中文译名。
    - 其它笔记（课程/Hub 等）：保持现有命名风格；后缀类型用连字符，例如 `增长理论-hub.md`。
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
  - `自动: 更新00_factor frontmatter`
  - `notes(02_Economy): 新增索罗模型卡片`
  - `util: refine optimize_frontmatter alias logic`
- Group related edits; avoid mixing content and config changes.
- PRs (if used): include summary, affected folders, and screenshots for visual changes.

## Security & Configuration Tips
- Do not commit secrets or personal tokens. Avoid manual edits in `.obsidian/`.
- Keep large media in `98_attachment/`; link via relative paths.
- When reorganizing files, update links or use Obsidian’s rename to preserve references.

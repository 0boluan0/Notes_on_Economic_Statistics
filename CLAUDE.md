# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an Obsidian vault for an academic note library maintained by a HUST 2022 Economics student. The vault contains course notes, learning records, and knowledge base for Mathematics, Economics, and Computer Science courses.

## Vault Structure

Numbered prefixes define folder ordering:

- **00_factor/** - Zettelkasten-style atomic concept repository with hub notes
- **00_inbox/** - Templates and unprocessed notes
- **01_Math/** - Mathematics courses (calculus, linear algebra, game theory, statistics, stochastic processes, time series)
- **02_Economy/** - Economics & Finance courses (econometrics, public finance, banking, securities, risk management, etc.)
- **03_Computer_Science/** - CS courses (CS50, CS61A, CSdiy)
- **04_method/** - Learning methodology and strategies
- **98_attachment/** - Media and PDF attachments (580+ files)
- **99_学习情况记录/** - Daily/weekly study logs (format: YYYY-MM-DD——ddd.md)
- **毕业论文/** - Graduation thesis workspace
- **Excalidraw/** - Visual diagram files

## Note Conventions

### Frontmatter
```yaml
---
date:
aliases:
科目:
---
```

### Content Structure
- Hierarchical headings (H1, H2, H3)
- LaTeX equations with `$$...$$` or `$...$`
- Internal links use `[[note name]]` syntax
- Embedded images use `![[image name]]` syntax
- Blockquotes for definitions and key concepts

### Language
- Primary content in Chinese
- English for CS-specific content
- Bilingual frontmatter fields

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

## Git Configuration

### Gitignore
The following directories are excluded from version control:
- `/99_学习情况记录/` - Daily study logs
- `/00_inbox/` - Templates and inbox
- `/毕业论文/` - Thesis work
- `/06_大创/` - Innovation projects

### Automated Commits
The vault uses obsidian-git plugin with automated daily commits:
- Commit message format: "自动: YYYY-MM-DD HH:MM"
- Triggered by cron plugin

## Obsidian Plugins

### Key Plugins for Navigation & Search
- **omnisearch** - Enhanced search
- **dataview** - Query and display notes
- **quick-explorer** - Quick navigation

### Visualization
- **obsidian-excalidraw-plugin** - Hand-drawn diagrams
- **obsidian-mind-map** - Mind maps from links
- **obsidian-charts** / **obsidian-chartsview-plugin** - Data visualization

### Productivity
- **obsidian-tasks-plugin** - Task management with custom statuses
- **obsidian-kanban** - Kanban boards
- **calendar** - Calendar view for notes
- **contribution-graph** - Activity visualization
- **templater-obsidian** - Dynamic templates with scripting

### Mathematics & Technical Writing
- **obsidian-latex-suite** - LaTeX shortcuts
- **obsidian-tikzjax** - TikZ diagrams
- **code-emitter** - Code block execution

### Automation
- **cron** - Scheduled tasks
- **quickadd** - Quick note creation

## Templates

Located in `00_inbox/`:

- **新建模版.md** - New note template with basic frontmatter
- **日记模版.md** - Daily note template with contribution graph and routine checklist
- **周记模版.md** - Weekly review template with goal tracking
- **add_done_item.md** - Templater script for auto-adding completed items

## Daily Notes Configuration

- Folder: `99_学习情况记录`
- Template: `00_inbox/日记模版.md`
- Date format: `YYYY-MM-DD——ddd` (with Chinese weekday)

## Knowledge Organization Strategy

### Course Notes
- Organized by subject area (Math/Economy/CS)
- Hierarchical structure matching university curriculum
- Heavy use of LaTeX for mathematical notation
- Cross-referenced between related topics

### Concept Repository (00_factor)
- Zettelkasten-style atomic notes
- Hub notes connect related concepts
- Covers financial concepts, risk metrics, economic theory, statistical concepts
 - 命名规范（知识点卡片）：`00_factor` 下的 concept/framework/procedure/system/proof/writing 以**英文文件名**为主；英文过长用**常见缩写**。在 `aliases` 中补**英文全称**与**中文译名**（可含常用缩写）。

### Learning Records
- Daily logs track activities, plans, and reflections
- Weekly summaries with goal review and habit tracking
- Contribution graph visualizes study consistency

## Knowledge Classification System

### `concept`（概念卡 / 名词卡 / 原子知识）
#concept
#### 核心定义

这页只回答：**“它是什么？”**
目标是把一个知识点压缩成**可检索的最小单位**，让你在任何地方提到它都能立刻跳转并想起来。

#### 允许写什么

- 一句话定义（最短不歧义）
- 关键符号/变量含义（例如回归式里每个量代表什么）
- 关键属性/结论（**不证明、不展开**）
- 最小例子（帮助理解）
- 最小反例/混淆项（帮助区分）
- 同义词/别名/中英对照
- 与谁容易混：链接到相近概念

#### 禁止写什么（写了就该拆出去）

- 任何“步骤1、步骤2”的执行流程 → 放`procedure`
- 任何“什么时候用、为什么成立、什么时候会翻车”的长解释 → 放`framework`
- 任何“怎么检验、怎么做稳健性、怎么排错、怎么复现”的工具箱 → 放`system`
- 任何推导链条/证明细节 → 放`proof`
- 任何段落模板/写作措辞/报告结构 → 放`writing`

#### 边界判定（最硬的一句）

**如果把这页压缩成几条“定义卡”，信息几乎不损失，那它就是`concept`。**

#### 典型例子

- 内生性（定义、来源列表一句话版）、异方差（定义）、自相关（定义）
- $ATT$、$ATE$、平行趋势（定义级）
- $RF$参数含义：`n_estimators`、`max_depth`、`max_features`、OOB是什么

#### 反例（不该放concept）

- “DID怎么做事件研究检验、怎么聚类标准误”→`system`
- “随机森林怎么调参、如何做对照实验”→`procedure/system`
- “为什么bagging能降方差”→`framework`（或写成推导再进`proof`）

#### 快速自检问题

1. 我这一页是不是主要在做“名词解释/符号解释”？
2. 读完能立刻复述定义吗？
3. 删除所有长解释只留定义与例子会不会仍然成立？若是 → `concept`

---

### `framework`（框架卡 / 选择与解释）
#framework
#### 核心定义

这页只回答：**“为什么这样？什么时候用？什么时候不要用？”**
它的目标是帮你在现实中做**选择**（选方法、选模型、选策略），并能讲清**直觉与边界**。

#### 允许写什么

- 该方法/模型解决的核心问题（任务定义）
- 直觉解释（为什么有效）
- 前提条件/关键假设（成立依赖什么）
- 适用场景 vs 不适用场景（边界）
- 失败模式（会在哪些数据/设定下翻车）
- 与其他方法的对比（何时选A而非B）
- 典型误解澄清（概念层面的“坑”）

#### 禁止写什么

- 具体执行步骤清单（比如“先分train/val，再调参…”）→`procedure`
- 一整套诊断/稳健性/排错工具箱 →`system`
- 证明推导细节 →`proof`
- 作业/论文段落模板 →`writing`

#### 边界判定（最硬的一句）

**如果读完它，你能回答“我该不该用它/我为什么这么用”，那它就是`framework`。**

#### 典型例子

- $DID$：识别逻辑、平行趋势为何关键、什么时候适合做DID、什么时候不适合
- $IV$：为什么能解决内生性、工具变量需要什么条件、什么时候会失败
- $RF$：为什么bagging+随机特征能提升泛化、什么时候RF不如线性模型或boosting

#### 反例

- “DID怎么画事件研究图、怎么做安慰剂检验”→`system`
- “随机森林参数怎么逐步调”→`procedure`
- “DID识别的数学推导”→`proof`

#### 快速自检问题

1. 这页能不能帮助我做选择（选工具/选设定）？
2. 这页是否在强调假设、边界、失败模式？
3. 如果我把所有“怎么做”删掉，它是否仍完整？若是 → `framework`

---

### `procedure`（流程卡 / 菜谱 / Checklist）
#procedure
#### 核心定义

这页只回答：**“怎么做（可执行步骤）？”**
目标是让你在作业/项目中**照着做就能产出**，减少“我懂原理但不知道从哪开始”。

#### 允许写什么

- 输入→输出的步骤清单（每步写“做什么”）
- 每一步的最低注意点（1–2条即可）
- 需要的工具/函数/命令（例如sklearn接口）
- 常见分支（if-then）：遇到某种情况改哪一步
- 最小可提交版本（MVP）怎么做（先跑通再加分）

#### 禁止写什么

- 长篇“为什么这样做”的原理阐述 →`framework`
- 大型诊断/稳健性/排错库 →`system`
- 推导证明 →`proof`
- 报告如何写 →`writing`

#### 边界判定（最硬的一句）

**如果这一页能被打印出来贴墙上，按步骤走就能做完任务，那它就是`procedure`。**

#### 典型例子

- $RF$训练流程：数据划分→基线→调参→评估→保存模型
- $DID$估计流程：构造变量→设定回归→选择标准误→输出表格
- “作业推进MVP流程”：先写骨架/先跑baseline→再迭代加分

#### 反例

- “为什么要聚类标准误、何时聚类到组/个体/时间”→常常是`system`（可信度/现实细节）
- “事件研究的理论解释”→`framework`或`proof`

#### 快速自检问题

1. 这页是不是以“Step 1/2/3…”为骨架？
2. 去掉解释后是否仍然可执行？
3. 这页是否主要为“减少启动成本”服务？若是 → `procedure`

---

### `proof`（证明/推导卡 / 数学链条）
#proof
#### 核心定义

这页只回答：**“为什么在数学上成立？”**
它的核心是**逻辑链条**：假设→推导→结论。重点在严密性。

#### 允许写什么

- 证明骨架（主线步骤）
- 关键引理/常用技巧（比如不等式、投影矩阵性质等）
- 条件在哪里用到（每一步依赖什么）
- 推导中的关键变形与解释（仍以推导为中心）

#### 禁止写什么

- 操作流程（怎么跑数据、怎么调参）→`procedure/system`
- 稳健性、诊断、排错、复现 →`system`
- 作业写作模板 →`writing`

#### 边界判定（最硬的一句）

**如果去掉推导链条，这页就不成立/不完整，那它就是`proof`。**

#### 典型例子

- $OLS$无偏/一致性推导
- $DID$在特定假设下识别$ATT$的推导骨架
- 统计定理证明、矩阵推导

#### 快速自检问题

1. 这页是否以“假设→推导→结论”为结构？
2. 这页是否必须逐步写数学逻辑才能成立？若是 → `proof`

---

### `system`（系统卡 / 可信度与落地 / 工具箱）
#system
#### 核心定义

这页只回答：**“如何在真实作业/研究/项目中跑通，并保证可信、不翻车？”**
它关注的是现实世界：数据问题、诊断、稳健性、排错、复现、实验纪律。

#### 允许写什么

- 诊断清单（如何证明结果可信）
- 稳健性工具箱（替代设定、安慰剂、敏感性分析）
- 标准误/误差结构/评估指标等“现实可信度细节”
- 常见翻车点与排错流程树（数据泄漏、样本选择、错设模型）
- 复现实验规范（seed、日志、版本、配置）
- 作业/论文 reviewer 会质疑什么、你怎么回应（方法论层面）

#### 禁止写什么

- 纯定义/术语解释（那是`concept`）
- 纯“为什么/何时用”的选择解释（那是`framework`）
- 纯步骤清单（那是`procedure`，除非步骤围绕诊断/稳健/排错）
- 推导证明（那是`proof`）
- 段落模板（那是`writing`）

#### 边界判定（最硬的一句）

**如果这一页的关键词是“诊断/稳健/排错/复现/可信度”，那它就是`system`。**

#### 典型例子

- $DID$可信度工具箱：事件研究、预趋势、安慰剂、溢出、聚类标准误、分期政策问题
- $RF$实验纪律：避免数据泄漏、交叉验证策略、重要性偏差、复现日志
- 面板回归：聚类怎么选、异方差/自相关怎么处理、稳健性怎么写

#### 快速自检问题

1. 这页是不是在回答“别人凭什么信你？”
2. 这页是否主要是“检查与防翻车”？若是 → `system`

---

### `writing`（写作卡 / 表达与呈现资产）
#writing
#### 核心定义

这页只回答：**“怎么写/怎么呈现，才能清楚、得分高、说服人？”**
它服务于交付物：作业、论文、报告、项目文档、考试作文。

#### 允许写什么

- 段落结构模板（结论→依据→稳健性→局限）
- 图表/结果解读话术（系数怎么解释、指标怎么解释）
- 常用表达句式库（中英皆可）
- 评分rubric拆解（IELTS/论文/报告）
- 常见写作错误与改写示例

#### 禁止写什么

- 执行流程（那是`procedure`）
- 诊断/稳健/排错细节（那通常是`system`，除非你写的是“如何在文中呈现稳健性”）
- 推导证明（那是`proof`）
- 纯定义（那是`concept`）

#### 边界判定（最硬的一句）

**如果这页能被你直接复制进作业/论文里，它就是`writing`。**

#### 典型例子

- $DID$结果段落怎么写
- ML实验报告怎么写（baseline→对照→ablation→结论）
- 论文“识别策略”段落模板

#### 快速自检问题

1. 这页是不是在提升“表达与说服”，而不是知识本体？
2. 这页是不是“拿去就能写出来”？若是 → `writing`

---

## Working with This Vault

When modifying notes:
1. Use existing naming conventions (Chinese titles, numbered prefixes)
2. Follow frontmatter structure with date, aliases, and subject
3. Use LaTeX for mathematical expressions
4. Link related notes with `[[ ]]` syntax
5. Place images in `98_attachment/`

When creating new notes:
1. Use appropriate folder based on subject (01_Math, 02_Economy, 03_Computer_Science)
2. Start from templates in `00_inbox/`
3. Follow hierarchical heading structure
4. Add appropriate frontmatter

When working with tasks:
1. Tasks plugin uses custom statuses
2. Use `==text==` for Templater variable substitution
3. Daily notes contain routine checklists and study blocks

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

### Learning Records
- Daily logs track activities, plans, and reflections
- Weekly summaries with goal review and habit tracking
- Contribution graph visualizes study consistency

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

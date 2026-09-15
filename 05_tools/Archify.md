---
date: 2026-09-10
aliases: []
tags: [tools, open-source, github]
---

# Archify

让 AI 将代码库或系统描述转为可交互、可导出的技术图，支持架构、工作流、时序、数据流和生命周期。

## 基本信息
- 输入链接：https://github.com/tt-a1i/archify
- 来源类型：GitHub 项目
- GitHub 仓库：[tt-a1i/archify](https://github.com/tt-a1i/archify)
- 官方网站：[https://tt-a1i.github.io/archify/](https://tt-a1i.github.io/archify/)
- 主要语言：JavaScript
- GitHub Stars：57045
- 开源协议：MIT License
- 最近更新：2026-09-10
- 最新发布：[v2.16.0](https://github.com/tt-a1i/archify/releases/tag/v2.16.0)
- 发布时间：2026-08-30
- 网页标题：Archify — Technical Diagrams from Plain English
- 页面描述：An agent skill for Cursor, Claude Code, Codex CLI, and OpenCode that turns plain-English descriptions into polished, explorable, self-contained technical diagrams with semantic focus, motion, and crisp export.

## 项目介绍（What it does）
- 核心机制：Agent 生成带类型的 JSON 中间表示（IR），Node.js 渲染与校验系统再确定性编译为 HTML/SVG。
- 交付形式：自包含 HTML，支持 PNG、SVG、WebM 和分享卡片导出。
- 适用场景：解释系统组件及边界、展示调用顺序、梳理数据管道与重试状态。
- 核验范围：本次阅读官方 README 和仓库元数据；未安装、未执行示例，效果与兼容性尚未实测。
- 仓库简介：Agent skill for beautiful, verifiable architecture, workflow, sequence, data-flow, and lifecycle diagrams—self-contained HTML with motion and crisp export.
- 主题标签：agent-skills, architecture-as-code, architecture-diagram, claude-skill, code-visualization, codex, coding-agents, data-flow-diagram

## 安装方法（Installation）
- 参考章节：1. Install, Installation options
- npx skills add tt-a1i/archify -g
- For an explicit, non-interactive Cursor install:
- npx -y skills add tt-a1i/archify --skill archify --agent cursor --global --copy --yes
- To try without installing:
- 推荐命令示例：
```bash
npx skills add tt-a1i/archify -g
```
```bash
npx -y skills add tt-a1i/archify --skill archify --agent cursor --global --copy --yes
```

## 首次使用（First run）
- 参考章节：2. Start from a description — no repository required
- Use Archify to draw: Browser -> API -> Redis cache -> PostgreSQL fallback.
- For source evidence, open a repository and ask:
- Analyze this repository, then use archify to create a high-level runtime architecture diagram.
- Show 8–12 core components, one primary path, external dependencies, and trust boundaries.
- 推荐命令示例：
```text
Use Archify to draw: Browser -> API -> Redis cache -> PostgreSQL fallback.
```
```text
Analyze this repository, then use archify to create a high-level runtime architecture diagram.
Show 8–12 core components, one primary path, external dependencies, and trust boundaries.
Put supporting detail in cards instead of adding more edges.
```

## 后续使用（Daily usage）

- 在对话中提出局部调整，例如添加 Redis、移动认证模块或高亮回滚路径；项目保留结构化图源以便继续修改。
- 查看图时可搜索节点、追踪已定义的上下游关系和路径，并播放预设讲解。
- 架构变更审阅可比较两个通过校验的快照，展示 Before / Delta / After。
- 使用依据：[README 的 Refine in chat 与 Choose the right diagram](https://github.com/tt-a1i/archify#choose-the-right-diagram)。

## 常见问题与排错（Troubleshooting）

- 图中的路径来自输入数据所定义的关系；不能据此确认真实运行时影响。架构差异也不直接证明风险或合并安全性。
- 无代码库也可以从文字描述开始；需要源码证据时，应让 Agent 阅读仓库并为组件与关系提供依据。
- 本次没有运行工具，因此没有已验证的本地故障或修复记录。

## 参考来源（References）
- https://github.com/tt-a1i/archify
- https://github.com/tt-a1i/archify/blob/main/README.md
- https://tt-a1i.github.io/archify/

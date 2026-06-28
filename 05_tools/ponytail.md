---
date: 2026-06-26
aliases: [Ponytail, AI agent minimal-code ruleset, AI 代理极简代码规则集]
tags: [tools, open-source, github]
---

# ponytail

Ponytail 是一个面向 AI coding agent 的规则集 / plugin，让代理在写代码前优先判断“是否根本不需要写”“项目里是否已有”“标准库或原生能力是否足够”，目标是减少过度工程化，同时不牺牲安全、校验、可访问性和错误处理。

## 基本信息
- 输入链接：https://github.com/DietrichGebert/ponytail
- 来源类型：GitHub 项目
- GitHub 仓库：[DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)
- 官方网站：[https://ponytail.dev](https://ponytail.dev)
- 主要语言：JavaScript
- GitHub Stars：59498
- 开源协议：MIT License
- 最近更新：2026-06-26
- 最新发布：[v4.8.3: lazy in subagents too](https://github.com/DietrichGebert/ponytail/releases/tag/v4.8.3)
- 发布时间：2026-06-24
- 网页标题：ponytail
- 页面描述：Ponytail makes AI coding agents write the least code that works. Stdlib over custom, native over deps, one line over fifty. 54% less code, safety never cut.

## 项目介绍（What it does）
- 核心目标：把“少写但写对”的判断流程注入到 Claude Code、Codex、Gemini CLI、Copilot CLI、OpenCode、Devin CLI 等 AI coding agent 中。
- 工作方式：代理在动手前沿着一条 ladder 判断：能否不做、能否复用项目已有代码、标准库是否已有、平台原生能力是否足够、已安装依赖是否可用、能否一行解决，最后才写最小可用实现。
- 适用场景：AI agent 经常过度封装、过早引入依赖、写不必要组件或重复造轮子时，用 Ponytail 作为默认约束。
- 重要边界：它不是 code golf；README 明确强调不能削减 validation、error handling、security、accessibility 等安全和质量要求。
- 主题标签：agent-skills, ai-agents, claude, claude-code, claude-code-plugin, cursor-rules, developer-tools, llm

## 安装方法（Installation）
- 前置条件：Claude Code 和 Codex plugin 会运行两个很小的 Node.js lifecycle hooks，因此 `node` 需要在 `PATH` 中。若没有 Node，skills 仍可用，但 always-on activation 不会生效。

Claude Code：

```text
/plugin marketplace add DietrichGebert/ponytail
```

```text
/plugin install ponytail@ponytail
```

Codex CLI / Codex desktop：

```bash
codex plugin marketplace add DietrichGebert/ponytail
codex
```

然后在 Codex 中打开 `/plugins`，选择 Ponytail marketplace 并安装 Ponytail；再打开 `/hooks`，检查并 trust 它的两个 lifecycle hooks。Codex desktop 安装后重启应用即可加载。

GitHub Copilot CLI：

```bash
copilot plugin marketplace add DietrichGebert/ponytail
copilot plugin install ponytail@ponytail
```

Gemini CLI：

```bash
gemini extensions install https://github.com/DietrichGebert/ponytail
```

OpenCode：在 `opencode.json` 中加入：

```json
{ "plugin": ["@dietrichgebert/ponytail"] }
```

instruction-only 方式：Cursor、Windsurf、Cline、GitHub Copilot editor、Aider、Kiro、Zed、VS Code Codex extension 等可复制仓库中对应规则文件，例如 `AGENTS.md`、`.cursor/rules/`、`.github/copilot-instructions.md`、`.kiro/steering/`。

## 首次使用（First run）
- Codex：安装后新开一个 thread。若已 trust hooks，Ponytail 会在每轮自动注入当前模式。
- Claude Code：按 README 要求分两次发送 marketplace add 和 install 命令；安装后新 session 中使用。
- Gemini CLI：安装 extension 后启动新会话，规则集会作为 always-on context 加载，并注册 `/ponytail` 系列命令。
- 检查当前模式：在支持命令的 host 中运行 `/ponytail`；Codex 中这些命令以 skill 形式使用，例如 `@ponytail-review`。

## 后续使用（Daily usage）
- 默认模式是 `full`。可用环境变量或配置文件修改新会话默认模式：
  - `PONYTAIL_DEFAULT_MODE=lite|full|ultra|off`
  - `~/.config/ponytail/config.json` 中的 `defaultMode`
- 常用命令：
  - `/ponytail [lite|full|ultra|off]`：切换强度；不带参数则显示当前模式。
  - `/ponytail-review`：审查当前 diff 中的过度工程化，给出可删除清单。
  - `/ponytail-audit`：审查整个 repo，而不只是当前 diff。
  - `/ponytail-debt`：收集被延后的 `ponytail:` shortcut，形成待处理清单。
  - `/ponytail-gain`：显示 benchmark 中的代码量、成本、速度收益。
  - `/ponytail-help`：显示命令速查。
- Codex 使用方式：README 说明在 Codex 中这些命令是 skills，用 `@` 调用，例如 `@ponytail-review`。
- 不支持 plugin command 的编辑器：通常只加载规则文件作为长期指令，不能使用 mode switch 或 hook。

## 常见问题与排错（Troubleshooting）
- 配置文件不是必需的；只在需要设置默认模式时才使用 `~/.config/ponytail/config.json` 或 `PONYTAIL_DEFAULT_MODE`。
- Codex / Claude Code hooks 不生效：检查 `node` 是否在非交互 shell 的 `PATH` 中，尤其是 Nix / nvm 用户。
- Codex desktop 没看到效果：安装后重启应用，并新开 thread。
- 只想用规则、不想装 plugin：复制仓库内对应 host 的规则文件，例如 `AGENTS.md` 或编辑器专用 rules 文件。
- 卸载：
  - Claude Code：`/plugin remove ponytail`
  - Codex：`codex plugin remove ponytail`
  - Devin CLI：`devin plugins remove ponytail`
  - Pi agent：`pi uninstall ponytail`
  - Cursor / Windsurf / Cline 等：删除复制过去的规则文件
- 完全清理：README 建议在 host remove command 前运行 `node scripts/uninstall.js`，用于移除 Ponytail 留下的 mode flag、配置文件和部分 statusLine 设置。

## 参考来源（References）
- https://github.com/DietrichGebert/ponytail
- https://github.com/DietrichGebert/ponytail/blob/main/README.md
- https://ponytail.dev

---
date: 2026-07-21
aliases: []
tags: [tools, open-source, github]
---

# DeepTutor: Lifelong Personalized Tutoring

DeepTutor: Lifelong Personalized Tutoring. https://deeptutor.info/.

## 基本信息
- 输入链接：https://github.com/HKUDS/DeepTutor
- 来源类型：GitHub 项目
- GitHub 仓库：[HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)
- 官方网站：[http://arxiv.org/abs/2604.26962](http://arxiv.org/abs/2604.26962)
- 主要语言：Python
- GitHub Stars：28517
- 开源协议：Apache License 2.0
- 最近更新：2026-07-21
- 最新发布：[v1.5.2](https://github.com/HKUDS/DeepTutor/releases/tag/v1.5.2)
- 发布时间：2026-07-19
- 网页标题：DeepTutor: Towards Agentic Personalized Tutoring
- 页面描述：Abstract page for arXiv paper 2604.26962: DeepTutor: Towards Agentic Personalized Tutoring

## 项目介绍（What it does）
- 核心目标：把聊天、解题、研究、可视化、测验、掌握度训练、知识库和长期记忆整合为一个本地运行的学习工作台。
- 仓库简介：DeepTutor: Lifelong Personalized Tutoring. https://deeptutor.info/.
- 主题标签：ai-agents, ai-tutor, clawdbot, cli-tool, deepresearch, interactive-learning, large-language-models, multi-agent-systems

## 是否值得本地部署

**当前建议：没有必要长期部署；值得做一次隔离的短期试用。**

原因：
- 当前 Academic vault + Obsidian + Codex 已覆盖笔记读写、资料整理、问答、研究和知识沉淀，DeepTutor 的 Chat、RAG、Co-Writer、Book 等能力与现有流程高度重叠。
- 真正新增的价值主要是结构化学习闭环：Guided Learning / Mastery Path、自动出题与题库、可视化的长期记忆，以及统一 Web 学习界面。只有这些功能能进入日常使用，维护一个额外系统才划算。
- DeepTutor 可以把现有 Obsidian vault 作为 Knowledge Center 的一种知识源，原地导航和写笔记，不需要重新上传和建立向量索引。这降低了试用成本，但同时带来误写风险；首次试用应使用 vault 副本或只读挂载。
- “本地部署”只表示应用和数据服务在本机运行，不等于推理完全离线。默认仍需配置 LLM；知识库通常还需 embedding。要完全离线，还需另行运行 Ollama、LM Studio、llama.cpp 或 vLLM，并选择本地 embedding 模型。

### 本机适配

- Apple Silicon M4 Max、64 GB 内存：远高于官方最低要求，运行应用和中等规模本地模型没有硬件障碍。
- 已有 Python 3.12、Node.js 25、Docker；官方最低要求为 Python 3.11+、Node.js 20+、4 GB 可用内存和 2 GB 可用磁盘。
- Ollama 已安装但当前服务未运行；若追求完全本地，需要启动服务并下载 LLM 与 embedding 模型。
- 当前剩余磁盘约 139 GiB，足够试用，但不宜无计划下载多个大型模型或重复建立大型索引。

### 建议的验证门槛

只做 7 天试用，并用三项结果决定是否保留：
1. 它能否基于一门真实课程连续完成“诊断 → 教学 → 测验 → 追踪掌握度”。
2. 对同一批课程笔记，它是否明显优于直接在 Codex 中使用 `teach` 工作流。
3. 每周是否至少主动使用 2 次；否则删除容器/环境，保留本工具笔记即可。

若试用，优先使用 PyPI 独立目录或 Docker volume，不要从源码部署，也不要直接连接唯一一份 Academic vault。先用少量复制出来的课程笔记验证 Guided Learning、Question Bank 和 Memory，再决定是否授予原 vault 写权限。

## 安装方法（Installation）
- 参考章节：Install backend + frontend deps
- python -m pip install -e .
- ( cd web && npm ci --legacy-peer-deps )
- deeptutor init
- deeptutor start
- 推荐命令示例：
```bash
Source installs run Next.js in dev mode against the local `web/` directory; everything else (config layout, ports, stop with `Ctrl+C`) matches Option 1.

<details>
<summary><b>Conda environment</b> (instead of <code>venv</code>)</summary>
```
```bash
</details>

<details>
<summary><b>Optional install extras</b> — dev / partners / matrix / math-animator</summary>
```

## 首次使用（First run）
- 参考章节：🚀 Get Started
- DeepTutor ships four installation paths. They all share one workspace layout: settings live in data/user/settings/ under the directory you launch from (or under DEEPTUTOR_HOME / deeptutor start --home if you set one explicitly). For the full app, the recommended flow is **pick a workspace directory → install → deeptutor init → deeptutor start**.
- Full local Web app + CLI, no clone required. Needs **Python 3.11+** and a **Node.js 20+** runtime on PATH (the packaged Next.js standalone server is spawned by deeptutor start).
- mkdir -p my-deeptutor && cd my-deeptutor
- pip install -U deeptutor
- 推荐命令示例：
```bash
mkdir -p my-deeptutor && cd my-deeptutor
pip install -U deeptutor
deeptutor init     # prompts for ports + LLM provider + optional embedding
deeptutor start    # starts backend + frontend; keep the terminal open
```

## 后续使用（Daily usage）
- 参考章节：py -3.11 -m venv .venv-cli ; .\.venv-cli\Scripts\Activate.ps1, ⌨️ DeepTutor CLI — Agent-Native Interface
- python3 -m venv .venv-cli && source .venv-cli/bin/activate
- python -m pip install --upgrade pip
- python -m pip install -e ./packaging/deeptutor-cli
- deeptutor init --cli
- 推荐命令示例：
```bash
`deeptutor init --cli` shares the same `data/user/settings/` layout as the full app but skips the backend/frontend port prompts and defaults embeddings to **off** (choose `Yes` if you plan to use `deeptutor kb …` or RAG tools). It still writes a complete runtime layout (`system.json`, `auth.json`, `integrations.json`, `model_catalog.json`, `main.yaml`, `agents.yaml`) and still prompts for the active LLM provider and model.

<details>
<summary><b>Common commands</b></summary>
```
```bash
deeptutor chat                                              # interactive REPL
deeptutor chat --capability deep_solve --kb my-kb --tool rag
deeptutor run chat "Explain the Fourier transform" --tool rag --kb textbook
deeptutor run deep_research "Survey 2026 papers on RAG" \
  --config mode=report --config depth=standard
```

## 常见问题与排错（Troubleshooting）
- 信息不足：未找到 FAQ / troubleshooting 章节。
- TODO: 增补常见报错、日志位置与修复路径。

## 参考来源（References）
- https://github.com/HKUDS/DeepTutor
- https://github.com/HKUDS/DeepTutor/blob/main/README.md
- http://arxiv.org/abs/2604.26962

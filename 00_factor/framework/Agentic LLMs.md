---
aliases:
  - "Agent 化LLM"
  - "LLM Agents"
tags:
  - llm
  - llm/route
---

# Agentic LLMs

Agentic LLMs 解释模型如何从单轮文本生成扩展为能调用工具、规划步骤、读取环境反馈并持续执行任务的系统。

## 核心直觉
Agent 化不是单个模型能力，而是模型、工具、记忆、环境和反馈循环的组合。

## 代表论文
- [[schick2023ToolformerLanguageModels]]
- [[yao2023ReActSynergizingReasoning]]
- [[patil2023GorillaLargeLanguage]]
- [[wang2023VoyagerOpenEndedEmbodied]]

## 边界
Agent 系统的主要风险来自错误传播、权限边界、工具失败和不可观测状态，不只是模型回答质量。

---
aliases:
  - "Agent 化LLM"
  - "LLM Agents"
tags:
  - llm
  - llm/route
type: framework
---

# Agentic LLMs

Agentic LLMs 解释模型如何从单轮文本生成扩展为能调用工具、规划步骤、读取环境反馈并持续执行任务的系统。

## 核心直觉
Agent 化不是单个模型能力，而是模型、工具、记忆、环境和反馈循环的组合。

## 什么时候用

- 任务需要多步规划、调用外部工具、读取中间结果或持续修改状态时。
- 需要判断“加一个 agent loop”是否真的优于一次性提示词时。

## 关键假设

- 模型能理解工具 schema，并能根据反馈修正下一步行动。
- 工具调用具有可观测结果，且权限、成本和副作用可以被约束。
- 任务可以拆成有限步，并存在可检查的中间状态或终止条件。

## 失败模式

- 规划错误会在后续步骤中累积；工具返回异常时可能反复重试。
- 权限过大、状态不可见或没有预算上限，会把局部错误放大为系统事故。
- 对简单、一次性任务使用 agent 会增加延迟、成本和不可预测性。

## 代表论文
- [[schick2023ToolformerLanguageModels]]
- [[yao2023ReActSynergizingReasoning]]
- [[patil2023GorillaLargeLanguage]]
- [[wang2023VoyagerOpenEndedEmbodied]]

## 边界
Agent 系统的主要风险来自错误传播、权限边界、工具失败和不可观测状态，不只是模型回答质量。

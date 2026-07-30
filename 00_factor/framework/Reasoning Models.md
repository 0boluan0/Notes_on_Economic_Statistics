---
aliases:
  - "推理模型"
  - "Reasoning LLMs"
tags:
  - llm
  - llm/route
type: framework
---

# Reasoning Models

Reasoning Models 解释 LLM 如何从直接回答走向显式中间步骤、搜索、验证和强化学习塑造的推理行为。

## 核心直觉
复杂问题需要把一次生成拆成可检查的中间状态；CoT、self-consistency、verifier、reasoning RL 都是在提高计算过程的可控性。

## 什么时候用

- 任务具有多步依赖、可验证答案或允许增加推理时计算时。
- 需要区分“模型知识不足”和“搜索、分解、验证不足”时。

## 关键假设

- 中间步骤能提供有用的搜索状态，且最终结果或过程存在可靠验证信号。
- 额外 token、延迟和采样成本换来的正确率提升足以抵消资源开销。

## 失败模式

- 更长的思维链可能只是更长的错误；错误验证器会把奖励投机固化。
- 对不可验证、开放式任务，显式推理不一定带来可靠提升，应比较直接回答基线。

## 代表论文
- [[wei2023ChainofThoughtPromptingElicits]]
- [[wang2023SelfConsistencyImprovesChain]]
- [[lightman2023LetsVerifyStep]]
- [[deepseek-ai2025DeepSeekR1IncentivizingReasoning]]

## 边界
推理输出更长不等于更真；需要 [[LLM Evaluation]]、验证器和任务反馈来判断质量。

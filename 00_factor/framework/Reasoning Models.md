---
aliases:
  - "推理模型"
  - "Reasoning LLMs"
tags:
  - llm
  - llm/route
---

# Reasoning Models

Reasoning Models 解释 LLM 如何从直接回答走向显式中间步骤、搜索、验证和强化学习塑造的推理行为。

## 核心直觉
复杂问题需要把一次生成拆成可检查的中间状态；CoT、self-consistency、verifier、reasoning RL 都是在提高计算过程的可控性。

## 代表论文
- [[wei2023ChainofThoughtPromptingElicits]]
- [[wang2023SelfConsistencyImprovesChain]]
- [[lightman2023LetsVerifyStep]]
- [[deepseek-ai2025DeepSeekR1IncentivizingReasoning]]

## 边界
推理输出更长不等于更真；需要 [[LLM Evaluation]]、验证器和任务反馈来判断质量。

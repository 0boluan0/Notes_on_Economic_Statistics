---
aliases:
  - "预训练范式"
  - "Pretraining"
tags:
  - llm
  - llm/route
---

# Pretraining Paradigm

Pretraining Paradigm 解释为什么先在大规模通用语料上学习，再通过少量任务数据或指令数据适配，成为 LLM 的基本生产方式。

## 核心直觉
预训练把语言、知识和模式压进通用参数空间；后续任务不再从零学习，而是在基础模型上激活和重排已有能力。

## 代表论文
- [[radfordImprovingLanguageUnderstanding]]
- [[devlin2019BERTPretrainingDeep]]
- [[brown2020LanguageModelsAre]]
- [[raffel2023ExploringLimitsTransfer]]

## 边界
预训练本身不保证模型听话、可靠或低成本；这些分别需要 [[Instruction Tuning and RLHF]]、[[LLM Evaluation]] 和 [[LLM Efficiency Engineering]]。

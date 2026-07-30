---
aliases:
  - "预训练范式"
  - "Pretraining"
tags:
  - llm
  - llm/route
type: framework
---

# Pretraining Paradigm

Pretraining Paradigm 解释为什么先在大规模通用语料上学习，再通过少量任务数据或指令数据适配，成为 LLM 的基本生产方式。

## 核心直觉
预训练把语言、知识和模式压进通用参数空间；后续任务不再从零学习，而是在基础模型上激活和重排已有能力。

## 什么时候用

- 解释基础模型的知识与迁移能力从何而来，或比较继续预训练与监督微调时。
- 需要判断数据、参数和算力投入对通用能力的边际收益时。

## 关键假设

- 训练语料足够多样、质量可控且与目标分布存在可迁移结构。
- 下一 token 预测损失与目标任务能力之间存在足够稳定的相关性。

## 失败模式

- 数据污染、重复、版权与隐私问题会影响可用性和评测可信度。
- 低损失不保证事实、指令遵循或推理；还需要后训练和任务级评估。

## 代表论文
- [[radfordImprovingLanguageUnderstanding]]
- [[devlin2019BERTPretrainingDeep]]
- [[brown2020LanguageModelsAre]]
- [[raffel2023ExploringLimitsTransfer]]

## 边界
预训练本身不保证模型听话、可靠或低成本；这些分别需要 [[Instruction Tuning and RLHF]]、[[LLM Evaluation]] 和 [[LLM Efficiency Engineering]]。

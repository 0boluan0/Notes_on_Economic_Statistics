---
aliases:
  - "Scaling Laws"
  - "规模化定律"
  - "缩放定律"
tags:
  - llm
  - llm/route
type: framework
---

# Scaling Law

Scaling Law 解释模型能力如何随参数量、数据量和训练算力变化，是大模型工业化的核心判断框架。

## 核心直觉
当损失随规模呈稳定幂律下降时，能力提升不再只靠结构创新，也可以靠更大模型、更好数据和更多算力的工程化组合。

## 什么时候用

- 做模型规模、数据量、训练算力或预算规划时。
- 判断一次能力提升更可能来自规模、数据质量、结构改进还是后训练时。

## 关键假设

- 训练分布、优化方法和评测任务在外推区间内没有发生根本变化。
- 计算预算、数据质量和模型容量之间按合理比例配置，而非只增加参数。

## 失败模式

- 幂律拟合可能只在局部区间成立，突发能力和瓶颈会导致外推失真。
- 训练损失下降不等于产品价值上升；推理成本、可靠性和对齐约束必须单独评估。

## 代表论文
- [[kaplan2020ScalingLawsNeural]]
- [[hoffmannTrainingComputeOptimalLarge]]
- [[brown2020LanguageModelsAre]]
- [[chowdhery2022PaLMScalingLanguage]]

## 边界
Scaling Law 说明趋势，不直接说明产品可用性；对齐、推理成本、数据质量和评测仍会改变最终价值。

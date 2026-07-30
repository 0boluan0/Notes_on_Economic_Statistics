---
aliases:
  - "指令微调与RLHF"
  - "Instruction Tuning"
  - "RLHF"
tags:
  - llm
  - llm/route
type: framework
---

# Instruction Tuning and RLHF

Instruction Tuning and RLHF 解释基础模型如何从“会续写文本”变成“能按人类意图完成任务”的助手。

## 核心直觉
预训练给模型能力，指令微调给交互格式，人类反馈或 AI 反馈给偏好排序，三者组合才形成 ChatGPT 式产品体验。

## 什么时候用

- 需要解释“基础模型会做”与“助手愿意按要求做”之间的差异时。
- 需要在 SFT、RLHF、DPO 等后训练路线之间建立选择关系时。

## 关键假设

- 指令数据覆盖目标任务，偏好标注能稳定反映目标行为。
- 训练目标与部署时的安全、事实性和帮助性指标足够一致。

## 失败模式

- 过拟合示范格式、奖励模型偏差或奖励投机，可能造成表面服从而非真实能力。
- 对齐训练可能损伤基础能力；必须用保留集和能力回归评测验证。

## 代表论文
- [[wei2022FinetunedLanguageModels]]
- [[ouyang2022TrainingLanguageModels]]
- [[bai2022TrainingHelpfulHarmless]]
- [[bai2022ConstitutionalAIHarmlessness]]

## 边界
RLHF 不是能力来源本身，它主要重塑输出偏好；复杂推理仍依赖 [[Reasoning Models]] 和训练/推理时计算。

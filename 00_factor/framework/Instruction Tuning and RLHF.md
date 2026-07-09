---
aliases:
  - "指令微调与RLHF"
  - "Instruction Tuning"
  - "RLHF"
tags:
  - llm
  - llm/route
---

# Instruction Tuning and RLHF

Instruction Tuning and RLHF 解释基础模型如何从“会续写文本”变成“能按人类意图完成任务”的助手。

## 核心直觉
预训练给模型能力，指令微调给交互格式，人类反馈或 AI 反馈给偏好排序，三者组合才形成 ChatGPT 式产品体验。

## 代表论文
- [[wei2022FinetunedLanguageModels]]
- [[ouyang2022TrainingLanguageModels]]
- [[bai2022TrainingHelpfulHarmless]]
- [[bai2022ConstitutionalAIHarmlessness]]

## 边界
RLHF 不是能力来源本身，它主要重塑输出偏好；复杂推理仍依赖 [[Reasoning Models]] 和训练/推理时计算。

---
aliases:
  - "大模型评测"
  - "LLM评测"
tags:
  - llm
  - llm/system
---

# LLM Evaluation

LLM Evaluation 关注如何判断模型能力、偏好、安全性、推理质量和真实产品体验。

## 诊断问题
- benchmark 是否覆盖目标任务？
- 指标是在测知识、推理、对齐、工具使用，还是人类偏好？
- 评测是否被训练数据污染？

## 代表论文
- [[hendrycks2021MeasuringMassiveMultitask]]
- [[lin2022TruthfulQAMeasuringHow]]
- [[liang2023HolisticEvaluationLanguage]]
- [[chiang2024ChatbotArenaOpen]]

## 风险点
排行榜会诱导过拟合；真实应用仍需要任务级评测和失败案例分析。

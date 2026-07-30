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

## 诊断与稳健性

- 分开报告能力、事实性、鲁棒性、安全性、工具使用和用户偏好，避免一个平均分掩盖关键失败。
- 对随机采样任务报告重复运行波动或置信区间，并抽查评分器与人工判断的一致性。
- 回归时先核对数据、提示词、模型版本、解码参数和评测脚本，而不是只看总分。

## 关联卡片

- [[LLM Big Picture-hub]]
- [[LLM Inference Optimization]]

## 复现规范

保存数据快照、提示词、评分规则、模型版本、解码参数、随机种子和原始输出；结果应可追溯到单条样本。

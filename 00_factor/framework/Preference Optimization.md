---
aliases:
  - "偏好优化"
  - "DPO family"
tags:
  - llm
  - llm/route
type: framework
---

# Preference Optimization

Preference Optimization 解释后训练如何在不一定显式训练 reward model 的情况下，让模型更偏向人类或规则认可的回答。

## 核心直觉
偏好数据比单一正确答案更贴近开放式对话；DPO、ORPO、SimPO、KTO 等方法都在简化或替代传统 RLHF 的优化链条。

## 什么时候用

- 目标是让回答更符合偏好、风格或安全规范，且可以收集 chosen/rejected 对时。
- 希望减少显式 reward model 或在线强化学习工程复杂度时。

## 关键假设

- 偏好对的质量、覆盖范围和标注一致性足以代表目标行为。
- 参考模型、温度和数据分布的变化不会让优化目标失真。

## 失败模式

- 模型可能学会迎合标注风格、长度或拒答模式，而非提升事实性和推理。
- 偏好数据分布外的任务可能退化；要同时做能力回归、事实性和安全评测。

## 代表论文
- [[rafailovDirectPreferenceOptimization]]
- [[hong2024ORPOMonolithicPreference]]
- [[meng2024SimPOSimplePreference]]
- [[ethayarajh2024KTOModelAlignment]]

## 边界
偏好优化会改善回答风格和安全性，但不会自动补齐事实、工具能力或深推理能力。

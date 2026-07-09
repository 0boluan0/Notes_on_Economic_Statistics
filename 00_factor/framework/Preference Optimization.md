---
aliases:
  - "偏好优化"
  - "DPO family"
tags:
  - llm
  - llm/route
---

# Preference Optimization

Preference Optimization 解释后训练如何在不一定显式训练 reward model 的情况下，让模型更偏向人类或规则认可的回答。

## 核心直觉
偏好数据比单一正确答案更贴近开放式对话；DPO、ORPO、SimPO、KTO 等方法都在简化或替代传统 RLHF 的优化链条。

## 代表论文
- [[rafailovDirectPreferenceOptimization]]
- [[hong2024ORPOMonolithicPreference]]
- [[meng2024SimPOSimplePreference]]
- [[ethayarajh2024KTOModelAlignment]]

## 边界
偏好优化会改善回答风格和安全性，但不会自动补齐事实、工具能力或深推理能力。

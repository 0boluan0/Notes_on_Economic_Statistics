---
aliases:
  - "专家混合"
  - "MoE"
tags:
  - llm
  - llm/concept
---

# Mixture of Experts

Mixture of Experts 是每个 token 只激活部分专家网络的稀疏模型结构。

## 最小定义
模型总参数很多，但通过路由器为每个 token 选择少数专家参与计算。

## 最小例子
DeepSeek-V3 使用 MoE 提高总容量，同时控制每 token 激活参数量。

## 相关卡片
[[LLM Efficiency Engineering]]、[[Distributed LLM Training]]

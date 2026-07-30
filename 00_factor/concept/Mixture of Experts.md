---
aliases:
  - "专家混合"
  - "MoE"
tags:
  - llm
  - llm/concept
  - concept
---

# Mixture of Experts

Mixture of Experts 是每个 token 只激活部分专家网络的稀疏模型结构。

## 最小定义
模型总参数很多，但通过路由器为每个 token 选择少数专家参与计算。

## 最小例子
DeepSeek-V3 使用 MoE 提高总容量，同时控制每 token 激活参数量。

## 相关卡片
[[LLM Efficiency Engineering]]、[[Distributed LLM Training]]
## 符号表达

将本概念记为 $C_{MixtureofExp}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。

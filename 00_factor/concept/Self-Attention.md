---
aliases:
  - Self-Attention
  - "自注意力"
  - "Self Attention"
  - "Self-attention"
tags:
  - concept
  - llm
  - llm/concept
---

# Self-Attention

Self-Attention 是让序列中每个 token 根据其他 token 的表示动态汇聚信息的机制。

## 最小定义
给定 query、key、value，attention 用 query-key 相似度决定从 value 中读取多少信息。

## 最小例子
在句子 “the animal did not cross the street because it was tired” 中，`it` 可以直接关注 `animal`，不用像 RNN 一样逐步传递很远。

## 相关卡片
[[Multi-Head Attention]]、[[Transformer Paradigm]]

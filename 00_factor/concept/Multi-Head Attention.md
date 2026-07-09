---
aliases:
  - "多头注意力"
  - "MHA"
tags:
  - llm
  - llm/concept
---

# Multi-Head Attention

Multi-Head Attention 是并行运行多组 attention，让不同 head 学习不同关系的机制。

## 最小定义
多个 attention head 分别在不同投影空间中读信息，最后拼接并投影回模型维度。

## 最小例子
一个 head 可能关注语法主谓关系，另一个 head 可能关注指代关系。

## 相关卡片
[[Self-Attention]]、[[Transformer Paradigm]]

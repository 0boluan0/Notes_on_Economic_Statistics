---
aliases:
  - "Encoder-only"
  - "仅编码器Transformer"
tags:
  - llm
  - llm/concept
---

# Encoder-only Transformer

Encoder-only Transformer 是使用双向上下文编码输入，主要服务理解、分类、检索等任务的 Transformer。

## 最小定义
模型可以同时看左右上下文，输出每个 token 或整段文本的表示。

## 最小例子
BERT 用 masked language modeling 学习双向表示，适合抽取式理解任务。

## 相关卡片
[[Pretraining Paradigm]]、[[Self-Attention]]

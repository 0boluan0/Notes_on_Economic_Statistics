---
aliases:
- Encoder-only Transformer
- Encoder-only
- 仅编码器Transformer
tags:
- llm
- llm/concept
- concept
---

# Encoder-only Transformer

Encoder-only Transformer 是使用双向上下文编码输入，主要服务理解、分类、检索等任务的 Transformer。

## 最小定义
模型可以同时看左右上下文，输出每个 token 或整段文本的表示。

## 最小例子
BERT 用 masked language modeling 学习双向表示，适合抽取式理解任务。

## 相关卡片
[[Pretraining Paradigm]]、[[Self-Attention]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Pretraining Paradigm]]、[[Self-Attention]]。

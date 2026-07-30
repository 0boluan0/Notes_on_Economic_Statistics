---
aliases:
  - Positional Encoding
  - "位置编码"
  - "Position Encoding"
tags:
  - concept
  - llm
  - llm/concept
---

# Positional Encoding

Positional Encoding 是给无递归的 Transformer 注入 token 顺序信息的方法。

## 最小定义
因为 self-attention 本身不区分位置，模型需要显式或隐式的位置表示来知道 token 顺序。

## 最小例子
`dog bites man` 和 `man bites dog` 有相同词集合，但位置不同导致语义不同。

## 相关卡片
[[Long Context LLMs]]、[[Transformer Paradigm]]

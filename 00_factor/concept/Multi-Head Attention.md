---
aliases:
  - "多头注意力"
  - "MHA"
tags:
  - llm
  - llm/concept
  - concept
---

# Multi-Head Attention

Multi-Head Attention 是并行运行多组 attention，让不同 head 学习不同关系的机制。

## 最小定义
多个 attention head 分别在不同投影空间中读信息，最后拼接并投影回模型维度。

## 最小例子
一个 head 可能关注语法主谓关系，另一个 head 可能关注指代关系。

## 相关卡片
[[Self-Attention]]、[[Transformer Paradigm]]
## 符号表达

将本概念记为 $C_{MultiHeadAtt}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。

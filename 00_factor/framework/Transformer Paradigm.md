---
aliases:
  - "Transformer 范式"
  - "Transformer"
tags:
  - llm
  - llm/route
type: framework
---

# Transformer Paradigm

Transformer Paradigm 解释为什么 self-attention 加位置编码的架构，取代了 RNN/seq2seq 成为现代 LLM 的底座。

## 核心直觉
Transformer 把序列建模从逐步递推改成 token 之间的并行信息路由，因此更适合 GPU 批量矩阵计算，也更容易随着数据、参数和算力扩展。

## 代表论文
- [[vaswaniAttentionAllYou]]
- [[devlin2019BERTPretrainingDeep]]
- [[brown2020LanguageModelsAre]]

## 边界
Transformer 不是所有序列问题的唯一答案；当上下文极长或推理成本受限时，[[Long Context LLMs]]、[[LLM Efficiency Engineering]] 和状态空间路线会重新进入讨论。

## 相关卡片
[[Self-Attention]]、[[Multi-Head Attention]]、[[Positional Encoding]]、[[Decoder-only Transformer]]、[[Encoder-only Transformer]]

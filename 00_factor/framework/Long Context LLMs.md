---
aliases:
  - "长上下文LLM"
  - "Long Context"
tags:
  - llm
  - llm/route
---

# Long Context LLMs

Long Context LLMs 解释模型如何处理更长输入，以及长上下文为什么会改变 RAG、代码、文档和 agent 工作流。

## 核心直觉
上下文窗口扩大可以减少切分和检索损失，但注意力成本、位置外推和有效利用能力会成为新瓶颈。

## 代表论文
- [[dai2019TransformerXLAttentiveLanguage]]
- [[beltagy2020LongformerLongDocumentTransformer]]
- [[dao2022FlashAttentionFastMemoryEfficient]]
- [[chen2024LongLoRAEfficientFinetuning]]

## 边界
更长上下文不等于更好记忆；模型可能看得到但用不好。

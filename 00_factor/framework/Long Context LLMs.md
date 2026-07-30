---
aliases:
  - "长上下文LLM"
  - "Long Context"
tags:
  - llm
  - llm/route
type: framework
---

# Long Context LLMs

Long Context LLMs 解释模型如何处理更长输入，以及长上下文为什么会改变 RAG、代码、文档和 agent 工作流。

## 核心直觉
上下文窗口扩大可以减少切分和检索损失，但注意力成本、位置外推和有效利用能力会成为新瓶颈。

## 什么时候用

- 原始材料之间的跨段依赖很强，盲目切块会损失语义时。
- 需要在长文档、代码仓库或多轮 agent 轨迹中保留更多上下文时。

## 关键假设

- 模型在目标长度上仍能定位关键证据，而不是只依赖近邻文本。
- 延迟、显存和输入成本允许使用更大的上下文窗口。

## 失败模式

- “看得到”不等于“用得到”：中间位置的信息可能被忽略，噪声也会稀释注意力。
- 用长上下文替代检索会增加成本；应以任务级召回、引用忠实性和答案质量验证。

## 代表论文
- [[dai2019TransformerXLAttentiveLanguage]]
- [[beltagy2020LongformerLongDocumentTransformer]]
- [[dao2022FlashAttentionFastMemoryEfficient]]
- [[chen2024LongLoRAEfficientFinetuning]]

## 边界
更长上下文不等于更好记忆；模型可能看得到但用不好。

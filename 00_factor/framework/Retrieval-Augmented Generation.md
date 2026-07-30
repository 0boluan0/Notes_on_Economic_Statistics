---
aliases:
  - "RAG"
  - "检索增强生成"
tags:
  - llm
  - llm/route
type: framework
---

# Retrieval-Augmented Generation

Retrieval-Augmented Generation 解释为什么外部知识库可以补足 LLM 参数记忆的时效性、可追溯性和专域覆盖。

## 核心直觉
RAG 把“知道什么”部分从模型参数中移到可更新的检索系统中，让生成依赖可引用上下文而不是完全依赖记忆。

## 什么时候用

- 知识更新频繁、来源需要追溯、或领域资料不适合全部写入模型参数时。
- 需要把“检索失败”和“生成失败”分开诊断时。

## 关键假设

- 知识库覆盖问题所需证据，切块、召回和重排能把证据送入上下文。
- 生成模型会忠实使用上下文，并能在证据不足时拒答或标注不确定性。

## 失败模式

- 召回不到证据、召回噪声或权限过滤错误，都会导致幻觉或错误引用。
- RAG 不是事实性保证；要分别评估召回率、上下文精度、答案正确性和引用忠实性。

## 代表论文
- [[guu2020REALMRetrievalAugmentedLanguage]]
- [[lewis2021RetrievalAugmentedGenerationKnowledgeIntensive]]
- [[izacard2022AtlasFewshotLearning]]
- [[asai2023SelfRAGLearningRetrieve]]

## 边界
RAG 解决知识注入，不自动解决推理、排序、权限和引用忠实性。

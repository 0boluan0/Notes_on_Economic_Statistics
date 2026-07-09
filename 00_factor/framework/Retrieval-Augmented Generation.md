---
aliases:
  - "RAG"
  - "检索增强生成"
tags:
  - llm
  - llm/route
---

# Retrieval-Augmented Generation

Retrieval-Augmented Generation 解释为什么外部知识库可以补足 LLM 参数记忆的时效性、可追溯性和专域覆盖。

## 核心直觉
RAG 把“知道什么”部分从模型参数中移到可更新的检索系统中，让生成依赖可引用上下文而不是完全依赖记忆。

## 代表论文
- [[guu2020REALMRetrievalAugmentedLanguage]]
- [[lewis2021RetrievalAugmentedGenerationKnowledgeIntensive]]
- [[izacard2022AtlasFewshotLearning]]
- [[asai2023SelfRAGLearningRetrieve]]

## 边界
RAG 解决知识注入，不自动解决推理、排序、权限和引用忠实性。

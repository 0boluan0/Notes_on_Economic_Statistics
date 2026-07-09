---
aliases:
  - "RAG流程"
  - "Retrieval-Augmented Generation Pipeline"
tags:
  - llm
  - llm/procedure
---

# RAG Pipeline

## Step 1: 构建知识库
清洗文档、切块、生成向量或索引。

## Step 2: 检索相关材料
根据用户问题召回候选上下文。

## Step 3: 重排与压缩
保留最相关、最可信、最适合放入上下文窗口的内容。

## Step 4: 生成并引用
模型基于检索上下文回答，并尽量给出来源。

## 输出
一个可更新、可追溯的知识增强回答流程。

## 相关卡片
[[Retrieval-Augmented Generation]]、[[Context Window]]

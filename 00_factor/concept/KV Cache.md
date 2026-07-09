---
aliases:
  - "KV缓存"
  - "Key-Value Cache"
tags:
  - llm
  - llm/concept
---

# KV Cache

KV Cache 是推理时缓存历史 token 的 key/value，避免每生成一个新 token 都重复计算全部历史。

## 最小定义
decode 阶段复用过去层的 key/value，只计算新 token 的增量。

## 最小例子
长对话每次续写只新增一个 token，KV cache 可以显著减少重复注意力计算。

## 相关卡片
[[LLM Inference Optimization]]、[[Long Context LLMs]]

---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
year: 2024
zotero_key: "gu2024MambaLinearTimeSequence"
llm_collection: "06 长上下文与效率"
paper_type: "efficiency-long-context"
tags:
  - llm/paper
  - llm/map
  - zotero
aliases:
  - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 一句话位置
这篇论文位于 `06 长上下文与效率`，以输入相关的选择性状态空间模型和硬件感知递推算法构建无注意力的 Mamba 架构；它用递推状态处理序列，属于与 Transformer 对照的架构效率路线，而不是 Transformer 的 KV 缓存优化。此笔记暂作索引，不做精读摘要。

<!-- bilingual-en:start -->
Mamba combines input-dependent selective state space models with hardware-aware recurrent computation in an attention-free architecture. Its recurrent state provides an architectural efficiency comparison with Transformers, not an optimization of their KV caches. This note remains an index entry, not a close-reading summary.
<!-- bilingual-en:end -->

## 路线
[[长上下文语言模型]], [[06_paper/LLM/LLM推理效率课程|LLM 推理效率]]

## 来源

- [原论文摘要与 §1](https://arxiv.org/html/2312.00752v2)：核验输入相关选择、递推计算和无注意力架构；据此区分于 Transformer 的 KV 缓存优化。

<!-- bilingual-en:start -->
[The abstract and Section 1](https://arxiv.org/html/2312.00752v2) support input-dependent selection, recurrence, and the attention-free architecture, establishing the distinction from Transformer KV-cache optimization.
<!-- bilingual-en:end -->

- [Zotero item](zotero://select/library/items/2KFY9XHL)
- [Zotero PDF](zotero://open-pdf/library/items/6KCAWPVL)
- DOI: `10.48550/arXiv.2312.00752`
- URL: http://arxiv.org/abs/2312.00752
- PDF attachment: `Gu and Dao - 2024 - Mamba Linear-Time Sequence Modeling with Selective State Spaces.pdf`

## Canvas
- [[LLM Big Picture]]
- [[06 长上下文与效率.canvas|06 长上下文与效率 Canvas]]

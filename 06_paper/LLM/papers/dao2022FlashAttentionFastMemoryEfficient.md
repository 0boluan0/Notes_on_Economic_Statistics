---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
year: 2022
zotero_key: "dao2022FlashAttentionFastMemoryEfficient"
llm_collection: "06 长上下文与效率"
paper_type: "efficiency-long-context"
tags:
  - llm/paper
  - llm/map
  - zotero
aliases:
  - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 一句话位置
这篇论文位于 `06 长上下文与效率`，提出用分块计算减少 GPU HBM 与片上 SRAM 间读写的精确注意力算法 FlashAttention；核心改变是执行与数据搬运方式。此笔记暂作索引，不做精读摘要。

<!-- bilingual-en:start -->
FlashAttention is an exact attention algorithm that uses tiling to reduce reads and writes between GPU HBM and on-chip SRAM. It changes execution and data movement. This note remains an index entry, not a close-reading summary.
<!-- bilingual-en:end -->

## 路线
[[长上下文语言模型]], [[FlashAttention]]

## 来源

- [原论文摘要](https://arxiv.org/abs/2205.14135)：核验精确注意力、分块计算与减少 HBM/SRAM 数据搬运的定位。

<!-- bilingual-en:start -->
[The paper abstract](https://arxiv.org/abs/2205.14135) supports exact attention, tiling, and reduced HBM/SRAM data movement.
<!-- bilingual-en:end -->

- [Zotero item](zotero://select/library/items/I4U5CGLY)
- [Zotero PDF](zotero://open-pdf/library/items/5X5Q98RP)
- DOI: `10.48550/arXiv.2205.14135`
- URL: http://arxiv.org/abs/2205.14135
- PDF attachment: `Dao et al. - 2022 - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness.pdf`

## Canvas
- [[LLM Big Picture]]
- [[06 长上下文与效率.canvas|06 长上下文与效率 Canvas]]

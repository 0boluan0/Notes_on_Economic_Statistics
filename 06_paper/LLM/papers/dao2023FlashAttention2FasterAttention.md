---
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
year: 2023
zotero_key: "dao2023FlashAttention2FasterAttention"
llm_collection: "06 长上下文与效率"
paper_type: "efficiency-long-context"
tags:
  - llm/paper
  - llm/map
  - zotero
aliases:
  - "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
---

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

## 一句话位置
这篇论文位于 `06 长上下文与效率`，FlashAttention-2 在精确注意力计算中减少非矩阵乘法运算，并改善 GPU thread block 与 warp 的工作划分，以提高并行度、减少共享内存通信。此笔记暂作索引，不做精读摘要。

<!-- bilingual-en:start -->
FlashAttention-2 reduces non-matmul operations and improves work partitioning across GPU thread blocks and warps in exact attention, increasing parallelism and reducing shared-memory communication. This note remains an index entry, not a close-reading summary.
<!-- bilingual-en:end -->

## 路线
[[长上下文语言模型]], [[FlashAttention2|FlashAttention-2]]

## 来源

- [原论文摘要](https://arxiv.org/abs/2307.08691)：核验非矩阵乘法、thread block 并行度与 warp 间通信三项改进。

<!-- bilingual-en:start -->
[The paper abstract](https://arxiv.org/abs/2307.08691) supports the changes to non-matmul work, thread-block parallelism, and inter-warp communication.
<!-- bilingual-en:end -->

- [Zotero item](zotero://select/library/items/I8PA6SYI)
- [Zotero PDF](zotero://open-pdf/library/items/CDHYSQCA)
- DOI: `10.48550/arXiv.2307.08691`
- URL: http://arxiv.org/abs/2307.08691
- PDF attachment: `Dao - 2023 - FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning.pdf`

## Canvas
- [[LLM Big Picture]]
- [[06 长上下文与效率.canvas|06 长上下文与效率 Canvas]]

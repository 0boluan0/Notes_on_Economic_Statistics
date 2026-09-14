---
title: "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models"
year: 2024
zotero_key: "chen2024LongLoRAEfficientFinetuning"
llm_collection: "06 长上下文与效率"
paper_type: "efficiency-long-context"
tags:
  - llm/paper
  - llm/map
  - zotero
aliases:
  - "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models"
---

# LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models

## 一句话位置
这篇论文位于 `06 长上下文与效率`，用改进的 LoRA 与 shifted sparse attention 降低长上下文微调成本；稀疏注意力用于微调，推理时保留标准全注意力，并兼容 FlashAttention-2 等实现。此笔记暂作索引，不做精读摘要。

<!-- bilingual-en:start -->
LongLoRA combines improved LoRA with shifted sparse attention to reduce long-context fine-tuning cost. Sparse attention is used during fine-tuning; inference retains standard full attention and remains compatible with implementations such as FlashAttention-2. This note remains an index entry, not a close-reading summary.
<!-- bilingual-en:end -->

## 路线
[[长上下文语言模型]], [[FlashAttention2|兼容的精确注意力实现]]

## 来源

- [原论文摘要与图 2](https://arxiv.org/html/2309.12307v2)：核验微调与推理的注意力差异，以及 FlashAttention-2 兼容性。

<!-- bilingual-en:start -->
[The abstract and Figure 2](https://arxiv.org/html/2309.12307v2) support the fine-tuning/inference distinction and FlashAttention-2 compatibility.
<!-- bilingual-en:end -->

- [Zotero item](zotero://select/library/items/K8LSDA9U)
- [Zotero PDF](zotero://open-pdf/library/items/XIA342KE)
- DOI: `10.48550/arXiv.2309.12307`
- URL: http://arxiv.org/abs/2309.12307
- PDF attachment: `Chen et al. - 2024 - LongLoRA Efficient Fine-tuning of Long-Context Large Language Models.pdf`

## Canvas
- [[LLM Big Picture]]
- [[06 长上下文与效率.canvas|06 长上下文与效率 Canvas]]

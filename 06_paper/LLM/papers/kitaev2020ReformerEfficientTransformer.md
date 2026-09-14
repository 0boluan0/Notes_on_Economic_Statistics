---
title: "Reformer: The Efficient Transformer"
year: 2020
zotero_key: "kitaev2020ReformerEfficientTransformer"
llm_collection: "06 长上下文与效率"
paper_type: "efficiency-long-context"
tags:
  - llm/paper
  - llm/map
  - zotero
aliases:
  - "Reformer: The Efficient Transformer"
---

# Reformer: The Efficient Transformer

## 一句话位置
这篇论文位于 `06 长上下文与效率`，用 LSH 近似注意力改变注意力计算，并用可逆残差层减少反向传播所需保存的激活；它为长序列提供结构与训练内存方面的效率路线。此笔记暂作索引，不做精读摘要。

<!-- bilingual-en:start -->
Reformer changes attention through an LSH-based approximation and reduces saved backpropagation activations with reversible residual layers. It provides an architectural and training-memory approach to long-sequence efficiency. This note remains an index entry, not a close-reading summary.
<!-- bilingual-en:end -->

## 路线
[[长上下文语言模型]], [[06_paper/LLM/LLM推理效率课程|LLM 推理效率]], [[自注意力]]

## 来源

- [原论文摘要与 §1](https://arxiv.org/html/2001.04451v2)：核验 LSH 近似注意力与可逆层节省训练激活的不同作用。

<!-- bilingual-en:start -->
[The abstract and Section 1](https://arxiv.org/html/2001.04451v2) distinguish approximate LSH attention from reversible layers' training-activation savings.
<!-- bilingual-en:end -->

- [Zotero item](zotero://select/library/items/RR7BFZFQ)
- [Zotero PDF](zotero://open-pdf/library/items/3CCSBVRL)
- DOI: `10.48550/arXiv.2001.04451`
- URL: http://arxiv.org/abs/2001.04451
- PDF attachment: `Kitaev et al. - 2020 - Reformer The Efficient Transformer.pdf`

## Canvas
- [[LLM Big Picture]]
- [[06 长上下文与效率.canvas|06 长上下文与效率 Canvas]]

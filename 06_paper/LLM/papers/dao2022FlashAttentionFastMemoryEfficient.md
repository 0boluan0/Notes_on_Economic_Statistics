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
  - "FlashAttention 论文中心"
---

# FlashAttention：论文脉络

Dao 等 · 2022 · 原始论文 v2 · [[FlashAttention 原文.pdf|打开原文]]

**这篇论文想解决什么？** 长序列注意力既慢又占显存。作者发现，反复读写巨大的中间矩阵会拖慢执行，于是重新安排计算，让数据尽量在更快的片上存储中完成处理，同时保持原来的注意力运算。

```mermaid
flowchart LR
    A[瓶颈<br/>中间矩阵反复读写] --> B[切入<br/>运算量与搬运量分开看]
    B --> C[方法<br/>分块合并＋反向重算]
    C --> D[验证<br/>结果等价、读写减少、加速]
```

## 沿着这条脉络往下读

| 此时要弄清的问题 | 论文里的展开 | 与已有知识的联系 |
| --- | --- | --- |
| 为什么“算得多”不足以解释“跑得慢”？ | [[FlashAttention - 为什么关注显存往返\|瓶颈与问题重述]] | [[注意力计算与显存\|区分计算量与显存占用]] |
| 不保留完整矩阵，怎么还算出同一个结果？ | [[FlashAttention - 分块如何保持精确\|读懂 Algorithm 1]] | [[在线Softmax合并\|分块结果如何正确合并]] |
| 哪些结果支持它的贡献，能推广到哪里？ | [[FlashAttention - 证据与贡献边界\|证明、实验与前人工作]] | [[FlashAttention\|算法的可复用定义]]；[[稠密注意力二次成本\|仍然存在的计算成本]] |

**关键证据：** 论文的一组 A100 注意力前向＋反向测试中，FlashAttention 做了更多运算，却显著减少显存读写和耗时。这个结果把“数据搬运也是瓶颈”的判断接到了实测上。[[FlashAttention 原文.pdf#page=6|看 Figure 2]]

**读完后，知识上应多出什么？** “注意力的平方级运算量”不能直接推出“必须保存平方级中间矩阵”，也不能单独预测运行速度。这篇论文把精确注意力的算法、存储层级和 GPU 实现一起设计；评价贡献还要考虑此前已有的低内存注意力工作。[[FlashAttention - 证据与贡献边界#它相对前人增加了什么|展开贡献判断]]

> [!info]- 原文与文献位置
> *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，arXiv:2205.14135v2。下列页码对应 Zotero 中这份 34 页 PDF。
> - [[FlashAttention 原文.pdf#page=4|p.4：标准算法与瓶颈]] · [[FlashAttention 原文.pdf#page=5|p.5：核心算法]] · [[FlashAttention 原文.pdf#page=6|p.6：IO 分析与 Figure 2]] · [[FlashAttention 原文.pdf#page=22|p.22：与前人方法的比较]]
> - [Zotero 条目](zotero://select/library/items/I4U5CGLY) · [Zotero 原文](zotero://open-pdf/library/items/5X5Q98RP) · [论文页面](https://arxiv.org/abs/2205.14135v2)
> - DOI：`10.48550/arXiv.2205.14135`
> - 放回研究脉络：[[长上下文语言模型]] · [[06 长上下文与效率.canvas|长上下文与效率 Canvas]] · [[LLM Big Picture]]

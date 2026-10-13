# Attention Is All You Need：论文批注

[[06_paper/Attention 阅读样例/Attention Is All You Need.pdf|在 Obsidian 阅读原文]] · [[06_paper/LLM/papers/vaswaniAttentionAllYou|论文索引]]

## 缩放点积注意力

§3.2.1 的式 (1) 把一次读取拆成三个步骤：Q 与 K 的点积得到匹配分数；除以键维度的平方根并经 Softmax 归一化；最后用权重汇总 V。这对应已有的[[缩放点积注意力]]原子。

<!-- bilingual-en:start -->
Equation (1) in Section 3.2.1 describes three steps: compute compatibility scores from Q and K, divide by the square root of the key dimension and normalize with Softmax, then use the weights to aggregate V. This corresponds to the existing [[缩放点积注意力|scaled dot-product attention]] atom.
<!-- bilingual-en:end -->

阅读公式时，[[Softmax]]回答“分数怎样变成总和为 1 的权重”；[[点积缩放]]回答“为什么采用这个缩放尺度，以及推导依赖哪些假设”。三个原子各自保留一种可单独使用的知识。

<!-- bilingual-en:start -->
[[Softmax]] explains how scores become weights summing to one; [[点积缩放|dot-product scaling]] explains the scale choice and the assumptions behind its derivation. The three atoms retain independently usable pieces of knowledge.
<!-- bilingual-en:end -->

## 原文定位


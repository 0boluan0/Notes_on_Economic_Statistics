---
aliases:
  - "Transformer and Attention"
  - "Self-Attention"
  - "Transformer"
status: source-checked
---

# Transformer 与注意力机制

> [!summary] 快速恢复
> **它解决什么：** 让序列中的每个 token 按当前任务直接选择并汇总其他 token 的信息，而不必像循环网络那样逐步传递。
> **具体锚点：** 在“银行没有批准贷款，因为它风险太高”中，“它”应重点读取“贷款”及其上下文；attention 学习这种数据依赖的读取权重。
> **核心难点：** attention 不是“理解力”本身，而是按 query–key 相似度加权 value；模型仍需位置、因果掩码、残差和训练目标才能成为语言模型。
> **为什么重要：** 预训练、长上下文、KV cache、多模态和大多数现代 LLM 都从这个计算骨架展开。
> **继续：** 先掌握一次 self-attention 的张量流，再读 [[预训练、指令微调与偏好优化#预训练：先学习续写分布|预训练]]；部署时转到 [[LLM 推理效率#Prefill 与 decoding|推理阶段]]。

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。

## Token 与表示

文本先被 tokenizer 切成 token 并映射为向量。tokenization 决定序列长度和词表粒度，但 token 本身没有固定“词义”；上下文表示由多层变换形成。输入表示还要携带位置信息，否则 attention 对排列近似不敏感。

## Self-attention

对表示矩阵 $X$ 做线性投影得到 $Q=XW_Q$、$K=XW_K$、$V=XW_V$：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

每个 query 与所有 key 比较，softmax 后形成读取 value 的权重。缩放项控制点积方差；$M$ 可屏蔽未来位置或 padding。权重能说明本层如何路由信息，但不能自动等同于完整、稳定的因果解释。

## Multi-head、残差与前馈层

多头注意力在不同投影子空间并行读取，再拼接；逐位置前馈网络对每个位置施加相同的非线性变换。残差连接和 normalization 稳定深层训练。层层堆叠后，一个 token 可组合多步上下文关系。

## 位置与序列顺序

原始 Transformer 加正弦位置编码；后续模型常用学习位置、相对位置或旋转位置编码。位置方案影响长度外推，但“换一种位置编码”不能单独保证模型在训练窗外仍能正确利用远距离信息。

## Encoder、decoder 与因果掩码

encoder 可双向读取整个输入，适合理解或编码；decoder 用 causal mask，只能读取当前位置及过去，适合自回归生成；encoder–decoder 还用 cross-attention 从输入读取。BERT 类模型偏 encoder，GPT 类模型偏 decoder，T5 类模型使用 encoder–decoder。

## Next-token prediction

decoder 语言模型最大化 $\sum_t\log p(x_t\mid x_{<t})$。训练可并行计算所有位置的损失，生成却通常逐 token 进行。这个目标学习语言、知识与模式，但不直接保证事实性、服从指令或安全性，后续对齐阶段解决的是另一层问题。

## 最小自检

### 用自己的话解释 query、key、value 各自做什么。

> [!answer]- 答案
> query 表示当前位置想找什么，key 表示每个位置可怎样被匹配，value 是匹配后实际被汇总的信息。
### 为什么 decoder 训练能并行而生成仍常逐 token？

> [!answer]- 答案
> 训练时完整目标序列已知，可用 causal mask 同时算所有位置；生成时下一个输入依赖刚生成的 token。
### 只增加 attention 层为什么还不构成语言模型？

> [!answer]- 答案
> 还需要 token/位置表示、前馈与残差结构、输出分布和训练目标；attention 只是信息路由机制。

## 来源与核验

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
- Vaswani et al. (2017), [[06_paper/LLM/papers/vaswaniAttentionAllYou|Attention Is All You Need]]：核验 Transformer 与多头注意力。

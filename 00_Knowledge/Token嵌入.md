---
aliases:
  - "Token 嵌入是把离散 token ID 映射为可学习向量的查表表示"
  - "Token embeddings map discrete token IDs to learnable vectors through a lookup table"
  - "Token embedding"
student_os: knowledge-atom
atom_id: LLM-TF-001
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# Token 嵌入是把离散 token ID 映射为可学习向量的查表表示

<!-- bilingual-en:start -->
*Token embeddings map discrete token IDs to learnable vectors through a lookup table*
<!-- bilingual-en:end -->

Token 嵌入把离散的 token ID 转换为连续向量：每个 ID 对应一个可学习矩阵中的一行。给定 tokenizer 产生的 ID 序列后，这一步为后续神经网络提供输入表示；token 怎样切分及计数，取决于另行指定的[[Token口径|编码管线]]。

<!-- bilingual-en:start -->
A token embedding converts a discrete token ID into a continuous vector by selecting a row from a learnable matrix. Given a sequence of IDs produced by a tokenizer, this operation supplies representations to subsequent neural layers. Token segmentation and counting depend on the specified [[Token口径|encoding pipeline]].
<!-- bilingual-en:end -->

## 从 ID 到表示矩阵

<!-- bilingual-en:start -->
*From IDs to a representation matrix*
<!-- bilingual-en:end -->

设词表有 $v$ 个条目，模型表示宽度为 $d_{\mathrm{model}}$，嵌入矩阵为 $E$。省略 batch，以行排列 token；第 $i$ 个位置的 ID 为 $t_i$，则查表结果和整段序列表示为

<!-- bilingual-en:start -->
Let the vocabulary contain $v$ entries and let the model width be $d_{\mathrm{model}}$. The embedding matrix is $E$. Omitting the batch dimension and arranging tokens in rows, the ID $t_i$ at position $i$ selects a vector; stacking these vectors gives the sequence representation:
<!-- bilingual-en:end -->

$$
E\in\mathbb R^{v\times d_{\mathrm{model}}},\qquad
e_i=E[t_i,:],\qquad
X_{\mathrm{emb}}=
\begin{bmatrix}e_1\\\vdots\\e_n\end{bmatrix}
\in\mathbb R^{n\times d_{\mathrm{model}}}.
$$

这里的 ID 只是行索引。例如 ID 为 7 和 8，并不因此比 7 和 100 的语义更接近；向量中学到的关系来自模型参数与训练，而不是编号的数值距离。

<!-- bilingual-en:start -->
An ID is a row index. IDs 7 and 8 are not necessarily more semantically similar than IDs 7 and 100. Relationships encoded in the vectors arise from the learned parameters and training, rather than numerical distances between IDs.
<!-- bilingual-en:end -->

## 同一 token 的起点与上下文

<!-- bilingual-en:start -->
*A shared starting vector and its context*
<!-- bilingual-en:end -->

假设一个 tokenizer 已将文本编码为 $[7,3,7]$。在同一张嵌入表中，第 1、3 个位置都读取 $E[7,:]$，所以纯 token 嵌入相同。后续引入[[位置编码]]和上下文计算后，这两个位置的表示可以不同；例如[[自注意力]]会根据可见上下文读取信息。查表表示与经过上下文化的隐藏表示是不同阶段的对象。

<!-- bilingual-en:start -->
Suppose a tokenizer has encoded a text as $[7,3,7]$. Positions 1 and 3 both retrieve $E[7,:]$ from the same table, so their token embeddings match. After [[位置编码|positional information]] and contextual computation are introduced, their representations can differ: [[自注意力|self-attention]], for example, reads from the available context. A lookup embedding and a contextual hidden representation belong to different processing stages.
<!-- bilingual-en:end -->

原始 Transformer 将查出的嵌入乘以 $\sqrt{d_{\mathrm{model}}}$。这是该模型的嵌入尺度选择；[[点积缩放|注意力分数的缩放]]则除以 $\sqrt{d_k}$，两者作用于不同对象。

<!-- bilingual-en:start -->
The original Transformer multiplies its lookup embeddings by $\sqrt{d_{\mathrm{model}}}$. This is that model's choice of embedding scale. [[点积缩放|Attention-score scaling]] divides by $\sqrt{d_k}$ and operates on a different quantity.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/html/1706.03762v7)，§3.4：支持使用可学习嵌入将输入与输出 token 转成 $d_{\mathrm{model}}$ 维向量，以及原始模型的嵌入缩放。查表矩阵记法与重复 ID 的例子是本卡对定义的展开。
  <!-- bilingual-en:start -->
  Section 3.4 supports learned embeddings for input and output tokens and the original model's embedding scale. The lookup notation and repeated-ID example unpack this definition.
  <!-- bilingual-en:end -->

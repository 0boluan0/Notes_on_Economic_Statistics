---
aliases:
  - "Transformer 编码器通过堆叠自注意力与逐位置前馈层，把输入序列映射为上下文表示序列"
  - Transformer encoder
student_os: knowledge-atom
atom_id: LLM-TF-014
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# Transformer 编码器通过堆叠自注意力与逐位置前馈层，把输入序列映射为上下文表示序列

<!-- bilingual-en:start -->
*A Transformer encoder maps an input sequence to contextual representations through stacked self-attention and position-wise feed-forward layers.*
<!-- bilingual-en:end -->

Transformer 编码器是把输入序列转换成上下文表示序列的网络模块。在原始 encoder–decoder 架构中，它接收带位置信息的源端 token 表示，通过多层自注意力与逐位置前馈运算产生 $Z$，再把 $Z$ 提供给解码器作为可以读取的输入记忆。每个有效输入位置都保留一个输出向量。

<!-- bilingual-en:start -->
A Transformer encoder is a network module that converts an input sequence into contextual representations. In the original encoder-decoder architecture, it receives source-token representations with positional information, applies stacked self-attention and position-wise feed-forward operations, and supplies the resulting sequence $Z$ to the decoder as readable source memory. It retains an output vector for each valid input position.
<!-- bilingual-en:end -->

## 一层怎样更新表示

<!-- bilingual-en:start -->
*How one layer updates the representations.*
<!-- bilingual-en:end -->

设第 $\ell$ 层输入为 $X^{(\ell)}\in\mathbb R^{n_{\mathrm{src}}\times d_{\mathrm{model}}}$。这一层先做多头 [[自注意力]]，再做 [[逐位置前馈网络]]；每个子层都按 [[Transformer后归一化]] 封装。省略 dropout，记 $R_F(U)=\operatorname{LN}(U+F(U))$，则

<!-- bilingual-en:start -->
Let layer $\ell$ receive $X^{(\ell)}\in\mathbb R^{n_{\mathrm{src}}\times d_{\mathrm{model}}}$. It first applies multi-head [[自注意力|self-attention]], then a [[逐位置前馈网络|position-wise feed-forward network]], with each sublayer wrapped in [[Transformer后归一化|original Transformer post-normalization]]. Omitting dropout, write $R_F(U)=\operatorname{LN}(U+F(U))$:
<!-- bilingual-en:end -->

$$
A^{(\ell)}=R_{\mathrm{SA}_{\ell}}\bigl(X^{(\ell)}\bigr),\qquad
X^{(\ell+1)}=R_{\mathrm{FFN}_{\ell}}\bigl(A^{(\ell)}\bigr).
$$

这里每个子层有自己的变换和 LayerNorm 参数。自注意力的 $Q,K,V$ 都由同一份 $X^{(\ell)}$ 投影而来，每个位置可以读取全部有效源位置。FFN 随后独立变换每一行；两步均保持 $n_{\mathrm{src}}\times d_{\mathrm{model}}$ 的形状。堆叠 $N$ 层后，输出为 $Z=X^{(N)}$。

<!-- bilingual-en:start -->
Each sublayer has its own transformation and LayerNorm parameters. Self-attention projects $Q,K,V$ from the same $X^{(\ell)}$, allowing each position to read every valid source position. The FFN then transforms each row independently. Both steps preserve the shape $n_{\mathrm{src}}\times d_{\mathrm{model}}$, and after $N$ layers the output is $Z=X^{(N)}$.
<!-- bilingual-en:end -->

## 输出为什么是一组向量

<!-- bilingual-en:start -->
*Why the output is a sequence of vectors.*
<!-- bilingual-en:end -->

例如输入有三个有效 token，编码后得到 $z_1,z_2,z_3$ 三个向量。第二个向量仍对应第二个位置，但可以综合三个输入位置的信息。随后，[[Transformer解码器]] 的 [[交叉注意力]] 从 $Z$ 产生 keys 和 values，让目标端各位置按当前查询读取这些源端表示。

<!-- bilingual-en:start -->
For three valid input tokens, the encoder produces three vectors $z_1,z_2,z_3$. The second vector still corresponds to the second position, but may incorporate information from all three positions. The [[Transformer解码器|Transformer decoder]] then obtains keys and values for [[交叉注意力|cross-attention]] from $Z$, allowing each target position to read source representations according to its current query.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification.*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/pdf/1706.03762)，§3、§3.1、§3.2.3、Figure 1：支持编码器输入与输出的序列对应、每层的两子层顺序、源端全位置读取，以及最终编码器输出作为交叉注意力 memory 的关系。
  <!-- bilingual-en:start -->
  Section 3, Sections 3.1 and 3.2.3, and Figure 1 establish the position-aligned encoder output, the two-sublayer order, attention across source positions, and the final encoder output's role as cross-attention memory.
  <!-- bilingual-en:end -->

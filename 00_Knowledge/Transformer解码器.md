---
aliases:
  - "原始 Transformer 解码器用目标前缀和编码器表示构造逐步生成所需的隐藏状态"
  - Original Transformer decoder
  - Transformer decoder in an encoder-decoder model
student_os: knowledge-atom
atom_id: LLM-TF-015
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# 原始 Transformer 解码器用目标前缀和编码器表示构造逐步生成所需的隐藏状态

<!-- bilingual-en:start -->
*The original Transformer decoder uses a target prefix and encoder representations to construct hidden states for step-by-step generation.*
<!-- bilingual-en:end -->

原始 Transformer 解码器是 encoder–decoder 架构中的目标端模块：它接收已知目标前缀的表示，并通过交叉注意力读取编码器输出，构造用于预测下一个目标 token 的隐藏状态。每层依次包含带因果掩码的自注意力、读取源端的交叉注意力、逐位置前馈网络。最终隐藏状态由另一个 [[语言模型输出头]] 转换为词表分布。

<!-- bilingual-en:start -->
The original Transformer decoder is the target-side module of an encoder-decoder architecture. It receives representations of the known target prefix and reads the encoder output through cross-attention to construct hidden states for predicting the next target token. Each layer applies causally masked self-attention, source-reading cross-attention, and a position-wise feed-forward network, in that order. A separate [[语言模型输出头|language-model output head]] converts the final hidden states into vocabulary distributions.
<!-- bilingual-en:end -->

## 三个子层接收什么

<!-- bilingual-en:start -->
*What the three sublayers receive.*
<!-- bilingual-en:end -->

令 $Z\in\mathbb R^{n_{\mathrm{src}}\times d_{\mathrm{model}}}$ 为 [[Transformer编码器]] 的最终输出，$H^{(\ell)}\in\mathbb R^{n_{\mathrm{tgt}}\times d_{\mathrm{model}}}$ 为当前解码器层输入。每个子层按 [[Transformer后归一化]] 封装；省略 dropout，记 $R_F(U)=\operatorname{LN}(U+F(U))$，则

<!-- bilingual-en:start -->
Let $Z\in\mathbb R^{n_{\mathrm{src}}\times d_{\mathrm{model}}}$ be the final [[Transformer编码器|encoder]] output, and $H^{(\ell)}\in\mathbb R^{n_{\mathrm{tgt}}\times d_{\mathrm{model}}}$ the current decoder-layer input. Each sublayer uses [[Transformer后归一化|original Transformer post-normalization]]. Omitting dropout, write $R_F(U)=\operatorname{LN}(U+F(U))$:
<!-- bilingual-en:end -->

$$
\begin{aligned}
A^{(\ell)}&=R_{\mathrm{CSA}_{\ell}}\bigl(H^{(\ell)}\bigr),\\
B^{(\ell)}&=R_{\mathrm{CA}_{\ell}(\,\cdot\,,Z)}\bigl(A^{(\ell)}\bigr),\\
H^{(\ell+1)}&=R_{\mathrm{FFN}_{\ell}}\bigl(B^{(\ell)}\bigr).
\end{aligned}
$$

第一步是带 [[因果注意力掩码]] 的 [[自注意力]]：$Q,K,V$ 都从目标端当前表示产生，可见范围到当前输入位置为止。第二步是 [[交叉注意力]]：queries 由第一步输出 $A^{(\ell)}$ 产生，keys 和 values 由 $Z$ 产生。第三步是 [[逐位置前馈网络]]，独立变换各目标位置。每个子层有自己的参数，所有子层都保留目标端的行数与 $d_{\mathrm{model}}$ 宽度。

<!-- bilingual-en:start -->
The first step is [[自注意力|self-attention]] with a [[因果注意力掩码|causal attention mask]]: target-side representations produce $Q,K,V$, and visibility extends through the current input position. The second step is [[交叉注意力|cross-attention]], with queries from $A^{(\ell)}$ and keys and values from $Z$. The third step is a [[逐位置前馈网络|position-wise FFN]] operating independently at each target position. Each sublayer has its own parameters, and all preserve the target row count and width $d_{\mathrm{model}}$.
<!-- bilingual-en:end -->

## 一个预测位置的两路条件

<!-- bilingual-en:start -->
*The two sources of conditioning at one prediction position.*
<!-- bilingual-en:end -->

设目标序列为 $y_1,y_2,y_3$，右移后的解码器输入是 $[\mathrm{BOS},y_1,y_2]$。用来预测 $y_3$ 的第三行可以经自注意力读取 $\mathrm{BOS},y_1,y_2$，也可以经交叉注意力读取全部有效源端位置。它由此构造以源序列和目标前缀 $y_{<3}$ 为条件的隐藏状态。

<!-- bilingual-en:start -->
For a target sequence $y_1,y_2,y_3$, the shifted decoder input is $[\mathrm{BOS},y_1,y_2]$. The third row, used to predict $y_3$, can read $\mathrm{BOS},y_1,y_2$ through self-attention and all valid source positions through cross-attention. Its hidden state is therefore conditioned on the source sequence and the target prefix $y_{<3}$.
<!-- bilingual-en:end -->

此处定义的是原始双栈架构中的解码器。decoder-only 模型怎样组织目标端计算、是否有独立源端 memory，由 [[Transformer架构可见性]] 对照说明。

<!-- bilingual-en:start -->
This definition concerns the decoder in the original two-stack architecture. [[Transformer架构可见性|Transformer architecture visibility]] explains how decoder-only models organize target-side computation and whether they have a separate source-memory stream.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification.*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/pdf/1706.03762)，§3、§3.1、§3.2.3、Figure 1：支持目标前缀与源端表示两路输入、三子层顺序、右移目标与因果掩码的配合，以及交叉注意力 queries 和 memory keys/values 的来源。公式中的交叉注意力输入位置按 Figure 1 的连线定位。
  <!-- bilingual-en:start -->
  Section 3, Sections 3.1 and 3.2.3, and Figure 1 establish the target-prefix and source-representation inputs, the three-sublayer order, shifted targets together with causal masking, and the sources of cross-attention queries and memory keys/values. The diagram locates the cross-attention input precisely in the displayed equations.
  <!-- bilingual-en:end -->

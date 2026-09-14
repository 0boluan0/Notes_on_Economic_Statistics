---
aliases:
  - "原始 Transformer 在每个子层变换后先做残差相加，再做层归一化"
  - Original Transformer post-normalization
  - Transformer Add and Norm
student_os: knowledge-atom
atom_id: LLM-TF-012
atom_type: mechanism
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# 原始 Transformer 在每个子层变换后先做残差相加，再做层归一化

<!-- bilingual-en:start -->
*After each sublayer transformation, the original Transformer performs residual addition followed by layer normalization.*
<!-- bilingual-en:end -->

原始 Transformer 的后归一化子层按固定顺序组合三个运算：可学习子层先算出变换结果，[[残差连接]] 把它与子层输入相加，[[LayerNorm]] 再处理相加后的表示。这正是原始架构图中每个子层旁边的 Add & Norm；“后”指归一化位于残差相加之后。

<!-- bilingual-en:start -->
An original Transformer post-normalization sublayer combines three operations in order: a learned sublayer computes a transformation, a [[残差连接|residual connection]] adds it to the sublayer input, and [[LayerNorm|layer normalization]] transforms the sum. This is the Add & Norm shown beside each sublayer in the original architecture diagram. “Post” locates normalization after residual addition.
<!-- bilingual-en:end -->

## 按括号读运算顺序

<!-- bilingual-en:start -->
*Read the computation from the parentheses.*
<!-- bilingual-en:end -->

先省略训练中的 dropout，设 $F$ 为一个保持输入形状的子层变换，则

<!-- bilingual-en:start -->
First omit training dropout. For a sublayer transformation $F$ that preserves the input shape,
<!-- bilingual-en:end -->

$$
R_F(X)=\operatorname{LN}\bigl(X+F(X)\bigr).
$$

先求 $F(X)$，再相加，最后归一化。$F$ 可以是 [[多头注意力]] 或 [[逐位置前馈网络]]；一层里每个子层分别完成自己的这套运算。[[Transformer编码器]] 每层有两个这样的子层，原始 [[Transformer解码器]] 每层有三个。

<!-- bilingual-en:start -->
Compute $F(X)$, add it to $X$, then normalize. The function $F$ can be [[多头注意力|multi-head attention]] or a [[逐位置前馈网络|position-wise FFN]], and each sublayer performs its own instance of this sequence. A [[Transformer编码器|Transformer encoder]] layer contains two such sublayers; an original [[Transformer解码器|Transformer decoder]] layer contains three.
<!-- bilingual-en:end -->

例如单行输入 $x=[1,2]$、子层结果 $F(x)=[0.5,-1]$。相加先得到 $[1.5,1]$，其均值为 $1.25$、方差为 $0.0625$。若该 LayerNorm 的 $\gamma=[1,1]$、$\beta=[0,0]$，完整输出为

<!-- bilingual-en:start -->
For a row $x=[1,2]$ and sublayer output $F(x)=[0.5,-1]$, addition first gives $[1.5,1]$, with mean $1.25$ and variance $0.0625$. If this LayerNorm has $\gamma=[1,1]$ and $\beta=[0,0]$, the complete output is
<!-- bilingual-en:end -->

$$
R_F(x)=\frac{[0.25,-0.25]}{\sqrt{0.0625+\varepsilon}}.
$$

原论文训练时还在子层输出上施加 dropout，再执行相加与归一化，因此包含该操作的写法是

<!-- bilingual-en:start -->
During training, the original paper additionally applies dropout to the sublayer output before addition and normalization. Including that operation gives
<!-- bilingual-en:end -->

$$
R_F(X)=\operatorname{LN}\bigl(X+\operatorname{Dropout}(F(X))\bigr).
$$

这张卡提供原始 Transformer 的子层布局，供读取和复现该架构时使用。

<!-- bilingual-en:start -->
This layout specifies the original Transformer sublayer for reading and reproducing that architecture.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification.*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/pdf/1706.03762)，§3.1 与 Figure 1：支持各子层的残差相加、随后层归一化的顺序；§5.4 的 Residual Dropout：支持在子层输出上先做 dropout、再相加与归一化。算例把此顺序与 [[LayerNorm]] 的公式直接结合。
  <!-- bilingual-en:start -->
  Section 3.1 and Figure 1 establish residual addition followed by layer normalization for each sublayer. Residual Dropout in Section 5.4 places dropout on the sublayer output before addition and normalization. The example directly combines this sequence with the [[LayerNorm|LayerNorm formula]].
  <!-- bilingual-en:end -->

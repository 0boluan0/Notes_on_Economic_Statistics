---
aliases:
  - "把独立的键值头改成共享头一般会改变模型输出"
  - "Replacing independent key-value heads with shared heads generally changes model outputs"
student_os: knowledge-atom
atom_id: LLM-INF-018
atom_type: boundary
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 把独立的键值头改成共享头一般会改变模型输出

<!-- bilingual-en:start -->
*Replacing independent key-value heads with shared heads generally changes model outputs*
<!-- bilingual-en:end -->

对已经训练好的标准[[多头注意力]]，把独立的 K/V 投影合并成[[多查询注意力]]或[[分组查询注意力]]，一般会改变模型计算的函数。因此，仅合并已生成的 K/V、选择其中一个头，或对投影取平均，都没有保留原输出的一般保证。

<!-- bilingual-en:start -->
For a trained standard [[多头注意力|multi-head attention]] layer, merging independent K/V projections into [[多查询注意力|multi-query attention]] or [[分组查询注意力|grouped-query attention]] generally changes the computed function. Merging cached K/V, selecting one head, or averaging projections therefore provides no general guarantee of preserving the original output.
<!-- bilingual-en:end -->

## 一个候选就足以给出反例

<!-- bilingual-en:start -->
*A single candidate is enough for a counterexample*
<!-- bilingual-en:end -->

考虑两个 attention 头，每头 value 宽度为 1，且只有一个可见候选。因此每个头的 Softmax 权重都是 1，与 query/key 的具体分数无关。令候选来源为行向量 $X_M=(1,0)$，两个 value 投影与输出投影为

<!-- bilingual-en:start -->
Consider two attention heads with value width one and a single visible candidate. Each Softmax weight is therefore one, regardless of its query/key score. Let the candidate source be the row $X_M=(1,0)$, with the following value projections and output projection:
<!-- bilingual-en:end -->

$$
W_1^V=\begin{pmatrix}0\\0\end{pmatrix},\qquad
W_2^V=\begin{pmatrix}2\\0\end{pmatrix},\qquad
W^O=I_2.
$$

原来的两个读出为 $X_MW_1^V=0$、$X_MW_2^V=2$，拼接并投影后输出 $(0,2)$。现在对 value 投影取平均，得到共享投影

<!-- bilingual-en:start -->
The original readouts are $X_MW_1^V=0$ and $X_MW_2^V=2$, producing $(0,2)$ after concatenation and output projection. Averaging the value projections gives
<!-- bilingual-en:end -->

$$
\overline W^V=\frac{W_1^V+W_2^V}{2}
=\begin{pmatrix}1\\0\end{pmatrix}.
$$

共享之后两个头都读出 1，最终输出变为 $(1,1)$。即使计算完全精确，模型输出仍改变；这个反例不依赖浮点误差，也不依赖长上下文。

<!-- bilingual-en:start -->
Both heads now return one, giving final output $(1,1)$. The result changes even with exact arithmetic; the counterexample requires neither floating-point error nor a long context.
<!-- bilingual-en:end -->

## 怎样理解转换后的训练

<!-- bilingual-en:start -->
*Interpreting training after conversion*
<!-- bilingual-en:end -->

GQA 原论文采用组内 K/V 投影平均，再继续预训练以适应新结构。这是获得可用新模型的方法，不是原函数不变的证明。若原来的投影已满足所需共享约束，特定转换可以保持计算；任意 MHA checkpoint 不自动满足这些条件。

<!-- bilingual-en:start -->
The GQA paper averages K/V projections within groups and then continues pretraining to adapt to the new structure. This produces an adapted model rather than proving function preservation. A particular conversion can preserve computation when the original projections already satisfy the required sharing constraints, but an arbitrary MHA checkpoint need not do so.
<!-- bilingual-en:end -->

与之比较，[[注意力计算与显存|FlashAttention 的精确执行优化]]保留注意力算子，改变分块和中间结果的存储方式。判断一种优化时，需要先确认它保持了同一个模型函数，还是引入了需要重新评测的结构或数值近似。

<!-- bilingual-en:start -->
By comparison, [[注意力计算与显存|FlashAttention's exact execution optimization]] preserves the attention operator while changing tiling and intermediate storage. An optimization should first be classified by whether it preserves the model function or introduces structural or numerical changes that require reevaluation.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Ainslie et al. (2023), *GQA*](https://arxiv.org/html/2305.13245v3)，§2.1–2.2：支持 K/V 投影平均后继续预训练的转换流程。单候选反例由原 MHA 与共享投影的公式直接构造，证明的是“不具有一般等价保证”，不预测某个 checkpoint 的质量损失。
  <!-- bilingual-en:start -->
  Sections 2.1–2.2 provide conversion by averaging K/V projections followed by continued pretraining. The single-candidate counterexample is constructed directly from the original and shared projection formulas. It disproves a general equivalence guarantee without predicting a particular checkpoint's quality loss.
  <!-- bilingual-en:end -->

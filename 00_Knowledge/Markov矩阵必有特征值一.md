---
aliases:
  - 有限 Markov 矩阵必有特征值一
  - Every finite stochastic matrix has eigenvalue one
  - Markov 矩阵的特征值一
student_os: knowledge-atom
atom_id: LA-EIG-046
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵]]"
  - "[[特征对]]"
related:
  - "[[Markov矩阵左右约定]]"
  - "[[Markov矩阵谱边界]]"
  - "[[Markov稳态分布]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 有限 Markov 矩阵必有特征值一
<!-- bilingual-en:start -->
*Every finite Markov matrix has eigenvalue one*
<!-- bilingual-en:end -->

> [!summary] 核心定理
> 对有限行随机或列随机矩阵 $P$，都有
> $$1\in\sigma(P).$$
> 约定只决定全一向量出现在右特征方程还是左特征方程中，不改变特征值 $1$ 的存在性。
>
> <!-- bilingual-en:start -->
> Every finite row- or column-stochastic matrix has eigenvalue one. The convention changes which side visibly carries the all-ones eigenvector, not the eigenvalue itself.
> <!-- bilingual-en:end -->

若 $P$ 是行随机矩阵，每行元素之和为一，所以
$$
P\mathbf1=\mathbf1.
$$
因此 $\mathbf1$ 是特征值 $1$ 的右特征向量。

若 $P$ 是列随机矩阵，每列元素之和为一，所以
$$
\mathbf1^TP=\mathbf1^T.
$$
这先说明 $1$ 是 $P^T$ 的特征值。又因为
$$
\det(tI-P^T)=\det((tI-P)^T)=\det(tI-P),
$$
$P$ 与 $P^T$ 具有相同的特征多项式，因此 $1$ 也是 $P$ 的特征值。
<!-- bilingual-en:start -->
For a row-stochastic matrix, $P\mathbf1=\mathbf1$. For a column-stochastic matrix, $\mathbf1^TP=\mathbf1^T$, so one is an eigenvalue of $P^T$; equality of the characteristic polynomials of $P$ and $P^T$ transfers it to $P$.
<!-- bilingual-en:end -->

这个定理只确定一个谱点。它没有说明其余特征值位于哪里，也没有说明固定点空间是否一维；前一个问题由[[Markov矩阵谱边界]]回答，后一个问题关系到[[Markov稳态唯一判据|稳态唯一性]]。
<!-- bilingual-en:start -->
This theorem identifies only one spectral point. The location of the remaining spectrum is handled by the [[Markov矩阵谱边界|spectral boundary for Markov matrices]], while the dimension of the fixed-point space is a separate stationarity question.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 对列随机矩阵，为什么 $\mathbf1^TP=\mathbf1^T$ 仍足以推出 $1$ 是 $P$ 的特征值？
>
> **答案：** 它先给出 $P^T$ 的特征值 $1$；$P$ 与 $P^T$ 的特征多项式相同，所以特征值也相同。
>
> <!-- bilingual-en:start -->
> For a column-stochastic matrix, why does $\mathbf1^TP=\mathbf1^T$ still imply that one is an eigenvalue of $P$?
>
> **Answer:** It first gives eigenvalue one for $P^T$; $P$ and $P^T$ have the same characteristic polynomial and hence the same eigenvalues.
> <!-- bilingual-en:end -->

## 来源与核验

- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/)：核对行随机与列随机矩阵都具有特征值 $1$，以及全一向量所在一侧随约定改变。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对列随机约定下 $\mathbf1^T P=\mathbf1^T$ 与特征值 $1$。
<!-- bilingual-en:start -->
- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/) was checked for eigenvalue one in both row- and column-stochastic conventions and for the convention-dependent side of the all-ones eigenvector.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]] was checked for $\mathbf1^TP=\mathbf1^T$ and eigenvalue one under the column-stochastic convention.
<!-- bilingual-en:end -->

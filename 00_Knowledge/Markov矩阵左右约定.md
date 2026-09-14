---
aliases:
  - "行随机与列随机约定决定全一向量和稳态向量位于特征方程的哪一侧"
  - Row-stochastic and column-stochastic conventions
  - Markov 矩阵左右特征向量约定
student_os: knowledge-atom
atom_id: LA-EIG-021
atom_set: eigenvalues-linear-dynamics
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵]]"
  - "[[特征对]]"
related:
  - "[[Markov稳态分布]]"
  - "[[Markov矩阵必有特征值一]]"
  - "[[Markov矩阵谱边界]]"
  - "[[DTMC时间齐次转移核]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 行随机与列随机约定决定全一向量和稳态向量位于特征方程的哪一侧
<!-- bilingual-en:start -->
*Row- and column-stochastic conventions determine which side contains the all-ones vector and the stationary vector*
<!-- bilingual-en:end -->

> [!summary] 先固定概率向量的方向
> - **列随机约定：** $P_{ij}=\Pr(X_{k+1}=i\mid X_k=j)$，列概率向量按 $p_{k+1}=Pp_k$ 演化；于是 $\mathbf1^TP=\mathbf1^T$，全一向量在左侧，稳态向量在右侧。
> - **行随机约定：** $P_{ij}=\Pr(X_{k+1}=j\mid X_k=i)$，行概率向量按 $p_{k+1}^T=p_k^TP$ 演化；于是 $P\mathbf1=\mathbf1$，全一向量在右侧，稳态向量在左侧。
> <!-- bilingual-en:start -->
> Column-stochastic matrices act on column probability vectors and place the all-ones vector on the left. Row-stochastic matrices act on row probability vectors and place it on the right.
> <!-- bilingual-en:end -->

两种约定互为转置，没有谁更正确。真正危险的是在同一段计算里混用两套式子：例如先写 $p_{k+1}=Pp_k$，随后却套用行随机矩阵的 $P\mathbf1=\mathbf1$。矩阵本身可能仍然合法，但概率总和守恒和稳态方程会被写到错误的一侧。

MIT 18.06SC 采用列随机约定，所以
$$
\mathbf1^TP=\mathbf1^T,
\qquad
P\pi=\pi.
$$
这里 $\mathbf1^T$ 是特征值 $1$ 的左特征向量，[[Markov稳态分布|稳态分布]] $\pi$ 是相应的右特征向量。若换成行随机约定，两个式子整体转置即可。
<!-- bilingual-en:start -->
Under the MIT column convention, $\mathbf1^T$ is a left eigenvector and a stationary distribution is a right eigenvector. Transposing the convention swaps the two sides without changing the underlying Markov evolution.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 若使用列概率向量且 $p_{k+1}=Pp_k$，检验概率总和守恒应写哪一个式子？
>
> **答案：** $\mathbf1^TP=\mathbf1^T$，于是 $\mathbf1^Tp_{k+1}=\mathbf1^Tp_k$。

## 来源与核验

- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/)：交叉核对行随机与列随机互为转置，以及特征值 $1$ 的左右位置。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对课程采用的列随机约定。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|MIT Lecture 24 transcript]]：核对全一向量、稳态向量与转置约定。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U2_S11_Recitation_Problem_Solving_Markov_Matrices.pdf|MIT Session 2.11 recitation]]：核对两状态列随机计算。

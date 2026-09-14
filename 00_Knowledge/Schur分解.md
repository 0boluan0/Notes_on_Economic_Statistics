---
aliases:
  - "Schur 分解把复方阵酉相似为上三角矩阵，并把实方阵正交相似为准上三角矩阵"
  - Schur decomposition
  - Schur 分解
student_os: knowledge-atom
atom_id: LA-EIG-035
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[相似矩阵]]"
  - "[[Unitary 矩阵]]"
  - "[[正交矩阵]]"
related:
  - "[[Schur分解存在性]]"
  - "[[Normal 矩阵谱定理]]"
  - "[[Jordan、Schur与SVD用途边界]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Schur 分解把复方阵酉相似为上三角矩阵，并把实方阵正交相似为准上三角矩阵
<!-- bilingual-en:start -->
*Schur decomposition unitarily triangularizes a complex square matrix and orthogonally reduces a real square matrix to quasi-triangular form*
<!-- bilingual-en:end -->

> [!summary] 核心定义
> 对复方阵 $A\in\mathbb C^{n\times n}$，若酉矩阵 $Q$ 与上三角矩阵 $T$ 满足
> $$A=QTQ^*.$$
> 就称它们给出 $A$ 的复 Schur 分解。实 Schur 分解写成 $A=QTQ^T$，其中 $Q$ 正交，$T$ 是由 $1\times1$ 与 $2\times2$ 对角块组成的实准上三角矩阵。
> <!-- bilingual-en:start -->
> A complex Schur decomposition writes $A=QTQ^*$ with $Q$ unitary and $T$ upper triangular. A real Schur decomposition uses an orthogonal $Q$ and a real quasi-triangular $T$ whose diagonal blocks have size one or two.
> <!-- bilingual-en:end -->

Schur 分解保持相似结构，因为 $Q^{-1}=Q^*$（实数情形为 $Q^{-1}=Q^T$）。复数情形中，$T$ 的对角元就是 $A$ 的特征值；实数情形的 $1\times1$ 块承载实特征值，$2\times2$ 块承载共轭复特征值对。

Schur 形中的 $T$ 通常只是上三角或准上三角，并不一定是对角矩阵。复数域中，$T$ 能取成对角矩阵当且仅当 $A$ 是 normal 矩阵。每个方阵是否都能找到上述分解，是单独的[[Schur分解存在性|存在定理]]。

平面旋转
$$R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$$
没有实特征向量基，却已经是一个合法的实 Schur $2\times2$ 块。转到复数域后，它可酉对角化为 $\operatorname{diag}(i,-i)$。
<!-- bilingual-en:start -->
Real Schur form keeps a conjugate pair inside one real two-dimensional invariant block.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 已知 $A=QTQ^*$ 是复 Schur 分解，$Q$ 与 $T$ 分别必须满足什么结构？
>
> **答案：** $Q$ 必须酉，$T$ 必须上三角；$T$ 不必是对角矩阵。

## 来源与核验

- [LAPACK Users' Guide, Eigenvalues and Schur Factorization](https://www.netlib.org/lapack/lug/node50.html)：核对复 Schur 形与实准上三角 Schur 形。
- [LAPACK Users' Guide, Nonsymmetric Eigenproblems](https://www.netlib.org/lapack95/lug95/node34.html)：核对 Schur 向量与不变子空间的数值角色。

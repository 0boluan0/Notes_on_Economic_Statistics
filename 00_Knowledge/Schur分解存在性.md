---
aliases:
  - "每个复方阵都有复 Schur 分解且每个实方阵都有实 Schur 分解"
  - Existence of Schur decompositions
  - Schur 分解存在性
student_os: knowledge-atom
atom_id: LA-EIG-044
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Schur分解]]"
  - "[[特征多项式底层域]]"
related:
  - "[[Normal 矩阵谱定理]]"
  - "[[Jordan标准形的域条件]]"
  - "[[Jordan、Schur与SVD用途边界]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 每个复方阵都有复 Schur 分解且每个实方阵都有实 Schur 分解
<!-- bilingual-en:start -->
*Every complex square matrix has a complex Schur decomposition, and every real square matrix has a real Schur decomposition*
<!-- bilingual-en:end -->

> [!summary] 存在定理
> 对每个 $A\in\mathbb C^{n\times n}$，都存在酉矩阵 $Q$ 与上三角矩阵 $T$，使
> $$A=QTQ^*.$$
> 对每个 $A\in\mathbb R^{n\times n}$，都存在实正交矩阵 $Q$ 与实准上三角矩阵 $T$，使
> $$A=QTQ^T,$$
> 其中 $T$ 的对角块只有 $1\times1$ 与 $2\times2$ 两种。
> <!-- bilingual-en:start -->
> Every complex square matrix is unitarily triangularizable. Every real square matrix is orthogonally reducible to real quasi-triangular form with one-by-one and two-by-two diagonal blocks.
> <!-- bilingual-en:end -->

复数情形可以用归纳法看清。复特征多项式至少有一个根，因此 $A$ 有单位特征向量 $q_1$。把 $q_1$ 扩充成酉矩阵 $Q_1=[q_1\ Q_2]$，则
$$
Q_1^*AQ_1=
\begin{bmatrix}
\lambda&*\\
0&A_2
\end{bmatrix}.
$$
对较小的方阵 $A_2$ 重复同一步骤，便得到上三角 $T$。这个证明只使用不变子空间，不要求 $A$ 有一组特征向量基，所以缺陷矩阵也有 Schur 分解。

实数情形不能逐个挑选非实特征向量，而是把一对共轭特征值留在同一个实二维不变子空间中；于是 $T$ 的对角线上可能出现 $2\times2$ 实块。平面旋转
$$
\begin{bmatrix}0&-1\\1&0\end{bmatrix}
$$
本身就是这样的块：它没有实特征向量，却完全符合实 Schur 形。
<!-- bilingual-en:start -->
The complex proof builds an orthonormal basis from one invariant direction at a time and proceeds by induction. The real theorem keeps a conjugate pair inside a real two-dimensional invariant block, which accounts for the two-by-two blocks in real Schur form.
<!-- bilingual-en:end -->

存在性不等于酉对角化。复 Schur 形总能取为上三角；只有当 $A$ 是[[Normal 矩阵|normal 矩阵]]时，才可把这个三角形进一步取成对角形。

> [!question]- 最小自检
> 一个实方阵没有任何实特征值，是否因此不存在实 Schur 分解？
>
> **答案：** 不是。共轭复特征值可由实准上三角形中的 $2\times2$ 对角块承载。

## 来源与核验

- [LAPACK Users' Guide, Eigenvalues and Schur Factorization](https://www.netlib.org/lapack/lug/node50.html)：核对任意复方阵的三角 Schur 形与任意实方阵的准上三角 Schur 形。
- [LAPACK Users' Guide, Nonsymmetric Eigenproblems](https://www.netlib.org/lapack95/lug95/node34.html)：核对实 Schur 形的 $1\times1$、$2\times2$ 对角块及数值角色。

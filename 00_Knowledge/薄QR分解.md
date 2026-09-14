---
aliases:
  - "对 m≥n 的矩阵薄 QR 分解写成 A=Q₁R₁其中 Q₁ 有 n 个标准正交列而 R₁ 是 n×n 上三角矩阵"
  - Thin QR factorization
  - Reduced QR factorization
  - 经济 QR 分解
student_os: knowledge-atom
atom_id: LA-PROJ-026
atom_set: orthogonal-projection-least-squares
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[QR分解]]"
  - "[[标准正交组]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
leads_to:
  - "[[薄QR最小二乘]]"
related:
  - "[[Gram-Schmidt正交化]]"
---

# 对 m≥n 的矩阵薄 QR 分解写成 A=Q₁R₁其中 Q₁ 有 n 个标准正交列而 R₁ 是 n×n 上三角矩阵
<!-- bilingual-en:start -->
*For an m-by-n matrix with m at least n, a thin QR factorisation writes A=Q₁R₁ with n orthonormal columns in Q₁ and an n-by-n upper-triangular R₁*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对 $A\in\mathbb R^{m\times n}$ 且 $m\ge n$，薄 QR 分解写成
> $$
> A=Q_1R_1,
> \qquad
> Q_1\in\mathbb R^{m\times n},
> \quad Q_1^TQ_1=I_n,
> \quad R_1\in\mathbb R^{n\times n}
> $$
> 其中 $R_1$ 上三角。它从完整 QR 中只保留重构 $A$ 所需的前 $n$ 个正交列与对应行块。
> <!-- bilingual-en:start -->
> A thin QR factorisation keeps the first $n$ orthonormal columns and the corresponding square upper-triangular block from a full QR factorisation of an $m\times n$ matrix with $m\ge n$.
> <!-- bilingual-en:end -->

若完整分解写成
$$
A=[Q_1\ Q_2]
\begin{bmatrix}R_1\\0\end{bmatrix},
$$
则下方零块说明 $Q_2$ 对重构 $A$ 没有贡献，因而可删去。薄分解中的 $Q_1$ 一般不是方阵，所以只有 $Q_1^TQ_1=I_n$；$Q_1Q_1^T$ 是到 $C(A)$ 的投影，而不是通常的 $I_m$。
<!-- bilingual-en:start -->
In a full factorisation, the rows associated with $Q_2$ multiply a zero block, so they are unnecessary for reconstructing $A$. The rectangular $Q_1$ satisfies $Q_1^TQ_1=I_n$, while $Q_1Q_1^T$ is a projector rather than the $m\times m$ identity.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

对
$$
A=\begin{bmatrix}1&1\\0&1\\0&0\end{bmatrix},
$$
可取
$$
Q_1=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix},
\qquad
R_1=\begin{bmatrix}1&1\\0&1\end{bmatrix}.
$$
完整 $Q$ 还可补上 $e_3$，但这个第三方向乘到零行，对 $A=Q_1R_1$ 没有贡献。
<!-- bilingual-en:start -->
For the displayed $3\times2$ matrix, $Q_1=[e_1\ e_2]$ and $R_1=\begin{bmatrix}1&1\\0&1\end{bmatrix}$. A full $Q$ may add $e_3$, but that direction multiplies a zero row and is unnecessary for reconstructing $A$.
<!-- bilingual-en:end -->

## 秩边界
<!-- bilingual-en:start -->
*Rank boundary*
<!-- bilingual-en:end -->

若 $A$ 满列秩，则 $R_1$ 可逆；若再约定其对角元为正，薄 QR 在实数情形下唯一。秩亏时仍可写薄形状的 QR，但 $R_1$ 必奇异，不能套用要求可逆三角因子的 [[薄QR最小二乘]] 求解规则。
<!-- bilingual-en:start -->
Full column rank makes $R_1$ nonsingular, and a positive-diagonal convention gives uniqueness in the real case. A rank-deficient matrix may still have a thin-shaped QR factorisation, but its triangular factor is singular and cannot support the invertible triangular solve used by [[薄QR最小二乘|thin-QR least squares]].
<!-- bilingual-en:end -->

> [!question]- 自检
> $Q_1\in\mathbb R^{m\times n}$、$m>n$ 且列标准正交时，为什么不能写 $Q_1^{-1}=Q_1^T$？
>
> **答案：** $Q_1$ 不是方阵，没有双侧逆；只有 $Q_1^TQ_1=I_n$，而 $Q_1Q_1^T$ 是列空间投影。
>
> <!-- bilingual-en:start -->
> **Question:** If $Q_1\in\mathbb R^{m\times n}$ has orthonormal columns and $m>n$, why can we not write $Q_1^{-1}=Q_1^T$?
>
> **Answer:** $Q_1$ is rectangular and has no two-sided inverse. Only $Q_1^TQ_1=I_n$ holds; $Q_1Q_1^T$ is the projector onto its column space.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary]]：核对 Gram–Schmidt 构造的长方 $Q_1$、上三角 $R_1$ 与 $Q_1^TQ_1=I$。
- [LAPACK Users' Guide: QR Factorization](https://www.netlib.org/lapack/lug/node40.html)：核对完整与薄 QR 的尺寸关系，以及满列秩时三角因子可逆。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 2.4 summary supports the rectangular orthonormal factor produced by Gram-Schmidt and the upper-triangular coefficient matrix.
- The LAPACK Users' Guide supports the dimensions of full and thin QR factorisations and nonsingularity of the triangular factor under full column rank.
<!-- bilingual-en:end -->

---
aliases:
  - "每个有限维实矩阵或复矩阵都存在奇异值分解"
  - 奇异值分解存在定理
  - Existence of the singular value decomposition
student_os: knowledge-atom
atom_id: LA-SVD-020
atom_set: singular-value-decomposition-low-rank
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异值分解]]"
  - "[[Hermitian 谱定理]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[SVD手算流程]]"
  - "[[奇异向量]]"
related:
  - "[[四个基本子空间]]"
---

# 每个有限维实矩阵或复矩阵都存在奇异值分解
<!-- bilingual-en:start -->
*Every finite-dimensional real or complex matrix has a singular value decomposition*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 对任意 $A\in\mathbb F^{m\times n}$，其中 $\mathbb F=\mathbb R$ 或 $\mathbb C$，总能找到酉矩阵 $U\in\mathbb F^{m\times m}$、$V\in\mathbb F^{n\times n}$ 和非负矩形对角矩阵 $\Sigma\in\mathbb R^{m\times n}$，使
> $$
> A=U\Sigma V^*.
> $$
> 因而 SVD 不要求矩阵方形、可逆、正规或可对角化。
> <!-- bilingual-en:start -->
> Every real or complex matrix admits an SVD, without requiring the matrix to be square, invertible, normal, or diagonalisable.
> <!-- bilingual-en:end -->

存在性的关键来自 Gram 矩阵 $A^*A$。它是 Hermitian 半正定矩阵，所以谱定理给出一组标准正交特征向量 $v_i$ 和非负特征值 $\lambda_i$。令 $\sigma_i=\sqrt{\lambda_i}$；对 $\sigma_i>0$，再定义
$$
u_i=\frac{Av_i}{\sigma_i}.
$$
这些 $u_i$ 彼此标准正交，并满足 $Av_i=\sigma_i u_i$。$A^*A$ 的零特征向量已经补齐 $\mathcal N(A)$ 中的右侧基；再用 $\mathcal N(A^*)$ 的标准正交基补齐 $U$，就得到完整的 $U$、$\Sigma$ 与 $V$。

这个论证既证明了分解存在，也解释了为什么奇异值非负、为什么左右奇异向量分居两个空间。把证明转成实际计算步骤见 [[SVD手算流程]]；零空间补基为何不唯一见 [[SVD零块补基不唯一]]。

> [!question]- 自检
> 一个 $5\times3$、秩为 $2$ 的矩阵既不方形也不可逆，为什么仍能保证存在 SVD？
>
> **答案：** 因为存在性来自 Hermitian 半正定矩阵 $A^*A$ 的谱定理，而不来自 $A$ 本身可逆或可对角化。

## 来源与核验

- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对实复矩阵 SVD 的无条件存在性、因子尺寸与酉性。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对由 $A^TA$ 的谱分解构造实矩阵 SVD 的证明。

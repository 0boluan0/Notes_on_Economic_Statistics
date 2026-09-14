---
aliases:
  - "任意矩阵的 $AA^+$ 是其列空间正交投影"
  - "任意矩阵的 AA+ 是其列空间正交投影"
  - AA plus column-space projector
  - Pseudoinverse projection identity
  - AA+ 投影公式
student_os: knowledge-atom
atom_id: LA-PROJ-008
atom_set: orthogonal-projection-least-squares
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[正交投影矩阵判别]]"
  - "[[Moore–Penrose 伪逆]]"
related:
  - "[[A+A行空间投影]]"
  - "[[最小二乘解唯一性]]"
leads_to:
  - "[[最小范数最小二乘解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
---

# 任意矩阵的 $AA^+$ 是其列空间正交投影
<!-- bilingual-en:start -->
*For every matrix $A$, $AA^+$ is the orthogonal projector onto its column space*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 对任意实矩阵 $A$ 及其 Moore–Penrose 伪逆 $A^+$，
> $$
> AA^+=P_{C(A)}.
> $$
> 因此 $AA^+b$ 是 $b$ 在 $C(A)$ 上唯一的正交投影与最佳拟合值；这个结论不要求 $A$ 满秩。
> <!-- bilingual-en:start -->
> For every real matrix $A$ and its Moore–Penrose pseudoinverse $A^+$, $AA^+=P_{C(A)}$. Hence $AA^+b$ is the unique orthogonal projection and best-fitting vector in $C(A)$, with no full-rank assumption on $A$.
> <!-- bilingual-en:end -->

## SVD 直接显出列空间投影
<!-- bilingual-en:start -->
*The SVD exposes the column-space projector directly*
<!-- bilingual-en:end -->

取紧 SVD
$$
A=U_r\Sigma_rV_r^T,\qquad A^+=V_r\Sigma_r^{-1}U_r^T,
$$
其中 $U_r$ 与 $V_r$ 分别张成列空间与行空间，则
$$
AA^+=U_rU_r^T.
$$
这是标准正交基的 $QQ^T$ 投影公式，所以自动对称且幂等，值域恰好是 $C(A)$。

这些式子按实矩阵书写；对复矩阵应把转置换为共轭转置，此时仍有 $AA^+=U_rU_r^*$，并投影到 $C(A)$。
<!-- bilingual-en:start -->
With the compact SVD $A=U_r\Sigma_rV_r^T$ and $A^+=V_r\Sigma_r^{-1}U_r^T$, the columns of $U_r$ span the column space and $AA^+=U_rU_r^T$. This is the orthonormal-basis projector $QQ^T$, so it is symmetric and idempotent with range $C(A)$. Over the complex numbers, transpose becomes conjugate transpose and $AA^+=U_rU_r^*$ still projects onto $C(A)$.
<!-- bilingual-en:end -->

## 满秩与单位矩阵的边界
<!-- bilingual-en:start -->
*Full-rank and identity boundary*
<!-- bilingual-en:end -->

因为 $AA^+$ 作用在输出空间，所以 $AA^+=I_m$ 当且仅当 $C(A)=\mathbb R^m$，也就是 $A$ 满行秩；满列秩并不足够。若 $A$ 秩亏，$AA^+$ 仍是定义良好的正交投影，拟合值 $AA^+b$ 仍唯一。输入空间中的对应恒等式是 [[A+A行空间投影|$A^+A=P_{C(A^T)}$]]；为什么 $A^+b$ 会在所有最小二乘系数中选出范数最小者，见[[最小范数最小二乘解]]。
<!-- bilingual-en:start -->
Because $AA^+$ acts on the output space, $AA^+=I_m$ exactly when $C(A)=\mathbb R^m$, equivalently when $A$ has full row rank; full column rank is not enough. If $A$ is rank deficient, $AA^+$ remains a well-defined orthogonal projector and the fitted vector $AA^+b$ remains unique. The corresponding input-space identity is [[A+A行空间投影|$A^+A=P_{C(A^T)}$]], while [[最小范数最小二乘解|the minimum-norm least-squares theorem]] explains why $A^+b$ removes the null-space freedom from the coefficients.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 为什么 $A$ 秩亏时，$AA^+b$ 仍然是唯一拟合值？
> <!-- bilingual-en:start -->
> Why is $AA^+b$ still the unique fitted vector when $A$ is rank deficient?
> <!-- bilingual-en:end -->
>
> **答案：** 秩亏只会使产生该拟合值的系数可能不唯一；列空间本身仍是子空间，而向子空间的正交投影始终唯一。
> <!-- bilingual-en:start -->
> **Answer:** Rank deficiency can make the coefficient vector non-unique, but the column space remains a subspace and orthogonal projection onto a subspace is unique.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8 summary]]：支持用 SVD 构造 $A^+$、行空间到列空间的可逆对应以及投影解释。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.3 SVD 定义伪逆|课程 3.8.3]] 与 [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.5 伪逆同时解决两类“最佳解”|课程 3.8.5]]：支持 $AA^+$ 的目标空间、唯一拟合值及其满秩边界。
- [[最小二乘解唯一性]]：支持秩亏时拟合值唯一而系数可能不唯一的通用线性代数表述。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 3.8 summary supports the SVD construction of $A^+$, the invertible correspondence from row space to column space, and the projection interpretation.
- Course Sections 3.8.3 and 3.8.5 support the target space of $AA^+$, the unique fitted vector, and its rank boundary.
- [[最小二乘解唯一性|The least-squares uniqueness criterion]] gives the same distinction between a unique fitted vector and potentially non-unique coefficients in general linear-algebra notation.
<!-- bilingual-en:end -->

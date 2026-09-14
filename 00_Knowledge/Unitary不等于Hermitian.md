---
aliases:
  - "Unitary 与 Hermitian 是不同的矩阵性质，任一方都不能单独推出另一方"
  - Neither unitarity nor Hermitian symmetry alone implies the other
student_os: knowledge-atom
atom_id: LA-SPD-030
atom_set: symmetric-positive-definite
atom_type: concept-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Unitary 矩阵]]"
  - "[[Hermitian 矩阵]]"
related:
  - "[[Normal 矩阵]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# Unitary 与 Hermitian 是不同的矩阵性质，任一方都不能单独推出另一方
<!-- bilingual-en:start -->
*Neither unitarity nor Hermitian symmetry alone implies the other*
<!-- bilingual-en:end -->

> [!summary] 核心区别
> Unitary 条件是 $U^*U=I$，它说共轭转置是逆，并保持内积。Hermitian 条件是 $A^*=A$，它说矩阵等于自己的共轭转置。两个等式回答不同问题，任一个都不能单独推出另一个。
> <!-- bilingual-en:start -->
> Unitarity, $U^*U=I$, says that the adjoint is the inverse and that inner products are preserved. Hermitian symmetry, $A^*=A$, says that the matrix equals its adjoint. These equations express different properties, and neither one alone implies the other.
> <!-- bilingual-en:end -->

## 两个最小反例
<!-- bilingual-en:start -->
*Two minimal counterexamples*
<!-- bilingual-en:end -->

标量矩阵 $U=[i]$ 满足
$$
U^*U=(-i)i=1,
$$
所以它 unitary；但 $U^*=[-i]\ne[i]=U$，所以它不 Hermitian。反过来，$A=[2]$ 满足 $A^*=A$，所以它 Hermitian；但 $A^*A=4\ne1$，所以它不 unitary。

<!-- bilingual-en:start -->
The scalar matrix $U=[i]$ is unitary because $U^*U=(-i)i=1$, but it is not Hermitian because $U^*=-i\ne i=U$. Conversely, $A=[2]$ is Hermitian but not unitary because $A^*A=4\ne1$.
<!-- bilingual-en:end -->

## 两种性质重合时
<!-- bilingual-en:start -->
*When the two properties overlap*
<!-- bilingual-en:end -->

若矩阵同时 Hermitian 且 unitary，则 $U^*=U$ 与 $U^*U=I$ 合在一起给出 $U^2=I$。因此它的特征值只能是 $1$ 或 $-1$。这说明两类有交集，但交集不等于任一整类。二者都是 [[Normal 矩阵]]，也不会因此变成同一性质。

<!-- bilingual-en:start -->
If a matrix is both Hermitian and unitary, then $U^*=U$ and $U^*U=I$ imply $U^2=I$, so its eigenvalues can only be $1$ or $-1$. The classes overlap, but their intersection is not either whole class. Both classes are normal, which still does not make them the same property.
<!-- bilingual-en:end -->

> [!question]- 自检
> $[i]$ 与 $[2]$ 分别反驳了哪一个错误推理？
>
> **答案：** $[i]$ 反驳“unitary $\Rightarrow$ Hermitian”；$[2]$ 反驳“Hermitian $\Rightarrow$ unitary”。
> <!-- bilingual-en:start -->
> Which false implication is refuted by $[i]$, and which by $[2]$?
>
> **Answer:** $[i]$ refutes “unitary $\Rightarrow$ Hermitian”, while $[2]$ refutes “Hermitian $\Rightarrow$ unitary”.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|MIT 18.06SC Session 3.2 summary]]：核对 Hermitian 与 unitary 的定义。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.2.1 复向量的长度与 Hermitian 内积|课程 3.2.1]]：核对 $A^*=A$、$U^*U=I$ 与共轭转置记号；两个标量反例由定义直接计算。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 3.2 summary was checked for the definitions of Hermitian and unitary matrices.
- Course Section 3.2.1 was checked for $A^*=A$, $U^*U=I$, and conjugate-transpose notation. Both scalar counterexamples are verified directly from those definitions.
<!-- bilingual-en:end -->

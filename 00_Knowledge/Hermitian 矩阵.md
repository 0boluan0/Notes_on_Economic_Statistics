---
aliases:
  - "复方阵等于自身共轭转置时称为 Hermitian 矩阵"
  - Hermitian matrix
  - 自伴矩阵
student_os: knowledge-atom
atom_id: LA-SPD-003
atom_set: symmetric-positive-definite
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[实对称矩阵]]"
  - "[[Unitary不等于Hermitian]]"
  - "[[Hermitian 谱定理]]"
  - "[[Hermitian二次型实值判据]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 复方阵等于自身共轭转置时称为 Hermitian 矩阵
<!-- bilingual-en:start -->
*A complex square matrix is Hermitian when it equals its conjugate transpose*
<!-- bilingual-en:end -->

> [!summary] 定义
> 对 $A\in\mathbb C^{n\times n}$，若
> $$A=A^*=\overline{A}^{\,T},$$
> 就称 $A$ 为 Hermitian（厄米、自伴）矩阵。逐项等价于 $a_{ij}=\overline{a_{ji}}$。
> <!-- bilingual-en:start -->
> A matrix $A\in\mathbb C^{n\times n}$ is Hermitian when $A=A^*=\overline{A}^{\,T}$, equivalently $a_{ij}=\overline{a_{ji}}$ for all indices.
> <!-- bilingual-en:end -->

主对角元必须为实数，因为 $a_{ii}=\overline{a_{ii}}$。实矩阵没有非零虚部，此时 $A^*=A^T$，所以实 Hermitian 矩阵恰好就是实对称矩阵。
<!-- bilingual-en:start -->
Every diagonal entry is real because it equals its own complex conjugate. For a real matrix, the conjugate transpose reduces to the transpose, so Hermitian and symmetric mean the same thing.
<!-- bilingual-en:end -->

例如
$$
\begin{bmatrix}2&i\\-i&3\end{bmatrix}
$$
是 Hermitian。只做转置会误判它；复数内积和二次型必须使用 $x^*$，因为非零复向量也可能满足 $x^Tx=0$。
<!-- bilingual-en:start -->
The displayed matrix is Hermitian. Transposition without conjugation would misclassify it. Complex inner products and quadratic forms must use $x^*$, because a nonzero complex vector can have $x^Tx=0$.
<!-- bilingual-en:end -->

> [!question]- 自检
> Hermitian 矩阵的对角元为什么不能含非零虚部？
>
> **答案：** $A=A^*$ 给出 $a_{ii}=\overline{a_{ii}}$，只有实数等于自身共轭。
>
> <!-- bilingual-en:start -->
> **Question:** Why can a diagonal entry of a Hermitian matrix not have a nonzero imaginary part?
>
> **Answer:** From $A=A^*$ we get $a_{ii}=\overline{a_{ii}}$, and only real numbers equal their own complex conjugates.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|MIT 18.06SC Session 3.2 summary]]：核对共轭转置与 Hermitian 定义。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.2.1 复向量的长度与 Hermitian 内积|课程 3.2.1]]：核对 $x^*$、对角元与实对称特例。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 3.2 summary was checked for conjugate transpose and the Hermitian definition.
- Course Section 3.2.1 was checked for the use of $x^*$, diagonal entries, and the real-symmetric special case.
<!-- bilingual-en:end -->

---
aliases:
  - "复方阵满足 $U^*U=I$ 时称为 unitary 矩阵"
  - "复方阵的列标准正交当且仅当它是 unitary 矩阵"
  - Unitary matrix
  - 酉矩阵
student_os: knowledge-atom
atom_id: LA-SPD-004
atom_set: symmetric-positive-definite
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[Unitary不等于Hermitian]]"
  - "[[正交矩阵]]"
  - "[[标准正交基投影公式]]"
  - "[[Normal 矩阵谱定理]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 复方阵满足 $U^*U=I$ 时称为 unitary 矩阵
<!-- bilingual-en:start -->
*A complex square matrix satisfying $U^*U=I$ is called unitary*
<!-- bilingual-en:end -->

> [!summary] 定义
> 对 $U\in\mathbb C^{n\times n}$，下列条件等价：
> $$U^*U=I,\qquad UU^*=I,\qquad U^{-1}=U^*.$$
> 此时称 $U$ 为 unitary（酉）矩阵。
> <!-- bilingual-en:start -->
> For $U\in\mathbb C^{n\times n}$, the conditions $U^*U=I$, $UU^*=I$, and $U^{-1}=U^*$ are equivalent. Such a matrix is unitary.
> <!-- bilingual-en:end -->

unitary 矩阵保持 Hermitian 内积：
$$
(Ux)^*(Uy)=x^*y,\qquad \|Ux\|_2=\|x\|_2.
$$
它是正交矩阵在复数域的推广；实 unitary 矩阵正是正交矩阵。
<!-- bilingual-en:start -->
A unitary matrix preserves the Hermitian inner product and Euclidean norm. It is the complex analogue of an orthogonal matrix, and a real unitary matrix is exactly orthogonal.
<!-- bilingual-en:end -->

“Unitary 矩阵”也只指方阵。若长方复矩阵满足 $U^*U=I$，只能说它的列标准正交；不能由此写 $U^{-1}=U^*$ 或 $UU^*=I$。矩形情形的投影结论见 [[标准正交基投影公式]]；与 Hermitian 的类别辨析见 [[Unitary不等于Hermitian]]。
<!-- bilingual-en:start -->
The term unitary matrix is reserved for square matrices. If a rectangular complex matrix satisfies $U^*U=I$, its columns are orthonormal, but one cannot infer $U^{-1}=U^*$ or $UU^*=I$. See [[标准正交基投影公式|the projection formula for an orthonormal basis]] for the resulting projection and [[Unitary不等于Hermitian|the distinction between unitary and Hermitian matrices]] for the category boundary.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么证明复矩阵保持长度时必须写 $U^*U$ 而不是 $U^TU$？
>
> **答案：** 复内积包含共轭；$\|Ux\|^2=x^*U^*Ux$。
>
> <!-- bilingual-en:start -->
> **Question:** Why must a proof that a complex matrix preserves length use $U^*U$ rather than $U^TU$?
>
> **Answer:** The complex inner product includes conjugation, so $\|Ux\|^2=x^*U^*Ux$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|MIT 18.06SC Session 3.2 summary]]：核对 unitary 条件与长度保持。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.2.1 复向量的长度与 Hermitian 内积|课程 3.2.1]]：核对 $U^{-1}=U^*$ 及其与正交矩阵的关系。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 3.2 summary was checked for the unitary condition and norm preservation.
- Course Section 3.2.1 was checked for $U^{-1}=U^*$ and the relation to orthogonal matrices.
<!-- bilingual-en:end -->

---
aliases:
  - "复方阵与其共轭转置可交换时称为 normal 矩阵"
  - Normal matrix
  - 正规矩阵
student_os: knowledge-atom
atom_id: LA-SPD-005
atom_set: symmetric-positive-definite
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[Hermitian 矩阵]]"
  - "[[Unitary 矩阵]]"
  - "[[Unitary不等于Hermitian]]"
  - "[[Normal 矩阵谱定理]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 复方阵与其共轭转置可交换时称为 normal 矩阵
<!-- bilingual-en:start -->
*A complex square matrix is normal when it commutes with its conjugate transpose*
<!-- bilingual-en:end -->

> [!summary] 定义
> 对 $A\in\mathbb C^{n\times n}$，若
> $$A^*A=AA^*,$$
> 就称 $A$ 为 normal（正规）矩阵。
> <!-- bilingual-en:start -->
> A complex square matrix $A$ is normal when it commutes with its adjoint: $A^*A=AA^*$.
> <!-- bilingual-en:end -->

Hermitian、skew-Hermitian 和 unitary 矩阵都 normal：前两类有 $A^*=\pm A$，后一类有 $A^*A=AA^*=I$。但 normal 是更大的类别。取 $A=\operatorname{diag}(2,i)$，则 $A^*A=AA^*=\operatorname{diag}(4,1)$，所以它 normal；这个乘积不是 $I$，且 $A^*\ne\pm A$，所以它既非 unitary，也非 Hermitian 或 skew-Hermitian。
<!-- bilingual-en:start -->
Hermitian, skew-Hermitian, and unitary matrices are all normal. The class is strictly larger. For $A=\operatorname{diag}(2,i)$, one has $A^*A=AA^*=\operatorname{diag}(4,1)$, so $A$ is normal; the product is not $I$, and $A^*\ne\pm A$, so it is neither unitary, Hermitian, nor skew-Hermitian.
<!-- bilingual-en:end -->

normal 不是“具有实特征值”的同义词；定义本身只要求 $A$ 与 $A^*$ 可交换。它为何恰好对应标准正交复特征基，由 [[Normal 矩阵谱定理]] 证明。该定理允许特征值为任意复数；例如实斜对称矩阵作为复矩阵是 normal，并且可以有非实的纯虚特征值。
<!-- bilingual-en:start -->
Normality is not another name for having real eigenvalues; the definition itself says only that $A$ commutes with $A^*$. [[Normal 矩阵谱定理|The spectral theorem for normal matrices]] proves its connection with orthonormal diagonalisation over $\mathbb C$, where eigenvalues may still be arbitrary complex numbers. A real skew-symmetric matrix is normal over $\mathbb C$ and may have nonreal, purely imaginary eigenvalues.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个 unitary 矩阵为什么一定 normal？
>
> **答案：** 因为 $A^*A=I=AA^*$。
>
> <!-- bilingual-en:start -->
> **Question:** Why is every unitary matrix normal?
>
> **Answer:** Because $A^*A=I=AA^*$.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT RES.18-011 Algebra I Lecture 27–28 notes](https://ocw.mit.edu/courses/res-18-011-algebra-i-student-notes-fall-2021/mit18_701f21_full_lec_new.pdf#page=136)：核对 normal 定义及 Hermitian/unitary 子类。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.2.1 复向量的长度与 Hermitian 内积|课程复内积约定]]：核对本库使用的共轭转置记号。
<!-- bilingual-en:start -->
- MIT RES.18-011 Lecture 27–28 notes were checked for the definition of normality and the Hermitian/unitary subclasses.
- The local course convention was checked for conjugate-transpose notation.
<!-- bilingual-en:end -->

---
aliases:
  - "Hermitian 矩阵有实特征值和标准正交特征基"
  - Hermitian spectral theorem
  - Spectral decomposition of a Hermitian matrix
student_os: knowledge-atom
atom_id: LA-SPD-008
atom_set: symmetric-positive-definite
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hermitian 矩阵]]"
  - "[[Unitary 矩阵]]"
related:
  - "[[Normal 矩阵谱定理]]"
  - "[[实对称矩阵谱定理]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# Hermitian 矩阵有实特征值和标准正交特征基
<!-- bilingual-en:start -->
*A Hermitian matrix has real eigenvalues and an orthonormal eigenbasis*
<!-- bilingual-en:end -->

> [!summary] Hermitian 谱定理
> 若 $A\in\mathbb C^{n\times n}$ 且 $A=A^*$，则全部特征值为实数，并存在 unitary 矩阵 $U$ 与实对角矩阵 $\Lambda$，使
> $$A=U\Lambda U^*.$$
> $U$ 的列是一组 Hermitian 内积下的标准正交特征基。
> <!-- bilingual-en:start -->
> If $A\in\mathbb C^{n\times n}$ is Hermitian, all its eigenvalues are real and $A=U\Lambda U^*$ for a unitary $U$ and a real diagonal $\Lambda$. The columns of $U$ form an orthonormal eigenbasis.
> <!-- bilingual-en:end -->

这是 normal 谱定理的特例，因为 $A=A^*$ 自动给出 $A^*A=AA^*$。Hermitian 条件额外迫使特征值为实：若 $Ax=\lambda x$，则
$$
\lambda x^*x=x^*Ax=(x^*Ax)^*=\overline\lambda x^*x,
$$
且 $x^*x>0$，所以 $\lambda=\overline\lambda$。
<!-- bilingual-en:start -->
This is a special case of the normal spectral theorem. The Hermitian condition also forces the spectrum to be real: comparing $x^*Ax=\lambda x^*x$ with its conjugate and using $x^*x>0$ gives $\lambda=\overline\lambda$.
<!-- bilingual-en:end -->

不能把 $U^*$ 换成 $U^T$。例如复特征向量可能满足 $x^Tx=0$；只有带共轭的内积才能表达长度与正交性。实对称谱定理则是在全部数据为实时的特例。
<!-- bilingual-en:start -->
The conjugate transpose cannot be replaced by the plain transpose. A nonzero complex vector may satisfy $x^Tx=0$, whereas the conjugated inner product correctly represents length and orthogonality. The real symmetric theorem is the real-valued special case.
<!-- bilingual-en:end -->

> [!question]- 自检
> Hermitian 矩阵比一般 normal 矩阵多保证了什么？
>
> **答案：** 两者都有 unitary 特征基；Hermitian 还保证所有特征值为实数。
>
> <!-- bilingual-en:start -->
> **Question:** What does a Hermitian matrix guarantee beyond what a general normal matrix guarantees?
>
> **Answer:** Both have a unitary eigenbasis, but a Hermitian matrix also has only real eigenvalues.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf|MIT 18.06SC Session 3.2 summary]]：核对 Hermitian 谱分解与实特征值。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.2.1 复向量的长度与 Hermitian 内积|课程 3.2.1]]：核对 $A=U\Lambda U^*$ 与共轭转置边界。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 3.2 summary was checked for Hermitian spectral decomposition and real eigenvalues.
- Course Section 3.2.1 was checked for $A=U\Lambda U^*$ and the conjugate-transpose boundary.
<!-- bilingual-en:end -->

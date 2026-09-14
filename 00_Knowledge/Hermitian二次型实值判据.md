---
aliases:
  - "复方阵是 Hermitian 矩阵当且仅当其二次型对每个复向量都取实值"
  - Real-valued quadratic-form criterion for Hermitian matrices
  - Hermitian quadratic-form test
student_os: knowledge-atom
atom_id: LA-SPD-034
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hermitian 矩阵]]"
  - "[[Hermitian 谱定理]]"
related:
  - "[[复二次型实部]]"
  - "[[二次型]]"
  - "[[二次型对称化]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 复方阵是 Hermitian 矩阵当且仅当其二次型对每个复向量都取实值
<!-- bilingual-en:start -->
*A complex square matrix is Hermitian exactly when its quadratic form is real for every complex vector*
<!-- bilingual-en:end -->

> [!summary] 判别准则
> 对 $A\in\mathbb C^{n\times n}$，
> $$
> A=A^*
> \iff
> x^*Ax\in\mathbb R\quad\text{对每个 }x\in\mathbb C^n.
> $$
> 量词必须覆盖全部复向量；只测试实向量或几个坐标向量都不够。
> <!-- bilingual-en:start -->
> A complex matrix is Hermitian exactly when $x^*Ax$ is real for every complex vector $x$. Testing only real vectors or only coordinate vectors is insufficient.
> <!-- bilingual-en:end -->

若 $A=A^*$，则
$$
\overline{x^*Ax}=x^*A^*x=x^*Ax,
$$
所以这个标量为实数。反过来，把 $A$ 分解为
$$
A=H+iS,
\qquad
H=\frac{A+A^*}{2},
\qquad
S=\frac{A-A^*}{2i},
$$
其中 $H$、$S$ 都是 Hermitian。若 $x^*Ax$ 对每个 $x$ 都为实，则它的虚部 $x^*Sx$ 对每个 $x$ 都为零。由 Hermitian 谱定理，取 $S$ 的任意单位特征向量便知对应特征值为零，因此 $S=0$，从而 $A=H=A^*$。
<!-- bilingual-en:start -->
Hermitian symmetry makes $x^*Ax$ equal to its complex conjugate. Conversely, write $A=H+iS$ with both $H$ and $S$ Hermitian. Real-valuedness forces $x^*Sx=0$ for every $x$; testing unit eigenvectors of $S$ shows all its eigenvalues vanish, so $S=0$ and $A$ is Hermitian.
<!-- bilingual-en:end -->

只检查实向量会漏掉关键信息。取
$$
A=\begin{bmatrix}0&1\\-1&0\end{bmatrix}.
$$
对每个实向量 $x$ 都有 $x^TAx=0$，但 $A$ 不是 Hermitian；对复向量 $z=(1,i)^T$，有 $z^*Az=2i$，立即暴露非实值。
<!-- bilingual-en:start -->
For every real vector $x$, one has $x^TAx=0$, yet $A$ is not Hermitian. The complex vector $z=(1,i)^T$ gives $z^*Az=2i$, which immediately reveals the nonreal value missed by tests on real vectors.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若只知道每个对角元 $a_{jj}$ 都为实数，为什么仍不能断言 $A$ 是 Hermitian？
>
> **答案：** 这只等价于在坐标向量 $e_j$ 上得到实值，完全没有检查非对角元是否满足 $a_{jk}=\overline{a_{kj}}$。
> <!-- bilingual-en:start -->
> If every diagonal entry $a_{jj}$ is real, why can we still not conclude that $A$ is Hermitian?
>
> **Answer:** This shows only that the quadratic value is real on each coordinate vector $e_j$; it does not test whether the off-diagonal entries satisfy $a_{jk}=\overline{a_{kj}}$.
> <!-- bilingual-en:end -->

## 来源与核验

- [UC Berkeley Math H110, *How to Recognize a Quadratic Form*](https://people.eecs.berkeley.edu/~wkahan/MathH110/qf.pdf)：核对复二次型与 Hermitian、skew-Hermitian 分解。
- [[Hermitian 谱定理]]：核对反向证明中“二次型恒为零推出 Hermitian 矩阵为零”的谱论步骤。
<!-- bilingual-en:start -->
- UC Berkeley Math H110 notes support the decomposition of a complex quadratic expression into Hermitian and skew-Hermitian contributions.
- [[Hermitian 谱定理|The Hermitian spectral theorem]] justifies the reverse implication through eigenvectors of $S$.
<!-- bilingual-en:end -->

---
aliases:
  - Normal matrix
  - 正规矩阵
student_os: knowledge-atom
atom_id: LA-SPD-005
atom_set: symmetric-positive-definite
atom_type: definition
status: source-checked
related:
  - "[[Hermitian 矩阵]]"
  - "[[Unitary 矩阵]]"
  - "[[Unitary不等于Hermitian]]"
  - "[[Normal 矩阵谱定理]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Normal matrices
<!-- bilingual-en:end -->
*正规矩阵*

<!-- bilingual-en:start -->
A **normal matrix** is a square matrix whose action commutes with its conjugate transpose: applying those two actions in either order gives the same result. This includes [[Hermitian 矩阵|Hermitian matrices]], which equal their conjugate transpose, and [[Unitary 矩阵|unitary matrices]], which preserve standard complex inner products, but also many others. The condition is useful because it identifies exactly the complex matrices that admit perpendicular, unit-length eigenvector coordinates; the [[Normal 矩阵谱定理|normal spectral theorem]] proves that connection.
<!-- bilingual-en:end -->

**正规矩阵**是与自身共轭转置可交换的方阵：以任意先后次序施加这两个作用，结果都一样。它包括等于自身共轭转置的 [[Hermitian 矩阵]]，以及保持标准复内积的 [[Unitary 矩阵|酉矩阵]]，也包括更多矩阵。这项条件的用途在于：它恰好找出能够选取标准正交特征向量坐标的复矩阵，这一联系由 [[Normal 矩阵谱定理]]证明。

<!-- bilingual-en:start -->
For a concrete example, let
<!-- bilingual-en:end -->

先看一个可以直接相乘的例子：

$$A=\begin{pmatrix}2&-3\\3&2\end{pmatrix},\qquad
A^*=\begin{pmatrix}2&3\\-3&2\end{pmatrix},\qquad
A^*A=AA^*=\begin{pmatrix}13&0\\0&13\end{pmatrix}.$$

<!-- bilingual-en:start -->
The star means [[矩阵转置|conjugate transpose]]: transpose the matrix, then conjugate every entry. Complex conjugation changes $a+bi$ into $a-bi$, where $a,b$ are real and $i^2=-1$. This example has real entries, so conjugation changes nothing and $A^*=A^T$. The two products agree, although $A^*\ne A$. Thus normality is weaker than requiring the matrix to equal its conjugate transpose.
<!-- bilingual-en:end -->

星号表示[[矩阵转置|共轭转置]]：先转置，再对每个元素取共轭。复共轭把 $a+bi$ 换成 $a-bi$，其中 $a,b$ 为实数，$i^2=-1$。本例的元素全为实数，共轭不改变元素，所以 $A^*=A^T$。两个乘积相等，但 $A^*\ne A$；因此，正规性比“矩阵等于其共轭转置”更弱。

> [!important] The defining equality · 定义等式
> <!-- bilingual-en:start -->
> Let $n\ge1$ be an integer and $A\in\mathbb C^{n\times n}$. The matrix is **normal** when
> <!-- bilingual-en:end -->
>
> 设 $n\ge1$ 为整数，$A\in\mathbb C^{n\times n}$。若满足下式，就称它为**正规矩阵**：
>
> $$\boxed{A^*A=AA^*.}$$
>
> <!-- bilingual-en:start -->
> These are equal $n\times n$ matrices, not merely equal determinants or traces. Equivalently, they send every input vector to the same output. For real $A$, the condition becomes $A^TA=AA^T$. No invertibility or positivity is required. The zero matrix and the [[对角矩阵|diagonal matrix]] $\operatorname{diag}(0,2)$ are normal; $\operatorname{diag}$ places the listed numbers on the main diagonal and zeros elsewhere.
> <!-- bilingual-en:end -->
>
> 要求两个 $n\times n$ 矩阵相等，不只是行列式或迹相等；等价地，它们对每个输入向量给出相同输出。实矩阵的条件就是 $A^TA=AA^T$。不要求可逆或正性。零矩阵与[[对角矩阵]] $\operatorname{diag}(0,2)$ 都是正规矩阵；$\operatorname{diag}$ 把列出的数放在主对角线上，其余位置为零。

<!-- bilingual-en:start -->
## Checking the whole product
<!-- bilingual-en:end -->
*比较完整乘积*

<!-- bilingual-en:start -->
For contrast, take $B=\begin{pmatrix}1&1\\0&1\end{pmatrix}$. Direct multiplication gives
<!-- bilingual-en:end -->

作为对照，取 $B=\begin{pmatrix}1&1\\0&1\end{pmatrix}$，直接相乘得到

$$B^*B=\begin{pmatrix}1&1\\1&2\end{pmatrix},\qquad
BB^*=\begin{pmatrix}2&1\\1&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
The first diagonal entries already differ, so $B$ is not normal. Its entries being real, and its diagonal entries being positive, do not decide this question. In particular, “normal” does not mean that the rows or columns have been normalized to length one.
<!-- bilingual-en:end -->

两个乘积的第一个对角元已经不同，所以 $B$ 不正规。元素全为实数、对角元为正，都不能决定正规性。“正规”也不表示各行或各列已经归一化成单位长度。

<!-- bilingual-en:start -->
There is a useful length interpretation of the defining equality. In the [[内积|standard complex inner product]], $x^*y=\sum_j\overline{x_j}y_j$ and the squared length is $\|x\|_2^2=x^*x=\sum_j|x_j|^2$, where $|x_j|$ is the complex modulus. If $A$ is normal, then for every $x$,
<!-- bilingual-en:end -->

定义等式也有长度解释。在[[内积|标准复内积]]中，$x^*y=\sum_j\overline{x_j}y_j$，长度平方为 $\|x\|_2^2=x^*x=\sum_j|x_j|^2$，其中 $|x_j|$ 是复数的模。若 $A$ 正规，则对每个 $x$，

$$\|Ax\|_2^2=x^*A^*Ax=x^*AA^*x=\|A^*x\|_2^2.$$

<!-- bilingual-en:start -->
This compares the two output lengths; it does not assert that either equals the input length. In the opening example, both output lengths are $\sqrt{13}\,\|x\|_2$.
<!-- bilingual-en:end -->

这里比较的是两个输出的长度，并未说它们等于输入长度。开头例子的两个输出长度都为 $\sqrt{13}\,\|x\|_2$。

<!-- bilingual-en:start -->
## Important subclasses, and a matrix outside them
<!-- bilingual-en:end -->
*常见子类与更大的范围*

<!-- bilingual-en:start -->
A [[Hermitian 矩阵|Hermitian matrix]] satisfies $A^*=A$, so both defining products are $A^2$. A **skew-Hermitian matrix** satisfies $A^*=-A$, so both products are $-A^2$. A [[Unitary 矩阵|unitary matrix]] satisfies $A^*A=AA^*=I_n$, where $I_n$ is the [[单位矩阵|identity matrix]]. Each condition therefore implies normality. Over the reals, symmetric and [[实反对称矩阵|skew-symmetric]] matrices are the corresponding first two subclasses.
<!-- bilingual-en:end -->

[[Hermitian 矩阵]]满足 $A^*=A$，所以定义两边均为 $A^2$。**skew-Hermitian（斜 Hermitian）矩阵**满足 $A^*=-A$，两边均为 $-A^2$。[[Unitary 矩阵|酉矩阵]]满足 $A^*A=AA^*=I_n$，其中 $I_n$ 是[[单位矩阵]]。因此三种条件都推出正规性。实数域中的对称矩阵与[[实反对称矩阵|反对称矩阵]]分别对应前两类。

<!-- bilingual-en:start -->
Normal matrices extend beyond all three subclasses. For example, take
<!-- bilingual-en:end -->

正规矩阵的范围比这三类更大。一个简单的复数例子是

$$D=\operatorname{diag}(2,i),\qquad D^*=\operatorname{diag}(2,-i),\qquad
D^*D=DD^*=\operatorname{diag}(4,1).$$

<!-- bilingual-en:start -->
Thus $D$ is normal, but $D^*\ne D$, $D^*\ne-D$ and $D^*D\ne I_2$. It is neither Hermitian, skew-Hermitian nor unitary. Its eigenvalues are the diagonal values $2$ and $i$: an [[特征对|eigenvalue]] is a scalar $\lambda$ with $Dv=\lambda v$ for some nonzero vector $v$. Normality permits nonreal eigenvalues. A real skew-symmetric example is the quarter-turn $R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$, which is normal and has eigenvalues $i,-i$ over $\mathbb C$.
<!-- bilingual-en:end -->

所以 $D$ 正规，但 $D^*\ne D$、$D^*\ne-D$、$D^*D\ne I_2$；它既不是 Hermitian，也不是 skew-Hermitian 或酉矩阵。其特征值是对角元 $2,i$：[[特征对|特征值]]指存在非零向量 $v$ 满足 $Dv=\lambda v$ 的标量 $\lambda$。正规性允许非实特征值。实反对称的例子是四分之一圈旋转 $R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$：它正规，在 $\mathbb C$ 上的特征值为 $i,-i$。

<!-- bilingual-en:start -->
The [[Normal 矩阵谱定理|spectral theorem]] explains why this algebraic definition supplies an orthonormal eigenbasis over $\mathbb C$. It also explains why a real normal matrix need not admit a real orthogonal diagonalization. The distinction between unitary and Hermitian structure is treated separately in [[Unitary不等于Hermitian|their comparison]].
<!-- bilingual-en:end -->

[[Normal 矩阵谱定理]]解释了这个代数定义为何能在 $\mathbb C$ 上提供标准正交特征基，也说明实正规矩阵为何未必可以实正交对角化。酉与 Hermitian 结构的区别另见 [[Unitary不等于Hermitian]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, fourth edition, 7.18–7.20, PDF pp.249–250](https://linear.axler.net/LADR4e.pdf#page=249): the normality definition, the opening real example and the output-length relation. The products above are also checked directly.
- [MIT 8.05, Lecture Notes 5, §6, PDF p.12](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/005979fa741c3ea2e0430456b70caf93_MIT8_05F13_Chap_05.pdf#page=12) and [MIT RES.18-011, Definitions 27.4–27.8, PDF pp.136–137](https://ocw.mit.edu/courses/res-18-011-algebra-i-student-notes-fall-2021/mit18_701f21_full_lec_new.pdf#page=136): adjoint notation and the Hermitian, skew-Hermitian and unitary subclasses.
- [[03_Positive Definite Matrices and Applications#3.2.1 复向量的长度与 Hermitian 内积|Permanent course Section 3.2.1]]: the convention $x^*y$, conjugate-linear in the first input. The mixed diagonal and nonnormal examples are verified directly.
<!-- bilingual-en:end -->

- Axler 7.18–7.20，PDF249–250：核对定义、开头实矩阵例子与输出长度关系；正文乘积另行直接核算。
- MIT8.05 第6节第12页与 RES.18-011 定义27.4–27.8，第136–137页：核对共轭转置记号及 Hermitian、skew-Hermitian、酉子类。
- 永久课程第3.2.1节核对 $x^*y$ 与第一个自变量共轭线性的约定；混合对角例子与非正规例子直接核算。

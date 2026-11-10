---
aliases:
  - Hermitian spectral theorem
  - Spectral decomposition of a Hermitian matrix
student_os: knowledge-atom
atom_id: LA-SPD-008
atom_set: symmetric-positive-definite
atom_type: theorem
status: source-checked
requires:
  - "[[Hermitian 矩阵]]"
  - "[[Unitary 矩阵]]"
related:
  - "[[Normal 矩阵谱定理]]"
  - "[[实对称矩阵谱定理]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# The Hermitian spectral theorem
<!-- bilingual-en:end -->
*Hermitian 谱定理*

<!-- bilingual-en:start -->
Complex matrix entries do not necessarily mean complex eigenvalues. A [[Hermitian 矩阵|Hermitian matrix]] has conjugate entries mirrored across its diagonal, $A=A^*$. This structure guarantees a complete set of mutually perpendicular unit eigendirections, each with a **real** multiplier. The Hermitian spectral theorem expresses this as a coordinate system in which the matrix is real and diagonal.
<!-- bilingual-en:end -->

矩阵里有复数，不代表特征值一定是非实数。[[Hermitian 矩阵]]沿对角线两侧的元素互为共轭，满足 $A=A^*$。这种结构保证：可以找出一整组彼此垂直的单位特征方向，每个方向上的乘数都为**实数**。Hermitian 谱定理把这件事写成一种坐标系：矩阵在其中是实对角矩阵。

<!-- bilingual-en:start -->
Here $*$ means conjugate transpose: transpose the array and replace each scalar $a+ib$ by $a-ib$, where $i^2=-1$. The complex [[内积|inner product]] is $u^*v$. Thus a vector has squared length $u^*u=\sum_j|u_j|^2$, and two vectors are perpendicular when $u^*v=0$. An [[特征对|eigenvector]] satisfies $Au=\lambda u$ with $u\ne0$.
<!-- bilingual-en:end -->

这里 $*$ 表示共轭转置：先转置数组，再把每个 $a+ib$ 换成 $a-ib$，其中 $i^2=-1$。复数[[内积]]采用 $u^*v$，所以长度平方为 $u^*u=\sum_j|u_j|^2$，垂直则表示 $u^*v=0$。[[特征对|特征向量]]满足 $Au=\lambda u$ 且 $u\ne0$。

> [!example] Complex directions, real stretches · 复方向，实伸缩
> $$A=\begin{pmatrix}2&i\\-i&2\end{pmatrix},\qquad
> u_1=\frac1{\sqrt2}\begin{pmatrix}i\\1\end{pmatrix},\qquad
> u_2=\frac1{\sqrt2}\begin{pmatrix}-i\\1\end{pmatrix}.$$
> <!-- bilingual-en:start -->
> The mirrored entries $i,-i$ are conjugates, so $A$ is Hermitian. Direct multiplication gives $Au_1=3u_1$ and $Au_2=u_2$. Moreover $u_1^*u_1=u_2^*u_2=1$ and $u_1^*u_2=0$. Consequently $A(c_1u_1+c_2u_2)=3c_1u_1+c_2u_2$, even when the coefficients $c_1,c_2$ are complex.
> <!-- bilingual-en:end -->
>
> 对称位置的 $i,-i$ 互为共轭，所以 $A$ 是 Hermitian。直接计算得 $Au_1=3u_1$、$Au_2=u_2$，且 $u_1^*u_1=u_2^*u_2=1$、$u_1^*u_2=0$。因此即使 $c_1,c_2$ 为复数，也有 $A(c_1u_1+c_2u_2)=3c_1u_1+c_2u_2$。

<!-- bilingual-en:start -->
## A unitary eigenbasis
<!-- bilingual-en:end -->
*由特征向量组成的酉基*

<!-- bilingual-en:start -->
For any finite $n\ge1$ and Hermitian $A\in\mathbb C^{n\times n}$, there are a [[Unitary 矩阵|unitary matrix]] $U=[u_1\ \cdots\ u_n]$ and a real [[对角矩阵|diagonal matrix]] $\Lambda$ such that
<!-- bilingual-en:end -->

对任意有限 $n\ge1$ 和 Hermitian 矩阵 $A\in\mathbb C^{n\times n}$，存在 [[Unitary 矩阵|酉矩阵]] $U=[u_1\ \cdots\ u_n]$ 与实[[对角矩阵]] $\Lambda$，使

$$U^*U=UU^*=I_n,\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n),\quad\lambda_j\in\mathbb R,
\qquad A=U\Lambda U^*.$$

<!-- bilingual-en:start -->
The identity $U^*U=I_n$, with $I_n$ the [[单位矩阵|identity matrix]], says that the columns are an [[标准正交组|orthonormal family]]. All $n$ columns together form a basis. For an input $z$, $U^*z$ gives its coordinates in this basis, $\Lambda$ multiplies each coordinate by the corresponding eigenvalue, and $U$ converts back. These eigenvalues can be positive, zero or negative; Hermitian does not by itself mean positive definite.
<!-- bilingual-en:end -->

$I_n$ 是[[单位矩阵]]；$U^*U=I_n$ 表示各列构成[[标准正交组]]，全部 $n$ 列合起来是一组基。输入为 $z$ 时，$U^*z$ 求新坐标，$\Lambda$ 按对应特征值逐项相乘，$U$ 再换回。特征值允许正、零或负；Hermitian 本身不等于正定。

<!-- bilingual-en:start -->
Conversely, if $A=U\Lambda U^*$ with unitary $U$ and **real** diagonal $\Lambda$, then $A^*=U\Lambda^*U^*=A$. This factorization therefore characterizes Hermitian matrices. It also gives the direction-by-direction expansion
<!-- bilingual-en:end -->

反过来，若 $A=U\Lambda U^*$，其中 $U$ 酉、$\Lambda$ 为**实**对角矩阵，则 $A^*=U\Lambda^*U^*=A$。所以这种分解恰好刻画 Hermitian 矩阵。它也给出逐方向的展开：

$$A=\sum_{j=1}^n\lambda_j u_ju_j^*,\qquad
Az=\sum_{j=1}^n\lambda_j(u_j^*z)u_j.$$

<!-- bilingual-en:start -->
The matrix $u_ju_j^*$ is the [[正交投影|orthogonal projector]] onto the complex line spanned by $u_j$: it retains that component and discards the perpendicular ones.
<!-- bilingual-en:end -->

$u_ju_j^*$ 是投向 $u_j$ 所张成复直线的[[正交投影]]：保留沿该方向的分量，丢弃垂直分量。

<!-- bilingual-en:start -->
## Why the eigenvalues are real and the basis is complete
<!-- bilingual-en:end -->
*特征值为什么实，基为什么完整*

<!-- bilingual-en:start -->
Suppose $Av=\lambda v$ and $v\ne0$. Taking the conjugate of the scalar $v^*Av$ gives $(v^*Av)^*=v^*A^*v=v^*Av$. Hence that scalar is real. But $v^*Av=\lambda v^*v$ and $v^*v>0$, so $\lambda$ is real. For eigenvectors $u,v$ with distinct eigenvalues $\lambda,\mu$, the same symmetry yields
<!-- bilingual-en:end -->

设 $Av=\lambda v$、$v\ne0$。标量 $v^*Av$ 的共轭为 $(v^*Av)^*=v^*A^*v=v^*Av$，所以它为实数。又因为 $v^*Av=\lambda v^*v$ 且 $v^*v>0$，所以 $\lambda$ 为实数。对分别对应不同特征值 $\lambda,\mu$ 的特征向量 $u,v$，同样的对称性给出

$$\lambda u^*v=(Au)^*v=u^*Av=\mu u^*v.$$

<!-- bilingual-en:start -->
The first equality uses the fact just proved that $\lambda=\overline\lambda$. Thus distinct eigenspaces are perpendicular.
<!-- bilingual-en:end -->

第一步使用了刚证明的 $\lambda=\overline\lambda$。于是不同特征空间彼此垂直。

> [!note]- Why no directions are missing · 为什么不会缺方向
> <!-- bilingual-en:start -->
> In positive dimension, the [[特征多项式底层域|complex characteristic polynomial]] has a root, so an eigenvector exists. Normalize one to $u$. Its [[正交补|orthogonal complement]] $u^\perp=\{z:u^*z=0\}$ has dimension $n-1$. For $z$ in that subspace,
> $$u^*Az=(Au)^*z=\lambda u^*z=0.$$
> Thus $A$ maps the subspace into itself. Expressing this restricted map in an orthonormal basis gives another Hermitian matrix, now of dimension $n-1$. Repeating the construction, with a scalar as the one-dimensional base case, supplies a complete orthonormal eigenbasis. Repeated eigenvalues do not invalidate this induction.
> <!-- bilingual-en:end -->
>
> 正维空间中，[[特征多项式底层域|复特征多项式]]有根，所以存在特征向量。把其中一个标准化为 $u$。其[[正交补]] $u^\perp=\{z:u^*z=0\}$ 维数为 $n-1$；对其中的 $z$，上式说明 $Az$ 仍在这个子空间内。用标准正交基表示这个限制映射，会得到一个 $n-1$ 维 Hermitian 矩阵。继续重复，以一维标量为归纳起点，就能找齐标准正交特征基。重特征值不会破坏这个归纳过程。

<!-- bilingual-en:start -->
The choice of basis need not be unique. Within a repeated [[特征空间|eigenspace]], [[Gram-Schmidt正交化|Gram–Schmidt orthogonalization]] stays inside that space. Even with distinct eigenvalues, a column can be multiplied by a complex scalar of magnitude one without changing its normalization or eigenvalue. Arbitrary eigenvectors for the same eigenvalue need not be perpendicular; the theorem guarantees that a suitable basis exists.
<!-- bilingual-en:end -->

基的选择不必唯一。在重[[特征空间]]中，[[Gram-Schmidt正交化]]始终留在该空间内部。即使特征值互异，也可以把某列乘以模长为一的复数，不改变其长度与特征值。同一特征值下随意取的向量未必垂直；定理保证的是存在一组合适的基。

<!-- bilingual-en:start -->
## Conjugation and the distinction from normal matrices
<!-- bilingual-en:end -->
*共轭与 normal 矩阵的区别*

<!-- bilingual-en:start -->
Plain transpose cannot replace conjugate transpose. For $v=(1,i)^T$, one gets $v^Tv=1+i^2=0$ but $v^*v=1+|i|^2=2$. The first expression fails to measure squared length. Similarly the required inverse of $U$ is $U^*$, not in general $U^T$.
<!-- bilingual-en:end -->

普通转置不能替代共轭转置。取 $v=(1,i)^T$，则 $v^Tv=1+i^2=0$，而 $v^*v=1+|i|^2=2$。前者不能衡量长度平方。同样，$U$ 的逆应为 $U^*$，一般不是 $U^T$。

<!-- bilingual-en:start -->
A [[Normal 矩阵|normal matrix]] satisfies $A^*A=AA^*$. Every Hermitian matrix has that property, so the Hermitian theorem is also a special case of the [[Normal 矩阵谱定理|complex normal spectral theorem]]. The general normal theorem permits complex eigenvalues: $\operatorname{diag}(i,1)$ is normal and already diagonal, but is not Hermitian. Hermitian structure adds the real-spectrum conclusion. When all entries are real, it reduces to the [[实对称矩阵谱定理|real symmetric spectral theorem]], with a real orthogonal basis.
<!-- bilingual-en:end -->

[[Normal 矩阵]]满足 $A^*A=AA^*$。每个 Hermitian 矩阵都有这一性质，所以本定理也是[[Normal 矩阵谱定理|复 normal 谱定理]]的特例。但一般 normal 定理允许复特征值：$\operatorname{diag}(i,1)$ 是 normal 且已经对角化，却不是 Hermitian。Hermitian 结构额外保证了实谱。全部元素为实时，它退化为[[实对称矩阵谱定理]]，并且可以选实正交基。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, fourth edition, 7.12 and 7.31, PDF pp.247 and 260–261](https://linear.axler.net/LADR4e.pdf#page=247): real eigenvalues for self-adjoint operators, complete unitary diagonalization for complex normal operators, and the real-spectrum distinction. The invariant-orthogonal-complement induction above gives a direct Hermitian proof; [4.12, PDF p.139](https://linear.axler.net/LADR4e.pdf#page=139) supplies complex eigenvalue existence.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=1|MIT Session 3.2 summary, pp.1–2]]: complex lengths and inner products, Hermitian matrices and unitary columns. The finite-dimensional completeness proof uses the textbook above.
- [[03_Positive Definite Matrices and Applications#3.2.1 复向量的长度与 Hermitian 内积|Permanent course Section 3.2.1]]: course notation for conjugate transpose and the nonzero vector with zero un-conjugated square. All displayed products, the converse and the normal counterexample are checked directly.
<!-- bilingual-en:end -->

- Axler 7.12、7.31（PDF247、260–261页）：核对自伴算子的实特征值、复 normal 算子的完整酉对角化以及实谱区别。正文用不变正交补给出直接的 Hermitian 证明；4.12（PDF139页）提供复特征值存在性。
- MIT Session3.2第1–2页：核对复长度与内积、Hermitian 矩阵、酉矩阵的列；完整的有限维证明依据上列教材。
- 永久课程3.2.1：核对共轭转置的课程记号及非零向量的无共轭平方可能为零。全部展示乘积、反向结论及normal反例直接核算。

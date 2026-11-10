---
aliases:
  - Spectral theorem for normal matrices
  - Unitary diagonalization criterion
student_os: knowledge-atom
atom_id: LA-SPD-006
atom_set: symmetric-positive-definite
atom_type: theorem
status: source-checked
requires:
  - "[[Normal 矩阵]]"
  - "[[Unitary 矩阵]]"
related:
  - "[[Hermitian 谱定理]]"
  - "[[可对角化判据]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# The spectral theorem for normal matrices
<!-- bilingual-en:end -->
*正规矩阵谱定理*

<!-- bilingual-en:start -->
An eigenvector coordinate system lets a matrix act by multiplying each coordinate separately. The **normal spectral theorem** tells us when those coordinates can also be chosen perpendicular and of unit length. For a complex square matrix, the exact condition is [[Normal 矩阵|normality]], $A^*A=AA^*$. No distinct-eigenvalue or invertibility assumption is needed.
<!-- bilingual-en:end -->

特征向量坐标让矩阵分别乘上各坐标，而**正规矩阵谱定理**回答：什么时候还能把这些坐标方向选成相互正交且长度为一？对复方阵，精确条件就是[[Normal 矩阵|正规性]] $A^*A=AA^*$。不要求特征值互异，也不要求矩阵可逆。

<!-- bilingual-en:start -->
For example, the real matrix $A=\begin{pmatrix}2&-3\\3&2\end{pmatrix}$ mixes the two original coordinates. In complex coordinates it has the following two unit eigenvectors:
<!-- bilingual-en:end -->

例如，实矩阵 $A=\begin{pmatrix}2&-3\\3&2\end{pmatrix}$ 会混合原来的两个坐标；进入复坐标后，可以找到两条单位特征向量：

$$q_1=\frac1{\sqrt2}\begin{pmatrix}i\\1\end{pmatrix},\qquad
q_2=\frac1{\sqrt2}\begin{pmatrix}-i\\1\end{pmatrix},\qquad
Aq_1=(2+3i)q_1,\quad Aq_2=(2-3i)q_2.$$

<!-- bilingual-en:start -->
Here $i^2=-1$, and the star denotes [[矩阵转置|conjugate transpose]]. Under the standard [[内积|complex inner product]] $x^*y$, both self-products $q_j^*q_j$ equal one, and $q_1^*q_2=((-i)(-i)+1)/2=0$. Thus these two vectors are an [[标准正交组|orthonormal basis]] of $\mathbb C^2$. The eigenvalues $2+3i$ and $2-3i$ are not real; orthonormal coordinates do not require real multipliers.
<!-- bilingual-en:end -->

这里 $i^2=-1$，星号表示[[矩阵转置|共轭转置]]。在标准[[内积|复内积]] $x^*y$ 下，两个自身内积 $q_j^*q_j$ 都为一，交叉内积 $q_1^*q_2=((-i)(-i)+1)/2=0$。所以这两个向量构成 $\mathbb C^2$ 的[[标准正交组|标准正交基]]。特征值 $2+3i,2-3i$ 不是实数；标准正交坐标不要求乘数为实数。

<!-- bilingual-en:start -->
## Three equivalent descriptions
<!-- bilingual-en:end -->
*三个等价描述*

<!-- bilingual-en:start -->
Let $n\ge1$ be an integer and $A\in\mathbb C^{n\times n}$. The following statements are equivalent:

- $A$ is normal: $A^*A=AA^*$.
- There are $n$ orthonormal [[特征对|eigenvectors]] $q_1,\ldots,q_n$ spanning $\mathbb C^n$.
- There is a [[Unitary 矩阵|unitary matrix]] $U$ and a complex [[对角矩阵|diagonal matrix]] $\Lambda$ such that $A=U\Lambda U^*$.
<!-- bilingual-en:end -->

设 $n\ge1$ 为整数，$A\in\mathbb C^{n\times n}$。以下三个陈述等价：

- $A$ 正规，即 $A^*A=AA^*$。
- 存在 $n$ 个标准正交的[[特征对|特征向量]] $q_1,\ldots,q_n$，张成整个 $\mathbb C^n$。
- 存在[[Unitary 矩阵|酉矩阵]] $U$ 与复[[对角矩阵]] $\Lambda$，使 $A=U\Lambda U^*$。

<!-- bilingual-en:start -->
More explicitly, $U=[q_1\ \cdots\ q_n]$, $U^*U=UU^*=I_n$, and $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$ lists the eigenvalues on the diagonal, with zeros off it. Here $I_n$ is the [[单位矩阵|identity]]. The equation $AU=U\Lambda$ says column by column that $Aq_j=\lambda_jq_j$, so the last two descriptions are equivalent. For input $x=Uz$, the new coordinate column is $z=U^*x$ and the output is $Ax=U\Lambda z$. This is [[对角化|diagonalization]] with a length-preserving change of coordinates.
<!-- bilingual-en:end -->

具体地，$U=[q_1\ \cdots\ q_n]$，$U^*U=UU^*=I_n$；$\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$ 将特征值放在主对角线，其余为零。$I_n$ 是[[单位矩阵]]。等式 $AU=U\Lambda$ 逐列就是 $Aq_j=\lambda_jq_j$，所以后两个描述等价。若输入 $x=Uz$，新坐标列为 $z=U^*x$，输出为 $Ax=U\Lambda z$。这属于保持长度的坐标变换下的[[对角化]]。

<!-- bilingual-en:start -->
## Why a unitary diagonalization implies normality
<!-- bilingual-en:end -->
*酉对角化为什么推出正规性*

<!-- bilingual-en:start -->
Assume $A=U\Lambda U^*$. Conjugate-transposing gives $A^*=U\Lambda^*U^*$; the diagonal entries of $\Lambda^*$ are the conjugates $\overline{\lambda_j}$. Multiplication gives
<!-- bilingual-en:end -->

假设 $A=U\Lambda U^*$。取共轭转置得到 $A^*=U\Lambda^*U^*$；$\Lambda^*$ 的对角元是 $\overline{\lambda_j}$。相乘得到

$$A^*A=U\Lambda^*\Lambda U^*,\qquad AA^*=U\Lambda\Lambda^*U^*.$$

<!-- bilingual-en:start -->
Both middle products are $\operatorname{diag}(|\lambda_1|^2,\ldots,|\lambda_n|^2)$, where $|\lambda_j|$ denotes complex modulus. Thus $A^*A=AA^*$. Zero or repeated eigenvalues cause no difficulty.
<!-- bilingual-en:end -->

中间的两个乘积均为 $\operatorname{diag}(|\lambda_1|^2,\ldots,|\lambda_n|^2)$，其中 $|\lambda_j|$ 表示复数模长，所以 $A^*A=AA^*$。零特征值或重复特征值都不妨碍这个计算。

<!-- bilingual-en:start -->
## Why normality removes the triangular off-diagonal entries
<!-- bilingual-en:end -->
*正规性怎样消去三角形中的非对角元*

<!-- bilingual-en:start -->
For the other direction, use the [[Schur分解存在性|complex Schur existence theorem]]: every complex square matrix has $A=QTQ^*$ with $Q$ unitary and $T=(t_{jk})$ [[三角矩阵|upper triangular]], meaning $t_{jk}=0$ for $j>k$. Schur's theorem supplies a whole orthonormal coordinate system before requiring eigenvectors in every direction; its linked proof establishes existence without assuming normality.
<!-- bilingual-en:end -->

反向使用[[Schur分解存在性|复 Schur 存在定理]]：每个复方阵都能写成 $A=QTQ^*$，其中 $Q$ 酉，$T=(t_{jk})$ 为上[[三角矩阵|三角矩阵]]，即 $j>k$ 时 $t_{jk}=0$。Schur 定理先提供完整的标准正交坐标系统，并未要求每个方向都是特征向量；链接中的存在性证明不以正规性为前提。

<!-- bilingual-en:start -->
If $A$ is normal, then $T=Q^*AQ$ is normal as well, because
<!-- bilingual-en:end -->

若 $A$ 正规，则 $T=Q^*AQ$ 也正规，因为

$$T^*T=Q^*A^*AQ=Q^*AA^*Q=TT^*.$$

<!-- bilingual-en:start -->
Compare the first diagonal entries of these two products. The first is the squared length of the first column of $T$; the second is the squared length of its first row:
<!-- bilingual-en:end -->

比较两边第一个对角元：左边是 $T$ 第一列的长度平方，右边是第一行的长度平方：

$$(T^*T)_{11}=|t_{11}|^2,\qquad
(TT^*)_{11}=|t_{11}|^2+\sum_{k=2}^n|t_{1k}|^2.$$

<!-- bilingual-en:start -->
Equality forces the displayed sum to be zero. Every summand is a nonnegative real number, so every $t_{1k}$ with $k>1$ is zero. The first column below the diagonal was already zero by triangularity. Consequently $T$ consists of a one-by-one block $t_{11}$ and a remaining upper-triangular block $T_1$, with no entries connecting the blocks. The equation $T^*T=TT^*$ then gives $T_1^*T_1=T_1T_1^*$ on the remaining block.
<!-- bilingual-en:end -->

相等迫使所列和为零。每项都是非负实数，因此 $k>1$ 的全部 $t_{1k}$ 都为零；第一列主对角线以下本来就因三角性而为零。于是 $T$ 分成一阶块 $t_{11}$ 与剩余上三角块 $T_1$，两块之间没有连接项。等式 $T^*T=TT^*$ 在剩余块上继续给出 $T_1^*T_1=T_1T_1^*$。

<!-- bilingual-en:start -->
Repeat the same argument on $T_1$. Each step removes another row and column; the one-dimensional case is already diagonal. Hence every off-diagonal entry of $T$ vanishes. Write this diagonal matrix as $\Lambda$ and take $U=Q$. We obtain $A=U\Lambda U^*$, completing both directions of the theorem.
<!-- bilingual-en:end -->

对 $T_1$ 重复同样论证，每一步去掉一行一列；一维情形本来就是对角形。因此 $T$ 的全部非对角元均为零。将它记为 $\Lambda$ 并令 $U=Q$，就得到 $A=U\Lambda U^*$，充要性的证明完成。

<!-- bilingual-en:start -->
## Repeated eigenvalues and normalization
<!-- bilingual-en:end -->
*重复特征值与归一化*

<!-- bilingual-en:start -->
The theorem guarantees a **choice** of orthonormal eigenbasis, not that every chosen pair of eigenvectors is orthogonal. For $A=2I_2$, both $(1,0)^T$ and $(1,1)^T$ are eigenvectors, but their inner product is one. Dividing the second by its length changes that product to $1/\sqrt2$, not zero. Within a repeated-eigenvalue [[特征空间|eigenspace]], choose an independent basis and use [[Gram-Schmidt正交化|Gram–Schmidt]] to subtract earlier components, then normalize. Linear combinations within the same eigenspace remain there. The zero combination is not an eigenvector.
<!-- bilingual-en:end -->

定理保证**可以选择**标准正交特征基，不是说任意挑出的两个特征向量都正交。对 $A=2I_2$，$(1,0)^T$ 与 $(1,1)^T$ 都是特征向量，但内积为一；把后者除以长度，内积也只是变成 $1/\sqrt2$，并不为零。在重复特征值的[[特征空间]]内部，可以先选线性无关的基，再用 [[Gram-Schmidt正交化]]减去之前方向的分量，最后归一化。同一特征空间内的线性组合仍在其中，但零组合不是特征向量。

<!-- bilingual-en:start -->
Distinct eigenspaces of a normal matrix are orthogonal. To read this from $A=U\Lambda U^*$, an eigenvector $v$ with eigenvalue $\lambda$ has coordinates $c=U^*v$ satisfying $(\lambda_j-\lambda)c_j=0$. Thus only positions with $\lambda_j=\lambda$ can be nonzero. Vectors for two distinct eigenvalues use disjoint positions and have zero inner product. Each repeated eigenvalue has as many orthonormal basis directions as its multiplicity among the diagonal entries. The basis is not unique: multiplying a column by a scalar of modulus one, or changing the orthonormal basis inside one eigenspace, preserves the decomposition when columns and diagonal entries remain paired.
<!-- bilingual-en:end -->

正规矩阵的不同特征空间相互正交。由 $A=U\Lambda U^*$ 可直接看出：特征值为 $\lambda$ 的特征向量 $v$，其坐标 $c=U^*v$ 满足 $(\lambda_j-\lambda)c_j=0$，所以只有 $\lambda_j=\lambda$ 的位置可以非零。两个不同特征值使用不相交的坐标位置，内积因此为零。每个重复特征值拥有的标准正交基方向数，等于它在对角线中出现的次数。基并不唯一：给某列乘模长为一的标量，或在同一特征空间内换一组标准正交基，只要列与对角元保持配对，就保留同一分解。

> [!warning] Ordinary diagonalization and the real field · 普通对角化与实数域
> <!-- bilingual-en:start -->
> The matrix $B=\begin{pmatrix}1&1\\0&2\end{pmatrix}$ has independent eigenvectors $(1,0)^T,(1,1)^T$, so it is diagonalizable. But
> <!-- bilingual-en:end -->
>
> $B=\begin{pmatrix}1&1\\0&2\end{pmatrix}$ 有线性无关的特征向量 $(1,0)^T,(1,1)^T$，因此可对角化；然而
>
> $$B^*B=\begin{pmatrix}1&1\\1&5\end{pmatrix}\ne
> \begin{pmatrix}2&2\\2&4\end{pmatrix}=BB^*.$$
>
> <!-- bilingual-en:start -->
> It is not normal, so no orthonormal eigenbasis exists. A merely invertible change-of-basis matrix does not supply the theorem's unitary condition.
> <!-- bilingual-en:end -->
>
> 它不正规，因而不存在标准正交特征基。换基矩阵仅仅可逆，不能替代本定理的酉条件。
>
> <!-- bilingual-en:start -->
> A real normal matrix need not be diagonalizable over the reals at all. The quarter-turn $R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ satisfies $R^TR=RR^T=I_2$, but its characteristic equation $\lambda^2+1=0$ has no real root. It has complex eigenvalues $i,-i$. The [[实对称矩阵谱定理|real symmetric theorem]] supplies real orthogonal diagonalization for real symmetric matrices. Conversely, $A=QDQ^T$ with real orthogonal $Q$ and real diagonal $D$ forces $A^T=A$. Thus real normality alone does not suffice; [[Hermitian 谱定理|Hermitian structure]] is the complex subclass that additionally forces the spectrum to be real.
> <!-- bilingual-en:end -->
>
> 实正规矩阵甚至未必能在实数域对角化。四分之一圈旋转 $R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ 满足 $R^TR=RR^T=I_2$，但特征方程 $\lambda^2+1=0$ 没有实根；复特征值为 $i,-i$。[[实对称矩阵谱定理]]保证实对称矩阵可以实正交对角化；反过来，若 $A=QDQ^T$ 且 $Q$ 实正交、$D$ 实对角，则必有 $A^T=A$。因此仅有实正规性不够；[[Hermitian 谱定理|Hermitian 结构]]则是在复数域中额外保证谱为实数的子类。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, fourth edition, 7.31 and 7.33, PDF pp.260–261](https://linear.axler.net/LADR4e.pdf#page=260): the complex equivalence, the triangular proof and the opening eigenbasis. The row/column block calculation above gives the induction explicitly.
- [MIT 8.05, Lecture Notes 5, §§5–6, PDF pp.12–15](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/005979fa741c3ea2e0430456b70caf93_MIT8_05F13_Chap_05.pdf#page=12): unitary diagonalization, both theorem directions, and choices inside repeated-eigenvalue eigenspaces. This card retains finite positive dimension and distinguishes nonzero eigenvectors from their whole eigenspace.
- [MIT RES.18-011, Lectures 27–28, PDF pp.137–139](https://ocw.mit.edu/courses/res-18-011-algebra-i-student-notes-fall-2021/mit18_701f21_full_lec_new.pdf#page=137): complex versus real coordinate scope. The theorem here does not impose invertibility; the real rotation boundary uses the explicit quarter-turn.
<!-- bilingual-en:end -->

- Axler7.31、7.33，PDF260–261：核对复数域等价定理、三角证明及开头特征基；正文用行列乘积和剩余块明确展开归纳。
- MIT8.05 第5–6节第12–15页：核对酉对角化、定理双向与重复特征值内部的选基自由；本卡保留有限正维数条件，并区分非零特征向量与整个特征空间。
- MIT RES.18-011 第137–139页：核对实复数域范围；本卡不添加可逆前提，旋转边界使用明确的四分之一圈例子。

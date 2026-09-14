---
aliases:
  - "若参数化的矩形 Jacobian 满列秩，则局部 k 维体积因子是 Gram 行列式的平方根"
  - Rectangular Jacobian volume factor
  - Gram determinant volume element
student_os: knowledge-atom
atom_id: LA-DET-020
atom_set: determinants
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jacobian矩阵]]"
  - "[[行列式体积与取向]]"
related:
  - "[[Jacobian行列式]]"
  - "[[奇异值与Gram谱]]"
part_of:
  - "[[行列式.canvas]]"
  - "[[多元微分.canvas|多元微分]]"
---

# 若参数化的矩形 Jacobian 满列秩，则局部 k 维体积因子是 Gram 行列式的平方根
<!-- bilingual-en:start -->
*A full-column-rank rectangular Jacobian scales local $k$-dimensional volume by the square root of its Gram determinant*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 设 $\phi:U\subseteq\mathbb R^k\to\mathbb R^n$ 可微，$1\le k<n$，并且 $J_\phi(u)$ 在所考察的点满列秩。参数化把输入空间中的微小 $k$ 维体积送入 $\mathbb R^n$ 时，一阶体积因子是
> $$
> \sqrt{\det\!\bigl(J_\phi(u)^T J_\phi(u)\bigr)}.
> $$
> $J_\phi(u)^T J_\phi(u)$ 是列向量的 Gram 矩阵：它记录这些切向量两两之间的内积，因此其 determinant 的平方根正是它们张成的 $k$ 维平行多面体体积。
>
> <!-- bilingual-en:start -->
> Let $\phi:U\subseteq\mathbb R^k\to\mathbb R^n$ be differentiable, with $1\le k<n$, and suppose $J_\phi(u)$ has full column rank. Its first-order local $k$-volume scale factor is $\sqrt{\det(J_\phi(u)^T J_\phi(u))}$. The Gram matrix records all pairwise inner products of the tangent columns, so the square root of its determinant is the $k$-volume of their parallelepiped.
> <!-- bilingual-en:end -->

矩形 Jacobian 本身没有普通 determinant。真正需要测量的是它的 $k$ 个列向量在输出空间里张成了多大的 $k$ 维体积，而不是把这个像误当成一个 $n$ 维体积。若 $\sigma_1,\ldots,\sigma_k$ 是 $J_\phi(u)$ 的奇异值，那么
$$
\sqrt{\det(J_\phi^T J_\phi)}=\prod_{i=1}^k\sigma_i.
$$
这也说明：只要秩下降，至少一个奇异值为零，局部 $k$ 维体积因子就变成零。

<!-- bilingual-en:start -->
A rectangular Jacobian has no ordinary determinant. The relevant object is the $k$-dimensional volume spanned by its tangent columns inside the output space, not an $n$-dimensional volume. Equivalently, the factor is the product of the $k$ singular values. A rank drop therefore makes the local $k$-volume factor zero.
<!-- bilingual-en:end -->

## 一个二维曲面例子

令
$$
\phi(u,v)=(u,v,u+v).
$$
它的 Jacobian 是 $3\times2$ 矩阵
$$
J_\phi=
\begin{pmatrix}
1&0\\
0&1\\
1&1
\end{pmatrix},
\qquad
J_\phi^TJ_\phi=
\begin{pmatrix}
2&1\\
1&2
\end{pmatrix}.
$$
所以参数平面中的单位小面积被一阶放大为
$$
\sqrt{\det(J_\phi^TJ_\phi)}=\sqrt{3}
$$
倍。这里不能写 $\det J_\phi$，因为 $J_\phi$ 不是方阵。

<!-- bilingual-en:start -->
For the displayed surface parametrisation, the $3\times2$ Jacobian has Gram matrix $\begin{psmallmatrix}2&1\\1&2\end{psmallmatrix}$. The local area factor is therefore $\sqrt3$; writing $\det J_\phi$ would be meaningless because the Jacobian is not square.
<!-- bilingual-en:end -->

## 与方阵 Jacobian 的关系

若把同一个 Gram 公式用于 $k=n$ 的方阵情形，则
$$
\sqrt{\det(J^TJ)}=|\det J|.
$$
因此 Gram 公式会还原 [[Jacobian行列式|方阵 Jacobian]] 的无符号体积因子，但它不会保留取向的正负号。满列秩是把 $\phi$ 当作正则参数化并使用通常曲线、曲面面积公式的条件；秩下降时，上式仍给出零，却不能把该点继续当作正则参数点。

<!-- bilingual-en:start -->
In the square case $k=n$, the same Gram formula reduces to $|\det J|$, recovering the unsigned square-Jacobian volume factor but not its orientation sign. Full column rank is the regularity condition used in the usual curve and surface area formulas; at a rank-deficient point the expression is zero, but the point is no longer a regular parametrisation point.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对曲线 $\gamma(t)=(t,t^2)$，为什么局部弧长因子是 $\sqrt{1+4t^2}$？
>
> **答案：** $J_\gamma=(1,2t)^T$，所以 $J_\gamma^TJ_\gamma=1+4t^2$；取平方根便得到速度，也就是一维体积因子。

## 来源与核验

- [MIT 18.101, Integration on Manifolds](https://math.mit.edu/classes/18.101/fa07/pub/int-density.pdf)：核对参数化子流形的体积密度 $\sqrt{\det((D\phi)^TD\phi)}$。
- [Stanford Math 52H, Volume and Gram Matrix](https://math.stanford.edu/~eliash/Public/52h-2013/52h-2013/52htext.pdf)：核对 Gram 矩阵 $C^TC$ 与其 determinant 给出的平方体积。
- [[奇异值与Gram谱]]：交叉核对 Gram 矩阵特征值与奇异值平方之间的关系。

<!-- bilingual-en:start -->
- MIT 18.101 was checked for the volume density $\sqrt{\det((D\phi)^TD\phi)}$ of a parametrised submanifold.
- Stanford Math 52H was checked for the relation between a Gram matrix $C^TC$ and squared volume.
- [[奇异值与Gram谱|The Gram-spectrum identity]] was cross-checked for the relation between the eigenvalues of a Gram matrix and squared singular values.
<!-- bilingual-en:end -->

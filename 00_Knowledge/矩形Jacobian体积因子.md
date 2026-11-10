---
aliases:
  - Rectangular Jacobian volume factor
  - Gram determinant volume element
student_os: knowledge-atom
atom_id: LA-DET-020
atom_set: determinants
atom_type: theorem
status: source-checked
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

<!-- bilingual-en:start -->
# Rectangular Jacobian volume factors
<!-- bilingual-en:end -->

*矩形 Jacobian 体积因子*

<!-- bilingual-en:start -->
A parametrization can use two numbers to locate a point in three-dimensional space. Its derivative then turns a tiny parameter square into a tangent parallelogram. The **rectangular Jacobian volume factor** measures that parallelogram’s area, or the corresponding higher-dimensional volume. The dimension being measured is the number of parameter directions, not automatically the dimension of the surrounding space.
<!-- bilingual-en:end -->

参数化可以用两个数定位三维空间中的一个点。它的导数会把很小的参数正方形变成切向平行四边形。**矩形 Jacobian 体积因子**测量这个平行四边形的面积，或相应的高维体积。测量的维数来自参数方向的个数，并不自动等于周围空间的维数。

<!-- bilingual-en:start -->
## A surface with two parameter directions
<!-- bilingual-en:end -->

## 一个二维曲面例子

<!-- bilingual-en:start -->
Consider $\phi(u,v)=(u,v,u+v)$. Changing only $u$ moves in direction $(1,0,1)$; changing only $v$ moves in direction $(0,1,1)$. These are the two columns of the [[Jacobian矩阵|Jacobian matrix]]. Form their pairwise [[内积|Euclidean inner products]]: each squared length is $2$, and their mutual dot product is $1$. With $T$ denoting [[矩阵转置|transpose]],
<!-- bilingual-en:end -->

考虑 $\phi(u,v)=(u,v,u+v)$。只改变 $u$ 时，移动方向是 $(1,0,1)$；只改变 $v$ 时，方向是 $(0,1,1)$。它们正是 [[Jacobian矩阵]]的两列。计算两两的[[内积|欧氏内积]]：各自的长度平方为 $2$，相互点积为 $1$。上标 $T$ 表示[[矩阵转置|转置]]，于是

$$J=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\qquad
G=J^TJ=\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad
\sqrt{\det G}=\sqrt{4-1}=\sqrt3.$$

<!-- bilingual-en:start -->
The $2\times2$ matrix $G$ is the **Gram matrix**: it records lengths and angles through dot products. The unit parameter square has image area $\sqrt3$. Multiplying the two column lengths would give $2$, which is too large because the columns are not perpendicular. This map is linear and one-to-one, so the same factor gives the exact image area for every parameter rectangle. A nonlinear parametrization generally uses a factor that varies from point to point.
<!-- bilingual-en:end -->

$2\times2$ 矩阵 $G$ 称为 **Gram 矩阵**，它通过点积记录长度与夹角。单位参数正方形的像面积为 $\sqrt3$。若直接乘两列的长度，会得到偏大的 $2$，因为这两列并不垂直。这个映射线性且一一，所以相同因子能给出任意参数矩形的精确像面积；非线性参数化的因子通常随位置变化。

<!-- bilingual-en:start -->
## The formula and its rank condition
<!-- bilingual-en:end -->

## 公式与秩条件

<!-- bilingual-en:start -->
Let $1\le k\le m$ be integers, $U\subseteq\mathbb R^k$ open, and $\phi:U\to\mathbb R^m$ [[全微分|differentiable]] at $a\in U$. Use standard Euclidean coordinates in both spaces. The derivative matrix $J=D\phi(a)$ has $m$ rows and $k$ columns; it sends a small parameter displacement to its first-order output displacement. Define
<!-- bilingual-en:end -->

设整数 $1\le k\le m$，$U\subseteq\mathbb R^k$ 为开集，$\phi:U\to\mathbb R^m$ 在 $a\in U$ 处[[全微分|可微]]。两边都采用标准欧氏坐标。导数矩阵 $J=D\phi(a)$ 有 $m$ 行、$k$ 列，把小的参数位移映为输出的一阶位移。定义

$$\boxed{\mathcal V_k(J)=\sqrt{\det(J^TJ)}.}$$

<!-- bilingual-en:start -->
The square root is nonnegative. It is the $k$-dimensional volume of the parallelepiped whose edges are the $k$ columns of $J$. The [[矩阵秩|rank]] counts independent column directions. When $\operatorname{rank}J=k$, the columns are independent, $J^TJ$ is [[Gram矩阵正定判据|positive definite]], and $\mathcal V_k(J)>0$. Such a point is called a regular parameter point. When $\operatorname{rank}J<k$, the linear image lies in fewer than $k$ dimensions and its $k$-volume, and this factor, are zero.
<!-- bilingual-en:end -->

平方根取非负值。它等于以 $J$ 的 $k$ 列为边的平行多面体的 $k$ 维体积。[[矩阵秩|秩]]计算独立列方向的个数。若 $\operatorname{rank}J=k$，各列独立，$J^TJ$ [[Gram矩阵正定判据|正定]]，故 $\mathcal V_k(J)>0$；这样的点称为正则参数点。若 $\operatorname{rank}J<k$，线性像落在不足 $k$ 维的空间中，其 $k$ 维体积及该因子都为零。

<!-- bilingual-en:start -->
In particular, a two-parameter surface in $\mathbb R^3$ has an **area** factor, not an ambient three-dimensional volume factor. The square Gram matrix always has size $k\times k$, even though $J$ is rectangular. If $m<k$, its columns cannot be independent and $\det(J^TJ)=0$; this zero expression does not supply an $m$-dimensional change-of-variables formula.
<!-- bilingual-en:end -->

特别地，三维空间中的双参数曲面得到的是**面积**因子，不是周围三维空间的体积因子。即使 $J$ 是矩形，Gram 矩阵也总是 $k\times k$ 方阵。若 $m<k$，各列不可能独立，$\det(J^TJ)=0$；这个零表达式不能充当 $m$ 维换元公式。

<!-- bilingual-en:start -->
## Why the Gram determinant is a squared volume
<!-- bilingual-en:end -->

## 为什么 Gram 行列式是体积的平方

<!-- bilingual-en:start -->
First suppose $J$ has full column rank. Choose an [[标准正交组|orthonormal basis]] for its $k$-dimensional column space: $k$ mutually perpendicular unit vectors spanning that space. Put them in the columns of an $m\times k$ matrix $Q$, so $Q^TQ=I_k$, where $I_k$ is the [[单位矩阵|identity matrix]]. Write each column of $J$ in this basis. The resulting coordinate matrix is a square $k\times k$ matrix $B$, and $J=QB$.
<!-- bilingual-en:end -->

先设 $J$ 满列秩。在它的 $k$ 维列空间中选一组[[标准正交组|标准正交基]]：即张成该空间的 $k$ 个相互垂直的单位向量。把它们排成 $m\times k$ 矩阵 $Q$ 的列，则 $Q^TQ=I_k$，其中 $I_k$ 为[[单位矩阵]]。把 $J$ 的每一列写成这组基下的坐标，得到 $k\times k$ 方阵 $B$，且 $J=QB$。

<!-- bilingual-en:start -->
The map $z\mapsto Qz$ preserves Euclidean lengths and angles, so it preserves $k$-volume. Therefore the volume spanned by $J$ is $|\det B|$, by the [[行列式体积与取向|square determinant-volume rule]]. Meanwhile, [[行列式乘法性|multiplicativity]] and [[行列式转置不变性|transpose invariance]] give
<!-- bilingual-en:end -->

映射 $z\mapsto Qz$ 保持欧氏长度和夹角，因此保持 $k$ 维体积。由[[行列式体积与取向|方阵行列式的体积解释]]，$J$ 各列围成的体积就是 $|\det B|$。另一方面，利用[[行列式乘法性]]及[[行列式转置不变性]]，有

$$J^TJ=B^TQ^TQB=B^TB,\qquad
\det(J^TJ)=\det(B^TB)=(\det B)^2.$$

<!-- bilingual-en:start -->
Taking the nonnegative square root proves the formula. If the columns are dependent, a nonzero coefficient vector $z$ satisfies $Jz=0$, hence $(J^TJ)z=0$: the Gram determinant is zero, matching the collapsed $k$-volume. Equivalently, the [[奇异值与Gram谱|Gram eigenvalues]] are the squares of the $k$ [[奇异值|singular values]] $\sigma_1,\ldots,\sigma_k$ of $J$ (including zeros), so the same factor is $\prod_{j=1}^k\sigma_j$. These are the stretches in mutually perpendicular principal directions, not generally the lengths of the original columns.
<!-- bilingual-en:end -->

取非负平方根即得公式。若各列相关，存在非零系数向量 $z$ 使 $Jz=0$，于是 $(J^TJ)z=0$：Gram 行列式为零，与坍缩的 $k$ 维体积相符。等价地，[[奇异值与Gram谱|Gram 特征值]]是 $J$ 的 $k$ 个[[奇异值]] $\sigma_1,\ldots,\sigma_k$（包括零值）的平方，所以该因子也等于 $\prod_{j=1}^k\sigma_j$。这些数是相互垂直的主方向上的伸缩量，通常不是原始各列的长度。

<!-- bilingual-en:start -->
## Square matrices and curves
<!-- bilingual-en:end -->

## 与方阵 Jacobian 的关系

<!-- bilingual-en:start -->
When $m=k$, the formula becomes the absolute [[Jacobian行列式|Jacobian determinant]]:
<!-- bilingual-en:end -->

当 $m=k$ 时，公式回到 [[Jacobian行列式]]的绝对值：

$$\mathcal V_k(J)=\sqrt{\det(J^TJ)}
=\sqrt{(\det J)^2}=|\det J|.$$

<!-- bilingual-en:start -->
This is an unsigned factor: for $J=\begin{pmatrix}2&1\\0&-3\end{pmatrix}$ it is $6$, although the signed determinant is $-6$. When $k=1$, $J$ is one column, so the Gram factor is its Euclidean length. For $\gamma(t)=(t,t^2)$,
<!-- bilingual-en:end -->

这是无符号因子：$J=\begin{pmatrix}2&1\\0&-3\end{pmatrix}$ 时它为 $6$，而有向行列式为 $-6$。当 $k=1$ 时，$J$ 只有一列，Gram 因子就是该列的欧氏长度。例如 $\gamma(t)=(t,t^2)$ 给出

$$D\gamma(t)=\begin{pmatrix}1\\2t\end{pmatrix},\qquad
\mathcal V_1(D\gamma(t))=\sqrt{1+4t^2}.$$

<!-- bilingual-en:start -->
This is the speed used in [[参数曲线弧长与重复描画|the arc-length integral]]. A zero factor indicates a degenerate derivative, not necessarily a singular image set: $\eta(t)=(t^3,0)$ has zero speed at $0$, but its image is a straight line.
<!-- bilingual-en:end -->

这正是[[参数曲线弧长与重复描画|弧长积分]]使用的速率。因子为零说明导数退化，不一定说明像点集本身奇异：$\eta(t)=(t^3,0)$ 在 $0$ 处速率为零，但它的像是一条直线。

> [!warning] A local factor does not count image points only once · 局部因子不保证像只覆盖一次
> <!-- bilingual-en:start -->
> For a sufficiently regular parametrization, integrating this factor accumulates parameterized volume. To identify it with the volume of the image counted once, check coverage as well. The unit circle $\gamma(t)=(\cos t,\sin t)$ on $[0,4\pi]$ has rank $1$ and factor $1$ everywhere, but its integral is $4\pi$, twice the image circumference $2\pi$. Full column rank is a local condition, not global one-to-one coverage. The standard surface formula on a one-to-one regular coordinate patch has that additional coverage condition.
> <!-- bilingual-en:end -->
>
> 对足够正则的参数化，积分这个因子累积的是按参数计算的体积。要把它认作像点集只计一次的体积，还需检查覆盖次数。单位圆 $\gamma(t)=(\cos t,\sin t)$ 在 $[0,4\pi]$ 上处处秩为 $1$、因子为 $1$，但积分为 $4\pi$，是像圆周长 $2\pi$ 的两倍。满列秩是局部条件，并非整体一一覆盖；标准曲面积分在一一的正则坐标片上使用时，另有这个覆盖条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Eliashberg, Stanford Math 52H, §§3.3–3.4, pp.31–34, proposition 3.3 and remark 3.4](https://math.stanford.edu/~eliash/Public/52h-2013/52h-2013/52htext.pdf#page=31): Euclidean parallelepiped volume, the Gram determinant and the rank boundary. The orthonormal-coordinate proof above supplies the squared determinant calculation explicitly.
- [MIT 18.101, *Integrating Densities on Manifolds*, pp.3–4, examples 1–3](https://math.mit.edu/classes/18.101/fa07/pub/int-density.pdf#page=3), and [Stanford Math 52H, example 10.6, pp.169–170](https://math.stanford.edu/~eliash/Public/52h-2013/52h-2013/52htext.pdf#page=169): the induced Euclidean volume factor for regular, one-to-one parametrizations, and the curve case.
- [Lebl, *Basic Analysis II*, §8.3, PDF pp.35 and 39–40](https://www.jirka.org/ra/realanal2.pdf#page=35): differentiability and the derivative matrix. The surface, curve, rank-zero and double-traversal examples are independently calculated.
<!-- bilingual-en:end -->

- Stanford Math52H §§3.3–3.4，第31–34页，命题3.3及注3.4：核对欧氏平行多面体体积、Gram 行列式及秩边界。本文用标准正交坐标完整写出行列式平方的计算。
- MIT18.101《Integrating Densities on Manifolds》第3–4页例1–3，以及Stanford例10.6、第169–170页：核对一一正则参数化的诱导欧氏体积因子与曲线情形。
- Lebl《Basic Analysis II》§8.3，PDF第35、39–40页：核对可微性与导数矩阵。曲面、曲线、零秩及重复走圆的例子均独立计算。

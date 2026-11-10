---
aliases:
- Gaussian density and singular support
- Multivariate normal density
- 多元正态密度
- 多元正态密度的正定边界
student_os: knowledge-atom
atom_id: STAT-MVN-003
atom_set: multivariate-normal
atom_type: theorem-boundary
status: source-checked
requires:
- '[[联合Gaussian]]'
- '[[正定二次型的椭球]]'
part_of:
- '[[多元正态分布.canvas]]'
related:
- '[[半正定主平方根]]'
- '[[Gaussian马氏平方律]]'
---

<!-- bilingual-en:start -->
# Gaussian densities and singular support
<!-- bilingual-en:end -->

*Gaussian 密度与奇异支撑*

<!-- bilingual-en:start -->
A [[联合Gaussian|multivariate Gaussian distribution]] can spread through a whole plane or space, or it can be confined to a line or another lower-dimensional set. These are both valid Gaussian models. The familiar density formula describes only the first situation. The [[协方差矩阵|covariance matrix]] tells us which situation we have.
<!-- bilingual-en:end -->

[[联合Gaussian|多元 Gaussian 分布]]可以在整个平面或空间里散布，也可以被限制在一条直线或其他低维集合上；两者都是合法的 Gaussian 模型。熟悉的密度公式只描述前一种情形，[[协方差矩阵]]能告诉我们属于哪一种。

> [!example] A Gaussian pair supported on a line · 落在直线上的 Gaussian 二元组
> <!-- bilingual-en:start -->
> Let $Z\sim N(0,1)$ be a [[正态分布|standard normal variable]], and set $X=(Z,2Z+1)^T$. Every realization satisfies $x_2=2x_1+1$. Both coordinates vary, but the pair has only one source of variation. Its mean and covariance are
>
> $$\mu=\begin{bmatrix}0\\1\end{bmatrix},\qquad
> \Sigma=\begin{bmatrix}1&2\\2&4\end{bmatrix},\qquad \det\Sigma=0.$$
>
> The distribution assigns probability one to a line of zero area. If it had an ordinary two-dimensional [[概率密度函数|density]], integrating that density over this zero-area line would give zero, a contradiction. Having normal marginal densities does not produce a joint density on the plane.
> <!-- bilingual-en:end -->
>
> 令 $Z\sim N(0,1)$ 为[[正态分布|标准正态变量]]，设 $X=(Z,2Z+1)^T$。每次实现都满足 $x_2=2x_1+1$；两个坐标都会变，但二元组只有一个波动来源。上式给出均值与协方差，且行列式为零。这个分布为一条面积为零的直线分配概率一；若存在通常的二维[[概率密度函数|密度]]，沿这个零面积集合积分只能得到零，产生矛盾。各边际有正态密度，并不保证平面上存在联合密度。

<!-- bilingual-en:start -->
## When the usual formula applies
<!-- bilingual-en:end -->

*通常的密度公式何时适用*

<!-- bilingual-en:start -->
Let $X\sim N_p(\mu,\Sigma)$ with $p\ge1$. A symmetric covariance is [[正定矩阵|positive definite]], written $\Sigma\succ0$, if $a^T\Sigma a>0$ for every nonzero real vector $a$. By the [[线性投影方差|projected-variance identity]], this says every nontrivial direction has positive variance. In that case the Gaussian has the following density with respect to ordinary $p$-dimensional volume (Lebesgue measure):
<!-- bilingual-en:end -->

设 $X\sim N_p(\mu,\Sigma)$，其中 $p\ge1$。对称协方差矩阵[[正定矩阵|正定]]，记作 $\Sigma\succ0$，是指每个非零实向量 $a$ 都使 $a^T\Sigma a>0$。由[[线性投影方差|投影方差恒等式]]，这表示每个非平凡方向都有正方差。此时 Gaussian 分布相对于通常的 $p$ 维体积（Lebesgue 测度）具有密度

$$
\boxed{f_X(x)=\frac{1}{(2\pi)^{p/2}(\det\Sigma)^{1/2}}
\exp\!\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right],
\qquad x\in\mathbb R^p.}
$$

<!-- bilingual-en:start -->
Here $x$ is a possible numerical vector, $x-\mu$ is its displacement from the mean, $\Sigma^{-1}$ is the [[逆矩阵|inverse matrix]], and $\det\Sigma$ is the [[行列式|determinant]]. Positive definiteness makes the inverse available and the determinant strictly positive. Some sources write $|\Sigma|$ for this determinant. The exponential is largest at $x=\mu$ and decreases with the covariance-adjusted squared displacement, not necessarily with ordinary Euclidean distance.
<!-- bilingual-en:end -->

这里 $x$ 是可能的数值向量，$x-\mu$ 是相对均值的偏移，$\Sigma^{-1}$ 是[[逆矩阵]]，$\det\Sigma$ 是[[行列式]]。正定性保证逆存在且行列式严格为正；一些来源将这个行列式写成 $|\Sigma|$。指数在 $x=\mu$ 处最大，并随协方差调整后的偏移平方增大而下降，不一定按普通欧氏距离下降。

<!-- bilingual-en:start -->
One derivation starts with $p$ independent standard normals $Z$ and writes $X=\mu+BZ$, where $BB^T=\Sigma$ and $B$ is square and invertible. The input needed to produce $x$ is $z=B^{-1}(x-\mu)$. A [[多元换元公式|multivariate change of variables]] divides the standard density by $|\det B|$. The identities $|\det B|=\sqrt{\det\Sigma}$ and $\|B^{-1}(x-\mu)\|_2^2=(x-\mu)^T\Sigma^{-1}(x-\mu)$ give the displayed formula.
<!-- bilingual-en:end -->

一种推导从 $p$ 个独立标准正态组成的 $Z$ 出发，写成 $X=\mu+BZ$，其中 $BB^T=\Sigma$，且 $B$ 是可逆方阵。为了产生 $x$，输入必须为 $z=B^{-1}(x-\mu)$。[[多元换元公式|多元密度换元]]把标准密度除以 $|\det B|$，再使用 $|\det B|=\sqrt{\det\Sigma}$ 与 $\|B^{-1}(x-\mu)\|_2^2=(x-\mu)^T\Sigma^{-1}(x-\mu)$，就得到公式。

<!-- bilingual-en:start -->
## Reading the ellipsoids
<!-- bilingual-en:end -->

*怎样读等密度椭球*

<!-- bilingual-en:start -->
Write the [[实对称矩阵谱定理|spectral decomposition]] as $\Sigma=Q\Lambda Q^T$. The columns $q_i$ of $Q$ are orthonormal eigenvectors; $\Lambda$ has positive eigenvalues $\lambda_i$ on its diagonal. Each displacement can be measured along those directions, giving
<!-- bilingual-en:end -->

把[[实对称矩阵谱定理|谱分解]]写成 $\Sigma=Q\Lambda Q^T$，其中 $Q$ 的列 $q_i$ 是标准正交特征向量，$\Lambda$ 的对角元素为正特征值 $\lambda_i$。沿这些方向测量偏移，得到

$$
(x-\mu)^T\Sigma^{-1}(x-\mu)
=\sum_{i=1}^p\frac{[q_i^T(x-\mu)]^2}{\lambda_i}.
$$

<!-- bilingual-en:start -->
Fixing this sum at $c>0$ gives an [[正定二次型的椭球|ellipsoid]] with semiaxis length $\sqrt{c\lambda_i}$ along $q_i$. A larger variance permits a longer displacement for the same density height. For $\mu=0$ and $\Sigma=\operatorname{diag}(4,1)$ in two dimensions, the density is $\frac1{4\pi}e^{-(x_1^2/4+x_2^2)/2}$. At $c=1$ the ellipse has semiaxes $2$ and $1$. At $c=0$ the level set is only the center; a negative $c$ is impossible.
<!-- bilingual-en:end -->

把这个和固定为 $c>0$，得到沿 $q_i$ 方向半轴长为 $\sqrt{c\lambda_i}$ 的[[正定二次型的椭球|椭球]]。方差更大的方向，在相同密度高度下允许更长的偏移。二维例子 $\mu=0$、$\Sigma=\operatorname{diag}(4,1)$ 的密度为 $\frac1{4\pi}e^{-(x_1^2/4+x_2^2)/2}$；$c=1$ 时椭圆半轴为 $2$、$1$，$c=0$ 时等值集只有中心，$c<0$ 则不可能。

<!-- bilingual-en:start -->
Off-diagonal covariances can rotate these principal directions. Individual variances alone do not determine the orientation. An equal-density surface is also not automatically a region of a specified probability: obtaining a coverage probability requires an additional calculation, treated in [[Gaussian马氏平方律|the Gaussian Mahalanobis-square law]].
<!-- bilingual-en:end -->

非对角协方差可能旋转这些主轴方向；仅看各自方差，不能确定朝向。等密度面也不会自动给出指定概率的区域：求覆盖概率还需要额外计算，见[[Gaussian马氏平方律]]。

<!-- bilingual-en:start -->
## Singular covariance changes the dimension of the support
<!-- bilingual-en:end -->

*奇异协方差改变支撑的维数*

<!-- bilingual-en:start -->
A Gaussian covariance may be merely [[半正定矩阵|positive semidefinite]]. If its [[矩阵秩|rank]] is $r<p$, exactly $r$ eigenvalues are positive; label those $\lambda_1,\ldots,\lambda_r$. Its square-root construction can be written as
<!-- bilingual-en:end -->

Gaussian 协方差可以只满足[[半正定矩阵|半正定]]。若其[[矩阵秩|秩]]为 $r<p$，就恰有 $r$ 个正特征值，将它们记为 $\lambda_1,\ldots,\lambda_r$；平方根构造可以写成

$$
X\overset{d}=\mu+\sum_{i=1}^{r}\sqrt{\lambda_i}\,Z_iq_i,
\qquad Z_1,\ldots,Z_r\text{ independent }N(0,1).
$$

<!-- bilingual-en:start -->
Only these $r$ directions vary. All probability lies in $\mu+\operatorname{range}(\Sigma)$, the [[列空间|column space]] shifted by the mean. This affine subspace is the **support**: every neighborhood of each of its points has positive probability, while points off it have a neighborhood of probability zero. When $r=0$, the sum is empty and $X=\mu$ almost surely. For $r<p$ the support has zero $p$-dimensional volume, so there is no density with respect to that volume. This proves the other half of the criterion: a Gaussian has a full-dimensional density **if and only if** $\Sigma\succ0$.
<!-- bilingual-en:end -->

只有这 $r$ 个方向会变化，全部概率都落在 $\mu+\operatorname{range}(\Sigma)$，即由均值平移后的[[列空间]]上。这个仿射子空间就是**支撑**：其中每个点的每个邻域都有正概率，而支撑外的点存在概率为零的邻域。$r=0$ 时求和为空，$X=\mu$ 几乎必然。$r<p$ 时支撑的 $p$ 维体积为零，因此不存在相对于该体积的密度。这也证明了判据的另一方向：Gaussian 具有全维密度，**当且仅当** $\Sigma\succ0$。

> [!warning] The reference volume must match the support · 密度所用的体积必须匹配支撑
> <!-- bilingual-en:start -->
> For $0<r<p$, the active orthonormal coordinates $q_i^T(X-\mu)$ do have an $r$-dimensional Gaussian density, with variances $\lambda_i$. This is a density along the support, not a $p$-dimensional density. Replacing the missing inverse by a pseudoinverse and the zero determinant by the product of positive eigenvalues does not by itself turn that expression into an ordinary density on $\mathbb R^p$; the reference measure must be changed and stated. In the line example, the nonzero eigenvalue is $5$ and the unit direction is $(1,2)^T/\sqrt5$, so the coordinate along the line has variance $5$.
> <!-- bilingual-en:end -->
>
> 当 $0<r<p$ 时，有效标准正交坐标 $q_i^T(X-\mu)$ 确实具有 $r$ 维 Gaussian 密度，方差分别为 $\lambda_i$；这是沿支撑的密度，不是 $p$ 维密度。仅仅把不存在的逆换成伪逆，把零行列式换成正特征值之积，并不能自动得到 $\mathbb R^p$ 上的普通密度，必须明确更换参考测度。开头直线例子的非零特征值为 $5$，单位方向为 $(1,2)^T/\sqrt5$，所以沿直线的坐标方差为 $5$。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J, Lecture 14, Definition 2 and the degenerate example, pp. 3–4; Theorem 1(d), pp. 5–6](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf#page=3): the positive-definite density, change-of-variables derivation and failure of a full-dimensional density under degeneracy.
- [MIT 18.443, Lecture 4, PDF pp. 2–3 / printed pp. 24–25](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf#page=2): spectral generation, positive-eigenvalue coordinates and density on a lower-dimensional support.
- [MIT 6.450, Chapter 7, §§7.3.2–7.3.5, PDF pp. 9–13 / printed pp. 207–211](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/49163236e20779bae41639ff9dec1ac4_book_7.pdf#page=9): nonsingular-density conditions, singular linear constraints and principal-axis contours. The note's contour parameter is adjusted to the quadratic form equalling $c$ here.
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.2. 多元正态密度及等密度曲线|Multivariate Statistics §1.2]]: course density and ellipsoid notation. The line and diagonal-covariance examples are computed explicitly above.
<!-- bilingual-en:end -->

- MIT 6.436J 第 14 讲定义 2、退化例子与定理 1(d)：核对正定密度、换元推导及退化时不存在全维密度。
- MIT 18.443 第 4 讲 PDF 第 2–3 页：核对谱生成、正特征值坐标与低维支撑上的密度。
- MIT 6.450 第 7 章第 7.3.2–7.3.5 节 PDF 第 9–13 页：核对非奇异密度条件、奇异线性约束与主轴等密度面；正文将等值参数统一为二次型等于 $c$。
- 多元统计课程第 1.2 节保留本课密度与椭球符号；直线和对角协方差例子在正文直接计算。

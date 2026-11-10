---
aliases:
  - "Negative-definite Hessian test"
  - "Hessian local maximum test"
student_os: knowledge-atom
atom_id: CALC-MV-010
atom_set: multivariable-differentiation
atom_type: sufficient-condition
status: source-checked
requires:
  - "[[多元Taylor近似]]"
related:
  - "[[Hessian 局部极小判据]]"
  - "[[凸优化全局性]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
  - "[[多元优化.canvas|多元优化]]"
---

<!-- bilingual-en:start -->
# Hessian test for a strict local maximum
<!-- bilingual-en:end -->

*Hessian 局部极大判据*

<!-- bilingual-en:start -->
For $f(x,y)=-x^2-2y^2$, every nonzero move away from $(0,0)$ lowers the output. A negative-definite Hessian describes this downward quadratic behavior in every combined direction. At a stationary point of a sufficiently smooth real-valued function, that quadratic decrease outweighs the smaller Taylor error and produces a strict local maximum.
<!-- bilingual-en:end -->

对 $f(x,y)=-x^2-2y^2$，从 $(0,0)$ 作任何非零位移，输出都会降低。负定 Hessian 描述了所有坐标共同变化时，每个方向上的这种二次下降。在足够光滑的实值函数的驻点处，二次下降压过更小的 Taylor 误差，从而给出严格局部极大。

<!-- bilingual-en:start -->
## State the point and neighborhood first
<!-- bilingual-en:end -->

*先明确内点、驻点与光滑邻域*

<!-- bilingual-en:start -->
Let $f:U\to\mathbb R$, where $U\subseteq\mathbb R^n$ is [[开集|open]], $n\ge1$ is a finite integer, and $a\in U$. Assume $f$ is **$C^2$ on an open neighborhood of $a$**: all partial derivatives through order two exist there and are continuous. The [[全微分#实值函数、梯度与全微分|gradient]] $\nabla f(a)$ is the column of first partial derivatives; $a$ is [[驻点|stationary]] when this column is zero. The [[Hessian矩阵|Hessian]] $H_f(a)$ is the matrix of second partial derivatives, symmetric by the [[Hessian对称条件|$C^2$ mixed-partial theorem]].

A real symmetric matrix $H$ is [[负定矩阵|negative definite]] when $h^THh<0$ for every real nonzero column $h$. Here $T$ means [[矩阵转置|transpose]], and $h$ is an input displacement. This tests all quadratic directions, including mixed changes of several coordinates.
<!-- bilingual-en:end -->

设 $f:U\to\mathbb R$，其中 $U\subseteq\mathbb R^n$ 是[[开集]]，$n\ge1$ 为有限整数，$a\in U$。假设 $f$ **在 $a$ 的某个开邻域上为 $C^2$**，即直到二阶的全部偏导在其中存在且连续。[[全微分#实值函数、梯度与全微分|梯度]] $\nabla f(a)$ 是一阶偏导组成的列向量；这一列为零时，$a$ 称为[[驻点]]。[[Hessian矩阵]] $H_f(a)$ 收集二阶偏导，由 [[Hessian对称条件|$C^2$ 混合偏导定理]]知其对称。

实对称矩阵 $H$ 为[[负定矩阵|负定]]，指每个非零实列向量 $h$ 都满足 $h^THh<0$。这里 $T$ 表示[[矩阵转置|转置]]，$h$ 是输入位移；这个条件检查所有二次方向，包括多个坐标同时变化的混合方向。

> [!important] A sufficient local test · 局部充分判据
> <!-- bilingual-en:start -->
> If $\nabla f(a)=0$ and $H_f(a)$ is negative definite under these assumptions, then $a$ is a **strict local maximum point**: one radius $\delta>0$ makes
> <!-- bilingual-en:end -->
>
> 在上述假设下，若 $\nabla f(a)=0$ 且 $H_f(a)$ 负定，则 $a$ 是**严格局部极大点**：存在同一个半径 $\delta>0$，使
>
> $$0<\|h\|_2<\delta\quad\Longrightarrow\quad a+h\in U,\qquad f(a+h)<f(a).$$
>
> <!-- bilingual-en:start -->
> The [[欧氏范数|Euclidean length]] is $\|h\|_2=(\sum_{i=1}^n h_i^2)^{1/2}$. Strictness excludes equal outputs at other points in this ball; it does not compare the entire domain. This is the distinction between [[局部极值与绝对极值|local and absolute extrema]].
> <!-- bilingual-en:end -->
>
> [[欧氏范数|欧氏长度]]为 $\|h\|_2=(\sum_{i=1}^n h_i^2)^{1/2}$。“严格”排除球内其他点与它取得相同值，却不比较整个定义域；这就是[[局部极值与绝对极值]]的区别。

<!-- bilingual-en:start -->
At an interior differentiable local extremum, each coordinate-line restriction has derivative zero, so $\nabla f(a)=0$ is [[无约束一阶条件|necessary]]. Curvature does not replace it: $f(x)=x-x^2$ has $f''(0)=-2$, yet $f(h)>f(0)$ for $0<h<1$. Conversely, zero gradient alone leaves the classification open. The open-domain condition also matters for this first-order argument; an endpoint or a constrained feasible set needs its own analysis.
<!-- bilingual-en:end -->

内部可微局部极值沿每条坐标直线的导数都为零，所以 $\nabla f(a)=0$ 是[[无约束一阶条件|必要条件]]。曲率不能代替它：$f(x)=x-x^2$ 虽有 $f''(0)=-2$，却在 $0<h<1$ 时满足 $f(h)>f(0)$。反过来，零梯度本身仍不能完成分类。这个一阶论证也依赖开放域的内点条件；端点或约束可行域需要另行分析。

<!-- bilingual-en:start -->
## Why the criterion works
<!-- bilingual-en:end -->

## 判据为何成立

<!-- bilingual-en:start -->
Put $H=H_f(a)$. Negative definiteness says $-H$ is [[正定矩阵|positive definite]]. On the unit sphere $\{u:\|u\|_2=1\}$, the continuous function $u\mapsto-u^THu$ attains a positive minimum $m>0$. The sphere is nonempty and [[Heine–Borel 定理的边界|compact in finite-dimensional Euclidean space]], so the [[极值定理|extreme value theorem]] gives an attained minimum, and positivity at its nonzero minimizing vector makes that minimum strictly positive. Rescaling any nonzero $h$ to a unit vector gives a bound with the **same** $m$ for every direction:
<!-- bilingual-en:end -->

记 $H=H_f(a)$。负定等价于 $-H$ [[正定矩阵|正定]]。在单位球面 $\{u:\|u\|_2=1\}$ 上，连续函数 $u\mapsto-u^THu$ 取得正的最小值 $m>0$：球面非空，由[[Heine–Borel 定理的边界|有限维欧氏空间的紧致性判据]]知其紧致，所以[[极值定理]]保证最小值真正取得；取得它的单位向量非零，故最小值严格为正。将任意非零 $h$ 缩放为单位向量，得到**同一个** $m$ 控制所有方向：

$$h^THh\le-m\|h\|_2^2.$$

<!-- bilingual-en:start -->
At the stationary point, [[多元Taylor近似|second-order Taylor approximation]] gives
<!-- bilingual-en:end -->

在驻点处，[[多元Taylor近似|二阶 Taylor 近似]]给出

$$f(a+h)-f(a)=\frac12h^THh+R(h),\qquad
\frac{R(h)}{\|h\|_2^2}\longrightarrow0\quad(h\to0,\ h\ne0).$$

<!-- bilingual-en:start -->
This is a limit over all displacement vectors. Therefore one sufficiently small radius inside the $C^2$ neighborhood makes $|R(h)|\le(m/4)\|h\|_2^2$ for every nonzero $h$ in that ball. Combining the bounds gives
<!-- bilingual-en:end -->

这是对所有位移向量的极限。因此，在 $C^2$ 邻域内存在同一个足够小的半径，使其中每个非零 $h$ 都满足 $|R(h)|\le(m/4)\|h\|_2^2$。合并两条估计：

$$f(a+h)-f(a)\le-\frac m2\|h\|_2^2+|R(h)|
\le-\frac m4\|h\|_2^2<0.$$

<!-- bilingual-en:start -->
Continuity of the Hessian supplies the all-direction error control. More explicitly, integration along $a+th$ gives
<!-- bilingual-en:end -->

Hessian 的连续性保证了针对所有方向的误差控制。具体地，沿 $a+th$ 积分有

$$R(h)=\int_0^1(1-t)h^T[H_f(a+th)-H_f(a)]h\,dt.$$

<!-- bilingual-en:start -->
If every entry difference inside a small ball is at most $\eta$ in absolute value, then expanding the quadratic form and using [[Cauchy–Schwarz不等式|Cauchy–Schwarz]] gives $|R(h)|\le(n\eta/2)\|h\|_2^2$. Continuity of the finitely many entries supplies one such ball for any $\eta>0$; choosing $\eta=m/(2n)$ proves the required estimate. The full entrywise estimate is also written out in the [[Hessian 局部极小判据|local-minimum criterion]]. Thus the proof controls one neighborhood, rather than selecting a different neighborhood for each line.
<!-- bilingual-en:end -->

若小球内每个矩阵元素之差的绝对值都不超过 $\eta$，展开二次型并使用 [[Cauchy–Schwarz不等式]]，就有 $|R(h)|\le(n\eta/2)\|h\|_2^2$。有限多个元素的连续性，使任意 $\eta>0$ 都有对应的共同小球；取 $\eta=m/(2n)$ 即得所需估计。完整的逐元素不等式也在[[Hessian 局部极小判据]]中展开。证明控制的是一个邻域，而不是为每条直线另选一个邻域。

<!-- bilingual-en:start -->
## Examples and matrix checks
<!-- bilingual-en:end -->

## 例子与矩阵检查

<!-- bilingual-en:start -->
For the opening example $f(x,y)=-x^2-2y^2$, the gradient is $(-2x,-4y)^T$ and the origin is stationary. Its Hessian is the [[对角矩阵|diagonal matrix]] with diagonal entries $-2,-4$:
<!-- bilingual-en:end -->

开头的 $f(x,y)=-x^2-2y^2$ 有梯度 $(-2x,-4y)^T$，所以原点是驻点。Hessian 是对角元为 $-2,-4$ 的[[对角矩阵]]：

$$H=\begin{pmatrix}-2&0\\0&-4\end{pmatrix},\qquad
h^THh=-2h_1^2-4h_2^2\le-2\|h\|_2^2.$$

<!-- bilingual-en:start -->
We may take $m=2$, and the remainder is zero. The criterion proves a strict local maximum. Directly, $-x^2-2y^2<0=f(0,0)$ for every other point in $\mathbb R^2$, so this particular maximum is also uniquely global.

Do not replace negative definiteness by negative diagonal entries. For
<!-- bilingual-en:end -->

可取 $m=2$，而且余项为零，所以判据给出严格局部极大。直接比较还可看到，$\mathbb R^2$ 中每个其他点都满足 $-x^2-2y^2<0=f(0,0)$，因此这个具体例子的最大点也在全局唯一。

不能把负定换成“对角元都负”。例如

$$A=\begin{pmatrix}-1&2\\2&-1\end{pmatrix},\qquad
v=\begin{pmatrix}1\\1\end{pmatrix},\quad
w=\begin{pmatrix}1\\-1\end{pmatrix}.$$

<!-- bilingual-en:start -->
Both diagonal entries are negative, but $v^TAv=2>0$ and $w^TAw=-6<0$. Indeed $Av=v$ and $Aw=-3w$, so its [[特征对|eigenvalues]] are $1$ and $-3$. It is indefinite, not negative definite. The quadratic function $q(z)=\tfrac12z^TAz$, for $z\in\mathbb R^2$, has Hessian $A$ and a stationary origin, but increases along $v$ and decreases along $w$.
<!-- bilingual-en:end -->

两个对角元都为负，却有 $v^TAv=2>0$、$w^TAw=-6<0$。实际上 $Av=v$、$Aw=-3w$，所以两个[[特征对|特征值]]是 $1$ 与 $-3$。矩阵不定，并非负定。对 $z\in\mathbb R^2$，二次函数 $q(z)=\tfrac12z^TAz$ 的 Hessian 就是 $A$，原点也是驻点，但沿 $v$ 上升、沿 $w$ 下降。

<!-- bilingual-en:start -->
## Local does not mean global
<!-- bilingual-en:end -->

## 局部不等于全局

<!-- bilingual-en:start -->
On $\mathbb R$, let $g(x)=-x^2+x^4$. Then $g'(0)=0$ and $g''(0)=-2$, so zero is a strict local maximum. In fact $g(x)=x^2(x^2-1)<0$ for $0<|x|<1$. Yet $g(2)=12>g(0)$ and $g(x)\to+\infty$ as $|x|\to\infty$. The pointwise Hessian test cannot rule out larger distant values. [[凸优化全局性|Concavity on a convex domain]] is a separate sufficient route from a local maximum to a global one; it is not the only possible route, since direct comparison may also prove globality.
<!-- bilingual-en:end -->

在 $\mathbb R$ 上令 $g(x)=-x^2+x^4$。有 $g'(0)=0$、$g''(0)=-2$，所以零点严格局部极大；事实上 $0<|x|<1$ 时，$g(x)=x^2(x^2-1)<0$。但 $g(2)=12>g(0)$，且 $|x|\to\infty$ 时 $g(x)\to+\infty$。一点的 Hessian 检验排除不了远处更大的值。[[凸优化全局性|凸域上的凹性]]是把局部最大升级为全局最大的一条独立充分途径，但不是唯一途径，直接比较也可能证明全局性。

<!-- bilingual-en:start -->
Nor does a strict local maximum guarantee a unique global maximizer. The function $-(x^2-1)^2$ has two stationary global maximum points, $1$ and $-1$, each with second derivative $-8$ and hence each a strict local maximum. Negative definiteness is also not necessary for strict local maximality: $-x^4$ has a strict maximum at zero with Hessian zero. If the Hessian is singular and semidefinite, consult [[半定Hessian无结论|the separate semidefinite boundary]]; failing this sufficient test is not a proof that no maximum exists.
<!-- bilingual-en:end -->

严格局部极大也不保证全局最大点唯一。函数 $-(x^2-1)^2$ 在 $1$ 与 $-1$ 有两个驻点，都是全局最大点，两处二阶导均为 $-8$，因此分别严格局部极大。负定也不是严格局部极大的必要条件：$-x^4$ 在零点严格极大，Hessian 却为零。若 Hessian 奇异且半定，应查看独立的[[半定Hessian无结论]]边界；充分判据不适用，并不证明不存在最大值。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[EC400 Slides Lecture 2.pdf#page=10|LSE EC400 SOFP Lecture 2, slides 10–14]]: the $C^2$ interior first-order condition, negative-definite local sufficient condition, and local/global boundary. The proof above explicitly controls the remainder in the lecture's Taylor argument.
- [MIT 18.S096, lecture 12, §12.1, PDF pp.1–2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=2) and [Lebl, Basic Analysis II, §8.6, PDF pp.60–63](https://www.jirka.org/ra/realanal2.pdf#page=60): the second-order approximation, continuous second partials and symmetry. The negative-definite proof is the positive-definite argument applied to $-f$.
- [MIT 18.101, Review of Topology, PDF pp.7–9, theorem 1.25](https://ocw.mit.edu/courses/18-101-analysis-ii-fall-2005/babd982be745679b6d691f78b1c18f53_lectures.pdf#page=7): compactness and attained extrema, used for the uniform positive minimum of $-u^THu$ on the nonempty unit sphere.
- [Boyd and Vandenberghe, Convex Optimization, §§3.1.3–3.1.4 and 4.2.2, PDF pp.83–85 and 152](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=83): convex-domain first/second-order conditions and the separate local-to-global argument. The displayed quadratic and quartic examples are verified directly above; [[多元Taylor近似|the multivariable Taylor card]] supplies the linked approximation statement.
<!-- bilingual-en:end -->

- [[EC400 Slides Lecture 2.pdf#page=10|LSE EC400 SOFP Lecture 2，第10–14页]]：$C^2$ 内点的一阶条件、负定 Hessian 的局部充分性及局部与全局边界；本文明确控制讲义 Taylor 论证中的余项。
- MIT 18.S096 讲义12 §12.1，PDF第1–2页，以及 Lebl《Basic Analysis II》§8.6，PDF第60–63页：二阶近似、连续二阶偏导与对称性。负定情形是对 $-f$ 使用正定论证。
- MIT 18.101《Review of Topology》PDF第7–9页、定理1.25：紧致性与极值取得，用于证明非空单位球面上的 $-u^THu$ 有统一的正最小值。
- Boyd 与 Vandenberghe《Convex Optimization》§§3.1.3–3.1.4、4.2.2，PDF第83–85、152页：凸域上的一二阶条件与独立的局部到全局论证。本文二次、四次例子均直接计算验证；[[多元Taylor近似]]提供所用近似的精确前置。

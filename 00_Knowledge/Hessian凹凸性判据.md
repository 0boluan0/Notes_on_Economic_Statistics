---
student_os: knowledge-atom
atom_id: OPT-TOOLS-015
aliases:
- Hessian criterion for convexity and concavity
- 二阶凹凸性判据
status: source-checked
---

<!-- bilingual-en:start -->
# Hessian criterion for convexity and concavity
<!-- bilingual-en:end -->
*Hessian 凹凸性判据*

<!-- bilingual-en:start -->
A concave function stays at or above each chord joining two points of its graph. Checking every pair of inputs directly can be difficult. The **Hessian criterion** replaces those chord comparisons with a curvature test: for a sufficiently smooth function on an open convex domain, test the second derivative at every point in every direction.
<!-- bilingual-en:end -->

凹函数的图像不低于连接任意两点的弦。直接比较所有输入对可能很难。**Hessian 凹凸性判据**把这些弦比较转成曲率检验：在开凸域上，对足够光滑的函数，检查每一点、每个方向的二阶导数。

<!-- bilingual-en:start -->
Here a [[凸集|convex domain]] contains the whole segment between any two of its points. [[凹凸性|Concavity]] means that for inputs $x,y$ in that domain and a weight $0\le\theta\le1$,
<!-- bilingual-en:end -->

这里，[[凸集|凸域]]包含任意两点之间的整条线段。[[凹凸性|凹性]]指对定义域中的输入 $x,y$ 以及权重 $0\le\theta\le1$，有

$$f((1-\theta)x+\theta y)\ge(1-\theta)f(x)+\theta f(y).$$

<!-- bilingual-en:start -->
The right side averages the two endpoint outputs. Convexity reverses this inequality. Neither definition itself requires derivatives; smoothness is an assumption of the test below.
<!-- bilingual-en:end -->

右侧是两个端点输出的加权平均。凸性把这个不等号反向。定义本身都不要求导数；光滑性是下面这个判据的假设。

<!-- bilingual-en:start -->
## The conditions and the matrix sign
<!-- bilingual-en:end -->
*适用条件与矩阵符号*

<!-- bilingual-en:start -->
Let $U\subseteq\mathbb R^n$ be nonempty, [[开集|open]] and convex, where $n\ge1$ is a finite integer. Let $f:U\to\mathbb R$ be $C^2$: all partial derivatives through order two exist and are continuous on $U$. Its [[Hessian矩阵|Hessian matrix]] $H_f(x)$ collects the second partial derivatives and is symmetric under these assumptions. For a real column vector $v$, the scalar $v^TH_f(x)v$ is its directional quadratic value; $T$ means [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

设 $U\subseteq\mathbb R^n$ 非空、为[[开集]]且为凸集，其中 $n\ge1$ 是有限整数。令 $f:U\to\mathbb R$ 为 $C^2$，即直到二阶的全部偏导在 $U$ 上存在且连续。[[Hessian矩阵]] $H_f(x)$ 收集二阶偏导，在这些假设下对称。对实列向量 $v$，标量 $v^TH_f(x)v$ 是相应方向的二次型值；$T$ 表示[[矩阵转置|转置]]。

> [!important] Every point and every direction · 每一点与每个方向
> <!-- bilingual-en:start -->
> Under the stated assumptions,
> <!-- bilingual-en:end -->
>
> 在上述假设下，
>
> $$f\text{ is concave on }U
> \iff v^TH_f(x)v\le0\quad\text{for all }x\in U,\ v\in\mathbb R^n.$$
>
> <!-- bilingual-en:start -->
> The matrix condition is called [[半负定矩阵|negative semidefiniteness]], written $H_f(x)\preceq0$. Convexity is equivalent to [[半正定矩阵|positive semidefiniteness]], with $v^TH_f(x)v\ge0$ everywhere. The zero vector gives equality automatically; the test includes vectors of any length.
> <!-- bilingual-en:end -->
>
> 矩阵条件称为[[半负定矩阵|半负定]]，写作 $H_f(x)\preceq0$。凸性则等价于处处[[半正定矩阵|半正定]]，即所有这些二次型值都满足 $v^TH_f(x)v\ge0$。零向量自动给出等号；检验允许任意长度的向量。

若 $U$ 为开凸集、$f\in C^2(U)$，则 $f$ 凹当且仅当 $H_f(x)\preceq0$ 对每个 $x\in U$ 成立；凸对应处处半正定。矩阵的半定号要求所有方向的二次型符号，不是元素逐项符号。

<!-- bilingual-en:start -->
For example, take $f(x_1,x_2)=-(x_1-x_2)^2$ on $\mathbb R^2$. Direct differentiation gives
<!-- bilingual-en:end -->

例如，在 $\mathbb R^2$ 上取 $f(x_1,x_2)=-(x_1-x_2)^2$。直接求导得到

$$H_f=\begin{pmatrix}-2&2\\2&-2\end{pmatrix},\qquad
v^TH_fv=-2(v_1-v_2)^2\le0.$$

<!-- bilingual-en:start -->
This proves the sign for every vector, including simultaneous changes of both coordinates. The positive off-diagonal entries do not prevent concavity. Along $x_1=x_2$, the function is constant at zero, so concavity here is not strict.
<!-- bilingual-en:end -->

这对所有向量都证明了所需符号，包括两个坐标同时改变的方向。非对角元为正并不妨碍凹性。沿 $x_1=x_2$，函数恒为零，因此这里的凹性并不严格。

<!-- bilingual-en:start -->
## From directional curvature to every chord
<!-- bilingual-en:end -->
*从方向曲率得到每条弦的不等式*

<!-- bilingual-en:start -->
Assume the Hessian is negative semidefinite everywhere. Choose any $x,y\in U$ and set $d=y-x$. The segment stays in $U$ by convexity. Restrict the function to that segment using the scalar parameter $t$:
<!-- bilingual-en:end -->

假设 Hessian 处处半负定。任取 $x,y\in U$，记 $d=y-x$。凸性保证整条线段都在 $U$ 内。用标量参数 $t$ 把函数限制到这条线段：

$$g(t)=f(x+td),\qquad 0\le t\le1.$$

<!-- bilingual-en:start -->
The domain is open and $f$ is $C^2$, so $g$ is twice continuously differentiable around the segment. The [[Hessian方向二阶导数|second derivative along a line]] gives
<!-- bilingual-en:end -->

定义域开放且 $f$ 为 $C^2$，因此 $g$ 在线段参数区间周围二阶连续可微。[[Hessian方向二阶导数|沿直线的二阶导数]]给出

$$g''(t)=d^TH_f(x+td)d\le0.$$

<!-- bilingual-en:start -->
The [[拉格朗日中值定理|mean value theorem]] applied to $g'$ shows that $g'$ is nonincreasing: if $s<t$, then $g'(t)-g'(s)=g''(c)(t-s)\le0$ for some $c\in(s,t)$. For $0<\theta<1$, apply the same theorem to $g$ on $[0,\theta]$ and $[\theta,1]$. There are $a\in(0,\theta)$ and $b\in(\theta,1)$ such that
<!-- bilingual-en:end -->

对 $g'$ 使用[[拉格朗日中值定理]]，可知 $g'$ 不增：若 $s<t$，则存在 $c\in(s,t)$，使 $g'(t)-g'(s)=g''(c)(t-s)\le0$。对 $0<\theta<1$，再分别在 $[0,\theta]$ 和 $[\theta,1]$ 上对 $g$ 使用中值定理。存在 $a\in(0,\theta)$、$b\in(\theta,1)$，满足

$$\frac{g(\theta)-g(0)}{\theta}=g'(a)
\ge g'(b)=\frac{g(1)-g(\theta)}{1-\theta}.$$

<!-- bilingual-en:start -->
Multiplying by the positive number $\theta(1-\theta)$ and collecting the two copies of $g(\theta)$ yields $g(\theta)\ge(1-\theta)g(0)+\theta g(1)$. This is the desired chord inequality for $f$. At $\theta=0,1$, or when $x=y$, equality holds directly. Since the endpoints were arbitrary, the conclusion is concavity on all of $U$.
<!-- bilingual-en:end -->

乘以正数 $\theta(1-\theta)$，合并两项 $g(\theta)$，得到 $g(\theta)\ge(1-\theta)g(0)+\theta g(1)$，这正是 $f$ 所需的弦不等式。$\theta=0,1$ 或 $x=y$ 时直接取等。由于端点任意，结论覆盖整个 $U$。

<!-- bilingual-en:start -->
## Why concavity forces the Hessian sign
<!-- bilingual-en:end -->
*为什么凹性反过来限制 Hessian*

<!-- bilingual-en:start -->
Now assume $f$ is concave. Fix a point $a\in U$ and a nonzero vector $v$. Openness permits $a+sv$ and $a-sv$ for all sufficiently small real $s$. Their midpoint is $a$, so concavity gives
<!-- bilingual-en:end -->

反过来，假设 $f$ 凹。固定 $a\in U$ 和非零向量 $v$。开放性保证对充分小的实数 $s$，$a+sv$ 与 $a-sv$ 都在域内。它们的中点是 $a$，由凹性有

$$f(a+sv)+f(a-sv)-2f(a)\le0.$$

<!-- bilingual-en:start -->
Let $h(s)=f(a+sv)$. The [[二次近似|one-variable second-order expansion]] gives $h(\pm s)=h(0)\pm h'(0)s+\tfrac12h''(0)s^2+o(s^2)$, where $o(s^2)$ divided by $s^2$ tends to zero. Adding cancels the linear terms. For nonzero $s$, division by $s^2>0$ and passage to the limit therefore give
<!-- bilingual-en:end -->

令 $h(s)=f(a+sv)$。[[二次近似|一元二阶展开]]给出 $h(\pm s)=h(0)\pm h'(0)s+\tfrac12h''(0)s^2+o(s^2)$，其中 $o(s^2)$ 除以 $s^2$ 趋于零。相加后一次项抵消。对非零 $s$ 除以正数 $s^2$ 再取极限，得到

$$v^TH_f(a)v=h''(0)
=\lim_{s\to0}\frac{h(s)+h(-s)-2h(0)}{s^2}\le0.$$

<!-- bilingual-en:start -->
The point and nonzero direction were arbitrary, and the zero direction is automatic. This proves necessity. Applying both directions to $-f$, whose Hessian is $-H_f$, proves the convex version.
<!-- bilingual-en:end -->

点与非零方向都任意，而零方向自动成立，因此必要性得证。对 $-f$ 应用这两个方向，并使用它的 Hessian 为 $-H_f$，便得到凸性的版本。

沿任意线段令 $g(t)=f(x+t(y-x))$，由[[Hessian方向二阶导数]]把条件转为 $g''(t)\le0$，再用[[导数判凹凸]]；反向限制到每个点的局部直线即可。

<!-- bilingual-en:start -->
## What must actually be checked
<!-- bilingual-en:end -->
*实际检验不能省略什么*

<!-- bilingual-en:start -->
Negative coordinate second derivatives alone do not suffice. For $F(x,y)=-x^2-y^2+4xy$, both diagonal Hessian entries are $-2$, but the direction $v=(1,1)^T$ gives $v^TH_Fv=4>0$. Indeed $F(t,t)=2t^2$ bends upwards. A check at one point also does not cover the domain: for $q(x)=-x^2+x^4$, $q''(0)=-2$ but $q''(1)=10$.
<!-- bilingual-en:end -->

仅有坐标二阶导为负还不够。对 $F(x,y)=-x^2-y^2+4xy$，两个 Hessian 对角元都是 $-2$，但方向 $v=(1,1)^T$ 给出 $v^TH_Fv=4>0$；实际上 $F(t,t)=2t^2$ 向上弯曲。一点的检查也不能覆盖全域：对 $q(x)=-x^2+x^4$，有 $q''(0)=-2$，却有 $q''(1)=10$。

<!-- bilingual-en:start -->
The sign condition must hold at every point and in every direction. Restricting to line segments uses [[Hessian方向二阶导数|the directional Hessian identity]] and [[导数判凹凸|the one-dimensional criterion]] to prove it. One-point curvature cannot establish global concavity. Lower-dimensional domains and included boundary points require appropriate directional or extension arguments.
<!-- bilingual-en:end -->

一个点的 Hessian 定号不足以证明全域凹性。低维可行域应检验可行方向；含边界时需要额外的连续延拓论证。

<!-- bilingual-en:start -->
For example, $-x^2+y^2$ restricted to the line $y=0$ is concave along that line even though the ambient Hessian is indefinite. On an included boundary, interior curvature alone cannot control assigned boundary values: the function on $[0,1]$ with $f(0)=1$ and $f(x)=0$ for $x>0$ has zero second derivative inside, but fails the chord inequality at $0,1$ and their midpoint. A continuous extension from an open convex domain to its closure does preserve concavity, by approaching both endpoints from inside and taking limits in the chord inequality. The strict counterpart has a separate [[严格凹与负定Hessian|sufficient condition and converse boundary]].
<!-- bilingual-en:end -->

例如，$-x^2+y^2$ 限制在直线 $y=0$ 上后，沿这条直线是凹的，虽然环境空间中的 Hessian 不定。若把边界纳入定义域，仅有内部曲率控制不了边界赋值：在 $[0,1]$ 上令 $f(0)=1$、$x>0$ 时 $f(x)=0$，内部二阶导为零，却对端点 $0,1$ 及其中点违反弦不等式。若从开凸域连续延拓到闭包，则可以从内部逼近两个端点，在弦不等式中取极限，从而保留凹性。严格情形另见[[严格凹与负定Hessian|负定的充分条件及其逆命题边界]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- The source states the Hessian criterion; the text makes the open convex domain and C2 assumptions explicit and checks all directions through line restrictions.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
- [Boyd and Vandenberghe, Convex Optimization, §§3.1.1, 3.1.3–3.1.4, PDF pp.81–85](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=81): chord and line-restriction definitions, the open convex domain, and the second-order equivalence. The proof here expands both directions using the mean value theorem and the local second-order expansion. The polynomial and boundary examples are computed directly.
- [[EC400 Slides Lecture 1.pdf#page=49|LSE EC400 SOFP, PDF p.49, slide 47]] and [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#7.4 二阶判据：一点的曲率与处处的曲率|the permanent course exposition, §7.4]] locate the course statement and its directional interpretation.
<!-- bilingual-en:end -->

- [[EC400 Slides Lecture 1.pdf#page=49|EC400 SOFP 原材料，PDF p.49]]：核对处处半负定与凹性的等价关系；正文明确开凸域和二阶连续可微条件，并用直线限制核对全方向要求。
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#7.4 二阶判据：一点的曲率与处处的曲率|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。
- [Boyd 与 Vandenberghe《Convex Optimization》§§3.1.1、3.1.3–3.1.4，PDF81–85页](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=81)：弦与直线限制定义、开凸域条件及二阶等价判据。正文用中值定理与局部二阶展开补全两个方向的证明；多项式和边界例子均直接计算。

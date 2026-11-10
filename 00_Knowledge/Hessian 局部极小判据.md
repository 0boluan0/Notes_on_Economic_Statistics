---
aliases:
  - "Positive-definite Hessian test"
  - "Hessian local minimum test"
student_os: knowledge-atom
atom_id: LA-SPD-022
atom_set: symmetric-positive-definite
atom_type: sufficient-condition
status: source-checked
requires:
  - "[[正定矩阵]]"
  - "[[多元Taylor近似]]"
related:
  - "[[正定二次型的椭球]]"
  - "[[半定Hessian无结论]]"
  - "[[凸优化全局性]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[多元微分.canvas]]"
---

<!-- bilingual-en:start -->
# Hessian test for a strict local minimum
<!-- bilingual-en:end -->

*Hessian 局部极小判据*

<!-- bilingual-en:start -->
For $f(x,y)=x^2+2y^2$, moving away from $(0,0)$ in any direction increases the output. The surface is a bowl. The Hessian test recognizes the same local behavior in functions that are not exactly quadratic: at a stationary point, a positive-definite Hessian makes the quadratic increase dominate the smaller approximation error throughout one sufficiently small neighborhood.
<!-- bilingual-en:end -->

对 $f(x,y)=x^2+2y^2$，从 $(0,0)$ 朝任何方向离开，输出都会增加，曲面像一个碗。Hessian 判据能在并非恰好为二次函数的情形中识别同样的局部行为：在驻点处，正定 Hessian 使二次增长压过更小的近似误差，而且在同一个足够小的邻域内成立。

<!-- bilingual-en:start -->
## What the conditions mean
<!-- bilingual-en:end -->

*先区分位置、斜率与曲率*

<!-- bilingual-en:start -->
Let $f:U\to\mathbb R$ be real-valued, where $U\subseteq\mathbb R^n$ is [[开集|open]], $n\ge1$ is a finite integer, and $a\in U$. Openness means a whole small ball around $a$ is allowed. Assume $f$ is **$C^2$ near $a$**: all partial derivatives through order two exist and are continuous on some open neighborhood of $a$.

The [[全微分#实值函数、梯度与全微分|gradient]] $\nabla f(a)$ is the column of first partial derivatives; **[[驻点|stationary]]** means $\nabla f(a)=0$. The [[Hessian矩阵|Hessian]] $H_f(a)=\nabla^2f(a)$ collects second partial derivatives. It is symmetric under the stated [[Hessian对称条件|$C^2$ hypothesis]]. For a real column displacement $h$, the scalar $h^TH_f(a)h$ combines curvature in all coordinates; $T$ means [[矩阵转置|transpose]]. [[正定矩阵|Positive definiteness]] means this scalar is positive for every $h\ne0$, not that all matrix entries are positive.
<!-- bilingual-en:end -->

设 $f:U\to\mathbb R$ 为实值函数，$U\subseteq\mathbb R^n$ 是[[开集]]，$n\ge1$ 为有限整数，$a\in U$。开放性保证 $a$ 周围整个小球内的输入都合法。假设 $f$ **在 $a$ 附近为 $C^2$**：直到二阶的全部偏导都在 $a$ 的某个开邻域内存在且连续。

[[全微分#实值函数、梯度与全微分|梯度]] $\nabla f(a)$ 是一阶偏导组成的列向量；**[[驻点]]**指 $\nabla f(a)=0$。[[Hessian矩阵]] $H_f(a)=\nabla^2f(a)$ 收集二阶偏导，在所述 [[Hessian对称条件|$C^2$ 条件]] 下对称。对实列位移 $h$，标量 $h^TH_f(a)h$ 汇总所有坐标共同变化时的曲率，$T$ 表示[[矩阵转置|转置]]。[[正定矩阵|正定]]要求每个 $h\ne0$ 都使这个标量为正，并非要求矩阵每个元素为正。

> [!important] The local sufficient condition · 局部充分条件
> <!-- bilingual-en:start -->
> Under these assumptions, if $\nabla f(a)=0$ and $H_f(a)$ is positive definite, then $a$ is a **strict local minimum point**. There is one radius $\delta>0$ such that
> <!-- bilingual-en:end -->
>
> 在上述假设下，若 $\nabla f(a)=0$ 且 $H_f(a)$ 正定，那么 $a$ 是**严格局部极小点**。存在同一个半径 $\delta>0$，使
>
> $$0<\|h\|_2<\delta\quad\Longrightarrow\quad a+h\in U,\qquad f(a+h)>f(a).$$
>
> <!-- bilingual-en:start -->
> Here $\|h\|_2=(\sum_{i=1}^n h_i^2)^{1/2}$ is the [[欧氏范数|Euclidean length]]. “Strict” excludes ties with other inputs inside this ball; “local” restricts the comparison to the ball. See [[局部极值与绝对极值|local and absolute extrema]] for this distinction.
> <!-- bilingual-en:end -->
>
> 其中 $\|h\|_2=(\sum_{i=1}^n h_i^2)^{1/2}$ 是[[欧氏范数|欧氏长度]]。“严格”排除这个球内其他输入与它并列；“局部”则把比较限制在球内，见[[局部极值与绝对极值]]。

<!-- bilingual-en:start -->
Stationarity and positive curvature have different jobs. At a differentiable interior local extremum, restricting $f$ to each coordinate line gives a one-variable interior extremum, so every first partial is zero: this is the [[无约束一阶条件|first-order necessary condition]]. Zero gradient alone does not establish a minimum. Conversely, positive curvature without stationarity is insufficient: $f(x)=x+x^2$ has $f''(0)=2$, but $f(h)-f(0)=h(1+h)<0$ for $-1<h<0$. Boundary points require their own feasible-direction analysis; for example $f(x)=x$ on $[0,\infty)$ has a minimum at zero with right derivative $1$.
<!-- bilingual-en:end -->

驻点与正曲率分工不同。可微函数在内部局部极值处，沿每条坐标直线限制后都是一元内部极值，因此所有一阶偏导为零，这就是[[无约束一阶条件|一阶必要条件]]。零梯度本身并不证明极小。反过来，正曲率也不能代替驻点：$f(x)=x+x^2$ 满足 $f''(0)=2$，但对 $-1<h<0$，有 $f(h)-f(0)=h(1+h)<0$。边界点需要另按可行方向分析；例如 $[0,\infty)$ 上的 $f(x)=x$ 在零点最小，右导数却为 $1$。

<!-- bilingual-en:start -->
## Why one neighborhood works for every direction
<!-- bilingual-en:end -->

*统一的二次增长压过统一的小余项*

<!-- bilingual-en:start -->
Write $H=H_f(a)$. First put directions on the unit sphere $S=\{u\in\mathbb R^n:\|u\|_2=1\}$, so they all have the same length. This sphere is nonempty, closed and bounded in finite-dimensional Euclidean space, hence [[Heine–Borel 定理的边界|compact]]. The continuous scalar function $u\mapsto u^THu$ attains its minimum on $S$ by the [[极值定理|extreme value theorem]]. Call that minimum $m$. It is **strictly positive**, because the minimizing unit vector is nonzero and $H$ is positive definite. Thus, putting $u=h/\|h\|_2$ for any $h\ne0$ gives the [[正定谱下界|uniform quadratic lower bound]]
<!-- bilingual-en:end -->

记 $H=H_f(a)$。先把方向放到单位球面 $S=\{u\in\mathbb R^n:\|u\|_2=1\}$ 上，使它们长度相同。这个球面非空，在有限维欧氏空间中闭且有界，因而由 [[Heine–Borel 定理的边界|Heine–Borel 定理]]知其紧致。连续标量函数 $u\mapsto u^THu$ 由[[极值定理]]在 $S$ 上取得最小值，记为 $m$。取得最小值的单位向量非零，而 $H$ 正定，所以 **$m>0$**。于是对任意 $h\ne0$，令 $u=h/\|h\|_2$，得到[[正定谱下界|统一的二次型下界]]：

$$h^THh=\|h\|_2^2u^THu\ge m\|h\|_2^2.$$

<!-- bilingual-en:start -->
Next use the [[多元Taylor近似|second-order Taylor approximation]] at the stationary point:
<!-- bilingual-en:end -->

再在驻点处使用[[多元Taylor近似|二阶 Taylor 近似]]：

$$f(a+h)-f(a)=\frac12h^THh+R(h),\qquad
\frac{R(h)}{\|h\|_2^2}\longrightarrow0\quad(h\to0,\ h\ne0).$$

<!-- bilingual-en:start -->
The last limit is over **all** small vectors, not only along one fixed line. In particular, there is a radius $\delta>0$, chosen inside the $C^2$ neighborhood, such that $|R(h)|\le(m/4)\|h\|_2^2$ whenever $0<\|h\|_2<\delta$. Consequently
<!-- bilingual-en:end -->

最后这个极限针对**所有**小向量，并不只沿一条固定直线成立。特别地，可以在 $C^2$ 邻域内选一个半径 $\delta>0$，使每个 $0<\|h\|_2<\delta$ 都满足 $|R(h)|\le(m/4)\|h\|_2^2$。因此

$$f(a+h)-f(a)\ge\frac m2\|h\|_2^2-|R(h)|
\ge\frac m4\|h\|_2^2>0.$$

<!-- bilingual-en:start -->
This proves the strict local minimum. Merely finding a positive quadratic value in each direction would leave a gap unless one also obtains a common lower bound and a common radius; [[逐条直线极小不推局部极小|separate minima along all lines]] do not in general settle the neighborhood comparison.
<!-- bilingual-en:end -->

这就证明了严格局部极小。若只逐方向找到正的二次值，却没有共同下界与共同半径，证明仍有缺口；一般而言，[[逐条直线极小不推局部极小|沿所有直线分别极小]]并不能解决整个邻域中的比较。

> [!note]- Where continuity of the Hessian enters · Hessian 连续性用在哪里
> <!-- bilingual-en:start -->
> Applying one-variable integration along the segment $a+th$, $0\le t\le1$, gives the exact remainder
> <!-- bilingual-en:end -->
>
> 沿线段 $a+th$（$0\le t\le1$）作一元积分，得到精确余项
>
> $$R(h)=\int_0^1(1-t)h^T[H_f(a+th)-H_f(a)]h\,dt.$$
>
> <!-- bilingual-en:start -->
> Given $\eta>0$, continuity of the finitely many Hessian entries provides a common small ball in which each entry difference has absolute value at most $\eta$. The entire segment lies in that ball. Expanding the quadratic form and using [[Cauchy–Schwarz不等式|Cauchy–Schwarz]], $(\sum_i|h_i|)^2\le n\sum_i h_i^2$, yields
> <!-- bilingual-en:end -->
>
> 任给 $\eta>0$，有限多个 Hessian 元素的连续性给出共同小球，使每个元素之差的绝对值不超过 $\eta$；整条线段都在球中。展开二次型，再用 [[Cauchy–Schwarz不等式]] $(\sum_i|h_i|)^2\le n\sum_i h_i^2$，得到
>
> $$|R(h)|\le\frac{\eta}{2}\Bigl(\sum_i|h_i|\Bigr)^2
> \le\frac{n\eta}{2}\|h\|_2^2.$$
>
> <!-- bilingual-en:start -->
> Because $\eta$ can be arbitrarily small, this proves the all-direction remainder limit used above. Taking $\eta=m/(2n)$ gives the needed $m/4$ bound. Discarding the remainder without such control would not prove the conclusion.
> <!-- bilingual-en:end -->
>
> 由于 $\eta$ 可以任意小，这就证明了上面针对所有方向的余项极限；取 $\eta=m/(2n)$ 即得所需的 $m/4$ 界。若未经这样的控制就直接丢掉余项，不能完成证明。

<!-- bilingual-en:start -->
## What this test does not decide
<!-- bilingual-en:end -->

*充分性、局部性与唯一性分别看*

<!-- bilingual-en:start -->
The opening bowl has $\nabla f(0,0)=0$ and $H=\begin{pmatrix}2&0\\0&4\end{pmatrix}$. For $h=(h_1,h_2)^T$, $h^THh=2h_1^2+4h_2^2\ge2\|h\|_2^2$, so one may take $m=2$; its Taylor remainder is zero. Direct comparison also shows this particular minimum is global.

For $g(x)=x^2-x^4$ on $\mathbb R$, however, $g'(0)=0$ and $g''(0)=2$ still give a strict local minimum, while $g(2)=-12<g(0)$. The Hessian at one point does not compare distant inputs. A separate sufficient route is [[凸优化全局性|convexity on a convex domain]], which makes any local minimum global; direct comparison over the whole domain can also establish globality.
<!-- bilingual-en:end -->

开头的碗形函数满足 $\nabla f(0,0)=0$，且 $H=\begin{pmatrix}2&0\\0&4\end{pmatrix}$。对 $h=(h_1,h_2)^T$，有 $h^THh=2h_1^2+4h_2^2\ge2\|h\|_2^2$，故可取 $m=2$；它的 Taylor 余项为零。直接比较还表明，这个具体例子的最小值是全局的。

但对 $\mathbb R$ 上的 $g(x)=x^2-x^4$，虽然 $g'(0)=0$、$g''(0)=2$ 仍给出严格局部极小，却有 $g(2)=-12<g(0)$。一点的 Hessian 并不比较远处输入。另一个充分途径是[[凸优化全局性|凸域上的凸性]]，它使局部极小成为全局极小；对整个定义域直接比较也可以证明全局性。

<!-- bilingual-en:start -->
Strict local minimality does not mean a unique global minimizing point. The function $(x^2-1)^2$ has stationary points at $1$ and $-1$ with second derivative $8$ at both; each passes this test, and both attain the global value zero. Finally, positive definiteness is sufficient but not necessary: $x^4$ has a strict minimum at zero with Hessian zero. When the Hessian is singular and semidefinite, use the separate [[半定Hessian无结论|semidefinite-Hessian boundary]] rather than declaring that a minimum is absent.
<!-- bilingual-en:end -->

严格局部极小不等于全局极小点唯一。函数 $(x^2-1)^2$ 在 $1$ 与 $-1$ 都是驻点，两处二阶导都为 $8$，都通过本检验，而且都取得全局值零。最后，正定只是充分条件而非必要条件：$x^4$ 在零点严格极小，Hessian 却为零。当 Hessian 奇异且半定时，应转到独立的[[半定Hessian无结论]]边界分析，不能据此宣布不存在极小值。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[EC400 Slides Lecture 2.pdf#page=10|LSE EC400 SOFP Lecture 2, slides 10–14]]: interior stationarity, definite-Hessian local sufficient conditions and local/global distinction. The uniform lower bound and explicit remainder estimate above complete the Taylor argument.
- [MIT 18.S096, lecture 12, §12.1, PDF pp.1–2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=2) and [Lebl, Basic Analysis II, §8.6, PDF pp.60–63](https://www.jirka.org/ra/realanal2.pdf#page=60): the Hessian Taylor term, $C^2$ and symmetry; Lebl's exercise 8.6.7 explicitly distinguishes a uniform neighborhood argument from separate line tests.
- [MIT 18.101, Review of Topology, PDF pp.7–9, theorem 1.25](https://ocw.mit.edu/courses/18-101-analysis-ii-fall-2005/babd982be745679b6d691f78b1c18f53_lectures.pdf#page=7): compactness of closed bounded Euclidean sets and attainment of extrema. Here these are applied to the nonempty unit sphere.
- [[MIT18_06SCF11_Ses3.3sum.pdf#page=1|MIT 18.06SC Session 3.3, pp.1–3]] and [[03_Positive Definite Matrices and Applications#3.3.3 Hessian 与极小值|the permanent course quadratic example]]: positive-definite quadratic forms and direct comparison for quadratics. [Boyd and Vandenberghe, §§3.1.3–3.1.4 and 4.2.2, PDF pp.83–85 and 152](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=83): the separate convexity route to global optimality, with a convex domain.
<!-- bilingual-en:end -->

- [[EC400 Slides Lecture 2.pdf#page=10|LSE EC400 SOFP Lecture 2，第10–14页]]：内部驻点、正负定 Hessian 的局部充分条件及局部与全局的区别。正文用统一下界与明确余项估计补全 Taylor 论证。
- MIT 18.S096 讲义12 §12.1，PDF第1–2页，以及 Lebl《Basic Analysis II》§8.6，PDF第60–63页：Hessian 二次 Taylor 项、$C^2$ 与对称性；Lebl 习题8.6.7明确区分统一邻域论证与逐条直线检验。
- MIT 18.101《Review of Topology》PDF第7–9页、定理1.25：欧氏空间闭有界集的紧致性，以及极值的取得；本文将其用于非空单位球面。
- [[MIT18_06SCF11_Ses3.3sum.pdf#page=1|MIT 18.06SC Session 3.3，第1–3页]]与[[03_Positive Definite Matrices and Applications#3.3.3 Hessian 与极小值|永久课程二次函数例子]]：正定二次型及二次函数的直接比较。Boyd 与 Vandenberghe《Convex Optimization》§§3.1.3–3.1.4、4.2.2，PDF第83–85、152页：在凸域上通过凸性证明全局最优的独立充分路线。

---
aliases:
  - "Local convergence of Newton method"
  - "Newton 法局部收敛"
  - "Newton quadratic convergence"
student_os: knowledge-atom
atom_id: CS-NR-008
atom_set: numerical-root-finding
atom_type: convergence-theorem
status: needs-review
requires:
  - "[[Newton迭代]]"
related:
  - "[[残差控制根误差]]"
  - "[[算法成本模型]]"
leads_to:
  - "[[Newton失效边界]]"
  - "[[求根策略]]"
part_of:
  - "[[数值求根.canvas|数值求根]]"
  - "[[导数的应用.canvas]]"
---

<!-- bilingual-en:start -->
# Local convergence of Newton iteration
<!-- bilingual-en:end -->
*Newton 局部收敛*

<!-- bilingual-en:start -->
Newton iteration can turn a fairly accurate root estimate into a much more accurate one. **Local convergence** explains this improvement after the starting guess is sufficiently close to a suitable root. It does not explain how every remote guess finds that neighborhood. The result below proves both that nearby iterates stay nearby and that their errors shrink.
<!-- bilingual-en:end -->

Newton 迭代可以把已经较准的根估计变得更准。**局部收敛**解释的是：初值已经足够靠近合适的根之后，为什么这种改善会持续发生。它不解释所有远处初值怎样进入这个邻域。下面的结论同时证明“后续点留在附近”和“误差不断缩小”。

<!-- bilingual-en:start -->
Recall the [[Newton迭代|Newton update]] and the [[Taylor余项|Taylor remainder]], which measures what a local polynomial approximation leaves out. Write $\alpha$ for a true root and $e_k=x_k-\alpha$ for the signed position error. A **simple root** here means $f(\alpha)=0$ and $f'(\alpha)\ne0$: the graph crosses zero with nonzero slope. $C^2$ means that the first two derivatives exist and are continuous near the root.
<!-- bilingual-en:end -->

先回顾[[Newton迭代]]与[[Taylor余项]]：后者度量局部多项式近似遗漏的部分。用 $\alpha$ 表示真实根，$e_k=x_k-\alpha$ 是带符号的位置误差。这里的**简单根**指 $f(\alpha)=0$ 且 $f'(\alpha)\ne0$，也就是图像以非零斜率穿过零。$C^2$ 表示根附近的一阶、二阶导数存在且连续。

> [!abstract] The central idea · 核心思路
> <!-- bilingual-en:start -->
> Newton's speed comes from a conditional error recurrence. The iterates must already lie near a simple root, the second derivative must be controlled, and the first derivative must stay away from zero. Then the next error is bounded by the square of the current error. These conditions do not say that an arbitrary starting point will enter that neighbourhood.
> <!-- bilingual-en:end -->
>
> Newton 法的“快”来自一个有条件的误差递推：迭代点先要落在简单根附近，函数要有受控的二阶曲率，导数还要远离 0。满足这些条件后，下一轮误差由当前误差的平方控制；这些条件没有说明任意初值都能进入该邻域。

<!-- bilingual-en:start -->
## The Taylor remainder supplies a squared-error bound
<!-- bilingual-en:end -->

## Taylor 余项给出误差平方界

<!-- bilingual-en:start -->
Assume $f(\alpha)=0$ and $f'(\alpha)\ne0$, and set $e_k=x_k-\alpha$. Let $I$ be an open interval containing $\alpha$, with $f$ twice continuously differentiable on $I$. For $x_k\in I$ with $f'(x_k)\ne0$, Taylor’s theorem gives a point $\xi_k$ between $x_k$ and $\alpha$ such that
<!-- bilingual-en:end -->

设 $f(\alpha)=0$，$f'(\alpha)\ne0$，并记 $e_k=x_k-\alpha$。取一个包含 $\alpha$ 的开区间 $I$，假设 $f$ 在 $I$ 上二阶连续可微。对任意满足 $f'(x_k)\ne0$ 的 $x_k\in I$，Taylor 公式给出某个介于 $x_k$ 与 $\alpha$ 之间的 $\xi_k\in I$，使

$$
e_{k+1}
=\frac{f''(\xi_k)}{2f'(x_k)}e_k^2.
$$

<!-- bilingual-en:start -->
To see the cancellation, expand at the current iterate but evaluate at the root: $0=f(x_k)-f'(x_k)e_k+\tfrac12f''(\xi_k)e_k^2$. Dividing by $f'(x_k)$ gives $f(x_k)/f'(x_k)=e_k-[f''(\xi_k)/(2f'(x_k))]e_k^2$. Subtracting this quotient from $e_k$ is exactly the Newton error update. The linear error term disappears; the remaining curvature term is proportional to $e_k^2$.
<!-- bilingual-en:end -->

消去过程可以逐步看清：在当前点展开，却代入真实根，得 $0=f(x_k)-f'(x_k)e_k+\tfrac12f''(\xi_k)e_k^2$。除以 $f'(x_k)$，便有 $f(x_k)/f'(x_k)=e_k-[f''(\xi_k)/(2f'(x_k))]e_k^2$。Newton 更新恰好从 $e_k$ 中减去这个商，因此一次误差项被消去，只留下与 $e_k^2$ 成比例的曲率项。

<!-- bilingual-en:start -->
If throughout the interval $I$ we have
<!-- bilingual-en:end -->

进一步，若在整个 $I$ 上

$$
|f'(x)|\ge m>0,
\qquad
|f''(x)|\le M,
$$

<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->

则

$$
|e_{k+1}|
\le \frac{M}{2m}|e_k|^2.
$$

<!-- bilingual-en:start -->
These bounds can be arranged on a sufficiently small interval around a $C^2$ simple root. Continuity of $f'$ keeps it close to its nonzero root value and therefore away from zero; continuity of $f''$ makes it bounded on a small closed surrounding interval. Restrict to its interior if needed. Choose these bounds first, then choose a still smaller radius below. No future iterate is assumed to remain there in advance.
<!-- bilingual-en:end -->

在 $C^2$ 简单根附近，可以先选一个足够小的区间取得这些界：$f'$ 连续，使它保持远离零；$f''$ 在周围一个小闭区间上连续，因而有界，需要时再取内部。先固定这些界，随后再缩小下面的半径；这里没有预先假设未来迭代点不会跑出去。

<!-- bilingual-en:start -->
Put $C=M/(2m)$. Choose $\rho>0$ so that $[\alpha-\rho,\alpha+\rho]\subset I$ and $q=C\rho<1$. If $e_0=0$, the initial guess is already the root. Otherwise, $|e_0|\le\rho$ implies
<!-- bilingual-en:end -->

令 $C=M/(2m)$。选半径 $\rho>0$，使闭区间 $[\alpha-\rho,\alpha+\rho]\subset I$ 且 $q=C\rho<1$。若 $e_0=0$，初值已经是根；否则，只要 $|e_0|\le\rho$，就有

$$
|e_1|\le C|e_0|^2\le q|e_0|<|e_0|\le\rho.
$$

<!-- bilingual-en:start -->
Apply the same argument inductively. Zero error is the trivial fixed case; every nonzero error contracts by at most $q<1$. Thus the iterates stay inside the closed interval, converge to the root and satisfy the squared-error bound throughout. “Sufficiently close” now has an explicit sufficient condition, instead of assuming the neighborhood is invariant.
<!-- bilingual-en:end -->

同一论证可归纳到每一轮；零误差是平凡固定情形，非零误差则至多按因子 $q<1$ 收缩。因此迭代点不会离开这个闭区间，误差趋于 0，并始终满足平方界。这里的“初值足够接近”有一个明确的充分条件，而不是先假定后续点会留在邻域。

> [!example] A concrete protected neighborhood · 一个具体不变邻域
> <!-- bilingual-en:start -->
> For $f(x)=x^2-2$ and $\alpha=\sqrt2$, use $I=(1,2)$, $m=2$, $M=2$, and hence $C=1/2$. Taking $\rho=1/4$ keeps the closed neighborhood inside $I$ and gives $q=1/8$. The initial value $1.5$ lies in this neighborhood, so the theorem applies to the example in the update card. The true error uses the unknown root; this is a mathematical convergence proof, not by itself an implementable stopping test.
> <!-- bilingual-en:end -->
>
> 对 $f(x)=x^2-2$、$\alpha=\sqrt2$，可用 $I=(1,2)$、$m=2$、$M=2$，故 $C=1/2$。取 $\rho=1/4$，闭邻域仍在 $I$ 内，且 $q=1/8$。初值 $1.5$ 在这个邻域内，因此更新卡中的例子满足定理。真实误差含有未知根；这是收敛证明，本身不是可直接执行的停止判据。

<!-- bilingual-en:start -->
## At least quadratic versus exactly quadratic
<!-- bilingual-en:end -->

## 至少二次与恰好二次

<!-- bilingual-en:start -->
Provided the subsequent errors are nonzero, the signed asymptotic coefficient is
<!-- bilingual-en:end -->

只要后续误差非零，带符号的渐近系数为

$$
\lim_{k\to\infty}\frac{e_{k+1}}{e_k^2}
=\frac{f''(\alpha)}{2f'(\alpha)}
$$

<!-- bilingual-en:start -->
If $f''(\alpha)\ne0$ and the iteration never hits the root in finitely many steps, the absolute error ratio tends to a positive finite constant, so the local order is exactly two. If $f''(\alpha)=0$, that ratio tends to zero: the squared bound remains valid, but the actual order may be higher. A finite exact hit ends the sequence, so no later error ratios need to be assigned. These are local error statements, not per-step evaluation costs or a global complexity result.
<!-- bilingual-en:end -->

若 $f''(\alpha)\ne0$ 且迭代没有有限步命中根，则绝对值比率趋向正的有限常数，Newton 法在该根附近的收敛阶恰为 2。若 $f''(\alpha)=0$，上述比率趋于 0；平方上界仍成立，但实际可能高于二阶，不能无条件说“阶恰为 2”。若有限步已经命中根，序列直接结束，也无需再给后续误差比率定义。这些都是局部误差结论，不是单步函数计算成本，也不是任意初值下的全局复杂度。

<!-- bilingual-en:start -->
Here “at least quadratic” refers to a bound $|e_{k+1}|\le C|e_k|^2$ for all sufficiently late steps. Exactly quadratic additionally requires a nonzero limiting ratio. For $f(x)=x+x^3$, the simple root is zero but $f''(0)=0$, and direct substitution gives $x_{k+1}=2x_k^3/(1+3x_k^2)$. Near zero the absolute cubic-error ratio tends to $2$: convergence is exactly cubic for a nonterminating sufficiently close run. The condition $f''(\alpha)=0$ alone does not prescribe a particular higher order. For an affine function with nonzero slope, Newton reaches the root in one step.
<!-- bilingual-en:end -->

这里“至少二次”指足够后面的每一步都有 $|e_{k+1}|\le C|e_k|^2$ 这一界；“恰好二次”还要求极限比率非零。以 $f(x)=x+x^3$ 为例，零是简单根但 $f''(0)=0$，直接代入可得 $x_{k+1}=2x_k^3/(1+3x_k^2)$。根附近的绝对值三次误差比率趋于 $2$，所以足够近且不有限终止的运行恰为三次收敛。仅凭 $f''(\alpha)=0$，不能指定一个更高阶数。对于斜率非零的仿射函数，Newton 一步便命中根。

<!-- bilingual-en:start -->
If $|e_{k+1}|\approx A|e_k|^2$ with $A>0$, define a decimal accuracy measure $D_k=-\log_{10}|e_k|$ in fixed units. Then $D_{k+1}\approx2D_k-\log_{10}A$: once the local regime dominates, the number of accurate digits roughly doubles. The error is squared; the digit count is not squared. Floating-point rounding eventually limits this pattern, as discussed in [[数值迭代停止条件|stopping criteria]].
<!-- bilingual-en:end -->

若 $|e_{k+1}|\approx A|e_k|^2$ 且 $A>0$，在固定单位下用 $D_k=-\log_{10}|e_k|$ 表示十进制精度量，则 $D_{k+1}\approx2D_k-\log_{10}A$。进入局部阶段后，准确位数大致翻倍；被平方的是误差，不是位数。浮点舍入最终会限制这个模式，见[[数值迭代停止条件]]。

<!-- bilingual-en:start -->
## Why the simple-root condition matters
<!-- bilingual-en:end -->

## 简单根条件不能删掉

<!-- bilingual-en:start -->
For $f(x)=x^3$, the root $\alpha=0$ has multiplicity three. At a nonzero iterate, ordinary Newton gives
<!-- bilingual-en:end -->

对 $f(x)=x^3$，根 $\alpha=0$ 的重数为 3。当 $x_k\ne0$ 时，普通 Newton 更新为

$$
x_{k+1}=x_k-\frac{x_k^3}{3x_k^2}
=\frac23x_k,
$$

<!-- bilingual-en:start -->
Therefore $|e_{k+1}|=(2/3)|e_k|$: convergence is only linear. The iterates may still tend to the root, but the squared-error theorem for a simple root does not transfer to this multiple root. From outside the local neighborhood, entry into it depends on the problem and the mechanisms in [[Newton失效边界|Newton failure boundaries]].
<!-- bilingual-en:end -->

因此 $|e_{k+1}|=(2/3)|e_k|$，只有线性收敛。迭代仍可能趋近根，但简单根附近的平方误差界不能移植到这个重根。若初值不在局部收敛邻域，序列能否进来还要由 [[Newton失效边界]] 和实际问题结构判断。

> [!question-] Try the distinction · 检验理解
> <!-- bilingual-en:start -->
> If errors decrease slowly in the first few Newton steps, does that refute the squared-error theorem near a simple root?
>
> **Answer:** No. Those steps may lie outside the required neighborhood. Check simplicity of the root, the derivative bound and that the error refers to the same true root. Exactly second-order convergence also requires a nonzero asymptotic coefficient.
> <!-- bilingual-en:end -->
>
> 一次 Newton 运行前几轮误差下降很慢，能否仅凭这一现象否定简单根附近的平方误差界？
>
> **答案：** 不能。前几轮可能还没有进入定理要求的局部邻域；也要核对根是否简单、导数是否远离 0，以及误差是否用同一个真实根定义。平方误差界只描述条件满足后的局部阶段；是否恰为二阶还要看渐近系数是否非零。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT 18.330, Chapter 4: Nonlinear equations, PDF pp. 2–4: the Taylor error identity and local squared-error estimate. This card supplies an invariant-neighborhood induction and distinguishes a zero from a nonzero asymptotic coefficient.
<!-- bilingual-en:end -->

- MIT 18.330, [*Introduction to Numerical Analysis, Chapter 4: Nonlinear equations*](https://ocw.mit.edu/courses/18-330-introduction-to-numerical-analysis-spring-2012/5b325bfa56a599794c7196de926844b0_MIT18_330S12_Chapter4.pdf)：核对简单根附近的 Newton 误差展开、平方误差界，以及渐近系数非零时恰为二阶的结论。重根反例由本卡把 $f(x)=x^3$ 直接代入更新式验证。

<!-- bilingual-en:start -->
- MIT 6.006 Lecture 12, linked lecture notes PDF pp. 1–2: the square-root relative-error recurrence and the approximate doubling of accurate digits.
<!-- bilingual-en:end -->

- MIT 6.006, [*Lecture 12: Square Roots, Newton's Method*](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-12-square-roots-newtons-method/)：交叉核验平方根 Newton 迭代的误差分析与精度增长。

<!-- bilingual-en:start -->
- The Newton-iteration card supplies the update; this card concerns its local error estimate and convergence-order boundaries.
<!-- bilingual-en:end -->

- [[Newton迭代]]：复用切线更新式；本卡只拥有局部平方误差界及其收敛阶边界。

<!-- bilingual-en:start -->
The MIT 18.330 PDF p. 3 image contains an inaccurate statement that the digit count is squared and mixes iteration indices in the displayed iterated bound. Neither is adopted. The local invariant-region proof here does not assume convergence before deriving it. The cubic and affine cases are direct substitutions.
<!-- bilingual-en:end -->

MIT 18.330 PDF 第 3 页原图把位数说成平方，并在迭代误差界中混用了下标；两处都不沿用。这里先证明局部区域不变，再推出收敛，不先假定收敛。三次与仿射情形由直接代入验证。

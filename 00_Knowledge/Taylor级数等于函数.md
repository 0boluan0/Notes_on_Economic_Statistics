---
aliases:
  - Convergence of a Taylor series to its function
student_os: knowledge-atom
atom_id: CALC-SER-024
atom_type: theorem
status: source-checked
part_of:
  - "[[无穷级数与幂级数.canvas]]"
requires:
  - "[[Taylor级数]]"
  - "[[Taylor余项]]"
related:
  - "[[光滑不代表解析]]"
leads_to:
  - "[[指数函数幂级数]]"
  - "[[正弦函数幂级数]]"
---

<!-- bilingual-en:start -->
# When a Taylor series equals its function
<!-- bilingual-en:end -->
*Taylor 级数等于函数的条件*

<!-- bilingual-en:start -->
A sequence of polynomial approximations represents a function exactly when its errors disappear. Merely seeing the approximations settle down is not enough: they could settle at the wrong value. The Taylor remainder records the difference from the intended function, so it gives a direct test of equality.
<!-- bilingual-en:end -->

一列多项式近似能够表示原函数，正是因为误差最终消失。仅仅看见这些近似值稳定下来还不够，它们可能稳定在错误的数值。Taylor 余项记录近似值与目标函数的差，因此能直接判定两者是否相等。

<!-- bilingual-en:start -->
For example, the Maclaurin partial sums of $e^x$ begin with $1$, $1+x$, and $1+x+x^2/2$. At $x=0.1$, these give $1$, $1.1$, and $1.105$. Computing further values may suggest a limit, but proving that the limit is $e^{0.1}$ requires control of the omitted part. The argument below provides that control for every fixed real input.
<!-- bilingual-en:end -->

例如，$e^x$ 的 Maclaurin 部分和从 $1$、$1+x$、$1+x+x^2/2$ 开始。在 $x=0.1$ 处，它们依次给出 $1$、$1.1$、$1.105$。继续计算可以提示存在极限，但要证明极限就是 $e^{0.1}$，仍须控制被省略的部分。下面的论证会对每个固定实数输入提供这种控制。

<!-- bilingual-en:start -->
## Keep the input fixed and let the order grow
<!-- bilingual-en:end -->
*固定输入，让阶数增加*

<!-- bilingual-en:start -->
Let $I$ be a nonempty open real interval containing the centre $a$, and let $f:I\to\mathbb R$ be smooth: every [[高阶导数|derivative of every order]] exists and is continuous on $I$. Write $P_{N,a}$ for the [[Taylor多项式|Taylor polynomial]] through order $N$, where $N$ is a nonnegative integer. Its [[Taylor余项|remainder]] is the exact difference
<!-- bilingual-en:end -->

设 $I$ 是包含中心 $a$ 的非空实开区间，$f:I\to\mathbb R$ 光滑，即每一阶[[高阶导数]]都在 $I$ 上存在且连续。对非负整数 $N$，用 $P_{N,a}$ 表示截至 $N$ 阶的 [[Taylor多项式]]；其 [[Taylor余项]]是精确差值

$$P_{N,a}(x)=\sum_{n=0}^{N}\frac{f^{(n)}(a)}{n!}(x-a)^n,
\qquad R_{N,a}(x)=f(x)-P_{N,a}(x).$$

> [!important] The exact criterion / 精确判据
> <!-- bilingual-en:start -->
> For each fixed $x\in I$, the [[Taylor级数|Taylor series]] converges to $f(x)$ if and only if its remainders there tend to zero:
> <!-- bilingual-en:end -->
>
> 对每个固定的 $x\in I$，[[Taylor级数]]收敛到 $f(x)$，当且仅当该处余项趋于零：
>
> $$f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n
> \quad\Longleftrightarrow\quad \lim_{N\to\infty}R_{N,a}(x)=0.$$

<!-- bilingual-en:start -->
Indeed, an [[无穷级数|infinite series]] equals $f(x)$ precisely when its partial sums $P_{N,a}(x)$ tend to $f(x)$. Subtracting them from the fixed number $f(x)$ gives the equivalence. The smooth-interval assumption lets us use derivative estimates below; this algebraic equivalence itself needs only the defined coefficients, the value $f(x)$ and the remainder identity. At the centre, $P_{N,a}(a)=f(a)$ for every $N$, so the remainder is already zero.
<!-- bilingual-en:end -->

证明直接来自[[无穷级数]]的定义：级数等于 $f(x)$，恰好是其部分和 $P_{N,a}(x)$ 趋于 $f(x)$。用固定数值 $f(x)$ 减去部分和，就得到上述等价关系。光滑区间条件便于使用下文的导数估计；这个代数等价本身只需系数、$f(x)$ 及余项恒等式都有定义。中心处对每个 $N$ 都有 $P_{N,a}(a)=f(a)$，余项已经为零。

<!-- bilingual-en:start -->
## Turn a finite-order error bound into a convergence proof
<!-- bilingual-en:end -->
*把有限阶误差界用于收敛证明*

<!-- bilingual-en:start -->
Fix $x\ne a$ in $I$. The whole closed segment joining $a$ to $x$ lies inside $I$. For each $N$, choose a finite nonnegative number $M_N$ that bounds $|f^{(N+1)}(t)|$ throughout that segment; $t$ is the varying point on it. Such a bound exists here because this derivative is continuous on a closed bounded segment. The [[Taylor余项|Lagrange remainder bound]] gives
<!-- bilingual-en:end -->

固定 $I$ 中的 $x\ne a$。连接 $a$ 与 $x$ 的整个闭线段都位于 $I$ 内。对每个 $N$，选有限非负数 $M_N$，使线段上每一点 $t$ 都满足 $|f^{(N+1)}(t)|\le M_N$。由于该导数在闭有界线段上连续，这样的界确实存在。[[Taylor余项|Lagrange 余项界]]给出

$$|R_{N,a}(x)|\le \frac{M_N|x-a|^{N+1}}{(N+1)!}.$$

<!-- bilingual-en:start -->
If the right-hand side tends to zero, the [[夹逼定理|squeeze theorem]] proves equality with the function at this $x$. The centre is handled directly, so no intermediate point is needed when $x=a$. Crucially, $M_N$ can grow with $N$: the entire product must tend to zero. This estimate is sufficient, not necessary for a particular chosen bound; an excessively large bound can fail to tend to zero even when the actual remainder does.
<!-- bilingual-en:end -->

若右端趋于零，由[[夹逼定理]]就得到该 $x$ 处与原函数相等。中心处直接处理，因此 $x=a$ 时无需寻找中间点。关键在于，$M_N$ 可以随 $N$ 增长，必须检查整个乘积的极限。这项估计是充分方法；对某个选定的上界而言，它并非必要条件，因为过大的上界可能不趋于零，而真实余项仍趋于零。

> [!example] Exponential: one bound works at every order / 指数函数：同一个界控制每一阶
> <!-- bilingual-en:start -->
> For $f(x)=e^x$ and $a=0$, [[指数函数求导|every derivative is $e^x$]]. On the segment from $0$ to a fixed $x$, each derivative is at most $e^{|x|}$ in magnitude. Therefore
> <!-- bilingual-en:end -->
>
> 对 $f(x)=e^x$、$a=0$，[[指数函数求导|每阶导数都是 $e^x$]]。在从 $0$ 到固定 $x$ 的线段上，每阶导数的绝对值都不超过 $e^{|x|}$。因此
>
> $$|R_{N,0}(x)|\le e^{|x|}\frac{|x|^{N+1}}{(N+1)!}.$$
>
> <!-- bilingual-en:start -->
> For $x\ne0$, put $b_N=|x|^{N+1}/(N+1)!$. Its ratio is $b_{N+1}/b_N=|x|/(N+2)$, eventually at most $1/2$. After that index the bound is repeatedly halved, so $b_N\to0$. For $x=0$, the remainder is identically zero. Thus $e^x=\sum_{n=0}^{\infty}x^n/n!$ for every real $x$, now with equality proved rather than inferred from the coefficients.
> <!-- bilingual-en:end -->
>
> 对 $x\ne0$，令 $b_N=|x|^{N+1}/(N+1)!$，相邻项之比为 $b_{N+1}/b_N=|x|/(N+2)$，最终不超过 $1/2$。从那一项开始，上界不断减半，所以 $b_N\to0$。$x=0$ 处余项恒为零。由此对每个实数 $x$ 都有 $e^x=\sum_{n=0}^{\infty}x^n/n!$；这里的等号经过了证明，而非仅从系数形式猜出。

<!-- bilingual-en:start -->
To prove equality throughout an interval, apply the criterion to every fixed point in that interval. The order needed for a chosen accuracy may depend on the point. Requiring one order to work for every point simultaneously is the stronger issue of uniform convergence; it is not an extra requirement in the displayed pointwise equivalence. A boundary point outside $I$ also needs its own defined function value and justified limiting argument.
<!-- bilingual-en:end -->

要证明整个区间上相等，就必须把判据用于区间中的每一个固定点。达到指定精度所需的阶数，可以随该点改变。要求同一个阶数同时控制所有点，是更强的一致收敛问题，并不是上述逐点等价中的额外要求。若要讨论 $I$ 之外的边界点，也须先有定义明确的函数值，再单独说明极限为何成立。

<!-- bilingual-en:start -->
The distinction is real: in [[光滑不代表解析|the smooth nonanalytic example]], every Taylor coefficient at zero vanishes. The zero series converges everywhere, but for each fixed nonzero $x$ the remainder is the positive number $e^{-1/x^2}$ at every order. Series convergence alone never identifies that wrong limit with the original function.
<!-- bilingual-en:end -->

这种区分有实际反例：[[光滑不代表解析]]中的函数在零点所有 Taylor 系数都为零。零级数处处收敛，但对每个固定非零 $x$，每一阶余项始终是正数 $e^{-1/x^2}$。仅有级数收敛，不能把这个错误的极限认作原函数。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [OpenStax Calculus2 §6.3, Theorem6.8 and Example6.16](https://openstax.org/books/calculus-volume-2/pages/6-3-taylor-and-maclaurin-series): vanishing remainders as the necessary and sufficient condition, and the exponential example.
- [Lebl, Basic Analysis I v6.3, Theorem4.3.2 and following discussion, PDF172–173](https://www.jirka.org/ra/realanal.pdf#page=172): finite-order Lagrange remainders and the distinction between smoothness, series convergence and representation. The existing [[Taylor余项|Taylor remainder]] card contains the finite-order theorem; this card explains its limiting use.
<!-- bilingual-en:end -->

- OpenStax §6.3，定理6.8及例6.16：余项趋零的充要条件及指数函数例子。
- Lebl《Basic Analysis I》v6.3，定理4.3.2及其后讨论，PDF172–173页：有限阶 Lagrange 余项，以及光滑、级数收敛和表示成立之间的区别。已有 [[Taylor余项]]承载有限阶定理，本卡解释其极限用途。

---
aliases:
  - p-integral at infinity
  - 尾部 p 积分判别
student_os: knowledge-atom
atom_id: CALC-IMP-004
atom_set: improper-integrals
atom_type: theorem
status: source-checked
requires:
  - "[[反常积分]]"
part_of:
  - "[[反常积分.canvas]]"
---

<!-- bilingual-en:start -->
# The p-integral test at infinity
<!-- bilingual-en:end -->
*无穷远 p 积分判别*

<!-- bilingual-en:start -->
A positive curve can get closer and closer to zero while the area beneath its endless tail keeps growing without bound. The **p-integral test** decides what happens for the family $x^{-p}=1/x^p$. Here $p$ is a fixed real exponent and $x>0$, so the power is real and positive. For example, $p=2$ gives $1/x^2$, while $p=1/2$ gives $1/\sqrt x$. Both decrease toward zero, but only the first has finite total area from $1$ to infinity.
<!-- bilingual-en:end -->

一条正曲线即使越来越接近零，它无限延伸的尾部面积仍可能不断增大、没有上界。**无穷远 p 积分判别**判断幂函数族 $x^{-p}=1/x^p$ 在这一端的累积。这里 $p$ 是固定实数指数，$x>0$，所以实数幂有定义且为正。例如 $p=2$ 给出 $1/x^2$，$p=1/2$ 给出 $1/\sqrt x$。两者都趋于零，却只有前者从 $1$ 到无穷远的总面积有限。

<!-- bilingual-en:start -->
To find that total, first stop at a finite point $B>a$, where $a>0$ is the starting point. The [[定积分|definite integral]] $\int_a^B x^{-p}\,dx$ is an ordinary finite-interval accumulation. The [[反常积分|improper integral]] is its limit as $B$ moves arbitrarily far to the right; convergence means that this limit is a **finite real number**, not that the curve merely becomes small.
<!-- bilingual-en:end -->

求这个总量时，先停在有限位置 $B>a$，其中 $a>0$ 是起点。[[定积分]] $\int_a^B x^{-p}\,dx$ 是普通有限区间上的累积；[[反常积分]]则让 $B$ 不断向右移动后取极限。收敛要求这个极限是**有限实数**，而不只是曲线看起来越来越小。

$$I_p(a):=\lim_{B\to\infty}\int_a^B x^{-p}\,dx,\qquad a>0,\ p\in\mathbb R.$$

> [!abstract] The tail threshold · 尾部的临界指数
> <!-- bilingual-en:start -->
> The integral converges exactly when $p>1$. In that case its value is $a^{1-p}/(p-1)$; for $p\le1$, the finite truncated integrals increase without bound.
> <!-- bilingual-en:end -->
>
> 当且仅当 $p>1$ 时收敛，此时积分值为 $a^{1-p}/(p-1)$；若 $p\le1$，有限截断积分随上限增大而无界增长。
>
> $$\int_a^\infty x^{-p}\,dx=
> \begin{cases}
> \dfrac{a^{1-p}}{p-1},&p>1,\\[4pt]
> +\infty\quad\text{(divergent)},&p\le1.
> \end{cases}$$

<!-- bilingual-en:start -->
## Why the threshold is one
<!-- bilingual-en:end -->
*为什么临界值是一*

<!-- bilingual-en:start -->
For $p\ne1$, the [[实数幂求导|real power rule]] shows that the derivative of $x^{1-p}/(1-p)$ is $x^{-p}$ on $x>0$. It is therefore an [[原函数|antiderivative]], and the finite-interval [[Newton-Leibniz公式|Fundamental Theorem of Calculus]] gives
<!-- bilingual-en:end -->

若 $p\ne1$，由[[实数幂求导]]，$x^{1-p}/(1-p)$ 在 $x>0$ 上的导数是 $x^{-p}$，所以它是一个[[原函数]]。对有限区间应用 [[Newton-Leibniz公式]]，得到

$$\int_a^B x^{-p}\,dx=\frac{B^{1-p}-a^{1-p}}{1-p}.$$

<!-- bilingual-en:start -->
If $p>1$, the exponent $1-p$ is negative, so $B^{1-p}\to0$ and the expression approaches $a^{1-p}/(p-1)$. If $p<1$, then $1-p>0$ and $B^{1-p}\to+\infty$; the denominator is positive, so the accumulated integral also tends to $+\infty$.
<!-- bilingual-en:end -->

若 $p>1$，指数 $1-p$ 为负，故 $B^{1-p}\to0$，上式趋于 $a^{1-p}/(p-1)$。若 $p<1$，则 $1-p>0$，$B^{1-p}\to+\infty$，分母也是正数，因此累积积分趋于 $+\infty$。

<!-- bilingual-en:start -->
At $p=1$, that power formula has a zero denominator and cannot be used. Instead, the [[自然对数|natural logarithm]] has derivative $1/x$, so
<!-- bilingual-en:end -->

当 $p=1$，幂公式的分母为零，不能代入。应改用[[自然对数]]的原函数公式：

$$\int_a^B\frac{dx}{x}=\ln B-\ln a=\ln(B/a)\longrightarrow+\infty.$$

<!-- bilingual-en:start -->
This is logarithmic divergence: slower growth than a positive power of $B$, but still unbounded. The proof never substitutes infinity as an ordinary input into an antiderivative.
<!-- bilingual-en:end -->

这就是对数发散：它比 $B$ 的正次幂增长得慢，却仍没有上界。整个推导都没有把无穷当作原函数的普通输入值。

> [!example] A finite area and a slowly growing one · 有限面积与缓慢发散
> $$\int_1^B x^{-2}\,dx=1-\frac1B\longrightarrow1,$$
> $$\int_1^B x^{-3/4}\,dx=4(B^{1/4}-1)\longrightarrow+\infty.$$
>
> <!-- bilingual-en:start -->
> Both integrands tend to zero. The difference is the rate at which the remaining area accumulates. In the convergent first example, the area still beyond $B$ is exactly $1/B$. More generally, when $p>1$, the uncomputed tail beyond $B$ equals $B^{1-p}/(p-1)$.
> <!-- bilingual-en:end -->
>
> 两个被积函数都趋于零，区别在于尾部面积累积的速度。第一个例子收敛，而且 $B$ 之后尚未计入的面积恰为 $1/B$。一般地，$p>1$ 时，$B$ 之后的尾部积分是 $B^{1-p}/(p-1)$。

<!-- bilingual-en:start -->
## Use the test at the correct endpoint
<!-- bilingual-en:end -->
*先确认检验的是哪一端*

<!-- bilingual-en:start -->
Changing the positive starting point changes the value but not this convergence threshold. For example, $\int_{10}^\infty x^{-3/2}\,dx=2/\sqrt{10}$. Any interval between two positive finite starting points contains no singularity of $x^{-p}$ and has a finite integral. This is the relevant [[反常积分敛散的局部性|locality of convergence]]; an additional bad point elsewhere cannot be discarded just because it lies in a finite interval.
<!-- bilingual-en:end -->

改变正的有限起点会改变数值，却不改变这一敛散门槛。例如 $\int_{10}^\infty x^{-3/2}\,dx=2/\sqrt{10}$。任意两个正的有限起点之间，$x^{-p}$ 都没有奇点，积分为有限数；这正是这里使用的[[反常积分敛散的局部性]]。但别处若还有坏点，不能只因它位于有限区间就略过。

<!-- bilingual-en:start -->
The power family is also a benchmark for [[反常积分极限比较判别|limit comparison]]. If a function is properly integrable on finite tail intervals, is positive for all sufficiently large $x$, and $f(x)/x^{-p}$ tends to a finite constant $C>0$, its tail has the same convergence behaviour. The precise ratio condition matters: simply being below $1/x$ does not guarantee convergence, since $1/(2x)$ is below it and still has a divergent integral. Within the power family, $p\le0$ does not even give a function tending to zero at infinity.
<!-- bilingual-en:end -->

幂函数族还可以作为[[反常积分极限比较判别]]的基准：若 $f$ 在尾部的有限区间上正常可积，充分大的 $x$ 处为正，且 $f(x)/x^{-p}\to C$，其中 $0<C<\infty$，它就与对应幂函数的尾部同敛散。必须核对这一准确条件；仅仅低于 $1/x$ 不保证收敛，因为 $1/(2x)$ 更小，积分却仍发散。在幂函数族内部，$p\le0$ 甚至不能让函数在无穷远趋于零。

<!-- bilingual-en:start -->
Near zero, increasing $p$ has the opposite effect on the curve. That endpoint is governed by [[零附近 p 积分判别|the p-integral test near zero]], where convergence requires $p<1$. The tail criterion alone cannot decide an integral whose interval also reaches zero.
<!-- bilingual-en:end -->

在零附近，增大 $p$ 对曲线的影响恰好相反；那里应使用[[零附近 p 积分判别]]，收敛条件是 $p<1$。如果积分区间还触及零，仅检验无穷尾部就不够。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses91e_Lecture_Notes.pdf#page=1|MIT 18.01SC Session 91e, p.1]]: the three cases $p<1$, $p=1$, $p>1$ and the value for starting point $1$. The general positive starting point is obtained by the finite-truncation calculation above.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93c_Lecture_Notes.pdf#page=1|Session 93c, p.1]]: the reversed ordering of the power curves near zero and at infinity.
- [OpenStax Calculus 2 §3.7](https://openstax.org/books/calculus-volume-2/pages/3-7-improper-integrals), infinite-interval definition, Example 3.47, Theorem 3.7 and Example 3.56: truncation limits, logarithmic divergence and the direction of positive comparison.
<!-- bilingual-en:end -->

- MIT 91e 第1页支持三个指数范围和起点为一的值；一般正起点由正文有限截断计算得到。
- MIT 93c 第1页支持幂函数在零附近和无穷远的大小顺序反转。
- OpenStax §3.7 的无穷区间定义、例3.47、定理3.7与例3.56支持截断极限、对数发散及正函数比较方向。

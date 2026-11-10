---
aliases:
  - p-series convergence criterion
student_os: knowledge-atom
atom_id: CALC-SER-013
atom_type: theorem
status: source-checked
requires:
  - "[[级数积分判别]]"
  - "[[无穷远 p 积分判别]]"
related:
  - "[[级数通项判别]]"
  - "[[调和数对数增长]]"
part_of:
  - "[[无穷级数与幂级数.canvas]]"
---

<!-- bilingual-en:start -->
# The p-series test
<!-- bilingual-en:end -->
*p级数判别*

<!-- bilingual-en:start -->
A **p-series** adds reciprocal powers of the positive integers: $1+1/2^p+1/3^p+\cdots$. The exponent $p$ controls how rapidly the contributions shrink. Its convergence criterion gives a useful dividing line: the accumulated sum is finite exactly when $p>1$.
<!-- bilingual-en:end -->

**p级数**把正整数的倒数幂依次相加：$1+1/2^p+1/3^p+\cdots$。指数 $p$ 控制每次贡献缩小的速度。它的收敛判据给出一个常用分界：累积总和恰在 $p>1$ 时有限。

<!-- bilingual-en:start -->
Fix a real number $p$; it is a parameter, not the index that changes during summation. The [[无穷级数|series]] $\sum_{n=1}^\infty n^{-p}$ converges when its partial sums $S_N=\sum_{n=1}^N n^{-p}$ approach a finite number as $N$ increases. Every term is positive and defined, since $n\ge1$.
<!-- bilingual-en:end -->

固定实数 $p$；它是参数，不是求和时不断变化的下标。[[无穷级数|级数]] $\sum_{n=1}^\infty n^{-p}$ 收敛，是指部分和 $S_N=\sum_{n=1}^N n^{-p}$ 随 $N$ 增大趋于有限数。由于 $n\ge1$，每项都为正且有定义。

> [!important] The threshold is one · 临界指数为一
>
> $$\sum_{n=1}^\infty\frac1{n^p}\text{ converges}\quad\Longleftrightarrow\quad p>1.$$
>
> <!-- bilingual-en:start -->
> For $p\le1$ the partial sums tend to $+\infty$. The equality case $p=1$ is divergent, even though its terms tend to zero.
> <!-- bilingual-en:end -->
>
> $p\le1$ 时部分和趋于 $+\infty$。等号情形 $p=1$ 仍然发散，尽管它的通项趋于零。

<!-- bilingual-en:start -->
## Why the boundary occurs there
<!-- bilingual-en:end -->
*为什么分界在这里*

<!-- bilingual-en:start -->
For $p>0$, use $f(x)=x^{-p}$ on $[1,\infty)$. This function is continuous and positive, with derivative $f'(x)=-p x^{-p-1}<0$. Thus the [[级数积分判别|integral test]] says the series and $\int_1^\infty x^{-p}\,dx$ have the same convergence behaviour. The [[无穷远 p 积分判别|power-tail calculation]] gives, for a finite cutoff $R>1$,
<!-- bilingual-en:end -->

当 $p>0$，取 $[1,\infty)$ 上的 $f(x)=x^{-p}$。函数连续、为正，且导数 $f'(x)=-p x^{-p-1}<0$。因此，[[级数积分判别]]说明级数与 $\int_1^\infty x^{-p}\,dx$ 同敛散。由[[无穷远 p 积分判别|幂函数尾积分计算]]，对有限截断 $R>1$，

$$\int_1^R x^{-p}\,dx=
\begin{cases}
\dfrac{R^{1-p}-1}{1-p},&p\ne1,\\[4pt]
\ln R,&p=1.
\end{cases}$$

<!-- bilingual-en:start -->
If $p>1$, the exponent $1-p$ is negative, so $R^{1-p}\to0$ and the integral has finite limit $1/(p-1)$. If $0<p<1$, the positive exponent makes $R^{1-p}\to+\infty$. If $p=1$, the [[自然对数|natural logarithm]] also tends to $+\infty$. These settle all positive $p$.
<!-- bilingual-en:end -->

若 $p>1$，指数 $1-p$ 为负，故 $R^{1-p}\to0$，积分趋于有限值 $1/(p-1)$。若 $0<p<1$，正指数使 $R^{1-p}\to+\infty$。若 $p=1$，[[自然对数]]也趋于 $+\infty$。这就处理了所有正的 $p$。

<!-- bilingual-en:start -->
For $p\le0$, no integral calculation is needed. When $p=0$, every term equals one. When $p<0$, $n^{-p}=n^{|p|}\ge1$ and increases without bound. In both cases $S_N\ge N\to\infty$; the terms also fail the necessary [[级数通项判别|vanishing-term condition]]. This argument avoids applying a decreasing-function test to the increasing case.
<!-- bilingual-en:end -->

对 $p\le0$，不需要计算积分。$p=0$ 时每项为一；$p<0$ 时 $n^{-p}=n^{|p|}\ge1$，并且无界增长。两种情况都有 $S_N\ge N\to\infty$；通项也不满足[[级数通项判别|必须趋零的必要条件]]。这样不必把递减函数的判别硬套到递增情形上。

<!-- bilingual-en:start -->
## Read the exponent before making the decision
<!-- bilingual-en:end -->
*认出指数再作判断*

<!-- bilingual-en:start -->
| Series | Exponent | Conclusion |
| --- | --- | --- |
| $\sum n^{-3/2}$ | $p=3/2>1$ | Converges |
| $\sum 1/\sqrt n$ | $p=1/2$ | Diverges |
| $\sum 1/n$ | $p=1$ | Diverges |
| $\sum n^2$ | $p=-2$ | Diverges; terms do not vanish |
<!-- bilingual-en:end -->

| 级数 | 指数 | 结论 |
| --- | --- | --- |
| $\sum n^{-3/2}$ | $p=3/2>1$ | 收敛 |
| $\sum 1/\sqrt n$ | $p=1/2$ | 发散 |
| $\sum 1/n$ | $p=1$ | 发散 |
| $\sum n^2$ | $p=-2$ | 发散，通项不趋零 |

<!-- bilingual-en:start -->
For $p=1$, the partial sums are the [[调和数|harmonic numbers]] $H_N$. Rectangle comparison gives $H_N\ge\ln(N+1)$, so they grow without bound even though $1/n\to0$. Their slow, logarithmic growth is explained in [[调和数对数增长|harmonic-number growth]]. A large finite numerical sum cannot by itself establish convergence.
<!-- bilingual-en:end -->

当 $p=1$，部分和就是[[调和数]] $H_N$。矩形比较给出 $H_N\ge\ln(N+1)$，所以即使 $1/n\to0$，部分和仍无界。它缓慢的对数增长见[[调和数对数增长]]；算出一个很大的有限部分和，本身不能证明收敛。

> [!warning] The integral value is not the series sum · 积分值不是级数和
> <!-- bilingual-en:start -->
> For $p=2$, $\int_1^\infty x^{-2}\,dx=1$, but the series already has first two terms $1+1/4>1$. The criterion proves finiteness without claiming equality of these totals. [[积分判别余项界|Integral remainder bounds]] estimate the missing tail after a partial sum.
> <!-- bilingual-en:end -->
>
> 当 $p=2$，$\int_1^\infty x^{-2}\,dx=1$，但级数仅前两项就有 $1+1/4>1$。判据证明的是总和有限，并没有说两种总量相等。[[积分判别余项界]]可估计有限求和后遗漏的尾部。

<!-- bilingual-en:start -->
Changing the starting index to any positive integer removes only finitely many defined terms and leaves the criterion unchanged. A series whose terms merely resemble $n^{-p}$ needs a justified [[级数直接比较判别|direct comparison]] or [[级数极限比较判别|limit comparison]]; resemblance alone is not this theorem's hypothesis. Factors such as $\ln n$ or alternating signs require their own analysis.
<!-- bilingual-en:end -->

把起点改为任意正整数，只去掉有限多个已定义项，不改变判据。若通项只是“像” $n^{-p}$，就需要有依据的[[级数直接比较判别]]或[[级数极限比较判别]]；看起来相似并不是本定理的前提。出现 $\ln n$ 因子或交错符号时，需要另作分析。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [OpenStax Calculus2 §5.3, “The p-Series”, equation5.9](https://openstax.org/books/calculus-volume-2/pages/5-3-the-divergence-and-integral-tests): classification for real $p$ and separate arguments for positive and nonpositive exponents. The displayed examples and $p=2$ unequal-sum check are calculated directly.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses91e_Lecture_Notes.pdf#page=1|MIT18.01SC Session91e, p.1]] and [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses95b_Lecture_Notes.pdf#page=1|Session95b, p.1]]: power-tail integrals and the sum–integral comparison used in the proof. The power antiderivative is applied at finite cutoffs before taking limits.
<!-- bilingual-en:end -->

- OpenStax §5.3“The p-Series”及式5.9支持全部实数 $p$ 的分类，以及正与非正指数的分开论证；表中例子及 $p=2$ 的数值不相等检查直接计算。
- MIT18.01SC 91e、95b 第1页分别支持幂函数尾积分及证明使用的和与积分比较。幂函数原函数先用于有限截断，再取极限。

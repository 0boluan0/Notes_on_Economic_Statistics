---
student_os: knowledge-atom
atom_id: PROB-CONC-011
aliases:
- Cauchy–Schwarz inequality for expectations
- Cauchy-Schwarz inequality for random variables
atom_type: theorem
status: source-checked
---

<!-- bilingual-en:start -->
# Cauchy–Schwarz inequality for expectations
<!-- bilingual-en:end -->

期望的Cauchy–Schwarz不等式

<!-- bilingual-en:start -->
The average product of two random quantities cannot be arbitrarily large when both have controlled average squared sizes. Cauchy–Schwarz makes this precise: the product’s expected value is bounded in magnitude by the product of the two root-mean-square sizes. It works even when the quantities are dependent.
<!-- bilingual-en:end -->

如果两个随机量的平均平方大小都受到控制，它们乘积的平均值就不可能任意大。Cauchy–Schwarz不等式把这点写成精确的界：乘积期望的绝对值，不超过两个均方根大小的乘积；两者相互依赖时也成立。

<!-- bilingual-en:start -->
Let $X,Y$ be real [[随机变量|random variables]] on the same [[概率空间|probability space]]. Their product $XY$ uses the two values from the same outcome. Assume $E[X^2]<\infty$ and $E[Y^2]<\infty$, called square integrability. The [[期望|expectation]] $E[X^2]$ is a probability-weighted average of squared values; it is not $(E[X])^2$. Its square root $\sqrt{E[X^2]}$ measures root-mean-square size.
<!-- bilingual-en:end -->

设实[[随机变量]]$X,Y$定义在同一个[[概率空间]]上，乘积$XY$使用同一次结果中的两个值。假设$E[X^2]<\infty$且$E[Y^2]<\infty$，称为平方可积。[[期望]]$E[X^2]$先平方再按概率平均，不是$(E[X])^2$；其平方根$\sqrt{E[X^2]}$表示均方根大小。

> [!important] The expectation bound · 期望界
> <!-- bilingual-en:start -->
> Under these assumptions, the product is integrable and $|E[XY]|\le\sqrt{E[X^2]E[Y^2]}$. Applying the same inequality to $|X|$ and $|Y|$ also yields $E|XY|\le\sqrt{E[X^2]E[Y^2]}$. Independence is not needed, and $E[XY]$ cannot generally be replaced by $E[X]E[Y]$.
> <!-- bilingual-en:end -->
>
> 在上述条件下，乘积可积，且$|E[XY]|\le\sqrt{E[X^2]E[Y^2]}$。把同一不等式用于$|X|$和$|Y|$，还得到$E|XY|\le\sqrt{E[X^2]E[Y^2]}$。这里不要求独立，通常也不能把$E[XY]$换成$E[X]E[Y]$。

<!-- bilingual-en:start -->
## Check the joint outcomes
<!-- bilingual-en:end -->

按共同结果计算

<!-- bilingual-en:start -->
Suppose two outcomes are equally likely, giving $(X,Y)=(1,2)$ or $(3,0)$. Then $E[XY]=1$, $E[X^2]=5$ and $E[Y^2]=2$. The inequality reads $1\le\sqrt{10}$. If instead $Y=2X$ in both outcomes, $E[XY]=2E[X^2]=10$ and the bound is $\sqrt{5\cdot20}=10$, so equality holds.
<!-- bilingual-en:end -->

若两个结果等可能，对应$(X,Y)=(1,2)$或$(3,0)$，则$E[XY]=1$、$E[X^2]=5$、$E[Y^2]=2$，不等式为$1\le\sqrt{10}$。如果改成两个结果中都满足$Y=2X$，那么$E[XY]=2E[X^2]=10$，上界也是$\sqrt{5\cdot20}=10$，达到等号。

<!-- bilingual-en:start -->
To see why independence cannot be silently assumed, let $X=Y$ be a fair $\pm1$ variable. Both means are zero, but $XY=1$ always, so $E[XY]=1\ne E[X]E[Y]=0$. Cauchy–Schwarz still applies and gives equality.
<!-- bilingual-en:end -->

再看为什么不能偷偷假定独立：令$X=Y$为公平取$\pm1$的变量。两者均值都是零，但$XY$恒等于1，故$E[XY]=1\ne E[X]E[Y]=0$。Cauchy–Schwarz仍然适用，并且取等号。

<!-- bilingual-en:start -->
## A nonnegative squared error proves the bound
<!-- bilingual-en:end -->

用非负平方误差证明

<!-- bilingual-en:start -->
First, $2|XY|\le X^2+Y^2$ follows from $(|X|-|Y|)^2\ge0$. Taking expectations shows that $E|XY|<\infty$, so expanding an expected square will not involve undefined terms. If $E[Y^2]=0$, then $Y=0$ almost surely and the result is immediate. Otherwise set $a=E[XY]/E[Y^2]$. Nonnegativity gives
<!-- bilingual-en:end -->

先由$(|X|-|Y|)^2\ge0$得到$2|XY|\le X^2+Y^2$。求期望可知$E|XY|<\infty$，因此接下来的平方展开不会含有未定义的项。若$E[Y^2]=0$，则$Y=0$几乎必然成立，结论直接成立。否则取$a=E[XY]/E[Y^2]$，由非负性得到

$$
0\le E[(X-aY)^2]
=E[X^2]-2aE[XY]+a^2E[Y^2]
=E[X^2]-\frac{(E[XY])^2}{E[Y^2]}.
$$

<!-- bilingual-en:start -->
Multiplying by $E[Y^2]>0$ and taking square roots proves the inequality. This parallels the [[Cauchy–Schwarz不等式|finite-vector proof]], with expectation replacing a coordinate sum. The initial integrability check is what makes the probabilistic expansion legitimate.
<!-- bilingual-en:end -->

乘以$E[Y^2]>0$并开平方就得到结论。它与[[Cauchy–Schwarz不等式|有限向量版本]]的证明相似，只是以期望代替坐标求和；起初的可积性检查保证了这个概率展开合法。

<!-- bilingual-en:start -->
## Equality means proportionality almost surely
<!-- bilingual-en:end -->

等号对应几乎必然成比例

<!-- bilingual-en:start -->
For the bound on $|E[XY]|$, when $E[Y^2]>0$, equality is equivalent to $E[(X-aY)^2]=0$. A nonnegative variable has expectation zero only if it is zero almost surely, so $X=aY$ almost surely, with one fixed real constant $a$. If either variable vanishes almost surely, equality also holds. Negative constants are allowed because the expected product is inside an absolute value.
<!-- bilingual-en:end -->

对$|E[XY]|$这一条界，当$E[Y^2]>0$时，等号等价于$E[(X-aY)^2]=0$。非负变量的期望为零意味着它几乎必然为零，所以$X=aY$几乎必然成立，其中$a$是一个固定实常数。任一变量几乎必然为零时也取等号；$a$允许为负，因为乘积期望外有绝对值。

> [!warning] Two bounds have different equality details · 两种界的等号细节不同
> <!-- bilingual-en:start -->
> The statement above is for $|E[XY]|$. For $E|XY|$, the corresponding proportionality is between $|X|$ and $|Y|$. For example, a fair sign $X=\pm1$ and $Y=1$ give $E|XY|=1$ but $|E[XY]|=0$. Keeping the absolute value in the correct place matters.
> <!-- bilingual-en:end -->
>
> 上面的等号条件针对$|E[XY]|$。若讨论$E|XY|$，对应的是$|X|$与$|Y|$成比例。例如公平符号$X=\pm1$配合$Y=1$，有$E|XY|=1$，但$|E[XY]|=0$；绝对值放在哪里非常重要。

<!-- bilingual-en:start -->
## Turning squared error into absolute error
<!-- bilingual-en:end -->

从平方误差控制绝对误差

<!-- bilingual-en:start -->
For a square-integrable random variable $Z$, take the two variables to be $|Z|$ and the constant 1. Since $E[1^2]=1$, the inequality gives $E|Z|\le\sqrt{E[Z^2]}$. Thus a sequence that [[均方收敛|converges in mean square]] also [[L1收敛|converges in L1]]: apply the bound to $Z=X_n-X$. The square-root bound tends to zero with the mean squared error. No independence between an approximation and its target is required.
<!-- bilingual-en:end -->

对平方可积的随机变量$Z$，把两个变量取为$|Z|$与常数1，由$E[1^2]=1$得到$E|Z|\le\sqrt{E[Z^2]}$。因此，[[均方收敛]]也能推出[[L1收敛|$L^1$收敛]]：令$Z=X_n-X$，右侧随平均平方误差一起趋零即可。近似值与目标之间不需要独立。

## 来源与核验

<!-- bilingual-en:start -->
- [Durrett, Probability: Theory and Examples, Theorem 1.5.2 and following Cauchy–Schwarz remark, printed pp. 25–26 / PDF pp. 33–34](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=33): the integral inequality and squared-polynomial proof; expectation uses probability as the integrating measure. Section 1.6, printed p. 28, identifies expectation with that integral.
- [Axler, Linear Algebra Done Right, Theorem 6.14, printed p. 189 / PDF p. 203](https://linear.axler.net/LADR4e.pdf#page=203): the nonnegative-residual argument and proportionality principle. Here the zero-expectation argument spells out the almost-sure equality condition.
- The joint two-outcome examples, integrability estimate and error bound are explicitly calculated above.
<!-- bilingual-en:end -->

- Durrett核对积分形式、平方证明及期望作为概率积分的解释；Axler核对剩余平方与成比例条件。本文将零期望对应的几乎必然条件、两点分布计算与误差估计展开。

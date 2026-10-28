---
aliases:
  - "Markov's inequality"
  - "Markov inequality"
  - "马尔可夫不等式"
student_os: knowledge-atom
atom_id: PROB-CONC-003
atom_set: probability-concentration
atom_type: theorem
status: source-checked
requires:
  - "[[期望]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[平移Markov界]]"
  - "[[Chebyshev不等式]]"
  - "[[指数Markov法]]"
---

<!-- bilingual-en:start -->
# Markov's inequality
<!-- bilingual-en:end -->
*Markov 不等式*

<!-- bilingual-en:start -->
**Markov's inequality** gives an upper bound on the chance that a nonnegative quantity is large, using only its average. For example, if a waiting time has mean ten minutes, it cannot be at least thirty minutes too often: each such wait already contributes at least thirty minutes to the average. The result limits a probability even when we do not know the full distribution.
<!-- bilingual-en:end -->

**Markov 不等式**只利用平均值，就能给一个非负量取大值的概率设上界。例如，平均等待时间是十分钟，那么等待至少三十分钟的情况不可能太频繁：每次这样的等待，都已经向平均值贡献了至少三十分钟。即使不知道完整分布，也能限制某个事件的概率。

<!-- bilingual-en:start -->
Let $X$ be a [[随机变量|random variable]], and write $\mu=E[X]$ for its [[期望|expectation]], the probability-weighted mean. The event $\{X\ge a\}$ means that the value reaches or exceeds a threshold $a$. This is an upper-tail event.
<!-- bilingual-en:end -->

设 $X$ 是[[随机变量]]，用 $\mu=E[X]$ 表示它的[[期望]]，也就是按概率加权的均值。事件 $\{X\ge a\}$ 表示取值达到或超过门槛 $a$，称为右尾事件。

> [!abstract] A bound from the mean · 用均值给出上界
> <!-- bilingual-en:start -->
> If $X\ge0$ almost surely, $E[X]=\mu<\infty$, and $a>0$, then
> <!-- bilingual-en:end -->
>
> 若 $X\ge0$ 几乎必然成立、$E[X]=\mu<\infty$，且 $a>0$，则
>
> $$P(X\ge a)\le\frac{\mu}{a}.$$
>
> <!-- bilingual-en:start -->
> “Almost surely” means with probability one: any outcomes on which $X<0$ together have probability zero. No variance, independence assumption or particular distributional shape is needed. Combining this with the fact that probabilities are at most one gives the usable bound $\min\{1,\mu/a\}$.
> <!-- bilingual-en:end -->
>
> “几乎必然”表示概率为一：所有使 $X<0$ 的结果合起来只占零概率。不需要方差、独立性假设或特定分布形状。再结合概率不超过一，可以把实际使用的上界写成 $\min\{1,\mu/a\}$。

> [!example] Ten minutes on average · 平均十分钟
> <!-- bilingual-en:start -->
> For a nonnegative wait with $E[X]=10$, the inequality gives $P(X\ge30)\le10/30=1/3$. This does not say that exactly one third of waits reach thirty minutes. A wait that is always ten minutes has tail probability zero. A different model, with waits zero and thirty minutes having probabilities $2/3$ and $1/3$, has the same mean and attains the bound.
> <!-- bilingual-en:end -->
>
> 若等待时间非负且 $E[X]=10$，就有 $P(X\ge30)\le10/30=1/3$。这并不表示恰好三分之一的等待达到三十分钟。如果每次都等十分钟，这个尾概率是零；如果以 $2/3$ 的概率等零分钟、以 $1/3$ 的概率等三十分钟，均值同样是十，尾概率却恰好达到上界。

<!-- bilingual-en:start -->
## Why the comparison works
<!-- bilingual-en:end -->
*为什么可以这样比较*

<!-- bilingual-en:start -->
Use the [[事件指示变量|indicator]] $I=\mathbf1_{\{X\ge a\}}$: it is one when the threshold is reached and zero otherwise. On the event, $aI=a\le X$. Outside the event, $aI=0\le X$ wherever $X$ is nonnegative. Therefore
<!-- bilingual-en:end -->

使用[[事件指示变量]] $I=\mathbf1_{\{X\ge a\}}$：达到门槛时它为一，否则为零。在事件内，$aI=a\le X$；在事件外，只要 $X$ 非负，就有 $aI=0\le X$。因此

$$aI\le X\quad\text{almost surely}.$$

<!-- bilingual-en:start -->
The comparison may fail on a probability-zero set if nonnegativity was assumed only almost surely. Such a set does not change either expectation. Taking expectations, and using that the [[指示变量期望|mean of an indicator is its event probability]], gives
<!-- bilingual-en:end -->

如果非负性只是假设几乎必然成立，这个比较在某个零概率集合上可能失败，但不影响两边的期望。取期望，再用[[指示变量期望|指示变量期望等于事件概率]]，得到

$$aP(X\ge a)=E[aI]\le E[X]=\mu.$$

<!-- bilingual-en:start -->
Dividing by the positive threshold proves the bound. If $\mu=0$, it says that every positive threshold has zero tail probability, and in fact $X=0$ almost surely. If $\mu>0$, setting $a=c\mu$ for $c>0$ gives $P(X\ge c\mu)\le1/c$; this is informative when $c>1$. The positive-mean condition matters because $a=0$ is not an allowed substitution.
<!-- bilingual-en:end -->

除以正门槛即可得到结论。如果 $\mu=0$，那么每个正门槛对应的尾概率都是零，事实上 $X=0$ 几乎必然成立。如果 $\mu>0$，取 $a=c\mu$、$c>0$，就得到 $P(X\ge c\mu)\le1/c$；$c>1$ 时才比概率不超过一更有信息。这里必须保留均值为正的条件，不能代入不合法的 $a=0$。

> [!warning] A nonnegative mean is not enough · 只有均值非负还不够
> <!-- bilingual-en:start -->
> Let $X=100$ with probability $0.1$, and $X=-100/9$ with probability $0.9$. Then $E[X]=0$, but $P(X\ge100)=0.1$. Negative values cancel positive contributions to the mean without eliminating the upper tail. Markov cannot be applied directly to this $X$.
> <!-- bilingual-en:end -->
>
> 令 $X$ 以 $0.1$ 的概率取 $100$、以 $0.9$ 的概率取 $-100/9$。此时 $E[X]=0$，但 $P(X\ge100)=0.1$。负值可以抵消均值中的正贡献，却没有消除右尾，因此不能直接对这个 $X$ 使用 Markov。

<!-- bilingual-en:start -->
For a nonnegative integer count $N$, the threshold $a=1$ gives $P(N\ge1)\le E[N]$: a small expected count limits the probability of at least one occurrence. If a deterministic lower bound is known, [[平移Markov界|shifting before applying Markov]] can help. Applying it to a squared distance from the mean instead leads to [[Chebyshev不等式|Chebyshev's inequality]].
<!-- bilingual-en:end -->

对非负整数计数 $N$，取 $a=1$ 得 $P(N\ge1)\le E[N]$：期望次数很小，会限制至少出现一次的概率。若知道确定下界，可以考虑[[平移Markov界|先平移再用 Markov]]；若改为对离均值的平方距离使用它，则得到[[Chebyshev不等式]]。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=924|MIT Mathematics for Computer Science, Theorem 20.1.1, printed p. 916]]: the nonnegative-variable inequality, positive threshold and indicator proof; the neighbouring example explains attainability. Corollary 20.1.2 is used here with an explicitly positive mean.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=2|MIT 6.042J Markov Bounds, PDF pp. 1–2]]: the course formulation and role of nonnegativity. The almost-sure version follows by making the comparison outside the null exceptional set. The waiting-time and negative-value examples are calculated directly.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=924|MIT Mathematics for Computer Science，定理 20.1.1，印刷页 916]]：支持非负变量的界、正门槛及指示变量证明；相邻例子说明上界可达到。本文使用推论 20.1.2 时显式保留均值为正的条件。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=2|MIT 6.042J Markov Bounds，PDF 第 1–2 页]]：支持课程表述及非负性的作用。几乎必然版本通过在零概率例外集之外比较得到；等待时间与负值反例直接核算。

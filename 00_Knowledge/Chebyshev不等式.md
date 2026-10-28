---
aliases:
  - "Chebyshev's inequality"
  - "Chebyshev inequality"
  - "切比雪夫不等式"
student_os: knowledge-atom
atom_id: PROB-CONC-005
atom_set: probability-concentration
atom_type: theorem
status: source-checked
requires:
  - "[[Markov不等式]]"
  - "[[方差]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[和的方差协方差项]]"
---

<!-- bilingual-en:start -->
# Chebyshev's inequality
<!-- bilingual-en:end -->
*Chebyshev 不等式*

<!-- bilingual-en:start -->
**Chebyshev's inequality** bounds the chance of being far from the mean using the variance. A mean locates the centre; a [[方差|variance]] averages squared distances from that centre. If this average squared distance is small, large distances cannot occur too often. The bound counts deviations on either side of the mean.
<!-- bilingual-en:end -->

**Chebyshev 不等式**利用方差，限制取值离均值很远的概率。均值给出中心，[[方差]]则平均各个取值离中心的平方距离。平均平方距离很小，就不允许大偏离过于频繁。这条界同时计算均值两侧的偏离。

<!-- bilingual-en:start -->
Write $\mu=E[X]$ for the finite [[期望|mean]] and $\sigma^2=\operatorname{Var}(X)$ for the variance. The [[标准差|standard deviation]] is $\sigma=\sqrt{\sigma^2}$. The absolute value $|X-\mu|$ is the distance from the mean, so a threshold $t>0$ means “at least $t$ away”, either above or below.
<!-- bilingual-en:end -->

用 $\mu=E[X]$ 表示有限[[期望|均值]]，用 $\sigma^2=\operatorname{Var}(X)$ 表示方差，[[标准差]]为 $\sigma=\sqrt{\sigma^2}$。绝对值 $|X-\mu|$ 表示离均值的距离，所以正门槛 $t$ 的含义是“至少偏离 $t$”，可以向上，也可以向下。

> [!abstract] A two-sided bound · 双侧偏离的上界
> <!-- bilingual-en:start -->
> If $\sigma^2<\infty$, then for every $t>0$,
> <!-- bilingual-en:end -->
>
> 若 $\sigma^2<\infty$，则对每个 $t>0$，
>
> $$P(|X-\mu|\ge t)\le\frac{\sigma^2}{t^2}.$$
>
> <!-- bilingual-en:start -->
> The variable $X$ may take negative values. No normal distribution or independence assumption is needed. The right side is an upper bound, not an exact probability; if it exceeds one, replace it with one.
> <!-- bilingual-en:end -->
>
> $X$ 可以取负值，不需要正态分布或独立性假设。右边是上界，不是准确概率；如果它超过一，可以改用一。

> [!example] A twenty-minute deviation · 偏离二十分钟
> <!-- bilingual-en:start -->
> Suppose a duration has $\mu=50$ minutes and variance $\sigma^2=25$ square minutes, so its standard deviation is five minutes. The event $|X-50|\ge20$ contains durations at most thirty or at least seventy minutes. Therefore
> <!-- bilingual-en:end -->
>
> 设一段时长的均值为 $\mu=50$ 分钟，方差为 $\sigma^2=25$ 平方分钟，所以标准差为五分钟。事件 $|X-50|\ge20$ 包含不超过三十分钟、或至少七十分钟的时长。因此
>
> $$P(|X-50|\ge20)\le\frac{25}{20^2}=\frac1{16}.$$
>
> <!-- bilingual-en:start -->
> The event $X\ge70$ is part of that two-sided event, so $P(X\ge70)\le1/16$ as well. This is a valid one-sided bound, but it may be loose because the calculation also allows for the lower tail. Equivalently, $P(30<X<70)\ge15/16$; the strict endpoints follow from taking the complement of the displayed event.
> <!-- bilingual-en:end -->
>
> 事件 $X\ge70$ 是这个双侧事件的一部分，所以也有 $P(X\ge70)\le1/16$。这个单侧上界合法，但计算还为左尾留出了空间，因此可能较松。等价地，$P(30<X<70)\ge15/16$；这里端点用严格不等号，是因为取了上面事件的补集。

<!-- bilingual-en:start -->
## Turn distance into a nonnegative variable
<!-- bilingual-en:end -->
*把距离变成非负变量*

<!-- bilingual-en:start -->
To see why the result holds, apply [[Markov不等式|Markov's inequality]] to $Y=(X-\mu)^2$. This new variable is nonnegative even when $X$ is negative, and its expectation is exactly the variance. Since $t>0$, the events $|X-\mu|\ge t$ and $Y\ge t^2$ are identical:
<!-- bilingual-en:end -->

要理解结论为何成立，对 $Y=(X-\mu)^2$ 使用[[Markov不等式]]。即使 $X$ 为负，这个新变量仍非负，且它的期望恰好就是方差。因为 $t>0$，事件 $|X-\mu|\ge t$ 与 $Y\ge t^2$ 完全相同：

$$P(|X-\mu|\ge t)=P(Y\ge t^2)\le\frac{E[Y]}{t^2}=\frac{\sigma^2}{t^2}.$$

<!-- bilingual-en:start -->
The squared denominator matches the squared distance being averaged. The units of $t$, $X$ and $\sigma$ agree; $t^2$ and the variance have the corresponding squared units. Their ratio is dimensionless, as a probability bound must be.
<!-- bilingual-en:end -->

分母平方，是因为被平均的量本身就是平方距离。$t$、$X$ 与 $\sigma$ 的单位相同，$t^2$ 和方差则带相应的平方单位，所以二者之比没有单位，才能作为概率上界。

<!-- bilingual-en:start -->
## Measuring deviation in standard deviations
<!-- bilingual-en:end -->
*按标准差衡量偏离*

<!-- bilingual-en:start -->
When $\sigma>0$, set $t=k\sigma$ with $k>0$. The formula becomes $P(|X-\mu|\ge k\sigma)\le1/k^2$. For $k=2,3,4$, the upper bounds are $1/4,1/9,1/16$. These are general guarantees, not the familiar normal-distribution percentages. Knowing only finite variance does not specify the distribution's shape.
<!-- bilingual-en:end -->

当 $\sigma>0$ 时，取 $t=k\sigma$、$k>0$，公式变成 $P(|X-\mu|\ge k\sigma)\le1/k^2$。$k=2,3,4$ 时，上界依次为 $1/4,1/9,1/16$。这是一般保证，并不是正态分布中常见的那些百分比；只知道方差有限，无法确定分布形状。

> [!warning] Zero variance and missing moments · 零方差与矩不存在的边界
> <!-- bilingual-en:start -->
> If $\sigma=0$, then $X=\mu$ almost surely, and every positive-distance event has probability zero. Do not substitute $t=k\sigma=0$ into a theorem that requires $t>0$: the event $|X-\mu|\ge0$ has probability one. If the variance is infinite or undefined, this finite-variance theorem gives no nontrivial bound; check the [[矩存在性的使用边界|required moments]] first.
> <!-- bilingual-en:end -->
>
> 若 $\sigma=0$，则 $X=\mu$ 几乎必然成立，每个正距离事件的概率都是零。不能把 $t=k\sigma=0$ 代入要求 $t>0$ 的定理，因为事件 $|X-\mu|\ge0$ 的概率反而是一。如果方差无限或没有定义，这条有限方差定理就不给出非平凡上界，应先检查[[矩存在性的使用边界|所需的矩]]。

<!-- bilingual-en:start -->
For a random sum, the remaining work is to calculate its variance correctly. The [[和的方差协方差项|variance-of-a-sum expansion]] includes covariance terms; Chebyshev itself does not remove them. Independence can help with that calculation, but it is not a premise of the inequality for a single $X$.
<!-- bilingual-en:end -->

面对随机和，另一个任务是正确算出它的方差。[[和的方差协方差项]]包含交叉协方差，Chebyshev 本身不会把它们消掉。独立性可以帮助计算方差，但它不是对单个 $X$ 使用这条不等式的前提。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=927|MIT Mathematics for Computer Science, Theorem 20.2.3 and Corollary 20.2.6, printed pp. 919 and 921]]: the squared-deviation derivation and standard-deviation form. Here the finite-moment, positive-threshold and positive-standard-deviation conditions are stated explicitly.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf#page=1|MIT 6.042J Chebyshev Bounds, PDF pp. 1–3]]: the transformation and bounds at two, three and four standard deviations. The duration example, units and zero-variance boundary are checked directly.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=927|MIT Mathematics for Computer Science，定理 20.2.3 与推论 20.2.6，印刷页 919、921]]：支持平方偏差推导和标准差倍数形式；本文显式说明有限矩、正门槛及标准差为正的条件。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf#page=1|MIT 6.042J Chebyshev Bounds，PDF 第 1–3 页]]：支持变换及二、三、四个标准差的上界。时长例子、量纲和零方差边界直接核验。

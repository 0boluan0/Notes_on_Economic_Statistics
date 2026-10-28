---
aliases:
  - "Shifted Markov inequality"
  - "Shifted Markov bound"
  - "平移 Markov 界"
student_os: knowledge-atom
atom_id: PROB-CONC-004
atom_set: probability-concentration
atom_type: procedure
status: source-checked
requires:
  - "[[Markov不等式]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
---

<!-- bilingual-en:start -->
# Shifted Markov bound
<!-- bilingual-en:end -->
*平移 Markov 界*

<!-- bilingual-en:start -->
A **shifted Markov bound** uses a known baseline before applying [[Markov不等式|Markov's inequality]]. If a job always takes at least one hundred minutes, those first hundred minutes cannot explain why some jobs take unusually long. Subtracting the guaranteed baseline leaves a nonnegative excess, whose mean can give a sharper bound on long completion times.
<!-- bilingual-en:end -->

**平移 Markov 界**先利用一个已知基线，再应用[[Markov不等式]]。如果一项工作至少耗时一百分钟，那么必需的这一百分钟不能解释为什么某些工作格外久。减去这段必需时间，剩下非负的超出量，就可以用它的均值更准确地限制长耗时的概率。

> [!example] Remove a known baseline · 减去已知基线
> <!-- bilingual-en:start -->
> Suppose a completion time $X$ has mean $E[X]=150$ minutes and satisfies $X\ge100$ with probability one. We want to bound $P(X\ge200)$. Direct Markov gives $150/200=3/4$. Set $Y=X-100$. Its mean is fifty minutes, and the same target event becomes $Y\ge100$, so
> <!-- bilingual-en:end -->
>
> 设完成时间 $X$ 的均值为 $E[X]=150$ 分钟，且 $X\ge100$ 以概率一成立。要限制 $P(X\ge200)$，直接用 Markov 得 $150/200=3/4$。改令 $Y=X-100$，其均值为五十分钟，同一个目标事件改写成 $Y\ge100$，于是
>
> $$P(X\ge200)=P(Y\ge100)\le\frac{50}{100}=\frac12.$$
>
> <!-- bilingual-en:start -->
> The numerator and the threshold must both be shifted. The event itself has not changed, and the extra information is the verified lower bound.
> <!-- bilingual-en:end -->
>
> 分子和门槛必须同时平移。事件本身没有改变，多用的信息就是已经确认的下界。

<!-- bilingual-en:start -->
## Transform the variable and the event together
<!-- bilingual-en:end -->
*变量与事件一起变换*

<!-- bilingual-en:start -->
Let $b$ be a fixed finite lower bound, so $P(X\ge b)=1$, and write $\mu=E[X]$ for the finite [[期望|mean]]. The new variable $Y=X-b$ is nonnegative almost surely. [[期望线性性|Linearity of expectation]] gives $E[Y]=\mu-b$. For a target $a>b$, the shifted threshold $a-b$ is positive, and
<!-- bilingual-en:end -->

设 $b$ 是固定的有限下界，即 $P(X\ge b)=1$，用 $\mu=E[X]$ 表示有限[[期望|均值]]。新变量 $Y=X-b$ 几乎必然非负，[[期望线性性]]给出 $E[Y]=\mu-b$。若目标 $a>b$，平移后的门槛 $a-b$ 为正，并且

$$\{X\ge a\}=\{X-b\ge a-b\}.$$

> [!abstract] The shifted bound · 平移后的上界
> <!-- bilingual-en:start -->
> Under these conditions, Markov applied to $X-b$ gives
> <!-- bilingual-en:end -->
>
> 在这些条件下，对 $X-b$ 使用 Markov 得到
>
> $$P(X\ge a)\le\frac{\mu-b}{a-b}.$$
>
> <!-- bilingual-en:start -->
> The lower-bound assumption ensures $\mu-b\ge0$. As with any probability bound, cap the right side at one. If $a\le b$, the target already occurs with probability one; the displayed division is not the way to handle that case.
> <!-- bilingual-en:end -->
>
> 下界条件保证 $\mu-b\ge0$。同其他概率上界一样，右边可以与一取较小值。如果 $a\le b$，目标事件已经以概率一发生，应直接得到概率一，不能套用上面的除法。

<!-- bilingual-en:start -->
When $b>0$ and $a>\mu$, this is strictly smaller than the direct bound $\mu/a$. The comparison is
<!-- bilingual-en:end -->

当 $b>0$ 且 $a>\mu$ 时，它严格小于直接上界 $\mu/a$。比较两者可得

$$\frac{\mu-b}{a-b}-\frac{\mu}{a}=\frac{b(\mu-a)}{a(a-b)}<0.$$

<!-- bilingual-en:start -->
If $b<a\le\mu$, both usable bounds are just one, so a positive baseline alone does not promise a useful improvement. A negative lower bound can have a different purpose: making Markov applicable at all. For example, if $X\ge-10$, $E[X]=0$ and $a=10$, shifting gives $P(X\ge10)\le(0+10)/(10+10)=1/2$, whereas direct Markov on a potentially negative $X$ is invalid.
<!-- bilingual-en:end -->

若 $b<a\le\mu$，两个可用上界都只剩下一，所以仅有正基线并不保证实质改善。负下界则可能发挥另一种作用：让原本不能使用 Markov 的变量变得可用。例如 $X\ge-10$、$E[X]=0$、$a=10$ 时，平移给出 $P(X\ge10)\le(0+10)/(10+10)=1/2$；直接对可能为负的 $X$ 使用 Markov 却不合法。

> [!warning] A likely floor is not a lower bound · 通常不低于不等于下界
> <!-- bilingual-en:start -->
> The assumption is $P(X\ge b)=1$. A sample minimum, a quantile or a statement such as $P(X\ge0)=0.99$ does not establish it. Even a small positive probability below $b$ can make $X-b$ negative, breaking the required comparison. Choose $b$ from a justified property of the model, not from a convenient observed value.
> <!-- bilingual-en:end -->
>
> 所需条件是 $P(X\ge b)=1$。样本最小值、分位数，或 $P(X\ge0)=0.99$ 这样的说法，都不能证明它。只要低于 $b$ 的概率仍为正，$X-b$ 就可能为负，从而破坏比较所需的条件。应根据模型中有依据的性质选择 $b$，不能为方便计算拿一个观测值充当下界。

<!-- bilingual-en:start -->
A known upper bound gives the corresponding lower-tail method. If $X\le u$ almost surely and $a<u$, apply Markov to the nonnegative variable $u-X$: $P(X\le a)\le(u-\mu)/(u-a)$. In both directions, first make the variable nonnegative, then rewrite the target event and check that its new threshold is positive.
<!-- bilingual-en:end -->

已知上界时，也有对应的左尾方法。若 $X\le u$ 几乎必然成立且 $a<u$，对非负变量 $u-X$ 用 Markov，就得 $P(X\le a)\le(u-\mu)/(u-a)$。两个方向都要先构造非负变量，再改写目标事件，并检查新门槛为正。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=925|MIT Mathematics for Computer Science, §20.1.2, printed pp. 917–918]] and [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session33.pdf|Session 33, §19.1.2]]: the lower-bound shift and the corresponding upper-bound transformation.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=3|MIT Markov Bounds, PDF p. 3]]: shifting both the variable and the threshold.
- [MIT 6.1200J Lecture 24 (2024), §3.1, p. 3](https://www.ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/mit6_1200j_s24_lec24.pdf#page=3): lower/upper shifts, including variables that may be negative. The strict-improvement condition and examples here are checked algebraically.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=925|MIT Mathematics for Computer Science，第 20.1.2 节，印刷页 917–918]] 与 [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session33.pdf|Session 33，第 19.1.2 节]]：支持下界平移及上界反向变换。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=3|MIT Markov Bounds，PDF 第 3 页]]：支持同时平移变量和门槛。
- [MIT 6.1200J Lecture 24（2024），第 3.1 节，第 3 页](https://www.ocw.mit.edu/courses/6-1200j-mathematics-for-computer-science-spring-2024/mit6_1200j_s24_lec24.pdf#page=3)：支持下界/上界变换，包括变量可能为负的情况。本文严格改善的条件及例子均直接代数核对。

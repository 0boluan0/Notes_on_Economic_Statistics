---
student_os: knowledge-atom
atom_id: PROB-LIM-019
aliases:
- Fatou's lemma
- Fatou’s lemma
- 法图引理
atom_type: theorem
status: source-checked
---

<!-- bilingual-en:start -->
# Fatou's lemma
<!-- bilingual-en:end -->

*Fatou 引理*

<!-- bilingual-en:start -->
A sequence of nonnegative random quantities may settle down on almost every outcome while keeping some of its average in increasingly rare, increasingly large values. **Fatou's lemma** tells us which inequality remains valid: the average of the eventual lower-limit values cannot exceed the lower limit of the averages. Equality needs more control.
<!-- bilingual-en:end -->

一列非负随机量可能在几乎每个结果上都逐渐稳定，却把一部分平均贡献留在越来越罕见、越来越大的数值中。**Fatou 引理**说明此时仍然成立的不等式方向：逐结果的下极限再取平均，不会超过这些平均数的下极限。若要得到等号，还需要更多控制。

<!-- bilingual-en:start -->
## Read a lower limit one outcome at a time
<!-- bilingual-en:end -->

*逐个结果理解下极限*

<!-- bilingual-en:start -->
For an ordinary sequence $a_1,a_2,\ldots$, look at all terms from position $m$ onward and take their greatest lower bound, $\inf_{n\ge m}a_n$. As we discard more early terms, this tail lower bound cannot decrease. Its limit is called the **lower limit**, written $\liminf_n a_n$. For $1,2,1,2,\ldots$ it equals $1$, although the sequence has no ordinary limit. When an ordinary limit exists, the lower limit equals it.
<!-- bilingual-en:end -->

对普通数列 $a_1,a_2,\ldots$，从第 $m$ 项开始看所有后续项，取它们的下确界，即最大下界 $\inf_{n\ge m}a_n$。丢掉更多早期项后，这个尾部下界不会下降。它的极限叫作**下极限**，记为 $\liminf_n a_n$。数列 $1,2,1,2,\ldots$ 没有普通极限，但下极限等于 $1$；若普通极限存在，下极限就等于它。

<!-- bilingual-en:start -->
For [[随机变量|random variables]] on one [[概率空间|probability space]], first fix the outcome $\omega$ and apply this operation to the numerical path $X_n(\omega)$. The result $\liminf_n X_n$ is itself a measurable, possibly extended-valued random quantity. Only after constructing it do we take its [[期望|expectation]]. The expression $\liminf_n E[X_n]$ instead takes an ordinary lower limit of the numerical expectations.
<!-- bilingual-en:end -->

对同一[[概率空间]]上的[[随机变量]]，先固定结果 $\omega$，把这个操作应用到数值路径 $X_n(\omega)$。得到的 $\liminf_nX_n$ 本身仍是可测随机量，可以取扩展值；构造出它以后，再取[[期望]]。而 $\liminf_nE[X_n]$ 则是对数值期望构成的数列取普通下极限。这是两种不同的运算顺序。

> [!important] The inequality goes this way · 不等式的方向
> <!-- bilingual-en:start -->
> If $X_n\ge0$ almost surely for every $n$, then
> <!-- bilingual-en:end -->
>
> 若每个 $n$ 都有 $X_n\ge0$ 几乎必然成立，则
>
> $$
> E\!\left[\liminf_{n\to\infty}X_n\right]
> \le \liminf_{n\to\infty}E[X_n].
> $$
>
> <!-- bilingual-en:start -->
> Both sides are defined in $[0,+\infty]$. No finite-expectation, monotonicity, independence or ordinary-convergence assumption is needed. Since there are countably many $X_n$, their null exceptional sets can be combined into one null set. If $X_n\to X$ [[几乎必然收敛|almost surely]], the left side is simply $E[X]$.
> <!-- bilingual-en:end -->
>
> 两边都在 $[0,+\infty]$ 中有定义。不要求期望有限，不要求单调、独立或普通极限存在。由于 $X_n$ 只有可数多个，它们的零概率异常集可以合成一个零概率集合。若 $X_n$ [[几乎必然收敛]]到 $X$，左边就变成 $E[X]$。

> [!example] A disappearing spike can keep its average · 消失的尖峰仍能保留平均贡献
> <!-- bilingual-en:start -->
> Draw $U$ once from the [[连续均匀分布|uniform distribution on $(0,1)$]], and reuse the same draw for all $n$. Let the [[事件指示变量|indicator]] select the shrinking interval:
> <!-- bilingual-en:end -->
>
> 从 $(0,1)$ 的[[连续均匀分布]]抽取一次 $U$，之后所有 $n$ 都使用同一次抽到的数。用[[事件指示变量|指示变量]]选中逐渐缩小的区间：
>
> $$
> X_n=n\mathbf1_{\{U\le1/n\}}.
> $$
>
> <!-- bilingual-en:start -->
> For each fixed $U>0$, eventually $1/n<U$, so the values are eventually zero. However, $X_n=n$ on an event of probability $1/n$, so $E[X_n]=n(1/n)=1$ for every $n$. Fatou therefore gives the strict inequality
> <!-- bilingual-en:end -->
>
> 对每个固定的 $U>0$，最终都会有 $1/n<U$，所以数值最终为零。但 $X_n$ 以 $1/n$ 的概率取 $n$，因此每个 $n$ 都有 $E[X_n]=n(1/n)=1$。于是 Fatou 给出的确实可以是严格不等式：
>
> $$
> E[\liminf_nX_n]=0<1=\liminf_nE[X_n].
> $$
>
> <!-- bilingual-en:start -->
> The height grows as fast as the probability shrinks. The [[L1有界不推一致可积|bounded-first-moment counterexample]] examines the corresponding failure of tail control.
> <!-- bilingual-en:end -->
>
> 高度增长得与概率缩小得一样快。[[L1有界不推一致可积]]进一步分析了这个例子为何缺少一致的尾部控制。

<!-- bilingual-en:start -->
## Build an increasing sequence underneath
<!-- bilingual-en:end -->

*从下面构造一列递增近似*

<!-- bilingual-en:start -->
For each positive integer $m$, define $Y_m=\inf_{n\ge m}X_n$. This is the tail lower bound on each path, so $0\le Y_m\uparrow\liminf_nX_n$. Countable infima of measurable variables remain measurable. The [[单调收敛定理|monotone convergence theorem]] therefore gives
<!-- bilingual-en:end -->

对每个正整数 $m$，定义 $Y_m=\inf_{n\ge m}X_n$。它是每条路径的尾部下界，因此 $0\le Y_m\uparrow\liminf_nX_n$。可测变量的可数下确界仍然可测，所以[[单调收敛定理]]给出

$$
E[\liminf_nX_n]=\lim_{m\to\infty}E[Y_m].
$$

<!-- bilingual-en:start -->
For every $n\ge m$, $Y_m\le X_n$, hence $E[Y_m]\le E[X_n]$. Taking the infimum over those $n$ and then increasing $m$ yields
<!-- bilingual-en:end -->

对每个 $n\ge m$，都有 $Y_m\le X_n$，于是 $E[Y_m]\le E[X_n]$。先对这些 $n$ 取下确界，再令 $m$ 增大，得到

$$
E[Y_m]\le\inf_{n\ge m}E[X_n],
\qquad
\lim_m E[Y_m]\le\lim_m\inf_{n\ge m}E[X_n]
=\liminf_nE[X_n].
$$

<!-- bilingual-en:start -->
This proves the desired inequality. The tail-infimum construction explains the direction: each $Y_m$ lies below every remaining $X_n$, so its average can only give a lower bound.
<!-- bilingual-en:end -->

这就证明了所需不等式。尾部下确界的构造也解释了方向：每个 $Y_m$ 都位于所有剩余 $X_n$ 的下方，所以它的平均只能提供下界。

<!-- bilingual-en:start -->
## Why nonnegativity matters
<!-- bilingual-en:end -->

*非负条件为什么重要*

<!-- bilingual-en:start -->
Negate the spike example: $Z_n=-n\mathbf1_{\{U\le1/n\}}$ still converges almost surely to zero, but $E[Z_n]=-1$. The claimed inequality would now read $0\le-1$, which is false. A bound on average magnitudes alone does not repair the missing hypothesis.
<!-- bilingual-en:end -->

把尖峰反例取负号：$Z_n=-n\mathbf1_{\{U\le1/n\}}$ 仍然几乎必然趋于零，但 $E[Z_n]=-1$。若直接套用公式，就会得到错误的 $0\le-1$。仅仅知道平均绝对值有界，不能补回缺少的条件。

<!-- bilingual-en:start -->
There is a useful extension: if one integrable random variable $H$ satisfies $X_n\ge H$ almost surely for every $n$, apply Fatou to $X_n-H\ge0$ and add back the finite number $E[H]$. This gives the same inequality for the $X_n$, whose expectations may be $+\infty$ but cannot have an infinite negative part. A separate lower bound for each $n$ is not the same as one common integrable lower bound.
<!-- bilingual-en:end -->

一个有用的推广是：若存在同一个可积随机变量 $H$，使每个 $n$ 都有 $X_n\ge H$ 几乎必然成立，就对 $X_n-H\ge0$ 使用 Fatou，再加回有限数 $E[H]$。于是仍得到原来的不等式；此时 $X_n$ 的期望可以为 $+\infty$，但负部不可能有无穷期望。每个 $n$ 各自有一个下界，不等于有同一个可积下界。

<!-- bilingual-en:start -->
Fatou is particularly useful for proving that a limit is integrable. If $X_n\to X$ almost surely and $\sup_n E|X_n|\le C<\infty$, apply it to $|X_n|$ to obtain $E|X|\le C$. It still does not prove $E|X_n-X|\to0$: the positive spike already disproves that conclusion. The stronger [[控制收敛定理|dominated convergence theorem]] adds a common integrable envelope and obtains that average-error conclusion.
<!-- bilingual-en:end -->

Fatou 特别适合用来证明极限可积。若 $X_n$ 几乎必然趋于 $X$，且 $\sup_nE|X_n|\le C<\infty$，对 $|X_n|$ 应用它即可得到 $E|X|\le C$。但它仍不保证 $E|X_n-X|\to0$，上面的正尖峰就反驳了这个结论。更强的[[控制收敛定理]]加入同一个可积包络，才得到平均误差趋零的结论。

<!-- bilingual-en:start -->
The integral version replaces expectation with integration under a measure $\mu$: for nonnegative measurable $f_n$, $\int\liminf_n f_n\,d\mu\le\liminf_n\int f_n\,d\mu$. Probability is the special case $\mu=P$; the same lower-bound reasoning applies.
<!-- bilingual-en:end -->

积分版本把期望替换成按测度 $\mu$ 积分：对非负可测 $f_n$，有 $\int\liminf_nf_n\,d\mu\le\liminf_n\int f_n\,d\mu$。概率是 $\mu=P$ 的特例，同样使用从下方估计的思路。

## 来源与核验

<!-- bilingual-en:start -->
- [Durrett, Theorem 1.5.5 and Example 1.5.6, printed pp. 26–27 / PDF pp. 34–35](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=34): inequality direction, nonnegative hypothesis, tail infima and the shrinking-spike strict inequality. The signed common-lower-bound extension follows by subtracting one integrable variable.
<!-- bilingual-en:end -->

- Durrett 定理 1.5.5、例 1.5.6 核对不等号方向、非负条件、尾部下确界和收缩尖峰的严格不等式；共同可积下界的推广由减去一个可积变量直接得到。

<!-- bilingual-en:start -->
- [MIT 18.175, Lecture 5, slide 11](https://ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/1c016821763783c34c48816b1ac66969_MIT18_175S14_Lecture5.pdf#page=11): Fatou statement and tail-infimum construction; slide 12 connects the result with monotone and dominated convergence. The proof here uses the independently established monotone-convergence card.
<!-- bilingual-en:end -->

- MIT 18.175 第 5 讲第 11 页核对 Fatou 陈述与尾部下确界构造，第 12 页连接单调和控制收敛。本卡证明使用已经独立建立的单调收敛定理。

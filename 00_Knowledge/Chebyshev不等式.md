---
aliases:
  - "Chebyshev 不等式由平方偏差上的 Markov 界得到"
  - Chebyshev's inequality
  - Chebyshev inequality
  - 切比雪夫不等式
student_os: knowledge-atom
atom_id: PROB-CONC-005
atom_set: probability-concentration
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov不等式]]"
  - "[[方差]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[和的方差协方差项]]"
---

# Chebyshev 不等式由平方偏差上的 Markov 界得到
<!-- bilingual-en:start -->
*Chebyshev's inequality follows by applying Markov to squared deviation*
<!-- bilingual-en:end -->

> [!summary] 定理
> 若 $E[X]=\mu$ 且 $\operatorname{Var}(X)=\sigma^2<\infty$，则对 $t>0$，
> $$P(|X-\mu|\ge t)\le\frac{\sigma^2}{t^2}.$$
> 它不要求 $X$ 非负、独立或服从某个特定分布。
> <!-- bilingual-en:start -->
> If $E[X]=\mu$ and $\operatorname{Var}(X)=\sigma^2<\infty$, then for $t>0$,
> $$P(|X-\mu|\ge t)\le\frac{\sigma^2}{t^2}.$$
> It does not require $X$ itself to be non-negative, independent, or drawn from a particular family of distributions.
> <!-- bilingual-en:end -->

令 $Y=(X-\mu)^2\ge0$。事件 $|X-\mu|\ge t$ 与 $Y\ge t^2$ 完全等价，因此 Markov 给

$$
P(|X-\mu|\ge t)
=P(Y\ge t^2)
\le\frac{E[Y]}{t^2}
=\frac{\operatorname{Var}(X)}{t^2}.
$$

这也解释了分母为何必须是 $t^2$：方差的单位是 $X$ 的单位平方。若 $\sigma>0$ 且 $k>0$，取 $t=k\sigma$ 可写成 $P(|X-\mu|\ge k\sigma)\le1/k^2$。若 $\sigma=0$，则 $X=\mu$ almost surely；此时应直接处理任意正阈值，不能把 $t=0$ 代回要求 $t>0$ 的公式。
<!-- bilingual-en:start -->
Let $Y=(X-\mu)^2\ge0$. The events $|X-\mu|\ge t$ and $Y\ge t^2$ are identical, so Markov gives the displayed derivation. It also explains why the denominator must be $t^2$: variance has the squared units of $X$. If $\sigma>0$ and $k>0$, taking $t=k\sigma$ gives $P(|X-\mu|\ge k\sigma)\le1/k^2$. If $\sigma=0$, then $X=\mu$ almost surely; positive thresholds should be handled directly rather than substituting the inadmissible value $t=0$.
<!-- bilingual-en:end -->

若只关心右尾 $P(X-\mu\ge t)$，因为

$$
\{X-\mu\ge t\}\subseteq\{|X-\mu|\ge t\},
$$

Chebyshev 仍给合法上界，但它同时为不关心的左尾付出了代价，可能较松。若方差不存在或为无穷，这个公式不给有限保证。
<!-- bilingual-en:start -->
For a one-sided target $P(X-\mu\ge t)$, the displayed event inclusion makes Chebyshev a valid upper bound, but it pays for the irrelevant lower tail and may be loose. If the variance does not exist or is infinite, the formula provides no finite guarantee.
<!-- bilingual-en:end -->

例如 $\mu=50$、$\sigma^2=25$ 时，$X\ge70$ 蕴含 $|X-50|\ge20$，所以

$$
P(X\ge70)\le\frac{25}{20^2}=\frac1{16}.
$$
<!-- bilingual-en:start -->
For example, if $\mu=50$ and $\sigma^2=25$, then $X\ge70$ implies $|X-50|\ge20$, so
$$
P(X\ge70)\le\frac{25}{20^2}=\frac1{16}.
$$
<!-- bilingual-en:end -->

> [!question]- 自检
> Chebyshev 中的 threshold $t$ 与 $X$、标准差、方差分别是什么单位？
> <!-- bilingual-en:start -->
> In Chebyshev's inequality, how do the units of the threshold $t$ compare with those of $X$, the standard deviation, and the variance?
> <!-- bilingual-en:end -->
>
> **答案：** $t$、$X$ 与标准差单位相同；$t^2$ 与方差单位相同，所以比值无量纲。
> <!-- bilingual-en:start -->
> **Answer:** The threshold $t$, $X$, and the standard deviation have the same units. The squared threshold and the variance have the same squared units, so their ratio is dimensionless.
> <!-- bilingual-en:end -->

**继续：** 对随机和使用 Chebyshev 前，先在 [[和的方差协方差项]] 中确认方差是否真的可以相加。
<!-- bilingual-en:start -->
**Continue with:** Before applying Chebyshev to a random sum, use [[和的方差协方差项|the full variance-of-a-sum expansion]] to determine whether the variances really add.
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] 定理 20.2.3 与推论 20.2.6：核对有限方差条件、平方偏差推导以及标准差倍数形式。
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf|MIT 6.042J Chebyshev Bounds slides]]：核对课程公式及量纲解释。
<!-- bilingual-en:start -->
- Theorem 20.2.3 and Corollary 20.2.6 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verify the finite-variance theorem, squared-deviation derivation, and standard-deviation form.
- The [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf|MIT 6.042J Chebyshev Bounds slides]] verify the course formula and dimensional interpretation.
<!-- bilingual-en:end -->

---
aliases:
- Danskin's theorem
- Danskin theorem
- 最大值函数的方向导数
student_os: knowledge-atom
atom_id: OPT-MV-026
atom_set: multivariable-optimization
atom_type: theorem
status: source-checked
requires:
- '[[值函数]]'
- '[[最优解对应]]'
- '[[方向导数]]'
related:
- '[[包络定理]]'
- '[[影子价格]]'
leads_to:
- '[[优化器跳跃不推价值不可微]]'
part_of:
- '[[多元优化.canvas|多元优化]]'
---

<!-- bilingual-en:start -->
# Danskin's theorem
<!-- bilingual-en:end -->

*Danskin 定理*

<!-- bilingual-en:start -->
Suppose you choose the most profitable option, then a price or other parameter changes slightly. How quickly can the best achievable payoff change, especially when several options tie? Danskin’s theorem answers this by comparing the parameter slopes of the options that are optimal now. It does not first require a smooth rule for which option you will choose.
<!-- bilingual-en:end -->

设你总是挑收益最高的方案，现在价格或其他参数稍微改变。最好能获得的收益会以多快的速度变化，尤其是几个方案恰好并列最优时？Danskin 定理比较**当前最优方案各自的参数斜率**来回答，不要求先找到一条光滑的最优选择轨迹。

<!-- bilingual-en:start -->
Here $x$ labels a choice, $\theta\in\mathbb R^p$ is the parameter, and $f(x,\theta)$ is the payoff. The [[值函数|value function]] records the best payoff; the [[最优解对应|optimizer correspondence]] records every choice achieving it. A [[方向导数|directional derivative]] measures change along a specified parameter vector $d$ using a positive step $t\downarrow0$. Differentiating with respect to $\theta$ means holding the choice $x$ fixed.
<!-- bilingual-en:end -->

这里 $x$ 是方案，$\theta\in\mathbb R^p$ 是参数，$f(x,\theta)$ 是收益。[[值函数]]记录最好收益，[[最优解对应]]收集所有达到它的方案。[[方向导数]]沿给定参数方向 $d$，用正步长 $t\downarrow0$ 衡量变化。对 $\theta$ 求偏导时，暂时把方案 $x$ 固定。

<!-- bilingual-en:start -->
## A precise compact-choice version
<!-- bilingual-en:end -->

## 一个条件明确的紧选择集版本

<!-- bilingual-en:start -->
Fix a nonempty set $X\subset\mathbb R^n$ that does not depend on $\theta$. Assume $X$ is [[紧致性|compact]] and $f:X\times\mathbb R^p\to\mathbb R$ is continuous. Compactness and continuity ensure a finite attained maximum, so $S(\theta)$ is nonempty and compact. For every $x$, require $f(x,\cdot)$ to be [[凹凸性|convex]] and differentiable on $\mathbb R^p$, and require the parameter gradient $\nabla_\theta f$ to be jointly continuous in $(x,\theta)$. Convexity here concerns the **parameter**, not the choice variable; $X$ need not be convex or contain an interval.
<!-- bilingual-en:end -->

固定不随 $\theta$ 变化的非空集合 $X\subset\mathbb R^n$。假设 $X$ [[紧致性|紧致]]，且 $f:X\times\mathbb R^p\to\mathbb R$ 连续；这保证最大值有限且能够取得，因而 $S(\theta)$ 非空且紧。对每个 $x$，要求 $f(x,\cdot)$ 在 $\mathbb R^p$ 上[[凹凸性|凸]]且可微，参数梯度 $\nabla_\theta f$ 关于 $(x,\theta)$ 联合连续。这里的凸性针对**参数**，不是选择变量；$X$ 可以是离散集合，不必凸。

> [!important] The active slopes · 当前最优方案的斜率
> <!-- bilingual-en:start -->
> Let $X$ be nonempty and compact, with $v(\theta)=\max_{x\in X}f(x,\theta)$ and optimizer set $S(\theta)$. Suppose $f$ is continuous, each $f(x,\cdot)$ is convex and differentiable, and $\nabla_\theta f$ is jointly continuous. Then $v'(\theta;d)=\max_{x\in S(\theta)}\nabla_\theta f(x,\theta)^Td$. Only current optimizers determine the first-order directional change in the value.
> <!-- bilingual-en:end -->
>
> 令 $X$ 为非空紧集，定义
> $$
> v(\theta)=\max_{x\in X}f(x,\theta),
> \qquad
> S(\theta)=\operatorname*{argmax}_{x\in X}f(x,\theta).
> $$
> 假设 $f$ 连续，$f(x,\cdot)$ 对每个 $x$ 凸且可微，并且 $\nabla_\theta f$ 联合连续。则对任意方向 $d$，
> $$
> v'(\theta;d)
> =\lim_{t\downarrow0}
> \frac{v(\theta+td)-v(\theta)}{t}
> =\max_{x\in S(\theta)}
> \nabla_\theta f(x,\theta)^Td.
> $$
> 只有当前最优解会决定价值的一阶方向变化。

<!-- bilingual-en:start -->
## Why the largest direct effect appears
<!-- bilingual-en:end -->

## 为什么是“最大的”直接效应

<!-- bilingual-en:start -->
All points in $S(\theta)$ share the same current value. After a small movement in direction $d$, their direct first-order changes may differ. The maximum-value function follows the branch with the largest first-order increase, which explains the maximum over active optimizers.
<!-- bilingual-en:end -->

在 $\theta$ 处，所有 $x\in S(\theta)$ 都给出同一当前价值。参数沿 $d$ 微小移动后，这些并列的最优选择可能产生不同的一阶增量。最大值函数会沿其中增加最快的那个分支移动，所以公式对活动最优解取最大。

<!-- bilingual-en:start -->
The phrase “current optimizers” matters: a losing option with a larger slope is not automatically relevant at first order. For a finite collection, a fixed payoff gap cannot disappear under an arbitrarily small movement. With infinitely many choices, compactness prevents a sequence of nearly winning choices from escaping; continuity puts every limiting winner in $S(\theta)$. The argument below makes that step precise.
<!-- bilingual-en:end -->

“当前最优”很关键：某个落后的方案斜率更大，并不自动决定一阶变化。有限个方案时，固定的收益差距不会在任意小的移动中消失；无限多个方案时，紧致性防止接近最优的方案逃向无穷，连续性保证其极限仍属于 $S(\theta)$。下面把这一步写清楚。

<!-- bilingual-en:start -->
## A proof by upper and lower comparisons
<!-- bilingual-en:end -->

## 用上下界比较证明

<!-- bilingual-en:start -->
Write $g(x)=\nabla_\theta f(x,\theta)$ and $m=\max_{x\in S(\theta)}g(x)^Td$. This maximum exists because the gradient is continuous and $S(\theta)$ is compact. Choose $a\in S(\theta)$ attaining $m$. Keeping that one choice available after the perturbation gives the lower comparison:
<!-- bilingual-en:end -->

记 $g(x)=\nabla_\theta f(x,\theta)$、$m=\max_{x\in S(\theta)}g(x)^Td$。由于梯度连续、$S(\theta)$ 紧，这个最大值能够取得。选取达到 $m$ 的 $a\in S(\theta)$，参数移动后仍可使用这个方案，于是有下界：

$$
\frac{v(\theta+td)-v(\theta)}t\ge\frac{f(a,\theta+td)-f(a,\theta)}t\longrightarrow m.
$$

<!-- bilingual-en:start -->
For the upper comparison, choose a new optimizer $x_t\in S(\theta+td)$. At the old parameter, $f(x_t,\theta)\le v(\theta)$, so the same difference quotient is at most $[f(x_t,\theta+td)-f(x_t,\theta)]/t$. Joint continuity of the gradient on the compact set $X$ times a short closed parameter segment gives a uniform expansion:
<!-- bilingual-en:end -->

对上界，取移动后的最优方案 $x_t\in S(\theta+td)$。在旧参数处它满足 $f(x_t,\theta)\le v(\theta)$，所以同一差商不超过 $[f(x_t,\theta+td)-f(x_t,\theta)]/t$。梯度在 $X$ 与一小段闭参数线段的乘积上联合连续，给出对所有 $x$ 一致的展开：

$$\nf(x,\theta+td)-f(x,\theta)=t g(x)^Td+t r_t(x),\qquad\sup_{x\in X}|r_t(x)|\longrightarrow0.
$$

<!-- bilingual-en:start -->
One can see the uniform remainder by integrating the gradient along the segment: $r_t(x)=\int_0^1[\nabla_\theta f(x,\theta+std)-g(x)]^Td\,ds$. Along any sequence $t\downarrow0$, [[紧致性的序列判别|compactness supplies a convergent subsequence]] of $x_t$, say $x_t\to\bar x$. Optimality gives $f(x_t,\theta+td)\ge f(z,\theta+td)$ for every fixed $z\in X$. Passing to the limit shows $f(\bar x,\theta)\ge f(z,\theta)$, hence $\bar x\in S(\theta)$. The upper comparison therefore has limiting slope at most $g(\bar x)^Td\le m$. Together the comparisons force the directional derivative to equal $m$.
<!-- bilingual-en:end -->

一致余项可以直接沿线段积分看出：$r_t(x)=\int_0^1[\nabla_\theta f(x,\theta+std)-g(x)]^Td\,ds$。对任意趋零的正步长序列，[[紧致性的序列判别|紧致性提供收敛子列]]，设该子列上 $x_t\to\bar x$。最优性给出：对每个固定的 $z\in X$，均有 $f(x_t,\theta+td)\ge f(z,\theta+td)$。取极限得 $f(\bar x,\theta)\ge f(z,\theta)$，故 $\bar x\in S(\theta)$。上界的极限斜率至多为 $g(\bar x)^Td\le m$，与下界合起来便得到结论。

<!-- bilingual-en:start -->
## When one ordinary gradient exists
<!-- bilingual-en:end -->

## 何时能得到普通梯度

<!-- bilingual-en:start -->
With a unique optimizer $x^*$, the formula reduces to $v'(\theta;d)=\nabla_\theta f(x^*,\theta)^Td$, and the value is differentiable with gradient $\nabla_\theta f(x^*,\theta)$ in this convex-parameter setting. No derivative of the optimizer is needed.
<!-- bilingual-en:end -->

若 $S(\theta)$ 只有一个元素 $x^*$，则公式化为
$$
v'(\theta;d)=\nabla_\theta f(x^*,\theta)^Td.
$$
在上述凸参数设定中，这给出 $v$ 的普通梯度 $\nabla v(\theta)=\nabla_\theta f(x^*,\theta)$。定理不要求先求 $dx^*/d\theta$。

<!-- bilingual-en:start -->
Uniqueness is sufficient, not necessary. Under the stated assumptions the value is differentiable exactly when all active parameter gradients agree. If they all equal $g$, the preceding compactness argument works uniformly for an arbitrary small vector $h$, yielding $v(\theta+h)-v(\theta)=g^Th+o(\|h\|)$. Conversely, if $v$ is differentiable, applying the directional formula to both $d$ and $-d$ forces every active gradient to have the same dot product with every $d$, hence to be $\nabla v(\theta)$. This conclusion uses the theorem’s uniform assumptions; existence of directional derivatives alone does not generally imply differentiability.
<!-- bilingual-en:end -->

唯一最优解是充分条件，不是必要条件。在本卡条件下，所有当前最优方案的参数梯度相同，恰好等价于值函数可微。若它们都等于 $g$，前面的紧致性论证可对任意小向量 $h$ 一致进行，得到 $v(\theta+h)-v(\theta)=g^Th+o(\|h\|)$。反过来，若 $v$ 可微，把方向公式同时用于 $d$ 和 $-d$，就迫使所有活动梯度对每个 $d$ 的内积都相同，从而都等于 $\nabla v(\theta)$。这里依赖定理的一致性条件；仅有各方向导数通常不足以推出可微。

<!-- bilingual-en:start -->
For example, with $X=\{-1,1\}$ and $f(x,\theta)=\theta^2$, both choices remain optimal and both parameter derivatives are $2\theta$. Nevertheless $v(\theta)=\theta^2$ is smooth. Compare this with the switching example in [[优化器跳跃不推价值不可微|optimizer jumps and value differentiability]].
<!-- bilingual-en:end -->

例如取 $X=\{-1,1\}$、$f(x,\theta)=\theta^2$，两个方案一直并列最优，参数导数都等于 $2\theta$，但 $v(\theta)=\theta^2$ 完全光滑。实际发生方案切换的例子见[[优化器跳跃不推价值不可微]]。

<!-- bilingual-en:start -->
## Minimal example: two active optimizers create a kink
<!-- bilingual-en:end -->

## 最小例子：两个活动最优解产生折点

> [!example] Two options · 两个方案
> <!-- bilingual-en:start -->
> For $X=\{-1,1\}$ and $f(x,\theta)=x\theta$, both choices are optimal at zero and their parameter gradients are $-1$ and $1$. Hence $v'(0;d)=\max\{-d,d\}=|d|$, consistent with the kink in $v(\theta)=|\theta|$. The directional derivative at $d=-1$ is $1$, whereas the ordinary scalar left derivative is $-1$; failure of linearity in $d$ shows that no ordinary gradient exists at zero.
> <!-- bilingual-en:end -->
>
> 取 $X=\{-1,1\}$ 与 $f(x,\theta)=x\theta$。在 $\theta=0$ 时，
> $$
> S(0)=\{-1,1\},
> $$
> 两个活动梯度分别为 $-1$ 和 $1$。因此
> $$
> v'(0;d)=\max\{-d,d\}=|d|.
> $$
> 特别地，$v'(0;1)=v'(0;-1)=1$，但价值 $v(\theta)=|\theta|$ 在零点没有普通导数。
>
> 这里要区分两种记号：方向 $d=-1$ 的方向导数是 $v'(0;-1)=1$；按标量变量通常定义的左导数则是
> $$
> \lim_{h\uparrow0}\frac{v(h)-v(0)}{h}=-1.
> $$
> 方向导数关于 $d$ 不是线性的，正是零点不存在一个普通梯度的信号。

<!-- bilingual-en:start -->
## Boundary of the formula
<!-- bilingual-en:end -->

## 公式的使用边界

<!-- bilingual-en:start -->
This version assumes a compact parameter-independent choice set and convex differentiable dependence on the parameter. Parameter-dependent constraints require the corresponding constrained envelope or perturbation theorem. Without compactness or attainment, the optimizer set may be empty and the active-optimizer formula cannot be used as written.
<!-- bilingual-en:end -->

这里采用的版本要求选择集紧且不随参数变化，并明确假设目标对参数凸且可微。若选择集 $X(\theta)$ 随参数变化，约束的直接参数效应必须通过相应的包络或扰动定理处理。若选择集不紧或最优值不取得，则 $S(\theta)$ 可能为空，上式也失去取最大的对象。

<!-- bilingual-en:start -->
Parameter convexity is the scope of the convex-analysis version recorded here, not a necessary hypothesis of every Danskin theorem. The uniform smooth comparison proof above also works without it. Convexity does additionally ensure that $v$ itself is convex: take the convex inequality for each fixed $x$, then maximize. Never replace “convex in $\theta$” by “concave in $x$” when checking this statement. Moving feasible sets instead belong to [[包络定理|constrained envelope results]].
<!-- bilingual-en:end -->

参数凸性是这里保留的凸分析版本的范围，并非所有 Danskin 定理都必须要求的条件；上面的一致光滑比较证明本身也适用于没有参数凸性的情况。参数凸性还保证 $v$ 凸：先对每个固定 $x$ 写凸性不等式，再取最大值即可。检查本版本时，不能把“关于 $\theta$ 凸”替换为“关于 $x$ 凹”。随参数移动的可行集应使用[[包络定理|约束包络结果]]。

> [!question] Recall · 自检
> <!-- bilingual-en:start -->
> With several current optimizers, why can one not report an arbitrary $\nabla_\theta f$ as the value's directional derivative?
> <!-- bilingual-en:end -->
>
> 在多个当前最优解中，为什么不能随便挑一个 $\nabla_\theta f$ 报告为价值的方向导数？
>
> <!-- bilingual-en:start -->
> **Answer:** Active optimizers can have different direct first-order effects. The maximum-value function follows the largest of those effects in the chosen direction.
> <!-- bilingual-en:end -->
>
> **答案：** 不同活动最优解可以有不同的直接一阶效应；最大值函数在该方向沿其中最大的一个变化。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

<!-- bilingual-en:start -->
- Bertsekas, *Convex Optimization Theory: A Summary*, Proposition 5.4.9 was checked for the compact-set convex-parameter version of Danskin's theorem and its active-maximizer directional derivative.
- Milgrom and Segal, “Envelope Theorems for Arbitrary Choice Sets,” Corollary 4, supports the compact smooth scalar-parameter extension and the comparison of active derivatives.
<!-- bilingual-en:end -->

- [Bertsekas, *Convex Optimization Theory: A Summary*, Proposition 5.4.9](https://web.mit.edu/dimitrib/OldFiles/www/convexdualitycondenced.pdf)：直接核对紧选择集、参数凸性、活动最优解与方向导数取最大的 Danskin 版本。
- [Milgrom and Segal, “Envelope Theorems for Arbitrary Choice Sets”](https://web.stanford.edu/~milgrom/publishedarticles/Envelope%20Theorems.pdf)：为经济学中更一般的最优值单侧导数提供补充边界。

<!-- bilingual-en:start -->
Source locations: Bertsekas Proposition 5.4.9 is on PDF51–52 (printed517–518), with the convex-parameter statement and the unique-optimizer conclusion. Milgrom–Segal Corollary4 is on PDF11–12 (printed592–593), supporting the compact smooth scalar-parameter extension and agreement of active derivatives. The former ASU PDF URL returned404 during this review; the author’s MIT copy above was opened instead. The proof and elementary examples here were worked out directly.
<!-- bilingual-en:end -->

定位核验：Bertsekas 命题5.4.9 位于PDF第51–52页（印刷页517–518），支持参数凸的版本与唯一最优解结论；Milgrom–Segal 推论4 位于PDF第11–12页（印刷页592–593），支持紧选择集上光滑标量参数的推广及活动导数一致的判据。旧ASU链接本次返回404，已改用并实际打开作者的MIT副本。本卡证明和初等例子另行推导核算。

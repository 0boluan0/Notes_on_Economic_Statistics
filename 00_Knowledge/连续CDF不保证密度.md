---
aliases:
  - "A continuous CDF need not have a density"
  - "连续 CDF 与密度的区别"
student_os: knowledge-atom
atom_id: PROB-RV-018
atom_set: random-variables-distributions-moments
atom_type: boundary
status: source-checked
requires:
  - "[[累积分布函数]]"
  - "[[概率密度函数]]"
related:
  - "[[PMF不等于密度]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

<!-- bilingual-en:start -->
# A continuous CDF without a density
<!-- bilingual-en:end -->
*连续 CDF 为什么仍可能没有密度*

<!-- bilingual-en:start -->
A cumulative distribution function can rise continuously from $0$ to $1$ without being the accumulated area under any probability density. **Having no jumps** and **being representable by a density** are different requirements. The Cantor distribution gives a concrete counterexample: each exact value has probability zero, yet all the probability lies in a set of total length zero.
<!-- bilingual-en:end -->

累积分布函数可以从 $0$ 连续地上升到 $1$，却无法表示成任何概率密度下的累积面积。**没有跳跃**和**能够用密度表示**是两个不同要求。Cantor 分布给出了具体反例：每个精确取值的概率都为零，全部概率却集中在一个总长度为零的集合里。

<!-- bilingual-en:start -->
The [[累积分布函数|CDF]] of a real-valued random variable $X$ is $F(x)=P(X\le x)$: it records the probability accumulated up to threshold $x$. A [[概率密度函数|density]] $f$ represents this accumulation through area:
<!-- bilingual-en:end -->

实值随机变量 $X$ 的[[累积分布函数|CDF]] 是 $F(x)=P(X\le x)$，记录累积到门槛 $x$ 为止的概率。[[概率密度函数|密度]] $f$ 则用面积表示这种累积：

$$F(x)=\int_{-\infty}^{x}f(t)\,dt,\qquad f\ge0,\qquad \int_{-\infty}^{\infty}f(t)\,dt=1.$$

<!-- bilingual-en:start -->
Here $t$ is the variable being integrated. A distribution with this representation is called **absolutely continuous** with respect to length, formally Lebesgue measure. For a CDF, absolute continuity is stronger than ordinary continuity. The construction below shows how that difference can arise.
<!-- bilingual-en:end -->

这里 $t$ 是被积分的变量。能够这样表示的分布称为相对于长度的**绝对连续分布**；这里的长度严格说是 Lebesgue 测度。对于 CDF，绝对连续比通常的连续更强。下面用一个具体构造解释两者的差别。

<!-- bilingual-en:start -->
## A set whose remaining length tends to zero
<!-- bilingual-en:end -->
*剩余长度逐渐趋于零的集合*

<!-- bilingual-en:start -->
Start with $[0,1]$ and remove its open middle third $(1/3,2/3)$, retaining the endpoints. Two closed intervals remain: $[0,1/3]$ and $[2/3,1]$. Remove the open middle third of each of these, leaving four intervals of length $1/9$. Continue the same operation inside every remaining interval.
<!-- bilingual-en:end -->

从 $[0,1]$ 开始，删去中间三分之一的开区间 $(1/3,2/3)$，保留端点。剩下两个闭区间：$[0,1/3]$ 与 $[2/3,1]$。再从每个区间中删去中间三分之一，得到四个长度为 $1/9$ 的区间。此后在每个剩余区间内不断重复。

<!-- bilingual-en:start -->
After $n$ rounds there are $2^n$ intervals, each of length $3^{-n}$. Their combined length is $(2/3)^n$, which tends to zero. The points never removed form the **Cantor set**, denoted $C$. Since $C$ fits inside the remaining intervals at every round, its length is bounded by $(2/3)^n$ for every $n$, and hence is zero. It is still nonempty: for instance, $0$ and $1$ are never removed.
<!-- bilingual-en:end -->

经过 $n$ 轮后，剩下 $2^n$ 个区间，每个长度为 $3^{-n}$，总长度为 $(2/3)^n$，趋于零。所有始终没有被删掉的点组成 **Cantor 集**，记为 $C$。因为 $C$ 包含在每一轮剩余的区间中，所以它的长度对每个 $n$ 都不超过 $(2/3)^n$，只能为零。但它仍然非空，例如 $0$ 和 $1$ 始终保留。

<!-- bilingual-en:start -->
## Put all the probability on those remaining points
<!-- bilingual-en:end -->
*让全部概率落在这些剩余的点上*

<!-- bilingual-en:start -->
Now generate a random point by repeatedly choosing between the left and right remaining thirds, with probability $1/2$ for each choice, [[相互独立|independently]] at every round. Follow only the chosen interval and repeat inside it. The nested closed intervals shrink to a single point $X$. This point is never in a removed middle third, so $X$ always belongs to $C$ and $P(X\in C)=1$.
<!-- bilingual-en:end -->

现在按以下方式生成随机点：每一轮以各 $1/2$ 的概率选择左侧或右侧剩余的三分之一区间，各轮选择[[相互独立]]。只沿着选中的区间继续，在其中重复同样的操作。这些一层套一层的闭区间最终缩到一个点 $X$。这个点始终不在被删去的中间部分，因此一定属于 $C$，也就是 $P(X\in C)=1$。

<!-- bilingual-en:start -->
After $n$ choices, each of the $2^n$ possible intervals has probability $(1/2)^n$. Any fixed point $x\in C$ lies in exactly one of those intervals, so
<!-- bilingual-en:end -->

经过 $n$ 次选择后，$2^n$ 个可能区间各有 $(1/2)^n$ 的概率。对于任意固定的 $x\in C$，它恰好属于其中一个区间，因此

$$0\le P(X=x)\le 2^{-n}\quad\text{for every }n,\qquad\text{so }P(X=x)=0.$$

<!-- bilingual-en:start -->
Points outside $C$ also have probability zero. Thus every point has zero probability. A CDF's jump at $x$ equals $P(X=x)$, so this distribution's CDF has no jumps and is continuous. Its total increase is still one: $F(0)=0$ and $F(1)=1$.
<!-- bilingual-en:end -->

$C$ 外的点同样只有零概率。因此，每个单点的概率都是零。CDF 在 $x$ 处的跳跃大小等于 $P(X=x)$，所以这个分布的 CDF 没有跳跃，是连续的。它的总增长仍然为一：$F(0)=0$，$F(1)=1$。

> [!abstract] Why no density can represent this law · 为什么它不可能有密度
> <!-- bilingual-en:start -->
> If a density $f$ existed, the probability assigned to $C$ would be its integral over $C$. An integrable density assigns zero probability to any set of length zero. This would give
> <!-- bilingual-en:end -->
>
> 假如存在密度 $f$，分配给 $C$ 的概率就应当等于在 $C$ 上对 $f$ 的积分。任何可积密度都会给长度为零的集合分配零概率，因此将有
>
> $$P(X\in C)=\int_C f(t)\,dt=0.$$
>
> <!-- bilingual-en:start -->
> But the construction gives $P(X\in C)=1$. This contradiction rules out an ordinary density, meaning a density with respect to Lebesgue measure. The distribution is called **singular continuous**: its CDF is continuous, and its probability is concentrated on a set of zero length.
> <!-- bilingual-en:end -->
>
> 但刚才的构造给出 $P(X\in C)=1$。这个矛盾排除了通常意义下的密度，也就是相对于 Lebesgue 测度的密度。这种分布叫作**奇异连续分布**：CDF 连续，而概率集中在一个零长度集合上。

<!-- bilingual-en:start -->
Length and probability must remain distinct here. A point selected uniformly from $[0,1]$ lands in $C$ with probability zero; the left-or-right construction lands there with probability one. The set is the same, but the probability law is different. Likewise, [[零概率不等于不可能|zero probability does not imply an empty event]].
<!-- bilingual-en:end -->

这里必须分清长度与概率。在 $[0,1]$ 上均匀选点时，落入 $C$ 的概率为零；按刚才的左右选择规则生成点时，落入 $C$ 的概率为一。集合没有变，概率分布变了。同样，[[零概率不等于不可能|零概率也不意味着事件为空]]。

> [!warning] Differentiating is not enough · 只求导还不够
> <!-- bilingual-en:start -->
> In every removed open interval the CDF is constant, since there is no probability there. Within $[0,1]$, these intervals account for all the length except the zero-length set $C$; outside $[0,1]$, the CDF is also constant. Thus $F'=0$ almost everywhere. Yet $F$ rises from $0$ to $1$: integrating that derivative cannot recover it. To obtain a density from a CDF by differentiation, verify absolute continuity, which guarantees the integral representation.
> <!-- bilingual-en:end -->
>
> 在每个被删去的开区间上，CDF 都是常数，因为那里没有概率。在 $[0,1]$ 内，这些区间占据了除零长度集合 $C$ 以外的全部长度；在 $[0,1]$ 外，CDF 同样是常数。因此 $F'=0$ 几乎处处成立。但 $F$ 仍从 $0$ 上升到 $1$，对这个导数积分无法恢复原函数。要通过 CDF 求导得到密度，应确认绝对连续性，从而保证积分表示成立。

<!-- bilingual-en:start -->
Discrete distributions, distributions with densities, mixtures and singular continuous distributions all have CDFs. The [[PMF不等于密度|choice between a PMF and a PDF]] therefore does not exhaust all possible probability laws; the CDF remains available when neither one represents the whole law.
<!-- bilingual-en:end -->

离散分布、有密度的分布、混合分布与奇异连续分布都有 CDF。因此，[[PMF不等于密度|PMF 与 PDF]] 并不是穷尽所有概率分布的二选一；当两者都无法表示完整分布时，CDF 仍然适用。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J Lecture 4, §4 and Appendix B, pp. 10–14](https://live.ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/9797310bed4c7f5b5d40d007783eec8d_MIT6_436JF18_lec04.pdf#page=13): supports the density representation, zero-length sets under a density, and the Cantor-set and independent-coin construction. The finite-round length and point-probability bounds above make that construction explicit.
- [Durrett, *Probability: Theory and Examples*, §1.2, Example 1.2.7, printed pp. 12–14](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=20): supports absolute continuity, the Cantor CDF, its flat intervals, and the failure of a density representation.
- [MIT 18.175 Lecture 3, slides 6 and 11](https://www.ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/827332a721e47b09cb0cbcfb4b60ecf6_MIT18_175S14_Lecture3.pdf#page=6): supports the general CDF–probability-measure correspondence and lists the Cantor law among possible distributions.
<!-- bilingual-en:end -->

- [MIT 6.436J Lecture 4，第 4 节与附录 B，第 10–14 页](https://live.ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/9797310bed4c7f5b5d40d007783eec8d_MIT6_436JF18_lec04.pdf#page=13)：支持密度表示、有密度时零长度集合的概率，以及 Cantor 集与独立抛硬币构造。本文用有限轮的长度和单点概率上界展开该构造。
- [Durrett，*Probability: Theory and Examples*，第 1.2 节、例 1.2.7，印刷页码 12–14](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=20)：支持绝对连续性、Cantor CDF、其水平区间及无法用密度表示的结论。
- [MIT 18.175 Lecture 3，第 6、11 张](https://www.ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/827332a721e47b09cb0cbcfb4b60ecf6_MIT18_175S14_Lecture3.pdf#page=6)：支持一般 CDF 与概率测度的对应关系，并将 Cantor 分布列为一种可能的分布。

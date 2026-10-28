---
aliases:
  - "Sum of Bernoulli variables and the Binomial law"
  - "Binomial law for Bernoulli sums"
  - "独立同分布 Bernoulli 变量之和"
student_os: knowledge-atom
atom_id: PROB-RV-008
atom_set: random-variables-distributions-moments
atom_type: theorem
status: source-checked
requires:
  - "[[Bernoulli分布]]"
  - "[[二项分布]]"
  - "[[相互独立]]"
related:
  - "[[期望线性性]]"
  - "[[和的方差协方差项]]"
  - "[[指示变量与随机计数.canvas|指示变量与随机计数]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

<!-- bilingual-en:start -->
# When a Bernoulli sum is binomial
<!-- bilingual-en:end -->
*Bernoulli 变量之和何时服从二项分布*

<!-- bilingual-en:start -->
Adding $0/1$ records counts how many of their events occurred. The standard theorem obtains a [[二项分布|binomial distribution]] from two assumptions: the records are mutually independent and share the same success probability. Each condition does a different job in the calculation.
<!-- bilingual-en:end -->

把一组 $0/1$ 记录相加，就得到了其中事件发生的次数。标准定理从两个假设出发，推出这个计数服从[[二项分布]]：这些记录相互独立，而且成功概率相同。两个条件在计算中各自承担不同作用。

<!-- bilingual-en:start -->
For trial $i$, let $X_i$ be $1$ on success and $0$ on failure. Saying $X_i\sim\operatorname{Bernoulli}(p)$ means its success probability is $p$; the [[Bernoulli分布|Bernoulli definition]] specifies this two-value law. Suppose there are $n$ trials and the whole collection is [[相互独立|mutually independent]], so the probability of any fixed pattern factors into the individual probabilities. Define $S=X_1+\cdots+X_n$, the number of successes. Then
<!-- bilingual-en:end -->

对第 $i$ 次试验，令 $X_i$ 在成功时为 $1$、失败时为 $0$。$X_i\sim\operatorname{Bernoulli}(p)$ 表示该次成功概率为 $p$，这个两点分布的定义见 [[Bernoulli分布]]。设一共有 $n$ 次试验，整组变量[[相互独立]]，所以任何固定的成功／失败组合的概率都能分解为各次概率的乘积。定义成功总次数 $S=X_1+\cdots+X_n$，则

> [!abstract] From individual records to the count · 从各次记录推出总次数
> $$X_1,\ldots,X_n\text{ mutually independent},\quad
> X_i\sim\operatorname{Bernoulli}(p)\text{ for every }i
> \quad\Longrightarrow\quad
> S\sim\operatorname{Binomial}(n,p).$$
> <!-- bilingual-en:start -->
> Here $n$ is a nonnegative integer and $0\le p\le1$. With $n=0$, the empty sum is defined to be zero. This is a sufficient-condition theorem. Without these assumptions, the marginal Bernoulli laws do not in general determine the distribution of the sum.
> <!-- bilingual-en:end -->
>
> 其中 $n$ 是非负整数，$0\le p\le1$；$n=0$ 时把空和定义为零。这是一条充分条件定理。缺少这些假设时，各变量的 Bernoulli 边际分布一般不足以决定和的分布。

<!-- bilingual-en:start -->
## Why both assumptions appear
<!-- bilingual-en:end -->
*两个条件怎样进入推导*

<!-- bilingual-en:start -->
First consider $0<p<1$ and a candidate count $k$ between $0$ and $n$. Choose one particular set of $k$ positions to be successes. Independence lets us multiply the trial probabilities. The common $p$ makes the resulting product $p^k(1-p)^{n-k}$, regardless of which positions were chosen.
<!-- bilingual-en:end -->

先考虑 $0<p<1$，以及 $0$ 到 $n$ 之间的候选次数 $k$。固定一组 $k$ 个成功位置：独立性允许把各次概率相乘；共同的 $p$ 则让乘积总是 $p^k(1-p)^{n-k}$，不取决于究竟选中了哪些位置。

<!-- bilingual-en:start -->
There are $\binom nk$ such choices, the [[组合数|number of ways to choose $k$ positions from $n$]]. They describe disjoint events, so adding their probabilities gives
<!-- bilingual-en:end -->

这样的选择一共有 $\binom nk$ 种，即[[组合数|从 $n$ 个位置中选择 $k$ 个的方法数]]。不同完整顺序对应互斥事件，所以把它们的概率相加，得到

$$P(S=k)=\binom nk p^k(1-p)^{n-k}.$$

<!-- bilingual-en:start -->
For three trials, exactly two successes can occur as $110$, $101$, or $011$. With common success probability $p=0.2$, each pattern has probability $0.2^2\times0.8=0.032$, so $P(S=2)=3\times0.032=0.096$. If $p=0$, all records are zero with probability one; if $p=1$, all are one. These give the corresponding degenerate binomial laws without a power-formula ambiguity.
<!-- bilingual-en:end -->

三次试验恰好成功两次，有 $110$、$101$、$011$ 三种顺序。若共同成功率 $p=0.2$，每种顺序的概率为 $0.2^2\times0.8=0.032$，因此 $P(S=2)=3\times0.032=0.096$。若 $p=0$，所有记录以概率一为零；若 $p=1$，所有记录以概率一为一。这直接给出相应的退化二项分布，无须处理幂公式的歧义。

<!-- bilingual-en:start -->
## Same marginal probabilities do not give independence
<!-- bilingual-en:end -->
*各自的成功率相同，还不等于独立*

> [!example] Copy one success record twice · 把同一条记录抄两遍
> <!-- bilingual-en:start -->
> Let $B$ be $0$ or $1$ with equal probability, and set $X_1=B$ and $X_2=B$. Each variable is Bernoulli$(1/2)$, but the two records always agree. Their sum $S=2B$ is $0$ or $2$, each with probability $1/2$, and can never be $1$. A Binomial$(2,1/2)$ variable instead gives probabilities $1/4,1/2,1/4$ at $0,1,2$.
> <!-- bilingual-en:end -->
>
> 让 $B$ 以相同概率取 $0$ 或 $1$，再令 $X_1=B$、$X_2=B$。各变量都是 Bernoulli$(1/2)$，但两条记录总是相同。总数 $S=2B$ 只能以各 $1/2$ 的概率取 $0$ 或 $2$，绝不可能为 $1$。而 Binomial$(2,1/2)$ 在 $0,1,2$ 上的概率分别是 $1/4,1/2,1/4$。

<!-- bilingual-en:start -->
Even in this dependent example, [[期望线性性|linearity of expectation]] gives $E[S]=E[X_1]+E[X_2]=1$. In general, common marginal success probability $p$ still implies $E[S]=np$. The distribution and variance need more information: the [[和的方差协方差项|variance of a sum includes covariance terms]]. For a larger collection, checking only [[两两独立|pairwise independence]] also does not establish the mutual independence used in the proof.
<!-- bilingual-en:end -->

即使在这个相依例子中，[[期望线性性]]仍给出 $E[S]=E[X_1]+E[X_2]=1$。一般而言，只要各边际成功率同为 $p$，仍有 $E[S]=np$。但整个分布与方差需要更多信息，[[和的方差协方差项|和的方差还包含协方差项]]。对更大的集合，只检查[[两两独立]]，也不足以建立上述证明用到的相互独立。

<!-- bilingual-en:start -->
## Different probabilities or sampling without replacement
<!-- bilingual-en:end -->
*成功率不同，或抽过的对象不放回时*

<!-- bilingual-en:start -->
If the variables are independent but trial $i$ has its own success probability $p_i$, their count has a **Poisson-binomial distribution**. It includes the ordinary binomial as the special case in which all $p_i$ agree. Different success positions can now have different probabilities, so a single pattern probability cannot simply be multiplied by $\binom nk$. The mean and variance are
<!-- bilingual-en:end -->

如果变量仍相互独立，但第 $i$ 次有自己的成功率 $p_i$，总数服从 **Poisson-binomial（泊松二项）分布**。所有 $p_i$ 相同时，普通二项分布是它的特例。现在不同成功位置可能对应不同概率，不能再用同一个顺序的概率简单乘以 $\binom nk$。其均值和方差为

$$E[S]=\sum_{i=1}^{n}p_i,\qquad
\operatorname{Var}(S)=\sum_{i=1}^{n}p_i(1-p_i).$$

<!-- bilingual-en:start -->
For example, independent records with success probabilities $0.2$ and $0.8$ give $P(S=2)=0.16$ and $E[S]=1$. Replacing both probabilities by their average $0.5$ preserves that mean but would give the incorrect value $P(S=2)=0.25$.
<!-- bilingual-en:end -->

例如，两条独立记录的成功率为 $0.2$ 与 $0.8$，则 $P(S=2)=0.16$、$E[S]=1$。把两个概率都换成平均值 $0.5$ 虽然保留了均值，却会把 $P(S=2)$ 错算成 $0.25$。

<!-- bilingual-en:start -->
Sampling uniformly from a finite population **without replacement** creates another boundary. Drawing a successful item removes it from the pool and changes the next conditional success probability, so the records are generally dependent. When both success and failure items are present, a success makes success on the next draw less likely; the indicators are negatively correlated. With a fixed number of successes in the population and a fixed sample size, the count has a **hypergeometric distribution**. Both cases show why the count model must follow the actual trial mechanism.
<!-- bilingual-en:end -->

从有限总体中**等概率、不放回抽样**是另一种边界。抽到一个成功对象后，它会离开总体，下一次的条件成功率因而改变，所以各次记录通常相依。当成功和失败两类对象都存在时，一次成功会降低下一次成功的概率，指示变量呈负相关。若总体中的成功对象数和抽样数量固定，成功计数服从**超几何分布**。这两种情形都说明，计数模型必须跟随真实的试验机制。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.05 Probability Reading, Class 4, Discrete Random Variables §§3.2–3.3](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf) and [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=875|MIT Mathematics for Computer Science, §§19.3.4, 19.5 and 20.3.4]]: support the independent common-probability count, linearity without independence, and variance-addition conditions. The copied-record and two-probability counterexamples are calculated directly.
- [Tang and Tang, *The Poisson Binomial Distribution—Old & New*, §1, p. 108](https://www.columbia.edu/~wt2319/PB.pdf): defines the independent unequal-probability Bernoulli sum and its probability law; its moments follow from Bernoulli moments and independence.
- [Penn State STAT 414, §7.4, Hypergeometric Distribution](https://online.stat.psu.edu/stat414/Lesson07): supports the fixed-population, fixed-sample-size model with uniform sampling without replacement.
<!-- bilingual-en:end -->

- [MIT 18.05 Probability Reading，Class 4 的 Discrete Random Variables 第 3.2–3.3 节](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf) 及 [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=875|MIT Mathematics for Computer Science，第 19.3.4、19.5、20.3.4 节]]：支持独立且共同成功率的计数模型、不要求独立的线性性及方差相加条件。复制记录与不同成功率的反例由模型直接计算。
- [Tang 与 Tang，*The Poisson Binomial Distribution—Old & New*，第 1 节，印刷第 108 页](https://www.columbia.edu/~wt2319/PB.pdf)：定义独立、成功率可不同的 Bernoulli 和及其概率分布；矩由 Bernoulli 矩和独立性推出。
- [Penn State STAT 414，第 7.4 节，Hypergeometric Distribution](https://online.stat.psu.edu/stat414/Lesson07)：支持有限总体、固定样本量、等概率不放回抽样的模型。

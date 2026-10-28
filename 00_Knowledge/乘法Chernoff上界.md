---
aliases:
  - Multiplicative Chernoff bound
  - Chernoff upper-tail bound for independent bounded sums
  - 乘法 Chernoff 界
student_os: knowledge-atom
atom_id: PROB-CONC-008
atom_set: probability-concentration
atom_type: theorem
status: source-checked
requires:
  - "[[指数Markov法]]"
  - "[[相互独立]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[Hoeffding不等式]]"
---

<!-- bilingual-en:start -->
# Multiplicative Chernoff upper-tail bound
<!-- bilingual-en:end -->

*乘法 Chernoff 上尾界*

<!-- bilingual-en:start -->
Suppose a total is built from many independent contributions, each between zero and one. The **multiplicative Chernoff bound** limits the chance that this total exceeds its average by a specified proportion. For example, if 1,000 fair coin tosses produce 500 heads on average, getting at least 600 means exceeding that average by 20%. The bound gives a probability ceiling for that event without adding up all the relevant binomial probabilities.
<!-- bilingual-en:end -->

假设一个总量由许多独立的小贡献相加而成，每项都在零和一之间。**乘法 Chernoff 界**控制的是：总量超过其平均值某个比例的概率有多大。例如，1,000 次公平抛币平均出现 500 个正面，至少出现 600 个就是超过平均值 20%。这个界不用把相应的二项概率逐项相加，就能为事件给出概率上限。

<!-- bilingual-en:start -->
## Turn the question into a relative threshold
<!-- bilingual-en:end -->

*把问题写成相对均值的阈值*

<!-- bilingual-en:start -->
Write $T_i$ for contribution $i$, $T=\sum_{i=1}^nT_i$ for the total, and $\mu=E[T]$ for its [[期望|expectation]], or probability-weighted average. For counting successes, each $T_i$ can be an [[事件指示变量|indicator]]: one for a success and zero otherwise. More generally, intermediate values such as $0.4$ are allowed. The variables must be [[相互独立|mutually independent]]: the whole collection must factorise, not merely each pair. They need not share a distribution.
<!-- bilingual-en:end -->

用 $T_i$ 表示第 $i$ 项贡献，$T=\sum_{i=1}^nT_i$ 是总量，$\mu=E[T]$ 是它的[[期望]]，即按概率加权的平均。统计成功次数时，$T_i$ 可以是[[事件指示变量|指示变量]]：成功记一，否则记零；一般情形也允许 $0.4$ 这样的中间值。变量必须[[相互独立]]，要求整个集合的联合结构能够分解，不能只检查每一对。各项不必同分布。

> [!important] Relative upper-tail guarantee · 相对上尾保证
> <!-- bilingual-en:start -->
> For $0\le T_i\le1$ almost surely and $c\ge1$,
> <!-- bilingual-en:end -->
>
> 各项几乎必然满足 $0\le T_i\le1$，且 $c\ge1$ 时，
> $$P(T\ge c\mu)\le\exp\!\left[-(c\ln c-c+1)\mu\right].$$
> <!-- bilingual-en:start -->
> Here $c$ is the threshold divided by the mean when $\mu>0$. The [[自然对数|natural logarithm]] is denoted by $\ln$, and $\exp(x)=e^x$ is the [[自然指数函数|natural exponential]].
> <!-- bilingual-en:end -->
>
> 当 $\mu>0$ 时，$c$ 是阈值除以均值。$\ln$ 表示[[自然对数]]，$\exp(x)=e^x$ 表示[[自然指数函数]]。

<!-- bilingual-en:start -->
If the excess is described as a proportion $\delta$, set $c=1+\delta$. The same statement becomes
<!-- bilingual-en:end -->

若用超出均值的比例 $\delta$ 描述偏差，令 $c=1+\delta$，同一结论就成为

$$P(T\ge(1+\delta)\mu)
\le\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu,
\qquad \delta\ge0.$$

> [!example] 600 heads in 1,000 tosses · 千次抛币至少六百正面
> <!-- bilingual-en:start -->
> For independent fair coins, $\mu=1{,}000\times\tfrac12=500$. The threshold 600 gives $c=600/500=1.2$ and $\delta=0.2$. Thus
> <!-- bilingual-en:end -->
>
> 对独立公平币，$\mu=1{,}000\times\tfrac12=500$。阈值 600 对应 $c=600/500=1.2$、$\delta=0.2$，因此
> $$P(T\ge600)\le e^{-(1.2\ln1.2-0.2)500}\approx8.33\times10^{-5}.$$
> <!-- bilingual-en:start -->
> The absolute excess is 100 heads, whereas the relative excess is $0.2$. The result is an [[尾界含义|upper guarantee]], not an estimate of the exact probability.
> <!-- bilingual-en:end -->
>
> 绝对偏差是 100 个正面，相对偏差则是 $0.2$。结果是[[尾界含义|概率上限保证]]，不是对精确概率的估计。

<!-- bilingual-en:start -->
## Why the assumptions matter
<!-- bilingual-en:end -->

*这些条件怎样进入推导*

<!-- bilingual-en:start -->
The [[指数Markov法|exponential Markov method]] first gives $P(T\ge c\mu)\le e^{-\lambda c\mu}E[e^{\lambda T}]$ for $\lambda>0$. Independence splits this exponential average into one factor per summand. On $0\le x\le1$, the exponential graph lies below the line joining its endpoint values: $e^{\lambda x}\le1+x(e^\lambda-1)$. Averaging this inequality and using $1+u\le e^u$ gives
<!-- bilingual-en:end -->

[[指数Markov法]]首先给出 $P(T\ge c\mu)\le e^{-\lambda c\mu}E[e^{\lambda T}]$（$\lambda>0$）。独立性把这个指数平均拆成每个加数各自的因子。在 $0\le x\le1$ 上，指数图像不高于连接端点的直线：$e^{\lambda x}\le1+x(e^\lambda-1)$。对它取平均，再用 $1+u\le e^u$，得到

$$E[e^{\lambda T}]=\prod_iE[e^{\lambda T_i}]
\le\exp\!\left[(e^\lambda-1)\mu\right].$$

<!-- bilingual-en:start -->
For $c>1$, choosing $\lambda=\ln c$ produces the theorem. The interval condition controls each factor; mutual independence lets us multiply them. Pairwise independence alone does not justify that step. The proof also covers continuous summands because the pointwise inequality can be averaged over any distribution on $[0,1]$.
<!-- bilingual-en:end -->

当 $c>1$ 时，取 $\lambda=\ln c$ 就得到定理。区间条件控制每个因子，相互独立则允许把因子相乘；只有两两独立不能支持这一步。逐点不等式可以对 $[0,1]$ 上的任意分布取平均，所以证明同样覆盖连续加数。

<!-- bilingual-en:start -->
At $c=1$ the bound is one. If $\mu=0$, non-negativity forces $T=0$ almost surely, so both $P(T\ge c\mu)$ and the bound equal one; a positive threshold instead has probability zero. For $c\mu>n$, the event is impossible because $T\le n$. A lower tail needs its own argument: inserting $c<1$ into this upper-tail statement is invalid. For an absolute excess $T-\mu\ge t$, compare [[Hoeffding不等式|Hoeffding's inequality]] under its own assumptions; neither bound is always tighter.
<!-- bilingual-en:end -->

$c=1$ 时上界为一。若 $\mu=0$，非负性迫使 $T$ 几乎必然为零，因此 $P(T\ge c\mu)$ 和上界都等于一；但任何正阈值的超越概率都是零。若 $c\mu>n$，由于 $T\le n$，事件不可能发生。下尾需要单独论证，不能把 $c<1$ 直接塞入这个上尾结论。若问题写成绝对超额 $T-\mu\ge t$，也可以在相应条件下比较 [[Hoeffding不等式]]；两种界不存在恒定的优劣顺序。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=943|MIT Mathematics for Computer Science, Theorem 20.5.1 and §§20.5.3, 20.5.6, printed pp. 935–940]]: theorem, coin example and exponential proof. The argument above explicitly averages the pointwise inequality, covering non-discrete summands as well. The zero-mean and support endpoints follow directly from non-negativity and $T\le n$.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=943|MIT Mathematics for Computer Science，定理 20.5.1、第 20.5.3、20.5.6 节，印刷页 935–940]]：核对定理、抛币例子和指数证明。正文把逐点不等式直接取平均，明确覆盖非离散加数；零均值及支持端点由非负性和 $T\le n$ 直接推出。

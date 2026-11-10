---
student_os: knowledge-atom
atom_id: PROB-RV-021
aliases:
  - Student t distribution
  - Student's t distribution
  - Student’s t distribution
  - t distribution
  - t 分布
  - 学生t分布
atom_type: definition
status: source-checked
---

<!-- bilingual-en:start -->
# Student t distribution
<!-- bilingual-en:end -->

*Student t 分布*

<!-- bilingual-en:start -->
A **Student t distribution** describes a normal fluctuation divided by an independent, randomly estimated scale. If that scale happens to be small, the standardized fluctuation can become large. This extra uncertainty gives the t distribution heavier tails than a standard normal distribution: large departures occur more often.

Its most familiar use is measuring how far a sample mean is from a proposed population mean when the population variance is unknown. The distribution itself is a reusable mathematical object; a sampling theorem is still needed before applying it to a particular statistic.
<!-- bilingual-en:end -->

**Student t 分布**描述一个正态波动除以独立的随机尺度后得到的结果。如果这次尺度估得很小，标准化后的波动就可能很大。这份额外的不确定性使 t 分布比标准正态分布的尾部更厚：大幅偏离出现得更频繁。

最熟悉的用途是：总体方差未知时，衡量样本均值偏离指定总体均值有多远。分布本身是可复用的数学对象；把它用于某个统计量，仍须有相应抽样定理。

<!-- bilingual-en:start -->
## A normal numerator and a random denominator
<!-- bilingual-en:end -->

*正态分子与随机分母*

<!-- bilingual-en:start -->
Let $Z$ have the [[正态分布|standard normal distribution]] $N(0,1)$, with mean zero and variance one. Let $V$ have the [[卡方分布|chi-square distribution]] with a finite positive degree of freedom $\nu$, and require $Z$ and $V$ to be [[随机变量独立|independent]]. The **central, standard t distribution** is defined by
<!-- bilingual-en:end -->

设 $Z$ 服从均值零、方差一的[[正态分布|标准正态分布]] $N(0,1)$；$V$ 服从有限正自由度 $\nu$ 的[[卡方分布]]，并要求 $Z,V$ [[随机变量独立|相互独立]]。**中央、标准 t 分布**定义为

$$
Z\sim N(0,1),\qquad V\sim\chi_\nu^2,\qquad
T=\frac{Z}{\sqrt{V/\nu}}\sim t_\nu,\qquad \nu>0.
$$

<!-- bilingual-en:start -->
Here $\sim$ means “has the distribution.” The parameter $\nu$ controls the denominator's variability; it is not the observed value of $T$. When $\nu$ is an integer, $V$ can be formed from $\nu$ independent standard normal squares, independent also of $Z$. Its positive-real-degree extension defines the same ratio for noninteger $\nu>0$, without interpreting a noninteger as a count of measurements. Since $V>0$ almost surely, the denominator is well-defined and positive almost surely.
<!-- bilingual-en:end -->

$\sim$ 表示“服从这个分布”。参数 $\nu$ 控制分母的波动，不是本次观测到的 $T$ 数值。整数 $\nu$ 时，$V$ 可由 $\nu$ 个独立标准正态变量的平方组成，而且它们还须与 $Z$ 独立。卡方分布的正实自由度延拓使同一比值也适用于非整数 $\nu>0$；非整数无需解释成测量个数。$V$ 几乎必然严格为正，所以分母也几乎必然有定义且为正。

> [!example] The denominator changes the same numerator · 同一个分子，分母不同
> <!-- bilingual-en:start -->
> Take $\nu=4$ and an illustrative recorded numerator $z=2$. If $v=1$, then $t=2/\sqrt{1/4}=4$. If $v=16$, then $t=2/\sqrt{16/4}=1$. The random scale changes the apparent size of the same numerator. These are illustrative realizations, not fixed denominators defining the entire distribution.
> <!-- bilingual-en:end -->
>
> 取 $\nu=4$，用 $z=2$ 作为一次记录的示意值。若 $v=1$，结果为 $4$；若 $v=16$，结果为 $1$。同样的分子在不同随机尺度下显得大小不同。这只是两次示意取值，不是用固定分母定义整个分布。

<!-- bilingual-en:start -->
Independence cannot be replaced by checking the two marginal distributions. If $V=Z^2$ and $\nu=1$, the displayed ratio becomes $Z/|Z|$, which is $-1$ or $1$ with equal probabilities. Its marginals are indeed normal and chi-square, but this discrete ratio is not a continuous $t_1$ variable.
<!-- bilingual-en:end -->

只检查两个边际分布不能替代独立性。若 $V=Z^2$、$\nu=1$，比值变成 $Z/|Z|$，等概率取 $-1$ 或 $1$。两个边际确实分别为正态与卡方，但这个离散比值不是连续的 $t_1$ 变量。

<!-- bilingual-en:start -->
## Symmetry, tails and the meaning of standard
<!-- bilingual-en:end -->

*对称性、尾部与“标准”的含义*

<!-- bilingual-en:start -->
The distribution is continuous on the whole real line and symmetric about zero. The denominator is positive, so it preserves the sign of $Z$; replacing $Z$ by $-Z$ reverses $T$ without changing its law. Zero is the median for every $\nu>0$.

Smaller degrees of freedom allow more uncertainty in the denominator. As $\nu$ grows, $V/\nu$ concentrates near one: its mean is one and its variance is $2/\nu$. Consequently $t_\nu$ approaches $N(0,1)$ in distribution, by [[Chebyshev不等式|concentration of the denominator]] and [[Slutsky定理|substitution of a scale converging to one]]. There is no universal finite degree at which the distributions become identical.
<!-- bilingual-en:end -->

这个分布在整条实数轴上连续，并关于零对称。分母为正，会保留 $Z$ 的符号；把 $Z$ 换成 $-Z$，只会让 $T$ 反号而不改变分布。每个 $\nu>0$ 下，中位数都是零。

较小的自由度意味着分母有更大的不确定性。$\nu$ 增大时，$V/\nu$ 集中到一附近：均值为一，方差为 $2/\nu$。结合[[Chebyshev不等式|分母的集中]]及[[Slutsky定理|趋于一的尺度代入]]，便有 $t_\nu$ 依分布趋向 $N(0,1)$。不存在某个通用的有限自由度，使两种分布从此完全相同。

> [!warning] Standard t does not mean unit variance · 标准 t 不等于方差为一
> <!-- bilingual-en:start -->
> The standard distribution fixes its location at zero and its scale at one. Its [[期望|mean]] exists only for $\nu>1$, when it is zero. Its [[方差|finite variance]] exists only for $\nu>2$, when
> $$\operatorname{Var}(T)=\frac{\nu}{\nu-2}.$$
> For $1<\nu\le2$, the mean exists but the second moment is infinite; for $0<\nu\le1$, even the mean is undefined. Symmetry alone does not ensure an expectation. For example, $t_4$ has variance $2$, not $1$.
> <!-- bilingual-en:end -->
>
> 标准分布固定位置为零、尺度为一。只有 $\nu>1$ 时[[期望|均值]]才存在，此时为零；只有 $\nu>2$ 时才有[[方差|有限方差]]，等于 $\nu/(\nu-2)$。$1<\nu\le2$ 时均值存在但二阶矩无穷；$0<\nu\le1$ 时连均值也没有定义。对称性本身不保证期望存在，例如 $t_4$ 的方差为 $2$，不是 $1$。

> [!note]- A density expression · 密度写法
> <!-- bilingual-en:start -->
> The [[概率密度函数|density]] is
> $$f_\nu(t)=\frac{\Gamma((\nu+1)/2)}
> {\sqrt{\pi\nu}\,\Gamma(\nu/2)}
> \left(1+\frac{t^2}{\nu}\right)^{-(\nu+1)/2},\qquad t\in\mathbb R.$$
> Here $\Gamma(a)=\int_0^\infty u^{a-1}e^{-u}\,du$ for $a>0$ supplies the normalizing constant; the same Gamma function appears in the chi-square density. Probabilities come from integrating this density over an interval. For large $|t|$, it is proportional asymptotically to $|t|^{-(\nu+1)}$, explaining why high moments can fail to exist.
> <!-- bilingual-en:end -->
>
> 上式是[[概率密度函数]]。其中 $a>0$ 时 $\Gamma(a)=\int_0^\infty u^{a-1}e^{-u}\,du$ 提供归一化常数，与卡方密度中的Gamma函数相同；区间概率由密度积分给出。当 $|t|$ 很大时，密度渐近正比于 $|t|^{-(\nu+1)}$，因此高阶矩可能不存在。

<!-- bilingual-en:start -->
## Why a sample mean leads to this distribution
<!-- bilingual-en:end -->

*样本均值为什么会用到它*

<!-- bilingual-en:start -->
Suppose $X_1,\ldots,X_n$ are [[独立同分布|iid]] $N(\mu,\sigma^2)$ with $\sigma>0$ and $n\ge2$. Define $\bar X=n^{-1}\sum_iX_i$ and $S^2=(n-1)^{-1}\sum_i(X_i-\bar X)^2$. These are the one-dimensional versions of the [[样本均值向量|sample mean]] and [[样本协方差矩阵|sample covariance]]. Normal sampling gives
<!-- bilingual-en:end -->

设 $X_1,\ldots,X_n$ [[独立同分布|iid]]，服从 $N(\mu,\sigma^2)$，其中 $\sigma>0$、$n\ge2$。定义 $\bar X=n^{-1}\sum_iX_i$、$S^2=(n-1)^{-1}\sum_i(X_i-\bar X)^2$，它们分别是[[样本均值向量]]和[[样本协方差矩阵]]的一维版本。正态抽样给出

$$
Z=\frac{\sqrt n(\bar X-\mu)}{\sigma}\sim N(0,1),\qquad
V=\frac{(n-1)S^2}{\sigma^2}\sim\chi_{n-1}^2.
$$

<!-- bilingual-en:start -->
The [[正态均值协方差独立|normal mean–variance independence theorem]] makes these two quantities independent. Substituting them into the defining ratio cancels the unknown $\sigma$:
<!-- bilingual-en:end -->

由[[正态均值协方差独立|正态均值与方差独立定理]]，这两个量相互独立。代入定义比值，未知的 $\sigma$ 恰好相消：

$$
\frac{Z}{\sqrt{V/(n-1)}}=\frac{\bar X-\mu}{S/\sqrt n}\sim t_{n-1}.
$$

<!-- bilingual-en:start -->
The denominator $S/\sqrt n$ is the estimated [[标准误含义|standard error of the mean]], not the spread $S$ of individual measurements. To replace $\mu$ in the numerator by a proposed $\mu_0$ and still use the **central** law, the null $\mu=\mu_0$ must hold. Dividing any dataset's mean by an estimated standard error does not automatically create an exact t law; the sampling assumptions and independence argument matter.
<!-- bilingual-en:end -->

分母 $S/\sqrt n$ 是均值的估计[[标准误含义|标准误]]，不是个体测量的散布 $S$。若把分子中的 $\mu$ 换成指定值 $\mu_0$，仍要使用**中央** t 律，就需要原假设 $\mu=\mu_0$ 成立。任意数据的均值除以估计标准误，并不会自动获得精确 t 分布；抽样假设与独立论证不可省略。

<!-- bilingual-en:start -->
For $0<\alpha<1$, let $q_{1-\alpha/2,\nu}$ be the $(1-\alpha/2)$ [[总体分位数|quantile]]. Symmetry gives
$$P\!\left(-q_{1-\alpha/2,\nu}\le T\le q_{1-\alpha/2,\nu}\right)=1-\alpha.$$
Thus a central 95% range uses the 0.975 quantile, leaving 2.5% in each tail. With $\nu=9$, the cutoff is approximately $2.262$, whereas a 5% one-sided upper tail uses the 0.95 quantile. A table headed “upper-tail probability” uses the complementary probability convention.
<!-- bilingual-en:end -->

对 $0<\alpha<1$，令 $q_{1-\alpha/2,\nu}$ 为 $1-\alpha/2$ [[总体分位数|分位数]]，对称性便给出上面的中央概率。因此95%中央范围使用0.975分位数，两边各留下2.5%。$\nu=9$ 时，截点约为 $2.262$；而单侧5%右尾使用0.95分位数。若表头标的是“右尾概率”，则采用互补的概率约定。

<!-- bilingual-en:start -->
Squaring connects this law to the [[F分布|F distribution]]: $Z^2\sim\chi_1^2$ remains independent of $V$, so $T^2\sim F_{1,\nu}$. The square loses the direction of departure; this is why a two-sided t comparison can be expressed as an upper-tail comparison of $T^2$.
<!-- bilingual-en:end -->

平方后便与[[F分布]]衔接：$Z^2\sim\chi_1^2$ 仍与 $V$ 独立，因此 $T^2\sim F_{1,\nu}$。平方会丢掉偏离的正负方向，所以双侧 t 比较可以写成对 $T^2$ 的右尾比较。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.443, Lecture 6, PDF pp. 5–6 / printed pp. 39–40](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/924817d33e1ccb6f3a6b944c985d0cdb_lecture6.pdf#page=5): independent normal/chi-square construction, density and the squared-t/F relation.
- [R stats manual, The Student t Distribution, Arguments and Details](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/TDist.html): positive real degrees, density, moment thresholds, independent-ratio parameterization and the iid normal sample-mean application.
- [SciPy, scipy.stats.t, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html): all-real density, positive degrees and location/scale convention; a location shift is distinct from a noncentral t construction.
- [NIST/SEMATECH, t Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm): symmetry, heavier tails, normal limit and quantile use. Its discussion uses integer degrees; R and SciPy give the real-degree extension. The ratio examples, missing-independence counterexample and tail-integrability boundaries follow directly from the construction and density.
<!-- bilingual-en:end -->

- MIT第5–6页（印刷39–40页）：核对独立正态/卡方构造、密度与平方后的F关系。
- R官方手册：核对正实自由度、密度、矩存在门槛、独立比值参数化，以及iid正态样本均值的应用。
- SciPy：核对实数轴密度、正自由度及位置—尺度约定；平移分布与非中央t构造不同。
- NIST：核对对称性、厚尾、正态极限与分位数用途；该页按整数自由度表述，实数延拓另由R与SciPy支持。数值例子、缺独立性反例及尾部可积性边界由构造和密度直接推出。

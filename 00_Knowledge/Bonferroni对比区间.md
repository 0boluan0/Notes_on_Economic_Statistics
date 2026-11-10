---
aliases:
- Bonferroni simultaneous intervals
- 多元均值的 Bonferroni 同时区间
- Bonferroni 线性组合区间
student_os: knowledge-atom
atom_id: STAT-HOT-005
atom_set: hotelling-mean-inference
atom_type: procedure-boundary
status: source-checked
requires:
- '[[Gaussian仿射闭包]]'
- '[[正态均值协方差独立]]'
- '[[样本协方差Wishart律]]'
- '[[Wishart抽样假设]]'
part_of:
- '[[Hotelling T² 与多元均值推断.canvas]]'
contrasts_with:
- '[[Hotelling置信椭球]]'
related:
- '[[单样本Hotelling T²]]'
- '[[Hotelling投影极值]]'
---

<!-- bilingual-en:start -->
# Bonferroni intervals for selected linear combinations
<!-- bilingual-en:end -->

*预设线性组合的 Bonferroni 同时区间*

<!-- bilingual-en:start -->
Suppose two measurements are taken on every item. Before collecting the data, we decide to estimate the first population mean and the difference between the two population means. We want the **whole set of intervals** to cover its targets, not just a good coverage rate for each interval separately. Bonferroni intervals achieve this by giving each planned interval only part of the total allowed error probability.
<!-- bilingual-en:end -->

假设每个物品都测量两项指标。收集数据前，我们决定估计第一项的总体均值，以及两项总体均值之差。希望获得的是**整组区间共同覆盖各自目标**的保证，而不只是每个区间单独有较高覆盖率。Bonferroni 区间把允许的总错误概率分给各个预设区间，借此保护整个比较组。

<!-- bilingual-en:start -->
A [[置信区间覆盖率|confidence level]] describes repeated sampling: the unknown target is fixed, and the interval changes when the sample changes. Write $\alpha$ for the allowed probability that **at least one** interval misses its target. For example, a 95% simultaneous guarantee means this failure probability is at most $\alpha=0.05$.
<!-- bilingual-en:end -->

[[置信区间覆盖率|置信水平]]描述重复抽样：未知目标固定，区间随样本变化。用 $\alpha$ 表示允许的“**至少一个**区间漏掉目标”的概率。比如 95% 同时覆盖保证，意味着整组中出现漏覆盖的概率至多为 $\alpha=0.05$。

<!-- bilingual-en:start -->
## Name the comparisons before constructing the intervals
<!-- bilingual-en:end -->

*先说明保护哪些比较，再构造区间*

<!-- bilingual-en:start -->
Let $X_1,\ldots,X_n$ be [[独立同分布|iid]] $p$-coordinate observations from a [[联合Gaussian|joint Gaussian]] population $N_p(\mu,\Sigma)$, where $n\ge2$, $p\ge1$, and $\Sigma\succ0$. A [[正定矩阵|positive-definite covariance]] gives positive variance to every nonzero direction. Let $\bar X$ be the [[样本均值向量|sample mean]] and $S$ the [[样本协方差矩阵|sample covariance]] with divisor $n-1$.

Fix a positive integer $m$ and nonzero coefficient vectors $a_1,\ldots,a_m\in\mathbb R^p$ before examining the data used for inference. Target $j$ is the scalar $\theta_j=a_j^T\mu$. These can be coordinate means, differences, or other specified linear combinations; the coefficients do not have to sum to zero for the result below. Here $p$ counts measured coordinates, whereas $m$ counts the comparisons to be protected.
<!-- bilingual-en:end -->

设 $X_1,\ldots,X_n$ 是来自[[联合Gaussian|联合 Gaussian]]总体 $N_p(\mu,\Sigma)$ 的[[独立同分布|iid]]观测，$n\ge2$、$p\ge1$、$\Sigma\succ0$。协方差[[正定矩阵|正定]]意味着每个非零方向都有正方差。$\bar X$ 为[[样本均值向量]]，$S$ 为分母 $n-1$ 的[[样本协方差矩阵]]。

在查看用于推断的数据之前，固定正整数 $m$ 及非零系数向量 $a_1,\ldots,a_m$。第 $j$ 个目标是标量 $\theta_j=a_j^T\mu$，可以是一个坐标均值、均值差或其他指定线性组合；下面的结论不要求系数和为零。$p$ 数的是测量坐标，$m$ 数的是要保护的比较，不能混为一谈。

<!-- bilingual-en:start -->
The estimate is $\hat\theta_j=a_j^T\bar X$. Its estimated [[标准误含义|standard error]], measuring the estimate's sampling variation, is
$$\widehat{\operatorname{se}}_j=\sqrt{\frac{a_j^TSa_j}{n}}.$$
For each fixed $a_j$, [[Gaussian仿射闭包|Gaussian linear combinations]] and [[正态均值协方差独立|mean–covariance independence]] give
$$\frac{a_j^T\bar X-a_j^T\mu}{\sqrt{a_j^TSa_j/n}}\sim t_{n-1}.$$
Here the [[Student t分布|Student t law]] $t_\nu$ is the distribution of $Z/\sqrt{V/\nu}$ for independent $Z\sim N(0,1)$ and $V\sim\chi_\nu^2$. The numerator represents mean error; the random denominator represents estimated variability. Positive population variance makes each displayed sample denominator positive almost surely.
<!-- bilingual-en:end -->

估计值为 $\hat\theta_j=a_j^T\bar X$，估计[[标准误含义|标准误]]为 $\sqrt{a_j^TSa_j/n}$，衡量该估计量的抽样变动。对每个固定 $a_j$，[[Gaussian仿射闭包|Gaussian 线性组合]]及[[正态均值协方差独立|均值—协方差独立性]]给出上述 $t_{n-1}$ 分布。[[Student t分布|Student t 分布]] $t_\nu$ 是独立 $Z\sim N(0,1)$、$V\sim\chi_\nu^2$ 的比值 $Z/\sqrt{V/\nu}$ 所服从的分布；分子描述均值误差，随机分母描述估计变动。正的总体投影方差保证各分母几乎必然为正。

> [!important] Divide the total error allowance across the family · 在整组比较中分配错误概率
> <!-- bilingual-en:start -->
> Choose a fixed $0<\alpha<1$ before inspecting the inference data. Let $t_\nu(u)$ mean the lower-tail probability-$u$ [[总体分位数|quantile]] of $t_\nu$. Equal allocation assigns error probability $\alpha/m$ to each two-sided interval:
> $$I_j=\left[a_j^T\bar X\ \pm\ t_{n-1}\!\left(1-\frac{\alpha}{2m}\right)\sqrt{\frac{a_j^TSa_j}{n}}\right],\qquad j=1,\ldots,m.$$
> The bracket with $\pm$ denotes the lower and upper endpoints. The extra factor $2$ splits each interval's error between its two tails. All $m$ targets are covered simultaneously with probability **at least** $1-\alpha$.
> <!-- bilingual-en:end -->
>
> 查看推断数据前固定 $0<\alpha<1$，用 $t_\nu(u)$ 表示 $t_\nu$ 分布的下尾概率为 $u$ 的[[总体分位数|分位数]]。等额分配时，每个双侧区间获得错误概率 $\alpha/m$，因此使用 $1-\alpha/(2m)$ 分位数；额外的 $2$ 把单个区间的错误分到左右两尾。$[\text{中心}\pm\text{半宽}]$ 表示上下两个端点。这 $m$ 个区间同时覆盖各自目标的概率**至少**为 $1-\alpha$。

<!-- bilingual-en:start -->
The reason is the [[并事件上界|union bound]]. Let $E_j$ be the event that interval $j$ misses $\theta_j$. Each marginal t interval has $P(E_j)=\alpha/m$, so
$$P\!\left(\bigcup_{j=1}^m E_j\right)\le\sum_{j=1}^m P(E_j)=\alpha.$$
Taking complements gives the simultaneous guarantee. The events need not be independent; they may overlap heavily because all intervals use the same data. Overlap can make the bound conservative, meaning the actual joint coverage exceeds the promised minimum. Fixed unequal allocations also work: choose $0<\alpha_j<1$ with $\sum_j\alpha_j\le\alpha$ and replace $\alpha/m$ by $\alpha_j$ in each interval.
<!-- bilingual-en:end -->

依据是[[并事件上界]]。令 $E_j$ 表示第 $j$ 个区间漏掉 $\theta_j$，单个 t 区间有 $P(E_j)=\alpha/m$，于是“至少一次漏掉”的概率不超过这些概率之和。取补集便得到同时保证。事件无需独立；所有区间使用同一数据，失败事件可以大量重叠。重叠可能使界保守，即实际联合覆盖率高于承诺的下限。预先固定的不等额分配也可以：选 $0<\alpha_j<1$、$\sum_j\alpha_j\le\alpha$，各区间用 $\alpha_j$ 替换 $\alpha/m$。

> [!example] A mean and a difference · 一个均值和一个均值差
> <!-- bilingual-en:start -->
> Suppose $n=25$, $\bar x=(5,3)^T$, and $S=\begin{bmatrix}4&1\\1&9\end{bmatrix}$. Before collecting data, choose $a_1=(1,0)^T$ and $a_2=(1,-1)^T$, so $m=2$. Their estimates are $5$ and $2$. Their sample variances are $4$ and $4+9-2(1)=11$, giving standard errors $0.4$ and $\sqrt{11}/5\approx0.663$.
>
> At $\alpha=0.05$, use $t_{24}(0.9875)\approx2.391$. The intervals are approximately $[4.044,5.956]$ for $\mu_1$ and $[0.414,3.586]$ for $\mu_1-\mu_2$. Each individually has 97.5% coverage; together they have at least 95% coverage. Their dependence does not invalidate the guarantee.
> <!-- bilingual-en:end -->
>
> 预先指定两个方向，分别估计 $\mu_1$ 与 $\mu_1-\mu_2$。估计值为 $5,2$；投影样本方差为 $4,11$，标准误为 $0.4$ 与约 $0.663$。取 $\alpha=0.05$，临界值为 $t_{24}(0.9875)\approx2.391$，得到约 $[4.044,5.956]$ 与 $[0.414,3.586]$。每个区间单独覆盖率为 97.5%，整组至少为 95%；两者依赖并不破坏保证。

<!-- bilingual-en:start -->
## The protected family determines what you may conclude
<!-- bilingual-en:end -->

*保护的比较组决定了结论范围*

<!-- bilingual-en:start -->
After constructing the protected family, displaying only a subset keeps its simultaneous protection: if all original intervals cover, every displayed one covers too. This is different from searching new directions in the data and then pretending that only the selected directions were planned. That search is not covered by this finite-family calculation. Nor do ordinary unadjusted 95% intervals generally give 95% joint coverage; for two independent such intervals it would be $0.95^2=0.9025$.
<!-- bilingual-en:end -->

构造好已受保护的比较组后，即使只展示其中一部分，仍保留同时保护：原组全部覆盖时，展示的各项当然也覆盖。这与看数据后搜索新方向、再假装只预设了选中的那些方向不同；后者不在本卡有限比较组的计算保证内。普通未调整的 95% 区间也一般不能提供 95% 联合覆盖：两个独立区间的联合覆盖只有 $0.95^2=0.9025$。

<!-- bilingual-en:start -->
The [[Hotelling置信椭球|Hotelling ellipsoid]] protects all directions at once under its stronger sample-rank condition $n>p$. Bonferroni protects the finite listed family and never inverts $S$. Thus, even when $p\ge n$ makes $S$ singular, each fixed nonzero direction still has a positive sample variance almost surely under the stated model, and these intervals remain exact marginally. No universal width ordering exists: Bonferroni's multiplier depends on $m$, while Hotelling's all-direction multiplier depends on $p$ as well as $n,\alpha$.
<!-- bilingual-en:end -->

[[Hotelling置信椭球]]在更强的样本秩条件 $n>p$ 下同时保护所有方向。Bonferroni 保护列出的有限比较组，而且不求 $S^{-1}$。因此，即使 $p\ge n$ 导致 $S$ 奇异，在上述模型下，每个事先固定的非零方向仍几乎必然具有正的样本方差，边际区间保持精确。两者也没有普遍的宽度排序：Bonferroni 乘子取决于 $m$，Hotelling 全方向乘子则取决于 $p$ 以及 $n,\alpha$。

<!-- bilingual-en:start -->
Normal sampling justifies the marginal t law. With non-Gaussian or dependent observations, the union bound remains true, but each interval needs its own valid coverage argument; the same t formula is not automatically exact. A zero coefficient vector has the known target zero and needs no error budget. Finally, failing to exclude a target is not proof of equality, and these intervals do not certify unlisted directions or replace the full-vector [[单样本Hotelling T²|Hotelling test]].
<!-- bilingual-en:end -->

正态抽样负责保证边际 t 律。面对非 Gaussian 或依赖观测，并界依然成立，但必须先为每个区间建立有效覆盖依据，不能自动把同一 t 公式称为精确。零系数向量对应已知目标零，无需分配错误预算。最后，未排除某目标不是相等的证明；这些区间也不认证未列出的方向，不等同于全向量的[[单样本Hotelling T²|Hotelling 检验]]。

## 来源与核验

<!-- bilingual-en:start -->
- [NIST/SEMATECH, §7.4.7.3, Bonferroni's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm): prespecified finite linear-combination families, arbitrary-event noncoverage bound, and at-least nominal simultaneous coverage. Its ANOVA variance formula is not substituted for the one-sample covariance used here.
- [Penn State STAT505, Lesson5 §5.2](https://online.stat.psu.edu/stat505/Lesson05) and [Lesson7 §7.1.4, “Bonferroni Intervals”](https://online.stat.psu.edu/stat505/Lesson07): one-sample t multipliers and comparison with all-direction intervals. General fixed directions here follow by applying the scalar interval to $a_j^TX_i$; the union-bound proof is shown explicitly.
- [MIT18.443 Lecture6, PDF p.5 / printed p.39](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/924817d33e1ccb6f3a6b944c985d0cdb_lecture6.pdf#page=5): Student t as an independent normal/chi-square ratio.
- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.6. Bonferroni 多重比较|Course §1.6]]: notation for $m$ comparisons, projection variance and the high-dimensional boundary. Numerical intervals are calculated directly.
<!-- bilingual-en:end -->

- NIST 核对预设有限线性组合、任意事件的漏覆盖并界与至少名义水平的同时覆盖；原文 ANOVA 方差公式未替代本卡的单样本协方差。
- Penn State 第5课5.2节及第7课7.1.4节核对单样本 t 乘子与全方向区间的区别；一般固定方向通过对 $a_j^TX_i$ 使用标量区间得到，并界在正文证明。
- MIT第6讲PDF第5页/印刷第39页核对 Student t 的独立正态/卡方比值定义。
- 课程第1.6节核对比较数、投影方差和高维边界；数值区间直接计算。

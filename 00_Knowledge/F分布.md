---
student_os: knowledge-atom
atom_id: PROB-RV-020
aliases:
  - F distribution
  - Fisher–Snedecor distribution
  - Fisher F distribution
  - F 分布
atom_type: definition
status: source-checked
---

<!-- bilingual-en:start -->
# F distribution
<!-- bilingual-en:end -->

*F 分布*

<!-- bilingual-en:start -->
An **F distribution** describes how large one average of squared standard normal measurements is relative to another, independent average. Both averages change across repetitions, so their ratio does too. The construction below starts with positive integer counts; a positive-real-degree extension is explained afterwards.
<!-- bilingual-en:end -->

**F 分布**描述两个独立的“标准正态测量值的平方平均”之间的大小比。两边的平均值都会随着重复测量变化，比值也会变化。下面先从正整数个测量值构造，再解释正实数自由度的延拓。

> [!example] Compare two squared averages · 比较两个平方平均
> <!-- bilingual-en:start -->
> Suppose five measurements are [[相互独立|mutually independent]], and each has the [[正态分布|standard normal distribution]] $N(0,1)$: mean zero and variance one. Put two measurements in the first group and three in the second. For illustrative recorded values $(1,2)$ and $(1,-1,2)$, their squared totals are
> <!-- bilingual-en:end -->
>
> 假设五个测量值[[相互独立]]，各自服从[[正态分布|标准正态分布]] $N(0,1)$，即均值为零、方差为一。第一组放两个，第二组放三个。用 $(1,2)$ 与 $(1,-1,2)$ 作为一次记录的示意值，两组平方总量为
>
> $$u=1^2+2^2=5,\qquad v=1^2+(-1)^2+2^2=6.$$
>
> <!-- bilingual-en:start -->
> Divide each total by its own group size before taking their ratio:
> <!-- bilingual-en:end -->
>
> 先把每组总量除以本组个数，再求比值：
>
> $$\frac{u/2}{v/3}=\frac{5/2}{6/3}=\frac54=1.25.$$
>
> <!-- bilingual-en:start -->
> The first squared average is $1.25$ times the second. Across repetitions this ratio has distribution $F_{2,3}$. The value $1.25$ is one possible result; the subscripts $2,3$ describe the sampling model.
> <!-- bilingual-en:end -->
>
> 第一组平方平均是第二组的 $1.25$ 倍。重复这个模型时，随机比值服从 $F_{2,3}$；$1.25$ 是一次可能结果，下标 $2,3$ 则描述抽样模型。

<!-- bilingual-en:start -->
## Two degrees of freedom, in a fixed order
<!-- bilingual-en:end -->

*两个自由度，顺序固定*

<!-- bilingual-en:start -->
A [[卡方分布|chi-square variable]] with integer degrees of freedom $r$ is a sum of $r$ independent standard normal squares. Let $U$ and $V$ be [[随机变量独立|independent random variables]] with chi-square distributions of positive degrees $r$ and $s$. The **central, standard F distribution** is defined by
<!-- bilingual-en:end -->

整数自由度为 $r$ 的[[卡方分布|卡方变量]]，是 $r$ 个独立标准正态变量的平方和。设 $U,V$ 是[[随机变量独立|相互独立的随机变量]]，分别服从正自由度 $r,s$ 的卡方分布。**中央、标准 F 分布**定义为

$$U\sim\chi_r^2,\qquad V\sim\chi_s^2,\qquad
\boxed{T=\frac{U/r}{V/s}\sim F_{r,s}.}$$

<!-- bilingual-en:start -->
The symbol $\sim$ means “has the distribution.” The first parameter $r$ belongs to the **numerator**, the second $s$ to the **denominator**. In the integer construction they count independent normal-square contributions. Neither parameter is the observed value of the ratio. The word “central” refers to the zero means of those underlying standard normal variables.
<!-- bilingual-en:end -->

$\sim$ 表示“服从这个分布”。第一个参数 $r$ 是**分子**自由度，第二个 $s$ 是**分母**自由度；在整数构造中，它们数的是独立正态平方的贡献个数，都不是观察到的比值。“中央”对应底层标准正态变量的均值为零。

<!-- bilingual-en:start -->
The division by degrees of freedom matters: $U/V$ alone equals $(r/s)T$, which differs from $T$ unless $r=s$. Reversing numerator and denominator gives the reciprocal,
<!-- bilingual-en:end -->

除以各自自由度这一步不可省略：单独的 $U/V$ 等于 $(r/s)T$，除非 $r=s$，否则不等于 $T$。交换分子、分母，得到倒数：

$$\frac1T=\frac{V/s}{U/r}\sim F_{s,r}.$$

<!-- bilingual-en:start -->
Thus $F_{2,3}$ and $F_{3,2}$ generally give different probabilities for the same threshold. Each chi-square variable is strictly positive with probability one, so $T$ is finite and positive with probability one. Its distribution is continuous on the positive range: zero is a boundary, and no single value has positive probability.
<!-- bilingual-en:end -->

因此，在同一个门槛下，$F_{2,3}$ 与 $F_{3,2}$ 给出的概率一般不同。两个卡方变量都以概率一严格为正，所以 $T$ 也以概率一有限且为正。它在正数范围上连续分布：零是边界，任何单独数值都没有正概率。

<!-- bilingual-en:start -->
Each normalized part has [[期望|expectation]] one: $E[U/r]=E[V/s]=1$. This does **not** make the expected ratio equal to one. A very small denominator can make the ratio large. For example, $F_{2,3}$ has mean $3$, despite the two normalized parts each having mean one.
<!-- bilingual-en:end -->

每个归一化部分的[[期望]]都是一：$E[U/r]=E[V/s]=1$。这**不能**推出比值的期望也是一；分母很小时，比值可以很大。例如，$F_{2,3}$ 的均值为 $3$，尽管上下两个归一化部分的均值都为一。

> [!warning] Matching marginal laws is not enough · 各自服从卡方还不够
> <!-- bilingual-en:start -->
> If $U=V$ is the same $\chi_r^2$ variable used twice, then $(U/r)/(V/r)=1$ with probability one. Both marginal laws are chi-square, but this constant ratio does not have the continuous $F_{r,r}$ law. The independence condition is part of the definition's construction.
> <!-- bilingual-en:end -->
>
> 如果 $U=V$ 是同一个 $\chi_r^2$ 变量重复使用，则 $(U/r)/(V/r)$ 以概率一恒等于 $1$。两边各自都服从卡方分布，但这个常量比值并不服从连续的 $F_{r,r}$ 分布；独立性是构造条件的一部分。

> [!note]- Positive real degrees of freedom · 正实数自由度
> <!-- bilingual-en:start -->
> The [[卡方分布|chi-square density]] also defines a distribution for every positive real degree, not only integers. Using two independent variables from that extension in the same ratio defines $F_{r,s}$ for all real $r>0,s>0$. A noninteger degree is a distribution parameter, not a fractional count of normal measurements. Zero degrees are excluded: even if $\chi_0^2$ is separately defined as a point mass at zero, dividing by zero does not define this F ratio.
> <!-- bilingual-en:end -->
>
> [[卡方分布|卡方密度]]对任意正实数自由度都能定义分布，不限于整数。把该延拓中的两个独立变量放进同一个比值，就得到所有实数 $r>0,s>0$ 下的 $F_{r,s}$。非整数自由度是分布参数，不是“分数个”正态测量值。这里不允许零自由度：即使另行把 $\chi_0^2$ 定义成零点质量，除以零仍不能定义上述 F 比值。

<!-- bilingual-en:start -->
## Reading an F critical value
<!-- bilingual-en:end -->

*怎样读 F 临界值*

<!-- bilingual-en:start -->
A [[总体分位数|quantile]] turns a probability into a threshold. For $T\sim F_{r,s}$ and $0<\alpha<1$, write $q_{1-\alpha}(r,s)$ for its $(1-\alpha)$ quantile. Because this distribution is continuous with a positive [[概率密度函数|density]] on $t>0$,
<!-- bilingual-en:end -->

[[总体分位数|分位数]]把概率转为门槛。对 $T\sim F_{r,s}$、$0<\alpha<1$，用 $q_{1-\alpha}(r,s)$ 表示其 $1-\alpha$ 分位数。由于它在 $t>0$ 上有正的[[概率密度函数|连续密度]]，

$$P\!\left(T\le q_{1-\alpha}(r,s)\right)=1-\alpha,\qquad
P\!\left(T>q_{1-\alpha}(r,s)\right)=\alpha.$$

<!-- bilingual-en:start -->
For a 5% upper-tail comparison, use the $0.95$ quantile with the correct **ordered pair** $(r,s)$. Some texts write this threshold as $F_{r,s}(0.95)$; the argument then denotes a probability level, not a measured ratio. Other tables label an upper-tail area instead, so check which probability is requested.
<!-- bilingual-en:end -->

做 5% 右尾比较时，应使用正确**有序参数对** $(r,s)$ 下的 $0.95$ 分位数。有些教材把这个门槛写作 $F_{r,s}(0.95)$，括号里的数表示概率水平，并非测量比值；另一些表按右尾面积标记，因此应先核清所用概率的含义。

<!-- bilingual-en:start -->
This reference law can calibrate statistics once a sampling theorem establishes it. For example, [[单样本Hotelling T²|the one-sample Hotelling theorem]] derives an F law for a specific rescaling of a multivariate mean statistic under its Gaussian assumptions. A nonnegative statistic, a ratio of squared quantities, or normalization by sample sizes alone does not establish an F distribution.
<!-- bilingual-en:end -->

只有抽样定理确立了相应分布，这个参考分布才能用来校准统计量。例如，[[单样本Hotelling T²|单样本 Hotelling 定理]]在其 Gaussian 假设下，推出多元均值统计量经过特定缩放后服从 F 分布。统计量非负、写成两个平方量的比，或除以样本量，本身都不足以确立 F 分布。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.443, Lecture 6, PDF pp. 3–5 / printed pp. 37–39](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/924817d33e1ccb6f3a6b944c985d0cdb_lecture6.pdf#page=3): independent chi-square ratio, ordered degrees of freedom, reciprocal relation and positive-range density. The two-group numerical example and the copied-variable counterexample are direct calculations here.
- [NIST/SEMATECH, F Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm): standard parameterization, density, quantile interpretation and mean $s/(s-2)$ for $s>2$, giving the $F_{2,3}$ mean used here. Its brief ratio introduction does not state independence; MIT and SciPy supply that condition explicitly.
- [SciPy, scipy.stats.f, Notes and probability methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html): independent-ratio definition for positive real numerator and denominator degrees, their order, density on $t>0$, and the distinction between cumulative-probability and upper-tail quantiles.
- [Penn State STAT 505, Lesson 7, §§7.1.1–7.1.3](https://online.stat.psu.edu/stat505/Lesson07): the linked Hotelling application requires its sampling assumptions and a specific rescaling; its tables use upper-tail cutoff notation.
<!-- bilingual-en:end -->

- MIT 第3–5页（印刷页37–39）核对独立卡方比值、有序自由度、倒数关系及正数范围上的密度；两组数值与复制变量反例由定义直接计算。
- NIST 核对标准参数化、密度、分位数解释及 $s>2$ 时的均值 $s/(s-2)$，本卡据此得到 $F_{2,3}$ 的均值。其简短比值介绍省略独立条件，此条件由 MIT 与 SciPy 明确补足。
- SciPy 核对正实数分子、分母自由度下的独立比值定义、参数顺序、正数范围密度，以及累计概率分位数和右尾分位数的区别。
- Penn State 第7.1.1–7.1.3节核对所链接的 Hotelling 应用需要抽样假设及特定缩放，并示范按右尾面积标记临界值的约定。

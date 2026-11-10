---
aliases:
  - White heteroskedasticity test
  - White test
student_os: knowledge-atom
atom_id: ECON-ERR-006
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
requires:
  - "[[异方差]]"
  - "[[残差协方差诊断]]"
related:
  - "[[Breusch-Pagan检验]]"
  - "[[异方差稳健协方差]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

<!-- bilingual-en:start -->
# White's heteroskedasticity test
<!-- bilingual-en:end -->
*White 异方差检验*

<!-- bilingual-en:start -->
A regression's errors can be small near the middle of a predictor's range and large at both ends. A straight-line check of error size against that predictor may miss the pattern. **White's test asks whether squared regression residuals can be explained jointly by the mean predictors, their squares and their pairwise products.** It searches several variance directions at once, without choosing a single narrow variance formula.
<!-- bilingual-en:end -->

回归误差可能在解释变量取值中间较小、两端较大。如果只沿该变量做直线检查，就可能漏掉图案。**White 检验联合考察：均值解释变量、它们的平方和两两乘积，能否解释回归残差平方。** 它同时查看多个方差方向，无需先选定一种狭窄的方差公式。

> [!example] Larger misses at both ends · 两端的偏离都更大
> <!-- bilingual-en:start -->
> Consider nine observations, three at each $x=-1,0,1$. Their outcomes equal $10+2x$ plus the listed deviations. An OLS line recovers $10+2x$, because the deviations sum to zero within each group. Thus they are also its [[回归残差|residuals]].
> <!-- bilingual-en:end -->
>
> 考虑九个观测，在 $x=-1,0,1$ 各有三个。观测结果等于 $10+2x$ 加上下表偏离。每组偏离和为零，所以 OLS 直线恰好拟合出 $10+2x$，这些偏离也就是[[回归残差]]。
>
> | $x$ | Residuals / 残差 | Squared residuals / 平方残差 | Average square / 平均平方 |
> |---|---|---|---|
> | $-1$ | $-2,0,2$ | $4,0,4$ | $8/3$ |
> | $0$ | $-1,0,1$ | $1,0,1$ | $2/3$ |
> | $1$ | $-2,0,2$ | $4,0,4$ | $8/3$ |
>
> <!-- bilingual-en:start -->
> Squared size has no linear tilt from left to right: regressing it on $1,x$ gives slope zero and $R^2=0$. Adding $x^2$ distinguishes the middle from both ends. The fitted squared size becomes $2/3+2x^2$. This explains why a [[Breusch-Pagan检验|BP check]] using only $x$ and a White check using $x,x^2$ can differ. A BP check that deliberately includes $x^2$ can also examine that direction.
> <!-- bilingual-en:end -->
>
> 从左到右，平方大小没有线性倾斜：对 $1,x$ 回归的斜率为零，$R^2=0$。加入 $x^2$ 后，就能把中间与两端区分开，拟合平方大小为 $2/3+2x^2$。这说明为什么只选 $x$ 的 [[Breusch-Pagan检验|BP 检查]]，可以与使用 $x,x^2$ 的 White 检查得到不同结果；主动把 $x^2$ 放进 BP，也能检查这个方向。

<!-- bilingual-en:start -->
## Build the auxiliary regression
<!-- bilingual-en:end -->
*构造辅助回归*

<!-- bilingual-en:start -->
Start with $n$ observations and a linear conditional-mean model containing an intercept and $p$ nonconstant predictors $x_{i1},\ldots,x_{ip}$. Fit it by [[普通最小二乘|OLS]], which minimizes the sum of squared residuals. For observation $i$, write $\hat u_i=y_i-\hat y_i$ and $a_i=\hat u_i^2$. Here $y_i$ is the observed outcome and $\hat y_i$ its fitted mean. The auxiliary outcome is $a_i$, not $y_i$.
<!-- bilingual-en:end -->

从 $n$ 个观测、含截距及 $p$ 个非常数解释变量 $x_{i1},\ldots,x_{ip}$ 的线性条件均值模型出发，用[[普通最小二乘|OLS]]最小化平方残差和。对第 $i$ 个观测，记 $\hat u_i=y_i-\hat y_i$、$a_i=\hat u_i^2$；$y_i$ 是观察结果，$\hat y_i$ 是拟合均值。辅助回归的结果变量是 $a_i$，不是 $y_i$。

<!-- bilingual-en:start -->
With two predictors, the full quadratic auxiliary design contains the columns $1,x_1,x_2,x_1^2,x_1x_2,x_2^2$. The product $x_1x_2$ lets the relationship with one predictor vary with the level of the other. Regress
<!-- bilingual-en:end -->

若原模型有两个解释变量，完整二次辅助设计含列 $1,x_1,x_2,x_1^2,x_1x_2,x_2^2$。乘积 $x_1x_2$ 允许一个变量对应的关系随另一个变量水平改变。拟合

$$a_i=\delta_0+\delta_1x_{i1}+\delta_2x_{i2}+\delta_3x_{i1}^2+\delta_4x_{i1}x_{i2}+\delta_5x_{i2}^2+v_i.$$

<!-- bilingual-en:start -->
The $\delta$ coefficients describe the auxiliary fit, and $v_i$ is what it leaves unexplained. [[异方差|Homoskedasticity]], together with the maintained mean and sampling assumptions, implies zero nonconstant slopes in the population counterpart. Test them jointly. This does not assert that the true variance must itself be a quadratic, nor that every nonconstant variance will have a nonzero projection onto these finitely many columns.
<!-- bilingual-en:end -->

$\delta$ 系数描述辅助拟合，$v_i$ 是剩余未解释部分。在维持的均值与抽样假设下，[[异方差|同方差]]意味着总体对应回归的非常数斜率为零；检验联合考察这些斜率。这没有要求真实方差一定是二次函数，也没有保证任意非恒定方差都会在这有限组列上有非零投影。

<!-- bilingual-en:start -->
Let $W$ contain the auxiliary columns, including the intercept. Set $q=\operatorname{rank}(W)-1$, where [[矩阵秩|rank]] counts independent columns. If $q>0$, $n>\operatorname{rank}(W)$ and the squared residuals vary, use the ordinary centred, unadjusted auxiliary $R^2$:
<!-- bilingual-en:end -->

用 $W$ 收集辅助列，包括截距。令 $q=\operatorname{rank}(W)-1$，其中[[矩阵秩|秩]]计算独立列数。当 $q>0$、$n>\operatorname{rank}(W)$，且平方残差有变化时，使用普通的、中心化且未经调整的辅助 $R^2$：

$$R^2=\frac{\sum_i(\hat a_i-\bar a)^2}{\sum_i(a_i-\bar a)^2},\qquad LM_W=nR^2,\qquad LM_W\xrightarrow{d}\chi_q^2\quad\text{under }H_0.$$

<!-- bilingual-en:start -->
Here $\bar a$ is the average squared residual, $\hat a_i$ the auxiliary fitted value, and the arrow means [[依分布收敛|convergence in distribution]] as sample size grows under the conditions below. Thus the [[卡方分布|chi-squared reference]] is asymptotic. In the nine-observation example, $\bar a=2$, total centred variation is $30$, and explained variation is $8$. Hence $R^2=4/15$, $LM_W=12/5$ and $q=2$. This is an arithmetic illustration, not a claim that the small dataset has reliable chi-squared calibration or establishes heteroskedasticity.
<!-- bilingual-en:end -->

这里 $\bar a$ 是平均残差平方，$\hat a_i$ 是辅助拟合值；箭头表示样本量增大时，在下述条件下[[依分布收敛]]，所以[[卡方分布|卡方参考]]是渐近的。九观测例中，$\bar a=2$、中心化总变化量为 $30$、解释量为 $8$，因此 $R^2=4/15$、$LM_W=12/5$、$q=2$。这是计算示范，不声称这份小数据已有可靠卡方校准，或足以确立异方差。

<!-- bilingual-en:start -->
## Count independent directions, not written terms
<!-- bilingual-en:end -->
*数独立方向，不是数写出来的项*

<!-- bilingual-en:start -->
With $p$ nonconstant mean predictors, the generic full quadratic list has $1+p+p(p+1)/2$ columns. This is only a maximum before dependencies are removed. For a binary indicator $d\in\{0,1\}$, $d^2=d$; including both adds no direction. If two category indicators cannot both be one, their product is the zero column. If the original mean already contains $x$ and $x^2$, different products can reproduce the same power. Sample-specific dependencies can reduce rank further.
<!-- bilingual-en:end -->

原均值有 $p$ 个非常数解释变量时，通常的完整二次列表含 $1+p+p(p+1)/2$ 列；这只是删除线性依赖前的最大数目。对于二元指标 $d\in\{0,1\}$，$d^2=d$，同时加入没有增加方向；两个互斥类别指标的乘积恒为零；若原均值已含 $x$、$x^2$，不同乘积也可能产生同一个幂。特定样本中的列依赖还能继续降低秩。

<!-- bilingual-en:start -->
For $p=20$, the generic list contains $231$ columns, so at most $230$ nonconstant directions are tested. A dataset with only $100$ observations cannot support that full-rank auxiliary regression. Even when a fit is possible, a large $q$ relative to $n$ consumes information and can weaken [[检验功效|power]], the probability of rejecting a false null under a specified alternative. Dropping interactions or using only fitted-value powers defines a different, narrower variant; report the actual columns and rank rather than calling every variant the same full White test.
<!-- bilingual-en:end -->

$p=20$ 时，一般列表含 $231$ 列，至多检验 $230$ 个非常数方向。只有 $100$ 个观测的数据，不可能支持这个满列秩辅助回归。即使能够拟合，$q$ 相对 $n$ 过大也会消耗信息，并可能降低[[检验功效]]，即在指定备择成立时拒绝假零假设的概率。删掉交互项或只用拟合值的幂，会形成另一个更窄的版本；应报告实际列和秩，而不是把所有版本都称作同一个完整 White 检验。

<!-- bilingual-en:start -->
## What justifies the chi-squared comparison?
<!-- bilingual-en:end -->
*卡方比较依靠什么条件？*

<!-- bilingual-en:start -->
The common $nR^2$ test needs more than equal variances. A transparent sufficient setting is a correct linear mean, a fixed number of predictors, bounded fixed design rows, and positive-definite limits for the normalized cross-product matrices of the original and nonredundant auxiliary designs. These [[Gram矩阵正定判据|cross-product matrices]] record inner products of columns; the positive-definite limit prevents directions from collapsing as the sample grows. Under the null, take [[独立同分布|iid]] mean-zero errors with positive variance, the [[矩存在性的使用边界|moment bound]] $E|u_i|^{4+\delta}<\infty$ for some $\delta>0$, and $\operatorname{Var}(u_i^2)>0$. These conditions control squared-error fluctuations and keep the tested directions nondegenerate. Random regressors require corresponding independence/exogeneity, moment and design conditions.
<!-- bilingual-en:end -->

常见 $nR^2$ 检验需要的不只是方差相等。一组清楚的充分条件是：线性均值正确，解释变量数量固定，固定设计各行有界；原设计和去掉冗余列后的辅助设计，其除以样本量的交叉乘积矩阵都有正定极限。这些[[Gram矩阵正定判据|交叉乘积矩阵]]记录各列内积，正定极限防止方向随样本增大而塌缩。零假设下取[[独立同分布|iid]]、均值零、方差为正的误差，并要求[[矩存在性的使用边界|矩条件]]：某个 $\delta>0$ 下 $E|u_i|^{4+\delta}<\infty$，且 $\operatorname{Var}(u_i^2)>0$。这些条件控制平方误差的波动，并使检验方向不退化。随机解释变量还需相应的独立性／外生性、矩与设计条件。

<!-- bilingual-en:start -->
White's original $nR^2$ corollary also imposes a common fourth moment across errors, alongside its other assumptions. Iid errors supply that common moment. Equal conditional variances alone do not control how the fourth moment changes with predictors, so they do not universally validate this simple chi-squared calibration. Nor does this version automatically handle serial or clustered dependence. There are more general moment-based tests, but they use a different covariance normalization.
<!-- bilingual-en:end -->

White 原文的 $nR^2$ 推论还在其他假设之外要求误差具有共同四阶矩；iid 误差提供这个共同矩。条件方差相同本身并不控制四阶矩如何随解释变量变化，所以不能普遍保证这个简单卡方校准正确。该版本也不自动处理序列相关或聚类依赖。更一般的矩检验存在，但需要不同的协方差标准化。

> [!warning] Rejection does not identify a unique cause · 拒绝不识别唯一原因
> <!-- bilingual-en:start -->
> For a fixed candidate mean $g(x)$ and finite conditional second moments,
> $$E[(Y-g(x))^2\mid x]=\operatorname{Var}(Y\mid x)+(E[Y\mid x]-g(x))^2.$$
> Thus a missed curve in the mean can make squared discrepancies predictable even if the true conditional variance is constant. Fitting $g$ from the same data adds estimation effects; the displayed identity is for fixed $g$. A White rejection is evidence against the restrictions together with their maintained assumptions, not proof of a unique variance mechanism. Nonrejection also does not establish a correct mean or constant variance.
> <!-- bilingual-en:end -->
>
> 对固定的候选均值 $g(x)$，在条件二阶矩有限时，上式把平方差分成条件方差和均值误差平方。因此即使真实条件方差恒定，遗漏的均值曲线也能让平方差可预测。若 $g$ 由同一份数据拟合，还会增加估计影响；所列恒等式针对固定的 $g$。White 拒绝是反对所检验限制及其维持假设的证据，不是某一种方差机制的证明；不拒绝同样不能确立均值正确或方差恒定。

<!-- bilingual-en:start -->
White's test produces a diagnostic statistic. [[异方差稳健协方差|White/HC covariance estimation]] estimates coefficient uncertainty; it is a different operation and need not wait for a test rejection. Neither the test nor HC estimation automatically chooses a new mean, [[加权最小二乘|WLS weights]], or a dependent-data correction. Use [[残差协方差诊断|diagnostic context]] and the [[误差协方差决策|sampling structure]] to make that decision. In modest samples the LM approximation can be poor; an auxiliary F output is a variant, not a generally exact finite-sample remedy. Selecting terms after inspecting many tests also requires acknowledging that selection.
<!-- bilingual-en:end -->

White 检验产生诊断统计量；[[异方差稳健协方差|White/HC 协方差估计]]估计系数不确定性，是另一项操作，不必等待检验拒绝。两者都不会自动选出新均值、[[加权最小二乘|WLS 权重]]或依赖数据的修正。应结合[[残差协方差诊断|诊断背景]]与[[误差协方差决策|抽样结构]]做决定。中小样本的 LM 近似可能较差；辅助 F 输出是一种版本，并不是普遍精确的有限样本补救。先尝试很多检验再挑变量，还必须承认这段选择过程。

## 来源与核验

<!-- bilingual-en:start -->
- [White (1980), §3, printed pp.822–826 / PDF pp.7–11](https://crooker.faculty.unlv.edu/econ441/econ_papers/White-Heteroskedasticity-Correction-1980.pdf#page=7), [original DOI](https://doi.org/10.2307/1912934): the auxiliary regression and Corollary 1, homokurtosis and nondegeneracy conditions, redundant terms and degrees of freedom, and mean-specification caveats. The fixed-design setting above is a sufficient specialization; the numerical example and squared-discrepancy identity are derived directly.
- [statsmodels `het_white`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html) and [its implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/diagnostic.html#het_white): squared residual outcome, full quadratic default, $nR^2$, effective model degrees of freedom, and the separate F variant.
- [Stata, *regress postestimation*, PDF pp.19 and 35](https://www.stata.com/manuals/rregresspostestimation.pdf#page=35): the growth in product columns and the full White auxiliary-regression formula.
- [[02_Economy/01_Econometrics/07_异方差.md#3.3. White 异方差检验|The permanent course section]] supplies the two-predictor notation. Its displayed chi-squared comparison is made explicitly asymptotic here, and its term count is refined to the independent-column rank.
<!-- bilingual-en:end -->

- White 第3节核对辅助回归与推论1、共同四阶矩及非退化条件、冗余项与自由度、均值错设边界；正文有界固定设计是充分特例，数值例子和平方差恒等式直接推导。
- statsmodels 文档与实际实现核对残差平方结果变量、默认完整二次项、$nR^2$、有效模型自由度以及另列的 F 版本。
- Stata 手册核对乘积列数量的增长与完整 White 辅助回归公式。
- 永久课程段提供两变量记号；本卡明确卡方比较是渐近的，并将项数精化为独立列的秩。

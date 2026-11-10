---
aliases:
  - Breusch-Pagan test
  - BP heteroskedasticity test
student_os: knowledge-atom
atom_id: ECON-ERR-005
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
requires:
  - "[[异方差]]"
  - "[[残差协方差诊断]]"
related:
  - "[[White异方差检验]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

<!-- bilingual-en:start -->
# Breusch–Pagan test
<!-- bilingual-en:end -->
*Breusch–Pagan 异方差检验*

<!-- bilingual-en:start -->
A regression may predict the average outcome well while making much larger errors in one group than another. The **Breusch–Pagan test** looks for this change in spread along variables chosen in advance. It fits the mean first, then asks whether those variables help explain the **squared residuals**. Squaring turns a large negative miss and a large positive miss into the same measure of size.
<!-- bilingual-en:end -->

回归可能很好地预测平均结果，但在某一组里偏离得比另一组大得多。**Breusch–Pagan 检验**沿着事先选择的变量，寻找这种分散程度的变化。它先拟合均值，再问这些变量能否解释**残差的平方**。平方让很大的负偏离和正偏离，都变成衡量大小的同一种数值。

> [!example] Same average residual, different squared size · 平均残差相同，平方大小不同
> <!-- bilingual-en:start -->
> Fit consumption on an intercept and a group indicator $z$, where $z=0$ labels one income group and $z=1$ the other. Four observations in each group give fitted means $20$ and $30$. Subtracting each fitted mean from its observations gives these [[回归残差|residuals]]:
> <!-- bilingual-en:end -->
>
> 用截距与组别指标 $z$ 拟合消费：$z=0$、$z=1$ 表示两个收入组，每组四个观测，拟合均值分别为 $20$、$30$。观测值减去所在组的拟合均值，得到这些[[回归残差]]：
>
> | Group / 组别 | Residuals / 残差 | Squared residuals / 残差平方 | Average square / 平均平方 |
> |---|---|---|---|
> | $z=0$ | $-1,1,-2,2$ | $1,1,4,4$ | $2.5$ |
> | $z=1$ | $-2,2,-3,3$ | $4,4,9,9$ | $6.5$ |
>
> <!-- bilingual-en:start -->
> Both groups have residual mean zero. Their average squares differ, however. A second regression of the eight squared residuals on $1,z$ fits $2.5+4z$. This **auxiliary regression** describes the size of the misses, rather than re-estimating the consumption mean. The table illustrates the calculation; eight observations do not establish a reliable large-sample test calibration.
> <!-- bilingual-en:end -->
>
> 两组的平均残差都为零，平均平方却不同。把八个残差平方再对 $1,z$ 回归，拟合结果为 $2.5+4z$。这个**辅助回归**描述偏离的大小，不是在重新估计消费均值。表格用于说明计算；八个观测本身不足以保证大样本参考分布可靠。

<!-- bilingual-en:start -->
## From squared residuals to a test statistic
<!-- bilingual-en:end -->
*怎样从平方残差得到检验统计量*

<!-- bilingual-en:start -->
Write the mean model as $y_i=x_i^\top\beta+u_i$ for observations $i=1,\ldots,n$. The vector $x_i$ lists the mean predictors, $\beta$ contains their coefficients, and $u_i$ is the unobserved error. [[普通最小二乘|Ordinary least squares]] chooses $\hat\beta$ to minimize the sum of squared misses. Its residual is $\hat u_i=y_i-x_i^\top\hat\beta$; it is an estimated substitute for $u_i$, not a newly drawn independent error.
<!-- bilingual-en:end -->

对 $i=1,\ldots,n$ 个观测，把均值模型写成 $y_i=x_i^\top\beta+u_i$。向量 $x_i$ 列出均值解释变量，$\beta$ 是对应系数，$u_i$ 是看不到的真实误差。[[普通最小二乘]]选择使平方偏离之和最小的 $\hat\beta$，残差为 $\hat u_i=y_i-x_i^\top\hat\beta$；它是 $u_i$ 的估计代用品，不是重新独立抽到的误差。

<!-- bilingual-en:start -->
With a correctly specified conditional mean, [[异方差|homoskedasticity]] means that the conditional error variance stays at one positive value $\sigma^2$. Choose $q$ variance predictors $z_{i1},\ldots,z_{iq}$, such as education and experience in a wage model, before looking for the most favorable test. They need not be exactly the same variables as the mean predictors. Regress $a_i=\hat u_i^2$ on them and an intercept:
<!-- bilingual-en:end -->

在条件均值设定正确时，[[异方差|同方差]]表示条件误差方差一直是同一个正数 $\sigma^2$。事先选定 $q$ 个方差解释变量 $z_{i1},\ldots,z_{iq}$，例如工资模型中的教育与经验，而不是试到检验最有利为止。它们不必与均值解释变量完全相同。令 $a_i=\hat u_i^2$，将其对这些变量和截距回归：

$$a_i=\alpha_0+\sum_{j=1}^{q}\alpha_jz_{ij}+v_i.$$

<!-- bilingual-en:start -->
Here $\alpha_0$ is the baseline fitted squared size, $\alpha_j$ is the auxiliary slope for predictor $j$, and $v_i$ is the auxiliary residual. The slopes are tested **jointly** against zero. In the intended variance model, the [[零假设与备择假设|null hypothesis]] says that no chosen direction changes the variance. Zero slopes do not characterize every possible constant-variance versus changing-variance alternative: a nonlinear pattern can be orthogonal to the selected columns.
<!-- bilingual-en:end -->

$\alpha_0$ 是拟合平方大小的基准，$\alpha_j$ 是第 $j$ 个变量的辅助斜率，$v_i$ 是辅助残差。检验**联合**考察各斜率是否为零。在所考察的方差模型内，[[零假设与备择假设|零假设]]表示选定方向都不改变方差。但斜率为零并不等价于排除所有异方差：某个非线性图案可能恰好与所选列正交。

<!-- bilingual-en:start -->
Let $\bar a=n^{-1}\sum_i a_i$ and let $\hat a_i$ be the auxiliary fitted values. The centred total variation is $T=\sum_i(a_i-\bar a)^2$, and the explained part is $E=\sum_i(\hat a_i-\bar a)^2$. For $T>0$, the ordinary, unadjusted coefficient of determination is $R^2=E/T$. The commonly used **Koenker studentized version** is
<!-- bilingual-en:end -->

令 $\bar a=n^{-1}\sum_i a_i$，$\hat a_i$ 为辅助拟合值。围绕平方残差的样本平均的总变化量为 $T=\sum_i(a_i-\bar a)^2$，被解释的部分为 $E=\sum_i(\hat a_i-\bar a)^2$。当 $T>0$ 时，普通、未经调整的决定系数为 $R^2=E/T$。常用的 **Koenker 学生化版本**为

$$LM_K=nR^2.$$

<!-- bilingual-en:start -->
In the table, $\bar a=4.5$, $E=4(2.5-4.5)^2+4(6.5-4.5)^2=32$, and $T=66$. Thus $R^2=16/33$ and $LM_K=128/33$. Under the appropriate null and large-sample conditions, compare $LM_K$ with a [[卡方分布|chi-squared distribution]] with $q$ degrees of freedom; larger values give stronger evidence against the tested restrictions. This is an asymptotic comparison, not an exact finite-sample distribution.
<!-- bilingual-en:end -->

表中 $\bar a=4.5$，$E=4(2.5-4.5)^2+4(6.5-4.5)^2=32$，$T=66$，所以 $R^2=16/33$、$LM_K=128/33$。在适当的零假设与大样本条件下，将 $LM_K$ 与自由度为 $q$ 的[[卡方分布]]比较；统计量越大，反对所检验限制的证据越强。这是渐近比较，不是有限样本精确分布。

<!-- bilingual-en:start -->
## Why the original and studentized versions differ
<!-- bilingual-en:end -->
*原始版本与学生化版本为什么不同*

<!-- bilingual-en:start -->
The original normal-theory BP statistic uses $\hat\sigma^2=n^{-1}\sum_i\hat u_i^2=\bar a$ and divides the explained variation by a Gaussian reference scale:
<!-- bilingual-en:end -->

原始、基于正态模型的 BP 统计量使用 $\hat\sigma^2=n^{-1}\sum_i\hat u_i^2=\bar a$，以 Gaussian 参考尺度去除被解释变化量的单位：

$$LM_{BP}=\frac{E}{2(\hat\sigma^2)^2},\qquad LM_K=\frac{E}{T/n}.$$

<!-- bilingual-en:start -->
A mean-zero Gaussian error of variance $\sigma^2$ has $\operatorname{Var}(u_i^2)=2\sigma^4$. Koenker's version instead estimates the spread of squared errors through $T/n$. The two statistics need not be numerically equal: the table gives $LM_{BP}=64/81$, whereas $LM_K=128/33$. Under the normal null their denominators converge to the same value, with the usual regularity conditions. “Studentized” here describes the test's scale estimate; it does not tell you to replace each residual by a leverage-adjusted [[学生化残差|studentized residual]].
<!-- bilingual-en:end -->

均值零、方差 $\sigma^2$ 的 Gaussian 误差满足 $\operatorname{Var}(u_i^2)=2\sigma^4$。Koenker 版本则通过 $T/n$ 估计平方误差自身的分散程度。两个统计量不必数值相等：本表中 $LM_{BP}=64/81$，而 $LM_K=128/33$。在正态零假设与通常正则条件下，二者的分母收敛到相同值。这里“学生化”说的是检验尺度的估计，不是要求逐个换用考虑杠杆值的[[学生化残差]]。

<!-- bilingual-en:start -->
One transparent sufficient setting is a correctly specified linear mean, a fixed number of mean and variance predictors, bounded fixed design rows with nonsingular limiting normalized cross-product matrices after redundant columns are removed, and [[独立同分布|iid]] mean-zero errors. For the studentized version require the [[矩存在性的使用边界|moment condition]] $E|u_i|^{4+\delta}<\infty$ for some $\delta>0$ and $\operatorname{Var}(u_i^2)>0$; normal errors with $\sigma^2>0$ supply these conditions for the original version. The [[Gram矩阵正定判据|cross-product matrices]] collect inner products between design columns; dividing by $n$ puts them on an average scale, and a nonsingular limit keeps independent directions distinguishable. Bounded design rows prevent an individual observation from dominating as the sample grows. Random designs need the corresponding exogeneity, moment and design conditions; merely saying that the observations are iid does not make their errors independent of the predictors. These are sufficient conditions, not the weakest possible theorem.
<!-- bilingual-en:end -->

一组清楚的充分条件是：线性均值正确，均值与方差解释变量的个数固定；固定设计各行有界，删除冗余列后，除以样本量的列交叉乘积矩阵有非奇异极限；误差[[独立同分布]]且均值为零。学生化版本还要求[[矩存在性的使用边界|矩条件]]：某个 $\delta>0$ 下 $E|u_i|^{4+\delta}<\infty$，且 $\operatorname{Var}(u_i^2)>0$；原始版本中，方差 $\sigma^2>0$ 的正态误差提供这些条件。[[Gram矩阵正定判据|交叉乘积矩阵]]收集设计各列的内积，除以 $n$ 后按平均尺度衡量；非奇异极限使独立方向仍可区分。设计各行有界可避免个别观测在样本增大时独占影响。随机设计还需相应的外生性、矩与设计条件；只说完整观测 iid，并不能让误差与解释变量独立。这是一组充分条件，不是最弱定理。

> [!warning] Check the directions and the calibration · 同时检查方向与校准
> <!-- bilingual-en:start -->
> Let $Z$ be the auxiliary matrix including its constant column. The tested dimension is $q=\operatorname{rank}(Z)-1$, where [[矩阵秩|rank]] counts independent columns. Remove redundant predictors rather than counting them twice. Require $q>0$, $n>\operatorname{rank}(Z)$, and nonconstant squared residuals for the usual calculation. A saturated auxiliary fit or $T=0$ does not provide the regular test described here.
>
> Relaxing Gaussianity does not make the test robust to arbitrary serial or clustered dependence, infinite fourth moments, or a growing auxiliary dimension. Some programs also report an auxiliary F statistic; it can help assess finite-sample sensitivity, but squaring fitted residuals does not create an exact classical Gaussian F regression. Report the version, predictors, effective degrees of freedom and reference distribution.
> <!-- bilingual-en:end -->
>
> 令 $Z$ 为包含常数列的辅助矩阵。检验维数为 $q=\operatorname{rank}(Z)-1$，其中[[矩阵秩|秩]]计算线性无关列数。冗余变量应删除，不能重复计数。通常计算要求 $q>0$、$n>\operatorname{rank}(Z)$，且平方残差不全相同。辅助拟合饱和或 $T=0$ 时，不能直接使用这里的常规检验。
>
> 放松 Gaussian 条件，不会使检验自动抵抗任意序列相关、聚类依赖、无穷四阶矩或不断增长的辅助维数。部分软件也报告辅助 F 统计量，可用于核对有限样本敏感性；但对拟合残差平方，并不会造出一个满足经典 Gaussian 精确 F 分布的回归。应报告版本、变量、有效自由度与参考分布。

<!-- bilingual-en:start -->
Nonrejection only means that this procedure did not detect its specified departure; variance may change nonlinearly or with omitted variables. [[White异方差检验|White's test]] uses a larger collection of squares and cross-products. Rejection, in turn, does not identify the economic cause: a wrong mean can also make squared residuals predictable. Return to [[残差协方差诊断|residual diagnosis]] and the [[误差协方差决策|covariance decision]] before choosing between [[异方差稳健协方差|HC inference]] and [[加权最小二乘|variance-based reweighting]]. A design can justify HC inference without a preliminary BP rejection; searching many variance models and reporting only the smallest p-value changes the testing procedure.
<!-- bilingual-en:end -->

不拒绝，只说明这套程序没有发现指定偏离；方差仍可能随变量非线性变化，或随未放入的变量变化。[[White异方差检验]]采用更大的一组平方项与交叉项。反过来，拒绝也不能识别经济来源：均值设错也能使平方残差可预测。应先回到[[残差协方差诊断]]与[[误差协方差决策]]，再区分[[异方差稳健协方差|HC 推断]]和[[加权最小二乘|基于方差的重新加权]]。研究设计可以不经 BP 预检就支持 HC 推断；尝试很多方差模型后只报告最小 p 值，则改变了检验程序。

## 来源与核验

<!-- bilingual-en:start -->
- [Stata, *regress postestimation*, PDF pp.18–19 and 34–35](https://www.stata.com/manuals/rregresspostestimation.pdf#page=18): normal, iid and F variants, the normalized squared-residual auxiliary regression, and their formulas.
- [statsmodels `het_breuschpagan`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html) and [its implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/diagnostic.html#het_breuschpagan): Koenker as the default, $nR^2$ versus explained normalized sum of squares divided by two, intercept/full-rank assumptions and finite-sample cautions.
- [White (1980), §3, printed pp.822–826 / PDF pp.7–11](https://crooker.faculty.unlv.edu/econ441/econ_papers/White-Heteroskedasticity-Correction-1980.pdf#page=7): squared-error moment conditions, nondegenerate auxiliary directions, and the limitations of a homoskedasticity-only interpretation. The bounded fixed-design setting above is a sufficient specialization of the auxiliary-moment argument; the group arithmetic is calculated directly.
- [Breusch and Pagan (1979)](https://doi.org/10.2307/1911963) and [Koenker (1981)](https://doi.org/10.1016/0304-4076(81)90062-2) are the historical original references. Their version distinction is checked here against the accessible official formulas and [Koenker's institutional abstract](https://experts.illinois.edu/en/publications/a-note-on-studentizing-a-test-for-heteroscedasticity/), rather than claiming access to both full papers.
- [[02_Economy/01_Econometrics/07_异方差.md#3.2. Breusch–Pagan 拉格朗日乘数检验|The permanent course section]] supplies local notation; its $nR^2$ recipe is identified here as the studentized version and its chi-squared reference as asymptotic.
<!-- bilingual-en:end -->

- Stata 手册第18–19、34–35页核对正态、iid 与 F 版本，以及标准化残差平方辅助回归和公式。
- statsmodels 文档与实际实现核对 Koenker 默认版本、两种统计量、截距／满秩要求及有限样本限制。
- White 第3节核对平方误差的矩、辅助方向非退化及同方差解释的边界；正文有界固定设计是辅助矩论证的一组充分特例，组别数值直接计算。
- Breusch–Pagan 与 Koenker 的 DOI 保留原始文献入口；版本区别以可访问的官方公式与 Koenker 机构摘要交叉核验，不声称已取得两篇全文。
- 永久课程段保留本地记号；本卡把其 $nR^2$ 步骤明确为学生化版本，把卡方参考明确为渐近分布。

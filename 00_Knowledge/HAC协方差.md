---
aliases:
  - HAC covariance estimator
  - Newey-West standard errors
  - 异方差自相关稳健标准误
student_os: knowledge-atom
atom_id: ECON-ERR-010
atom_set: regression-error-covariance
atom_type: covariance-estimator
status: source-checked
requires:
  - "[[协方差失效下的OLS]]"
  - "[[误差序列相关]]"
related:
  - "[[标准误口径匹配]]"
  - "[[聚类标准误边界]]"
  - "[[自协方差与ACF]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

<!-- bilingual-en:start -->
# HAC covariance
<!-- bilingual-en:end -->

*HAC 协方差*

<!-- bilingual-en:start -->
A run of unusually high sales this month and next month may reflect the same persistent shock. Counting the two months as independent pieces of information can misstate the uncertainty of a regression coefficient. **HAC** means *heteroskedasticity and autocorrelation consistent*: it estimates coefficient uncertainty while allowing unequal error variances and specified forms of dependence over time. It keeps the [[普通最小二乘|OLS]] coefficients and adds information from products of residual contributions at different time lags. [[异方差稳健协方差|HC covariance]] uses only the same-observation products.
<!-- bilingual-en:end -->

这个月和下个月销售额都异常偏高，可能来自同一个持续冲击。把两个月当成相互独立的信息，可能误判回归系数的不确定性。**HAC** 是 *heteroskedasticity and autocorrelation consistent* 的缩写：在允许误差方差不同及一定时间依赖的条件下，估计系数的不确定性。它保留[[普通最小二乘|OLS]] 系数，把不同时距的残差贡献乘积加入计算；[[异方差稳健协方差|HC 协方差]]只使用同一观测的乘积。

> [!example] Four ordered observations · 四个按时间排序的观测
> <!-- bilingual-en:start -->
> Fit a constant to $y=(1,1,-1,-1)^T$. The fitted mean is $0$, so these four values are also the residuals. Their squared sum is $4$. Products of adjacent residuals sum to $1\cdot1+(-1)\cdot1+(-1)\cdot(-1)=1$.
>
> A Newey–West calculation retaining lag one with weight $1/2$ uses $4+2(1/2)(1)=5$ instead of $4$. Divide by $4^2$, because the fitted mean averages four observations: the estimated variance is $5/16$, compared with HC0's $4/16$. The standard errors are $\sqrt5/4$ and $1/2$; the fitted mean remains $0$. Four observations illustrate the arithmetic, not the reliability of asymptotic inference.
> <!-- bilingual-en:end -->
>
> 对 $y=(1,1,-1,-1)^T$ 只拟合一个常数，拟合均值是 $0$，四个值也就是残差。残差平方和为 $4$；相邻残差乘积之和为 $1\cdot1+(-1)\cdot1+(-1)\cdot(-1)=1$。
>
> Newey–West 计算若保留一阶滞后、权重取 $1/2$，就把 $4$ 改成 $4+2(1/2)(1)=5$。拟合均值平均了四个观测，因此再除以 $4^2$，估计方差为 $5/16$，而 HC0 为 $4/16$。相应标准误是 $\sqrt5/4$ 和 $1/2$，拟合均值仍为 $0$。四个观测仅展示算法，不说明渐近推断已经可靠。

<!-- bilingual-en:start -->
## From lagged products to a covariance matrix
<!-- bilingual-en:end -->

*从滞后乘积到协方差矩阵*

<!-- bilingual-en:start -->
For $T$ observations in time order and $k$ regression coefficients, write $y_t=x_t^T\beta+u_t$. The column vector $x_t$ contains observation $t$'s predictors, including $1$ for an intercept; the design matrix $X$ stacks the rows $x_t^T$. Here the superscript $T$ means [[矩阵转置|transpose]]. Assume $X$ has [[线性无关|linearly independent columns]] and $T>k$. Compute the [[回归残差|residual]] $\hat u_t=y_t-x_t^T\hat\beta$ and its contribution $g_t=x_t\hat u_t$, a $k$-component column vector. This is a contribution to the OLS estimating equations, not a claim about a likelihood score.
<!-- bilingual-en:end -->

对按时间排列的 $T$ 个观测、$k$ 个回归系数，写 $y_t=x_t^T\beta+u_t$。列向量 $x_t$ 包含第 $t$ 个观测的解释变量；含截距时也包含 $1$，设计矩阵 $X$ 由行 $x_t^T$ 叠成。上标 $T$ 表示[[矩阵转置]]。假定 $X$ 各列[[线性无关]]且 $T>k$。先计算[[回归残差]] $\hat u_t=y_t-x_t^T\hat\beta$，再令 $g_t=x_t\hat u_t$，它是含 $k$ 个分量的列向量，也是 OLS 估计方程的逐观测贡献；这里并未把它认作似然得分。

<!-- bilingual-en:start -->
Lag $\ell$ pairs an observation with the one $\ell$ periods earlier. Use **unnormalised sums**, with no division by $T$:
<!-- bilingual-en:end -->

滞后 $\ell$ 把一个观测与早 $\ell$ 期的观测配对。以下采用**未除以 $T$ 的求和口径**：

$$
\widehat G_0=\sum_{t=1}^{T}g_tg_t^T,
\qquad
\widehat G_\ell=\sum_{t=\ell+1}^{T}g_tg_{t-\ell}^T.
$$

<!-- bilingual-en:start -->
Choose an integer maximum lag $0\le L<T$. The common **Newey–West/Bartlett** version weights lag $\ell$ by $w_\ell=1-\ell/(L+1)$: nearer pairs receive more weight. Including both $\widehat G_\ell$ and its transpose accounts for both directions of a pair and makes the middle matrix symmetric:
<!-- bilingual-en:end -->

选整数最大滞后 $0\le L<T$。常用的 **Newey–West/Bartlett** 版本取 $w_\ell=1-\ell/(L+1)$，距离较近的配对权重更高。同时加入 $\widehat G_\ell$ 及其转置，纳入一对观测的两个方向，并使中间矩阵对称：

$$
\widehat S_L=\widehat G_0+
\sum_{\ell=1}^{L}w_\ell(\widehat G_\ell+\widehat G_\ell^T),
\qquad
\widehat V_{\mathrm{HAC}}
=(X^TX)^{-1}\widehat S_L(X^TX)^{-1}.
$$

<!-- bilingual-en:start -->
The outer matrices translate these contributions into coefficient uncertainty, following the [[协方差失效下的OLS|general OLS covariance formula]]. The $j$th [[标准误含义|standard error]] is $\sqrt{(\widehat V_{\mathrm{HAC}})_{jj}}$. Bartlett's construction is [[半正定矩阵|positive semidefinite]]: every real coefficient combination has nonnegative estimated variance. It need not be positive definite, and nonnegativity alone does not establish correct statistical calibration. Other HAC kernels exist; HAC is a broader family than this particular formula.
<!-- bilingual-en:end -->

两侧矩阵按[[协方差失效下的OLS|一般 OLS 协方差公式]]，把逐观测贡献转换成系数不确定性。第 $j$ 个[[标准误含义|标准误]]是 $\sqrt{(\widehat V_{\mathrm{HAC}})_{jj}}$。Bartlett 构造是[[半正定矩阵|半正定的]]，所以任意实系数组合的估计方差非负；它不一定正定，而且非负并不证明统计校准正确。HAC 还有其他核函数版本，范围比这一条具体公式更广。

> [!note] Check the normalisation · 检查归一化口径
> <!-- bilingual-en:start -->
> With $L=0$ and **no finite-sample correction**, this is exactly HC0. If instead $\widehat\Omega_L=\widehat S_L/T$ and $\widehat Q=X^TX/T$ are averages, the same formula is $\widehat V_{\mathrm{HAC}}=T^{-1}\widehat Q^{-1}\widehat\Omega_L\widehat Q^{-1}$. Equivalently, putting $\widehat\Omega_L$ between two $(X^TX)^{-1}$ factors requires an extra factor $T$. Do not mix sums and averages. Software may additionally multiply by $T/(T-k)$; that is a separate correction, so its $L=0$ output need not equal uncorrected HC0.
> <!-- bilingual-en:end -->
>
> $L=0$ 且**不作小样本修正**时，公式恰好就是 HC0。若采用平均口径 $\widehat\Omega_L=\widehat S_L/T$、$\widehat Q=X^TX/T$，同一公式是 $\widehat V_{\mathrm{HAC}}=T^{-1}\widehat Q^{-1}\widehat\Omega_L\widehat Q^{-1}$。等价地，若把 $\widehat\Omega_L$ 放在两个 $(X^TX)^{-1}$ 之间，外面须再乘 $T$。求和与平均不能混用。软件还可能另乘 $T/(T-k)$；这是独立的小样本修正，因此它在 $L=0$ 时未必等于未经修正的 HC0。

<!-- bilingual-en:start -->
## What “consistent” requires
<!-- bilingual-en:end -->

*“一致”需要什么条件*

<!-- bilingual-en:start -->
The object being estimated is the covariance governing the large-sample fluctuation of OLS. A common [[宽平稳定义|covariance-stationary]] setting, with finite second moments, constant means and lag-only covariances, makes this concrete: let $z_t=x_tu_t$ have mean zero and lag covariance $\Gamma_\ell=E[z_tz_{t-\ell}^T]$. If these covariances are absolutely summable (each matrix entry has a finite sum of absolute values over lags), their long-run sum is
<!-- bilingual-en:end -->

被估计的对象是支配 OLS 大样本波动的协方差。用一个常见的[[宽平稳定义|协方差平稳]]情形说明，也就是二阶矩有限、均值不变且协方差只依赖时距：令 $z_t=x_tu_t$ 均值为零，滞后协方差为 $\Gamma_\ell=E[z_tz_{t-\ell}^T]$。若这些协方差绝对可和（每个矩阵位置上的元素取绝对值后，沿滞后求和有限），长期协方差之和为

$$
\Omega=\Gamma_0+\sum_{\ell=1}^{\infty}(\Gamma_\ell+\Gamma_\ell^T).
$$

<!-- bilingual-en:start -->
For this route with a fixed number $k$ of coefficients, one also needs a stable invertible design limit $X^TX/T\xrightarrow{p} Q$ ([[依概率收敛|convergence in probability]]), a suitable central limit theorem giving $T^{-1/2}\sum_tz_t$ a limiting mean-zero Gaussian distribution with covariance $\Omega$, and control of the error from replacing $u_t$ by residuals. Then the limiting Gaussian law of $\sqrt T(\hat\beta-\beta)$ has covariance $Q^{-1}\Omega Q^{-1}$. “Weak dependence” is a description, not a complete assumption list: moment bounds, the rate at which dependence decays, and the bandwidth rule must support these steps. Absolute summability alone does not supply every needed theorem. The [[样本均值方差判据|sample-mean covariance sum]] explains the scalar version of the long-run target.
<!-- bilingual-en:end -->

走这条系数个数 $k$ 固定的论证路线，还需要稳定且可逆的设计极限 $X^TX/T\xrightarrow{p}Q$（[[依概率收敛]]）、让 $T^{-1/2}\sum_tz_t$ 趋于零均值、协方差为 $\Omega$ 的 Gaussian 分布的适用中心极限定理，以及用残差替换 $u_t$ 所带来误差的控制。此时 $\sqrt T(\hat\beta-\beta)$ 的极限 Gaussian 分布具有协方差 $Q^{-1}\Omega Q^{-1}$。“弱依赖”只是描述，不是一份完整假设：矩条件、依赖衰减速度和带宽规则都必须支持这些步骤；仅有协方差绝对可和并不能自动提供所有所需定理。[[样本均值方差判据]]解释了长期协方差目标的标量版本。

<!-- bilingual-en:start -->
For general short-memory dependence, the usual consistency route lets $L=L_T$ grow with $T$, but slowly enough that noisy long-lag estimates do not dominate; each fixed lag's weight then tends to one. The original Newey–West sufficient theorem assumes explicit moment, smoothness, mixing and parameter-estimation conditions and uses $L_T\to\infty$ with $L_T/T^{1/4}\to0$. This is one sufficient theorem, not a universal optimal bandwidth or a claim that a growth rate by itself guarantees validity. “Mixing” here quantifies how dependence between distant blocks of observations decays. Keeping a Bartlett bandwidth fixed is not generally consistent for the long-run target.
<!-- bilingual-en:end -->

对于一般短记忆依赖，通常的一致性路线让 $L=L_T$ 随 $T$ 增长，但增长足够慢，使较远滞后的估计噪声不至于主导结果；每个固定滞后的权重也因而趋近一。Newey–West 原文的一组充分条件包含明确的矩、光滑性、混合性和参数估计条件，并使用 $L_T\to\infty$、$L_T/T^{1/4}\to0$。这是一条充分定理，不是普遍最优带宽，也不是只要增长速度满足就必然有效；其中“混合性”量化相距很远的两段观测之间依赖如何衰减。把 Bartlett 带宽固定住，一般不能一致估计长期目标。

> [!warning] Even a mild AR(1) can defeat a fixed bandwidth · 温和的 AR(1) 也能说明固定带宽的问题
> <!-- bilingual-en:start -->
> Consider a stationary zero-mean process $u_t=\tfrac12u_{t-1}+\varepsilon_t$, with independent, identically distributed zero-mean Gaussian innovations of positive finite variance, independent of past errors. Independence of the new innovation gives the recursion $\gamma_\ell=(1/2)\gamma_{\ell-1}$. Thus its [[自协方差与ACF|lag covariance]] is $\gamma_\ell=\gamma_0(1/2)^\ell$, where $\gamma_0=\operatorname{Var}(u_t)$. For an intercept-only regression, the true long-run variance is $\gamma_0+2\sum_{\ell\ge1}\gamma_\ell=3\gamma_0$. Keeping $L=1$ gives a Bartlett middle average tending to $\gamma_0+2(1/2)\gamma_1=1.5\gamma_0$, only half the target. More data do not remove this fixed truncation-and-weighting error. The sum uses the [[几何级数|geometric series]].
> <!-- bilingual-en:end -->
>
> 考虑平稳、零均值过程 $u_t=\tfrac12u_{t-1}+\varepsilon_t$，其中创新项独立同分布、服从零均值、正有限方差的 Gaussian 分布，并独立于过去误差。创新项独立于过去误差，给出递推 $\gamma_\ell=(1/2)\gamma_{\ell-1}$，因此它的[[自协方差与ACF|滞后协方差]]为 $\gamma_\ell=\gamma_0(1/2)^\ell$，$\gamma_0=\operatorname{Var}(u_t)$。只含截距的回归中，真正的长期方差是 $\gamma_0+2\sum_{\ell\ge1}\gamma_\ell=3\gamma_0$。固定 $L=1$ 时，Bartlett 中间平均量却趋于 $\gamma_0+2(1/2)\gamma_1=1.5\gamma_0$，只有目标的一半。增加观测并不能消除这个固定的截断与加权误差；求和使用了[[几何级数]]。

<!-- bilingual-en:start -->
## Choose a time scale, and check the model
<!-- bilingual-en:end -->

*选择时间尺度，同时检查模型*

<!-- bilingual-en:start -->
A short bandwidth omits dependence; a very long one adds noisy estimates. State the data frequency, the anticipated dependence horizon, the chosen kernel and bandwidth rule, any finite-sample correction, and sensitivity to reasonable alternatives. Information criteria used to select a dynamic mean or ARMA order are not, without a separate argument, a generic HAC-bandwidth rule. With monthly data, $L=1$ includes only one-month products: it does not allow arbitrary dependence throughout a year.
<!-- bilingual-en:end -->

短带宽会遗漏依赖，过长带宽则引入嘈杂估计。应说明数据频率、预期依赖时间尺度、核与带宽规则、小样本修正，以及对合理替代选择的敏感性。用于动态均值或 ARMA 阶数的信息准则，不能在没有额外论证时直接充当通用 HAC 带宽规则。月度数据取 $L=1$ 只纳入相隔一个月的乘积，并不等于允许一年内任意依赖。

<!-- bilingual-en:start -->
Ordinary time-series HAC treats rows as consecutive equally spaced observations of one series. Concatenating different firms' panel records creates artificial neighbours and misses other relationships; the [[标准误口径匹配|covariance structure must match the design]]. HAC also supplies no remedy for a coefficient targeting the wrong quantity through omitted variables or endogeneity. If the goal is the conditional mean and residuals reveal omitted predictable dynamics, revisit that mean; if the target is explicitly a linear projection, justify inference for that target. Breaks, long memory and extreme persistence need their own theory. HAC standard errors alone do not turn a statistic into an exact finite-sample $t$ or $F$ pivot.
<!-- bilingual-en:end -->

普通时间序列 HAC 把各行视为单一序列中连续、等间隔的观测。把不同企业的面板记录串成一列，会造出虚假的相邻关系并遗漏其他关联，须让[[标准误口径匹配|协方差结构匹配设计]]。HAC 也不能修复遗漏变量或内生性导致的错误系数目标。若目标是条件均值，而残差显示遗漏了可预测动态，应重新检查均值设定；若明确研究线性投影，则应为该目标论证推断。结构突变、长记忆和极强持久性需要各自的理论；仅换成 HAC 标准误，不会让统计量自动成为精确有限样本 $t$ 或 $F$ 枢轴量。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Newey and West, original NBER working paper T0055](https://www.nber.org/system/files/working_papers/t0055/t0055.pdf), PDF pp. 4–8, equations (4)–(5), Theorems 1–2: lag-product normalisation, Bartlett positive semidefiniteness and explicit sufficient consistency conditions. The journal version is [Newey and West (1987)](https://doi.org/10.2307/1913610); verification here uses the accessible original working paper.
- [statsmodels sandwich covariance source](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html), `weights_bartlett`, `S_hac_simple`, `cov_hac_simple`: unnormalised sums, equal spacing, and the separate $T/(T-k)$ implementation correction.
- [Penn State STAT 501, Time Series and Autocorrelation](https://online.stat.psu.edu/stat501/Topic2TSandAutocorr.html), autoregressive-error section: stationary AR(1) setup. The $\rho=1/2$ long-run comparison and four-observation example are derived and checked here.
- [[02_Economy/01_Econometrics/08_自相关.md#5.3. Newey–West/HAC 修正推断|Local course section 5.3]]: course notation, HC0 reduction and application boundaries.
<!-- bilingual-en:end -->

- [Newey–West 原始 NBER 工作论文 T0055](https://www.nber.org/system/files/working_papers/t0055/t0055.pdf) PDF 第 4–8 页、式 (4)–(5)、定理 1–2：核对滞后乘积归一化、Bartlett 半正定性及明确的一致性充分条件。期刊版为 [Newey–West (1987)](https://doi.org/10.2307/1913610)，本次实际核验使用可访问的原始工作论文。
- [statsmodels sandwich covariance 源码](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html)：核对 `weights_bartlett`、`S_hac_simple`、`cov_hac_simple` 中的未归一化求和、等间隔索引和独立的 $T/(T-k)$ 软件修正。
- [Penn State STAT 501 时间序列与自相关](https://online.stat.psu.edu/stat501/Topic2TSandAutocorr.html) 的自回归误差部分：核对平稳 AR(1) 设定；$\rho=1/2$ 的长期方差比较及四观测例子在本卡独立推导、复算。
- [[02_Economy/01_Econometrics/08_自相关.md#5.3. Newey–West/HAC 修正推断|本地课程第 5.3 节]]：核对课程符号、退化到 HC0 及使用边界。

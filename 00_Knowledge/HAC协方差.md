---
aliases:
  - "HAC 协方差用加权得分交叉乘积和估计异方差和弱序列依赖下的不确定性"
  - "HAC 协方差用加权样本自协方差估计异方差和弱序列依赖下的长期不确定性"
  - HAC covariance estimator
  - Newey-West standard errors
  - 异方差自相关稳健标准误
student_os: knowledge-atom
atom_id: ECON-ERR-010
atom_set: regression-error-covariance
atom_type: covariance-estimator
status: source-checked
mastery_state: unassessed
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

# HAC 协方差用加权得分交叉乘积和估计异方差和弱序列依赖下的不确定性
<!-- bilingual-en:start -->
*A HAC covariance estimator uses weighted score cross-product sums to estimate uncertainty under heteroskedasticity and weak serial dependence*
<!-- bilingual-en:end -->

> [!summary] Newey–West 在估计什么
> 令第 $t$ 个 OLS 得分贡献为列向量 $g_t=x_t\hat u_t$。这里采用与常见软件三明治实现一致的**未除以 $T$ 的交叉乘积和**：
> $$
> \widehat G_0=\sum_{t=1}^{T}g_tg_t',
> \qquad
> \widehat G_\ell=\sum_{t=\ell+1}^{T}g_tg_{t-\ell}'\quad(\ell\ge1),
> $$
> 再构造
> $$
> \widehat S_L=\widehat G_0+
> \sum_{\ell=1}^{L}w_\ell
> (\widehat G_\ell+\widehat G_\ell'),
> $$
> 并放入三明治外壳
> $$
> \widehat{\operatorname{Var}}_{HAC}(\hat\beta)
> =(X'X)^{-1}\widehat S_L(X'X)^{-1}.
> $$
> Bartlett 权重 $w_\ell=1-\ell/(L+1)$ 让较远滞后的贡献逐渐衰减。
>
> <!-- bilingual-en:start -->
> With $g_t=x_t\hat u_t$, define unnormalised lag cross-product sums $\widehat G_0=\sum_t g_tg_t'$ and $\widehat G_\ell=\sum_{t=\ell+1}^T g_tg_{t-\ell}'$. A kernel-weighted sum $\widehat S_L$ is then placed directly inside the OLS sandwich. No hidden factor of $T$ is missing under this convention.
> <!-- bilingual-en:end -->

这个归一化口径可以直接做退化核对。若 $L=0$ 且不加任何小样本修正，
$$
\widehat S_0=\sum_{t=1}^{T}x_tx_t'\hat u_t^2
=X'\operatorname{diag}(\hat u_1^2,\ldots,\hat u_T^2)X,
$$
所以三明治恰好退化为 HC0。也可以把 $\widehat\Gamma_\ell=T^{-1}\widehat G_\ell$ 定义为平均样本自协方差，但那时外壳前必须相应乘回 $T$；不能把“除以 $T$ 的 $\Gamma_\ell$”和“未乘 $T$ 的外壳”混在同一公式里。

<!-- bilingual-en:start -->
The normalisation can be checked algebraically. With $L=0$ and no finite-sample correction, the middle matrix becomes $X'\operatorname{diag}(\hat u_t^2)X$, so the sandwich is exactly HC0. An averaged convention $\widehat\Gamma_\ell=T^{-1}\widehat G_\ell$ is also valid only if the sandwich restores the corresponding factor of $T$; the two conventions must not be mixed.
<!-- bilingual-en:end -->

它保留 OLS 系数，只改变推断；“H”允许条件方差变化，“AC”允许一定跨期相关。有效性不是说误差可以任意依赖，而依赖弱相关、适当矩条件、正确的长期平均目标，以及带宽 $L$ 随样本量恰当增长等渐近条件。结构突变、长记忆、极强持久性或错误均值模型都可能使普通 HAC 近似失真。
<!-- bilingual-en:start -->
HAC retains OLS coefficients. Its validity does not permit arbitrary dependence: it relies on weak dependence, appropriate moments, a stable long-run target, and suitable bandwidth asymptotics. Breaks, long memory, extreme persistence, or a wrong mean equation can invalidate the usual approximation.
<!-- bilingual-en:end -->

带宽是偏差—方差选择。$L$ 太小会漏掉仍有意义的滞后协方差；太大会引入许多噪声估计并降低精度。它应由数据频率、依赖时间尺度和预先说明的自动规则共同决定，并报告敏感性。信息准则通常用于选择动态均值/ARMA 阶数，不能无说明地当作 Newey–West 带宽的通用规则。
<!-- bilingual-en:start -->
Bandwidth trades omitted dependence against noisy long-lag covariance estimates. Choose it from data frequency, the dependence horizon, and a stated automatic rule, and report sensitivity. Information criteria for dynamic mean or ARMA order are not a universal HAC-bandwidth rule.
<!-- bilingual-en:end -->

实现还要匹配索引：普通时间序列 HAC 假定观测按连续等间隔时间排序。面板数据同时有组内与时间维度时，应使用与该结构匹配的面板/多维方法，而不是把所有行串成一条时间序列。HAC 也不会修复遗漏动态；若 BG 或残差图显示均值可预测，应先问是否需要重设模型。
<!-- bilingual-en:start -->
Implementation must match indexing. Ordinary time-series HAC assumes observations are ordered along one equally spaced series. Panel dependence requires an estimator designed for its group and time structure. HAC also cannot repair omitted dynamics in the conditional mean.
<!-- bilingual-en:end -->

> [!question]- 自检
> 月度数据取 $L=1$ 得到 HAC 标准误，是否已经允许“一年内任意相关”？
>
> **答案：** 没有。$L=1$ 只把一阶样本协方差放入核窗口；若一年是相关窗口，应有依据地考虑更长带宽并做敏感性检查。

## 来源与核验

- [Newey and West (1987), *A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*](https://doi.org/10.2307/1913610)：核对正半定 HAC 构造与一致性目标。
- [statsmodels `S_hac_simple` / `cov_hac_simple` 源码](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html)：核对内部中间矩阵使用未除以 $T$ 的当期与滞后交叉乘积和，以及 $L=0$ 时与 White/HC0 中间矩阵一致。
- [[02_Economy/01_Econometrics/08_自相关.md#5.3. Newey–West/HAC 修正推断|本地课程：Newey–West]]与[[标准误口径匹配]]：核对课程公式和 HC/HAC/cluster 的口径边界。

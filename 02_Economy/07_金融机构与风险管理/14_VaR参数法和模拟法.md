
# 1. VaR的定义与方差-协方差法原理（含正态分布假设）
<!-- bilingual-en:start -->
*1. Definition of VaR and the Variance–Covariance Method, Including the Normality Assumption*
<!-- bilingual-en:end -->

和前面一样[[VaR定义|VaR]]
<!-- bilingual-en:start -->
This is the same VaR concept introduced earlier.
<!-- bilingual-en:end -->

# 2. 单一资产VaR计算公式与例子
<!-- bilingual-en:start -->
*2. Formula and Example for the VaR of a Single Asset*
<!-- bilingual-en:end -->

对于单一资产，若头寸市值为$V$，资产日收益波动率（标准差）为$\sigma$，则在均值近似为0的情况下，一天$\alpha$置信水平VaR计算公式为： 
<!-- bilingual-en:start -->
For a single asset with position value $V$ and daily return volatility, or standard deviation, $\sigma$, and with the mean return approximated as zero, one-day VaR at confidence level $\alpha$ is:
<!-- bilingual-en:end -->

$$ VaR_{1天} = z_{\alpha}\,\sigma\,V $$

其中$z_{\alpha}$是正态分布$(1-\alpha)$右尾概率的分位数（例如$\alpha=99\%$时$z_{0.99}=2.33$，$\alpha=95\%$时$z_{0.95}=1.65$)。该公式表示资产价值变化的标准差乘以相应置信水平下的标准正态系数.
<!-- bilingual-en:start -->
Here $z_{\alpha}$ is the standard-normal quantile leaving probability $(1-\alpha)$ in the right tail. For example, $z_{0.99}=2.33$ when $\alpha=99\%$, and $z_{0.95}=1.65$ when $\alpha=95\%$. The formula multiplies the standard deviation of the asset's value change by the normal critical value for the chosen confidence level.
<!-- bilingual-en:end -->

<!-- GB-31:start -->
> [!note] GB-31｜单资产公式的头寸符号
> 上式沿用多头市值 $V\ge0$。若 $V$ 表示带正负号的线性敞口，零均值正态模型下标准差应为 $|V|\sigma$，而不是负的 $V\sigma$；非零均值还须单独加入损失均值。见 [[方差协方差VaR]]，此式不是适用于任意收益分布的 [[VaR定义]]。
> <!-- bilingual-en:start -->
> The displayed formula assumes a nonnegative long-position value. With signed linear exposure $V$, zero-mean normal P&L has standard deviation $|V|\sigma$; a nonzero loss mean must be added separately. See [[方差协方差VaR|variance–covariance VaR]]. This is a model formula, not the general [[VaR定义|definition of VaR]].
> <!-- bilingual-en:end -->
<!-- GB-31:end -->

# 3. 多资产组合VaR计算：协方差矩阵与线性组合原理
<!-- bilingual-en:start -->
*3. Multi-Asset Portfolio VaR: Covariance Matrices and Linear Combinations*
<!-- bilingual-en:end -->

对于包含多种资产的投资组合，假设资产价格变化近似线性且收益服从联合正态分布，则组合损益仍为正态。可通过协方差矩阵计算组合的总体方差：
<!-- bilingual-en:start -->
For a portfolio of several assets, suppose price changes are approximately linear and returns are jointly normal. Portfolio profit and loss is then also normal, and its total variance can be calculated from the covariance matrix:
<!-- bilingual-en:end -->

$$ \sigma_p^2 = \sum_{i=1}^{n}\sum_{j=1}^{n} V_i\,V_j\,\sigma_i\,\sigma_j\,\rho_{ij}, $$

其中$V_i$和$\sigma_i$分别为第$i$项资产的头寸价值和日波动率，$\rho_{ij}$为资产$i$与$j$收益的相关系数。组合标准差$\sigma_p$为上述方差的平方根。对于两个资产的特殊情况，组合日标准差公式为：
<!-- bilingual-en:start -->
Here $V_i$ and $\sigma_i$ are the value and daily volatility of position $i$, while $\rho_{ij}$ is the correlation between the returns on assets $i$ and $j$. Portfolio standard deviation $\sigma_p$ is the square root of this variance. For two assets, the daily portfolio standard deviation is:
<!-- bilingual-en:end -->

$$ \sigma_{X+Y} = \sqrt{(V_X\sigma_X)^2 + (V_Y\sigma_Y)^2 + 2\,\rho_{XY}\,V_X\sigma_X\,V_Y\sigma_Y}\,. $$

由此，一天VaR $= z_{\alpha}\,\sigma_p$。当相关系数$\rho_{ij}<1$时，组合方差小于各单项方差之和，因此组合VaR小于单项VaR之和，体现了分散化降低风险的作用 
<!-- bilingual-en:start -->
One-day VaR is therefore $z_{\alpha}\,\sigma_p$. When $\rho_{ij}<1$, diversification makes portfolio VaR smaller than the sum of stand-alone VaRs, although the precise comparison is between standard deviations rather than raw variances.
<!-- bilingual-en:end -->

<!-- GB-32:start -->
> [!note] GB-32｜分散化比较的是标准差，不是方差之和
> 对两项非退化多头敞口，$0<\rho<1$ 时协方差项为正，因此组合方差反而大于两项方差之和；但组合标准差小于两项标准差之和。零均值联合正态、$\alpha>1/2$ 的同口径 VaR 随标准差成比例，才得到这里的 VaR 分散化结论。一般带符号头寸须保留协方差项及符号，见 [[和的方差协方差项]]。
> <!-- bilingual-en:start -->
> For two nondegenerate long exposures with $0<\rho<1$, the positive covariance term makes portfolio variance greater than the sum of component variances, while its standard deviation is smaller than the sum of component standard deviations. The VaR comparison follows under zero-mean joint normality with a common confidence level above one half. Retain signed exposures and every covariance term; see [[和的方差协方差项|the variance of a sum]].
> <!-- bilingual-en:end -->
<!-- GB-32:end -->

# 4. 利率类资产的VaR估算：现金流映射、主成分分析（PCA）
<!-- bilingual-en:start -->
*4. VaR for Interest-Rate Positions: Cash-Flow Mapping and Principal Component Analysis (PCA)*
<!-- bilingual-en:end -->

**现金流映射法（Cash-flow Mapping）**：将非标准现金流按期限桶线性映射至标准节点，便于后续久期/方差等统一口径计算。~~略~~
<!-- bilingual-en:start -->
**Cash-flow mapping:** Map non-standard cash flows linearly into standard maturity buckets so that duration, variance, and other risk measures can be calculated on a consistent set of nodes. ~~Details omitted in the original outline.~~
<!-- bilingual-en:end -->

 **1  原始现金流梳理**
<!-- bilingual-en:start -->
**1. Organise the original cash flows**
<!-- bilingual-en:end -->

| **发生时点 $t$ (年)** | **现金流 $CF_t$ (美元)** | **备注**                        |
| ---------------- | ------------------- | ----------------------------- |
| $0.3$            | $50,000$            | 第一次半年息 $$1,000,000\times5\%$$ |
| $0.8$            | $50,000$            | 第二次半年息                        |
| $0.8$            | $1,000,000$         | 偿还本金                          |
| **合计**           | **$1,100,000$**     | –                             |
<!-- bilingual-en:start -->
| **Time $t$ (years)** | **Cash flow $CF_t$ (USD)** | **Description** |
| --- | ---: | --- |
| $0.3$ | $50,000$ | First semiannual coupon |
| $0.8$ | $50,000$ | Second semiannual coupon |
| $0.8$ | $1,000,000$ | Principal repayment |
| **Total** | **$1,100,000$** | – |
<!-- bilingual-en:end -->

> 把 $0.8$ 年那两笔加在一起，可视作一张 **面值 $1,050,000$** 、到期 $0.8$ 年的零息债；而 $0.3$ 年那笔直接是一张 **面值 $50,000$** 的 0.3 年零息债。
> <!-- bilingual-en:start -->
> Combine the two cash flows at $0.8$ years and treat them as a zero-coupon bond with face value **$1,050,000$** and maturity $0.8$ years. The cash flow at $0.3$ years is itself a zero-coupon bond with face value **$50,000$** and maturity 0.3 years.
> <!-- bilingual-en:end -->

<!-- GB-33:start -->
> [!note] GB-33｜现金流表的息票周期
> 本表按已给每半年 $50000$ 美元保留后续映射。这样 $5\%$ 必须是每半年的息票率；若题意是年票面利率 $5\%$、一年付息两次，则每次应为 $25000$ 美元。不能混用这两种口径，见 [[债券付息周期换算]]。
> <!-- bilingual-en:start -->
> The mapping below retains the given USD 50,000 semiannual cash flows. They imply a 5% coupon per half-year; a 5% annual coupon paid semiannually would instead pay USD 25,000 each time. See [[债券付息周期换算|coupon-period conversion]].
> <!-- bilingual-en:end -->
<!-- GB-33:end -->

 **2  为什么要拆到“标准桶”？**
<!-- bilingual-en:start -->
**2. Why map the cash flows into standard buckets?**
<!-- bilingual-en:end -->

- 银行／资产负债管理系统通常只设 **标准期限节点**（3 M、6 M、1 Y、2 Y …）。
- 将任意 $t$ 的现金流 **线性插值** 到最近上下两个节点，可以为后续久期、缺口和 [[压力影响估计|利率压力影响估计]] 提供共同期限节点；它本身不决定压力情景或估计方法。
<!-- bilingual-en:start -->
- Bank and asset–liability-management systems generally use only **standard maturity nodes**, such as 3 M, 6 M, 1 Y, and 2 Y.
- Linearly interpolating a cash flow at any maturity $t$ between its two nearest nodes creates common maturity nodes for duration, gap analysis, and [[压力影响估计|interest-rate stress-impact estimation]]; it does not by itself determine the stress scenario or estimation method.
<!-- bilingual-en:end -->

本文用 _最简单_ 的 **线性权重法**：
<!-- bilingual-en:start -->
The note uses the simplest approach, **linear weights**:
<!-- bilingual-en:end -->

$$
w=\frac{t_2-t_\text{eff}}{t_2-t_1},\qquad
1-w=\frac{t_\text{eff}-t_1}{t_2-t_1}.
$$

其中 $t_\text{eff}$ 是原现金流到期，$t_1,t_2$ 是相邻标准节点。
<!-- bilingual-en:start -->
Here $t_\text{eff}$ is the maturity of the original cash flow, and $t_1,t_2$ are the adjacent standard nodes.
<!-- bilingual-en:end -->

 **3  把 0.3 年的 $50,000$ 拆进 3 M 与 6 M 桶**
<!-- bilingual-en:start -->
**3. Map the $50,000$ cash flow at 0.3 years into the 3 M and 6 M buckets**
<!-- bilingual-en:end -->

- 上邻节点：$t_1=0.25$ (=3 M)
- 下邻节点：$t_2=0.5$ (=6 M)
$$
w=\frac{0.5-0.3}{0.5-0.25}=0.8\qquad 1-w=0.2.
$$
<!-- bilingual-en:start -->
- Earlier node: $t_1=0.25$ (=3 M)
- Later node: $t_2=0.5$ (=6 M)
<!-- bilingual-en:end -->

> **权重理解**：0.3 年离 0.25 年很近（80%）；离 0.5 年较远（20%）。
> <!-- bilingual-en:start -->
> **Interpreting the weights:** 0.3 years is close to 0.25 years, so the 3 M node receives 80%; it is farther from 0.5 years, so the 6 M node receives 20%.
> <!-- bilingual-en:end -->

对应面值：
<!-- bilingual-en:start -->
The corresponding face values are:
<!-- bilingual-en:end -->

| **桶** | **面值 (美元)**              |
| ----- | ------------------------ |
| 3 M   | $0.8\times50,000=40,000$ |
| 6 M   | $0.2\times50,000=10,000$ |
<!-- bilingual-en:start -->
| **Bucket** | **Face value (USD)** |
| --- | ---: |
| 3 M | $0.8\times50,000=40,000$ |
| 6 M | $0.2\times50,000=10,000$ |
<!-- bilingual-en:end -->

 **4  把 0.8 年的 $1,050,000$ 拆进 6 M 与 1 Y 桶**
<!-- bilingual-en:start -->
**4. Map the $1,050,000$ cash flow at 0.8 years into the 6 M and 1 Y buckets**
<!-- bilingual-en:end -->

- 上邻节点：$t_1=0.5$ (=6 M)    
- 下邻节点：$t_2=1$ (=1 Y)
$$
w=\frac{1-0.8}{1-0.5}=0.4,\qquad 1-w=0.6.
$$
<!-- bilingual-en:start -->
- Earlier node: $t_1=0.5$ (=6 M)
- Later node: $t_2=1$ (=1 Y)
<!-- bilingual-en:end -->

对应面值：
<!-- bilingual-en:start -->
The corresponding face values are:
<!-- bilingual-en:end -->

| **桶** | **面值 (美元)**                  |
| ----- | ---------------------------- |
| 6 M   | $0.4\times1,050,000=420,000$ |
| 1 Y   | $0.6\times1,050,000=630,000$ |
<!-- bilingual-en:start -->
| **Bucket** | **Face value (USD)** |
| --- | ---: |
| 6 M | $0.4\times1,050,000=420,000$ |
| 1 Y | $0.6\times1,050,000=630,000$ |
<!-- bilingual-en:end -->

**5  汇总映射结果**
<!-- bilingual-en:start -->
**5. Aggregate the mapped cash flows**
<!-- bilingual-en:end -->

|**标准桶**|**3 M**|**6 M**|**1 Y**|**总计**|
|---|---|---|---|---|
|面值 (美元)|$40,000$|$10,000+420,000=430,000$|$630,000$|**$1,100,000$**|
<!-- bilingual-en:start -->
| **Standard bucket** | **3 M** | **6 M** | **1 Y** | **Total** |
| --- | ---: | ---: | ---: | ---: |
| Face value (USD) | $40,000$ | $10,000+420,000=430,000$ | $630,000$ | **$1,100,000$** |
<!-- bilingual-en:end -->

- 映射后总面值仍是 $1,100,000$，保证 **现金量守恒**；
- 每个桶内都视作 **零息债**，后续贴现、[[修正久期|久期]]/[[债券凸性|凸性]]、缺口分析即可直接调用系统已有工具。
<!-- bilingual-en:start -->
- The mapped face values still total $1,100,000$, so the amount of cash flow is conserved.
- Treat each bucketed amount as a zero-coupon bond. Existing tools can then be used directly for discounting, [[修正久期|duration]], [[债券凸性|convexity]], and gap analysis.
<!-- bilingual-en:end -->

<!-- duration-source-note:start MAP01 -->
> [!note] 校注：映射后的风险仍需与原现金流比较
> 上述线性分配保留了总现金量，却不自动保留现值、久期、凸性或损益方差。贴现因子对期限通常不是线性的，因此把一笔支付移到两侧节点后，贴现加总一般不等于原支付的现值。工具算出的是**映射后**现金流的风险；要用于原头寸，还须按相同曲线和冲击做[[估值近似验证|映射前后的估值与敏感度检查]]。
>
> <!-- bilingual-en:start -->
> The linear allocation preserves total cash, not automatically present value, duration, convexity, or P&L variance. Discount factors are generally nonlinear in maturity, so discounting the two replacement payments need not recover the original present value. The tools measure the **mapped** cash flows; applying the result to the original position requires [[估值近似验证|valuation and sensitivity checks before and after mapping]] under the same curves and shocks.
> <!-- bilingual-en:end -->
<!-- duration-source-note:end -->

**主成分分析法（[[主成分分析.canvas|PCA]]）**：用少数主成分近似收益率曲线变动；平移、斜率和[[曲线形变|曲率]]是常见的经验性命名，不是 PCA 自动识别的固定机制。实际使用还要说明尺度与样本期，并检查[[主成分不等于潜变量|方向命名边界]]和[[PCA稳定性|跨期稳定性]]。~~略~~
<!-- bilingual-en:start -->
**Principal component analysis ([[主成分分析.canvas|PCA]]):** Use a small number of components to approximate yield-curve movements. Level, slope, and [[曲线形变|curvature]] are common empirical labels, not mechanisms that PCA identifies automatically. The scale, sample window, [[主成分不等于潜变量|interpretation boundary]], and [[PCA稳定性|out-of-period stability]] still require explicit checks. ~~Details omitted in the original outline.~~
<!-- bilingual-en:end -->

## 5. 非线性资产（期权等）的VaR估算方法：[[Delta-Gamma价格近似|Delta近似法]]、二次模型与Cornish-Fisher展开
<!-- bilingual-en:start -->
*5. VaR for Nonlinear Positions: [[Delta-Gamma价格近似|Delta Approximation]], Quadratic Models, and the Cornish–Fisher Expansion*
<!-- bilingual-en:end -->

**Delta近似法（线性模型）**：对于期权等非线性衍生品，价格变动和基础资产之间的关系非线性，直接应用方差-协方差法容易失准。**Delta法**通过泰勒展开一阶项，将小幅价格变动下的期权损益近似为线性函数：$\Delta P \approx \delta\,\Delta S$，其中$\delta$为期权的Delta（即价格对标的资产价格变化的一阶敏感度），$\Delta S$是标的资产价格变动这相当于将期权头寸视作持有$\delta$股标的资产的等效头寸，然后用线性组合的方法计算VaR 。[[方差协方差VaR|Delta]]-Normal法的实现步骤：先求出组合对各基础风险因子的Delta敞口，再将$\delta_i S_i$作为等效资产价值，利用协方差矩阵求取组合方差并计算VaR。这种方法计算简便，适用于Delta支配风险、Gamma和其他高阶效应可以忽略的情况。然而，Delta近似忽略了期权的非线性特征，当价格大幅波动时误差增大，对于深度价内/价外期权或持有大量期权的组合，线性假设可能低估尾部风险。
<!-- bilingual-en:start -->
**[[Delta-Gamma价格近似|Delta approximation]] (linear model):** An option's price is a nonlinear function of the underlying asset, so a variance–covariance calculation applied directly to the option can be inaccurate. The Delta method keeps the first-order Taylor term and approximates a small price change by $\Delta P \approx \delta\,\Delta S$. Here $\delta$ is the option's Delta, its first-order sensitivity to the underlying price, and $\Delta S$ is the change in that price. The option is therefore treated as an equivalent position of $\delta$ units of the underlying. In Delta–Normal VaR, first calculate each Delta exposure, treat $\delta_i S_i$ as an equivalent linear position, and then use the covariance matrix to obtain portfolio variance and VaR. The method is quick and works when Delta dominates and Gamma and other higher-order effects are negligible. It becomes unreliable for large market moves, deeply in- or out-of-the-money options, or portfolios with substantial optionality, where it may understate tail risk.
<!-- bilingual-en:end -->

**二次模型（Delta-Gamma法）**：为提高非线性资产VaR估计精度，可在泰勒展开中保留二阶项。对单一标的资产期权，有$\Delta P \approx \delta\,\Delta S + \frac{1}{2}\gamma\,(\Delta S)^2$，其中$\gamma$为期权的Gamma（二阶敏感度。对于多因子组合，可将每个期权价值变化展开为对其相关单一风险因子的$\delta$和$\gamma$项的和（假设不同资产依赖独立的风险因子）。二次项引入了$\Delta S^2$使损益分布不再对称。当标的资产收益$\Delta S$近似正态时，$\delta\,\Delta S$项呈正态分布，而$\gamma\,(\Delta S)^2$项会产生偏度和峰度，使组合损益分布相对于正态出现**偏斜和厚尾**。具体而言：
- 如果$\gamma>0$（组合具有正Gamma，例如持有看涨期权），损益分布左尾比相应正态分布更窄。极端负收益出现概率降低，因此直接用正态假设算出的VaR会偏保守（偏大）
- 如果$\gamma<0$（组合Gamma为负，如卖出期权），损益分布左尾比正态更厚。出现巨大损失的概率高于正态预测，正态假设下计算的VaR将会偏低，低估尾部风险 
<!-- bilingual-en:start -->
**Quadratic model (Delta–Gamma method):** Retaining the second-order Taylor term gives $\Delta P \approx \delta\,\Delta S + \frac{1}{2}\gamma\,(\Delta S)^2$, where $\gamma$ is Gamma, the second-order sensitivity. For a multi-factor portfolio, each option's change can be expanded into Delta and Gamma terms for its relevant risk factors. The squared term $(\Delta S)^2$ makes profit and loss asymmetric. If $\Delta S$ is approximately normal, $\delta\,\Delta S$ is normal but $\gamma\,(\Delta S)^2$ creates skewness and excess kurtosis.
- If $\gamma>0$, as for a long option position, the left tail of profit and loss can be thinner than its linear-normal approximation, so linear-normal VaR may be conservative.
- If $\gamma<0$, as for a short option position, the left tail can be heavier than the normal approximation, so linear-normal VaR may understate tail risk.
<!-- bilingual-en:end -->

<!-- GB-34:start -->
> [!note] GB-34｜等效金额要匹配协方差的输入坐标
> 若 $\Sigma$ 是比例收益的协方差矩阵，才以 $\delta_iS_i$ 作敞口；若是绝对价格变化的协方差，直接用 $\delta_i$。利率、波动率等因子不必先转成百分比收益，尤其零值或负值附近不能机械相除。见 [[风险因子映射]]、[[方差协方差VaR]]。另外，深度价内/价外并不自动意味着局部曲率最大；普通期权临近到期、接近平值时 Gamma 往往更显著。应按实际冲击做 [[估值近似验证]]，而不是只按实值程度决定线性化是否失效。
> <!-- bilingual-en:start -->
> Use $\delta_iS_i$ only with a covariance matrix of proportional returns; use $\delta_i$ with absolute price changes. Rates and volatility need not be converted to returns, especially near zero or negative values. See [[风险因子映射|risk-factor mapping]] and [[方差协方差VaR|variance–covariance VaR]]. Deep moneyness does not automatically imply the largest local curvature; near-expiry, near-the-money options often have larger Gamma. Use [[估值近似验证|validation under the actual shocks]], not moneyness alone.
> <!-- bilingual-en:end -->
<!-- GB-34:end -->

<!-- GB-35:start -->
> [!note] GB-35｜跨因子曲率与“更厚尾”的比较基准
> 因子随机独立不意味着估值函数的 [[交叉Gamma]] 为零。省略混合二阶项须核验它们为零，或验证其贡献可忽略；估值可分离是混合导数为零的一个充分条件。完整形式见 [[多因子二阶损益近似]]。单因子二次近似中，$\gamma>0$ 使每个情景的损益不低于同一 $\delta\Delta S$ 线性近似，$\gamma<0$ 则相反，所以可比较这两个近似的损失分位数；不能据此对任意重新匹配均值、方差的“相应正态分布”作同样断言。一般多因子二次型也不必总有非零偏度。
> <!-- bilingual-en:start -->
> Independent factors do not imply zero [[交叉Gamma|mixed valuation derivatives]]. Verify that mixed terms vanish or have negligible contributions before omitting them; separability is one sufficient condition for zero mixed derivatives. See [[多因子二阶损益近似|the full multifactor second-order form]]. For a single factor, positive Gamma raises every quadratic-approximation P&L relative to the same linear approximation, while negative Gamma lowers it. That ordering compares those two approximations, not every normal distribution with rematched moments. A general multifactor quadratic form need not have nonzero skewness.
> <!-- bilingual-en:end -->
<!-- GB-35:end -->

Major [[Gamma|Gamma]] combination: Right tail thickening > Linear VaR tends to be overestimated T toi [EEL A (normal approx.) 0.40 to ti A+T(+T) ! i --- linear 99% VaR = -2.33 0.35 ti ss real 99% VaR = -1.79 0.30 |i ti ri 0.25 i ri 0.20 ti iG ti ti 0.15 i ri 0.10 fa 13 1 1 0.05 1 1 0.00 -4 -2 0 2 4 6
Major Gamma combination: right tail thickening. Linear VaR tends to be **overestimated**.\n- Normal-approx linear 99% VaR: $-2.33$\n- Real 99% VaR: $-1.79$

Minor Gamma combination: Left tail thickening > Linear VaR tends to be underestimated 7 [EE A (normal approx.) 1 0.40 A+T(CD 1 --- linear 99% VaR = -2.33 ! 0.35 | …… real 99% VaR = -2.87 | 0.30 po i po pa 0.25 po po pa po 0.20 poy pa po pa 0.15 pa pa pa it 0.10 i pa 0.05 pot iat 0.00 -6 -4 -2 0 2 4
Minor Gamma combination: left tail thickening. Linear VaR tends to be **underestimated**.\n- Normal-approx linear 99% VaR: $-2.33$\n- Real 99% VaR: $-2.87$

<!-- GB-36:start -->
> [!note] GB-36｜保留图像转写，区分损益分位点与损失 VaR
> 上述图像转写里的 Major/Minor 应按上下文读作正/负 Gamma，而不是两种正式术语。$-2.33,-1.79,-2.87$ 是图示损益的左尾分位点；按 $L=-\Delta V$ 的损失约定，对应 VaR 为 $2.33,1.79,2.87$。这些图示数字不是本课所有期权组合的通用参数。见 [[VaR定义]]。
> <!-- bilingual-en:start -->
> In the preserved diagram transcription, Major/Minor refers contextually to positive/negative Gamma, not standard terminology. The negative numbers are lower-tail P&L quantiles; with loss $L=-\Delta V$, the corresponding loss VaRs are 2.33, 1.79 and 2.87. They are illustrative diagram values, not universal portfolio parameters. See [[VaR定义|the VaR convention]].
> <!-- bilingual-en:end -->
<!-- GB-36:end -->

二次模型要求计算组合损益分布的二阶矩、三阶矩等统计量，然后根据偏度、峰度对VaR进行修正。
<!-- bilingual-en:start -->
A quadratic model requires the second and third moments, and sometimes higher moments, of portfolio profit and loss. VaR can then be adjusted for the resulting skewness and kurtosis.
<!-- bilingual-en:end -->

**Cornish-Fisher展开**：这是一种利用分布矩（矩阶）来近似求解分位数的方法 。在VaR计算中，常用Cornish-Fisher展开根据分布的偏度和峰度对正态分位数进行调整，从而估计非正态分布的VaR。基本思想是：设损益分布的标准化偏度为$\gamma_3$（即三阶中心矩），则调整后的$\alpha$分位数近似为：
<!-- bilingual-en:start -->
**Cornish–Fisher expansion:** This method approximates a distribution's quantiles from its moments. In VaR work, it adjusts a normal quantile for skewness and kurtosis to approximate a non-normal quantile. If standardised skewness is $\gamma_3$, the adjusted $\alpha$-quantile is approximately:
<!-- bilingual-en:end -->

$$ z_{\text{adj}} = z_{\alpha} + \frac{1}{6}(z_{\alpha}^2 - 1)\,\gamma_3 + \cdots $$

（上式省略了涉及峰度的高阶项）。其中$z_{\alpha}$为正态分布的$\alpha$分位数，$z_{\text{adj}}$为考虑偏度修正后的等效分位数。如果分布偏度$\gamma_3$为负（左偏，厚尾在左侧），则$(z_{\alpha}^2-1)\gamma_3$项为负，使$z_{\text{adj}} < z_{\alpha}$，表明实际分位数在左尾更极端，VaR应比正态估计更大；反之，$\gamma_3$为正（右偏），$z_{\text{adj}} > z_{\alpha}$，对应VaR降低。通过Cornish-Fisher公式，可以在已知组合损益的一、二、三阶矩的情况下，调整正态VaR的结果以更贴近真实分布的VaR 。需要注意当分布偏度、峰度很大时，该近似的精度可能降低，但它提供了一个相对简单的修正思路。
<!-- bilingual-en:start -->
The displayed formula omits higher-order terms involving kurtosis. Here $z_{\alpha}$ is the normal $\alpha$-quantile and $z_{\text{adj}}$ is its skewness-adjusted counterpart. The sign must be interpreted consistently: for a profit-and-loss distribution, VaR uses a lower-tail quantile, whereas a positive loss variable uses an upper-tail quantile. Negative skewness moves the lower P&L quantile farther into the left tail and generally raises loss VaR; under the loss convention, the corresponding skewness sign is reversed. Cornish–Fisher can improve on normal VaR when the first few moments are estimated reliably, but the approximation may become non-monotonic or inaccurate when skewness or kurtosis is large.
<!-- bilingual-en:end -->

<!-- GB-37:start -->
> [!note] GB-37｜标准化偏度与尾部方向
> 标准化偏度是 $\gamma_3=E[(Y-\mu_Y)^3]/\sigma_Y^3$，不是三阶中心矩本身；须有有限的相应矩及 $\sigma_Y>0$。若 $Y$ 是损益，置信水平 $\alpha$ 的损失 VaR 应使用 $z_{1-\alpha}$，而非 $z_\alpha$：
> $$\operatorname{VaR}_\alpha(-Y)\approx-\left[\mu_Y+\sigma_Y\left(z_{1-\alpha}+\frac{z_{1-\alpha}^2-1}{6}\gamma_3\right)\right].$$
> 上述偏度修正方向还取决于 $z^2-1$ 的符号，在本节高置信水平左尾有 $|z|>1$。二次重估模型不强制使用 Cornish–Fisher；也可在情景上计算二次近似损益再读取分位数。CF 是进一步的分位数近似，须检验而非保证更准确；参见下题14.10的同口径计算、[[VaR定义]] 与 [[估值近似验证]]。
> <!-- bilingual-en:start -->
> Standardized skewness divides the third central moment by $\sigma_Y^3$ and requires the relevant finite moments and positive variance. For P&L $Y$, loss VaR uses the lower-tail normal quantile $z_{1-\alpha}$, as displayed. The correction's direction also depends on $z^2-1$; the high-confidence left tails here have $|z|>1$. A quadratic valuation model does not require Cornish–Fisher: its scenario P&Ls can be sampled directly. CF adds a quantile approximation whose accuracy must be checked. Compare Exercise 14.10, [[VaR定义|VaR]] and [[估值近似验证|approximation validation]].
> <!-- bilingual-en:end -->
<!-- GB-37:end -->

实际风险管理中，针对期权组合可以结合Delta-Gamma方法和Cornish-Fisher展开来估计VaR：先用Delta-Gamma近似计算组合损益的期望、方差和偏度，再用Cornish-Fisher公式调整正态VaR。对于更复杂情况（例如存在重要的高阶Greeks或者非连续性），通常需要采用数值模拟方法（如蒙特卡罗）求解。
<!-- bilingual-en:start -->
In practice, option-portfolio VaR can combine Delta–Gamma and Cornish–Fisher: use the Delta–Gamma approximation to calculate the mean, variance, and skewness of profit and loss, then adjust the normal quantile. Portfolios with important higher-order Greeks, discontinuities, or strong path dependence generally require full numerical methods such as Monte Carlo simulation.
<!-- bilingual-en:end -->

>[!example] 例题
> ** 某投资组合包含两种股票期权，分别基于微软公司和AT&T公司的股票。组合对这两只股票的Delta敞口为：微软$\delta_{MSFT}=1000$，AT&T $\delta_{AT}=20000$（表示在小幅变动下，该期权组合相当于多头持有1000股微软股票和20000股AT&T股票的敞口 )。微软现价$S_{MSFT}=120$美元，AT&T现价$S_{AT}=30$美元。假设微软股价日波动率2%，AT&T股价日波动率1%，两只股票日收益相关系数$\rho=0.3$。求该组合的一天95% VaR和五天95% VaR（假设收益正态独立）。  
> **解答:** 先计算等效线性头寸价值的日标准差：  
> - 微软部分：等效持股价值$=1000\times\$120=\$120000$，其日标准差$\approx120000\times2\%=\$2400$。  
> - AT&T部分：等效持股价值$=20000\times\$30=\$600000$，日标准差$\approx600000\times1\%=\$6000$。  
><!-- bilingual-en:start -->
>A portfolio contains options on Microsoft and AT&T shares. Its Delta exposures are $\delta_{MSFT}=1000$ and $\delta_{AT}=20000$, so for small moves it behaves like a long position in 1,000 Microsoft shares and 20,000 AT&T shares. Current prices are $S_{MSFT}=120$ dollars and $S_{AT}=30$ dollars. Daily volatilities are 2% and 1%, respectively, and the return correlation is $\rho=0.3$. Find one-day and five-day 95% VaR under the joint-normal, independent-over-time assumption.
>**Solution:** First calculate the daily standard deviation of each equivalent linear position:
>- Microsoft: value $=1000\times\$120=\$120000$; daily standard deviation $\approx120000\times2\%=\$2400$.
>- AT&T: value $=20000\times\$30=\$600000$; daily standard deviation $\approx600000\times1\%=\$6000$.
><!-- bilingual-en:end -->

两部分的协方差$=\rho\times2400\times6000=0.3\times14,400,000=\$4,320,000$。因此组合日方差为：
<!-- bilingual-en:start -->
The covariance contribution between the two positions is $\rho\times2400\times6000=0.3\times14,400,000=\$4,320,000$. Hence daily portfolio variance is:
<!-- bilingual-en:end -->

$\sigma_p^2 = 2400^2 + 6000^2 + 2\times4,320,000 = 50,400,000\ (\$^2)$，

组合日标准差$\sigma_p=\sqrt{50,400,000}\approx\$7100$。95%置信水平下，一天VaR $=1.65\times7100\approx\$11700$（约1.17万美元）；五天VaR按$\sqrt{5}$放大，$=11700\times\sqrt{5}\approx\$26150$（约2.62万美元）。相比直接将两头寸VaR相加（微软\$3960 + AT&T \$9900 ≈ \$13,860），组合VaR更低，再次验证了相关性<1时分散化的作用。
<!-- bilingual-en:start -->
Daily portfolio standard deviation is $\sigma_p=\sqrt{50,400,000}\approx\$7100$. At 95% confidence, one-day VaR is $1.65\times7100\approx\$11700$, or about USD 11,700. Scaling by $\sqrt{5}$ gives five-day VaR $=11700\times\sqrt{5}\approx\$26150$, or about USD 26,150. The sum of the two stand-alone VaRs is approximately Microsoft $\$3960$ plus AT&T $\$9900$, or $\$13,860$; the lower portfolio VaR illustrates diversification when correlation is below 1.
<!-- bilingual-en:end -->

> **注意：**上述计算采用Delta线性近似，未考虑期权的Gamma等二阶效应。当标的资产波动较大或组合包含显著Gamma敞口时，应使用二次模型或模拟法以获得更准确的VaR估计。
> <!-- bilingual-en:start -->
> **Note:** This calculation uses a linear Delta approximation and ignores second-order effects such as Gamma. When the underlying moves substantially or the portfolio has material Gamma exposure, use a quadratic or full-revaluation simulation method.
> <!-- bilingual-en:end -->

<!-- GB-38:start -->
> [!note] GB-38｜微软与 AT&T 例的独立性和单位
> 已给横截面相关系数 $0.3$，故“独立”只能在这里额外指跨交易日的增量假设，不能说两只股票彼此独立。$4320000$ 是协方差，单位为美元平方，不是美元；方差与约 $7100$ 美元的标准差计算保留。五天按 $\sqrt5$ 放大还采用零均值、稳定协方差和冻结线性敞口，见 [[VaR时间缩放]]。
> <!-- bilingual-en:start -->
> With cross-sectional correlation 0.3, independence can only refer here to the additional assumption across trading days, not independence between the stocks. The covariance of 4,320,000 has units of USD squared. The variance and approximately USD 7,100 standard deviation are retained. Five-day scaling also assumes zero mean, stable covariance and frozen linear exposures; see [[VaR时间缩放|VaR time scaling]].
> <!-- bilingual-en:end -->
<!-- GB-38:end -->

## 6. [[风险蒙特卡洛|蒙特卡罗模拟法]]：原理、步骤与优缺点
<!-- bilingual-en:start -->
*6. [[风险蒙特卡洛|Monte Carlo Simulation]]: Principle, Procedure, Advantages, and Limitations*
<!-- bilingual-en:end -->

风险蒙特卡罗在展望期上模拟当前组合的损益分布，最小而完整的流程是：

1. 在外层指定风险因子在**[[风险模拟P-Q分工|真实世界概率测度]] $P$** 下的[[风险因子联合生成|终点边际分布与横截面依赖]]；若任务需要逐期演化，再另外规定[[风险因子时间动态|跨时点动态]]。
2. 从 $P$ 抽取联合情景。只有当损益取决于中间触碰、平均或现金流顺序时，才必须按[[风险模拟路径依赖|路径依赖要求]]把有序路径保留到估值层；终点型头寸可以直接使用经验证的终点联合情景。
3. 在每个情景下保持**当前组合及持仓数量固定**并重新估值。若衍生品估值本身需要风险中性定价，则在给定该情景状态后，于内层条件地使用**[[风险模拟P-Q分工|风险中性测度]] $Q$** 计算贴现期望；外层 $P$ 与内层条件 $Q$ 的职责不能混淆。
4. 汇总情景损益，从经验分布读取所需尾部分位数得到 VaR；ES 按 [[ES定义|最坏固定尾部概率质量]]求平均，只有无并列且尾部数量恰好对齐等特殊情形，才可简写为严格超过 VaR 的样本均值。

全重估能够处理非线性；[[估值近似验证|经验证的近似重估]]可降低成本，因此速度取决于估值方式而不只是“是否使用蒙特卡罗”。[[路径数不修模型|增加路径数]]只会降低**抽样误差**；错误的分布、依赖结构、参数或估值模型造成的**模型误差**不会因路径数增加而消失，仍需回测、敏感性分析和压力测试。
<!-- bilingual-en:start -->
Risk Monte Carlo estimates the horizon profit-and-loss distribution of today's portfolio through the following minimum complete procedure:

1. Under the **[[风险模拟P-Q分工|real-world probability measure]] $P$**, specify the [[风险因子联合生成|terminal margins and cross-sectional dependence]] of risk factors; when the task needs period-by-period evolution, separately specify their [[风险因子时间动态|time dynamics]].
2. Draw joint scenarios from $P$. Preserve an ordered path through valuation only when profit and loss depends on intermediate crossings, averages, or cash-flow order, as required by [[风险模拟路径依赖|path-dependent instruments]]. A validated terminal joint scenario is enough for a terminal-state-only position.
3. Keep the **current portfolio and position quantities fixed** and revalue them in every scenario. If derivative valuation itself requires risk-neutral pricing, condition on the simulated state and use the **[[风险模拟P-Q分工|risk-neutral measure]] $Q$** inside that valuation to compute a discounted expectation. Do not confuse the outer risk distribution under $P$ with the inner conditional valuation under $Q$.
4. Aggregate scenario P&Ls and read VaR from the required empirical tail quantile. Compute ES as the average of the [[ES定义|fixed worst tail probability mass]]; only special cases with no ties and exactly aligned tail counts reduce to the sample mean strictly beyond VaR.

Full revaluation captures nonlinearity; [[估值近似验证|validated approximation methods]] can reduce cost, so speed depends on the valuation method rather than on the Monte Carlo label alone. [[路径数不修模型|More paths reduce]] **sampling error**, but they do not remove **model error** caused by a misspecified distribution, dependence structure, calibration, or pricing model. Backtesting, sensitivity analysis, and stress testing remain necessary.
<!-- bilingual-en:end -->

## 7. 不同VaR方法的对比分析（优劣、适用场景）
<!-- bilingual-en:start -->
*7. Comparing VaR Methods: Strengths, Limitations, and Appropriate Uses*
<!-- bilingual-en:end -->

常用的VaR计量方法主要有参数法（[[方差协方差VaR|方差-协方差法]]）、历史模拟法和蒙特卡罗模拟法。它们各有优缺点，在不同情境下适用性不同：
<!-- bilingual-en:start -->
The main VaR methods are the parametric [[方差协方差VaR|variance–covariance method]], historical simulation, and Monte Carlo simulation. Each has different strengths, weaknesses, and suitable applications:
<!-- bilingual-en:end -->

- **方差-协方差法（[[方差协方差VaR|参数法]]）**：计算快速，理解和实现简单。只需估计均值、方差和相关系数等参数，就能得到风险值，便于日常风险监控和报告。VaR提供了统一的风险度量语言，管理者和投资者易于理解，对监管资本计算也有参考价值 。然而，该方法**假定收益分布形状**（通常正态），存在模型风险。当资产收益呈现厚尾或偏态时，正态假设会低估极端风险。另外参数法主要基于**线性近似**，无法准确处理期权等非线性产品（Gamma风险、波动率风险被忽略）。**适用场景**：组合以线性资产为主、收益分布接近正态，例如股票+债券的传统投资组合在正常市场波动情况下，可采用参数法快速估计VaR；也常用于高频实时风险估计（因计算简便）。对于包含少量期权的组合，可在参数法基础上做Delta近似，但需警惕误差。
<!-- bilingual-en:start -->
- **[[方差协方差VaR|Variance–covariance method]] ([[方差协方差VaR|parametric method]]):** Fast, transparent, and easy to implement. Once means, variances, and correlations have been estimated, VaR can be produced quickly for routine monitoring and reporting. Its main weaknesses are model risk and linearity. A normal distribution can understate risk when returns are skewed or heavy-tailed, and a linear approximation misses option Gamma, volatility risk, and other nonlinear effects. **Best suited to:** portfolios dominated by linear assets with approximately elliptical return distributions, such as conventional stock-and-bond portfolios in ordinary market conditions. A Delta approximation can accommodate limited optionality, but the approximation error must be monitored.
<!-- bilingual-en:end -->

- **[[历史模拟法]]**：把历史风险因子变动施加到当前固定组合，再从情景损益分布读取分位数。它不预设参数分布，可保留样本中已经出现的偏度、厚尾和因子依赖；只有在每个情景下全重估时，才会自然捕捉组合非线性，若使用近似重估则速度更快但会引入近似误差。核心限制是历史窗口：未观察过的冲击不会出现，结构变化会削弱代表性。100 个观察值在 1% 尾部**期望只有约 1 个点**，这只是尾部样本量的直观说明，不等于获得可信的压力情景或稳定的 99% 分位数。**适用场景**：历史较长且具有代表性、当前组合能在历史冲击下可靠重估；不适合缺乏历史的新策略或明显不同于过去的市场状态。
<!-- bilingual-en:start -->
- **[[历史模拟法|Historical simulation]]:** Apply historical risk-factor changes to today's fixed portfolio and take a quantile of the scenario P&L distribution. It imposes no parametric distribution and can preserve skewness, heavy tails, and dependence already observed in the sample. It captures portfolio nonlinearity only when each scenario uses full repricing; approximation-based repricing is faster but adds approximation error. Its central weakness is the historical window: unobserved shocks cannot appear, and structural change can make the sample unrepresentative. With 100 observations, the 1% tail contains only about one observation in expectation; that is an indication of tail-sample quantity, not evidence of a credible stress scenario or a stable 99% quantile. **Best suited to:** portfolios with a long, representative history and reliable scenario valuation; unsuitable for new strategies or regimes unlike the past.
<!-- bilingual-en:end -->

- **[[风险蒙特卡洛|蒙特卡罗模拟法]]**：能够用设定的联合随机模型生成历史之外的情景，并通过全路径和全重估处理非线性、路径依赖产品；也可使用经验证的估值近似换取速度。主要成本是计算量、校准复杂度和模型风险。增加路径数可改善尾部分位数的抽样精度，但**路径很多不等于满足监管要求**：模型、数据、估值近似、回测和压力测试仍需验证。**适用场景**：非线性或路径依赖组合，以及需要探索历史中未出现但模型认为可能的联合情景。
<!-- bilingual-en:start -->
- **[[风险蒙特卡洛|Monte Carlo simulation]]:** A specified joint stochastic model can generate scenarios outside the historical sample, while full-path simulation and full revaluation handle nonlinear and path-dependent products; validated valuation approximations may instead trade some accuracy for speed. The main costs are computation, calibration complexity, and model risk. More paths improve sampling precision in the tail, but **a large path count does not by itself meet regulatory requirements**: models, data, valuation approximations, backtesting, and stress testing still require validation. **Best suited to:** nonlinear or path-dependent portfolios and joint scenarios that have not appeared in the historical record but are plausible under the model.
<!-- bilingual-en:end -->

三种方法的速度排序并非固定：是否全重估通常比方法名称更决定计算成本。Basel 市场风险框架不偏好某一种具体 VaR 方法，并允许使用经过独立验证、受到适当控制的估值近似；监管可接受性来自整个模型与验证体系，而非选用某一算法。
<!-- bilingual-en:start -->
There is no fixed speed ranking across the three methods: whether scenarios require full revaluation often matters more than the method's label. The Basel market-risk framework does not prefer one specific VaR technique and permits independently validated, appropriately controlled valuation approximations. Regulatory acceptability depends on the complete model and validation framework, not on choosing a particular algorithm.
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->

## 14.1

>[!question] 
>假定某投资组合由价值为100,000美元资产A的投资以及价值为100,000美元资产B的投资构成，假定两种资产的日波动率均为1%，两项投资回报的相关系数为0.3，投资组合5天展望期的97%的VaR和ES为多少?
><!-- bilingual-en:start -->
>A portfolio invests USD 100,000 in asset A and USD 100,000 in asset B. Both assets have daily volatility of 1%, and their returns have correlation 0.3. What are the portfolio's five-day 97% VaR and ES?
><!-- bilingual-en:end -->

| **步骤**    | **关键计算**                                                                                                                                                                | **说明**               |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| 1. 配置与已知  | 组合总市值$$V=100000+100000=200000$$权重$$w_A=w_B=\frac{100000}{200000}=0.5$$日波动率$$\sigma_A=\sigma_B=0.01$$[[相关系数]]$$\rho=0.3$$                                               | 写明输入参数               |
| 2. 组合日方差  |$$\sigma_p=\sqrt{w_A^2\sigma_A^2+w_B^2\sigma_B^2+2w_Aw_B\rho\sigma_A\sigma_B}=\sqrt{0.000065}=0.00806226$$| 得出日波动率$$0.806226\%$$  |
| 3. 5 天波动率 |$$\sigma_{5d}=\sigma_p\sqrt{5}=0.01802776$$                                                                                                          | √时距缩放                |
| 4. 查分位点   |$$z_{0.97}=1.88079,\qquad \phi(z)=0.06804195$$                                                                                                                                                     | 正态分布$97\%$单侧$z$值及密度 |
|5. 计算$$VaR$$|$$VaR_{0.97}=z_{0.97}\sigma_{5d}V=1.88079\times0.01802776\times200000=\$6,781.30$$|负向损失界|
|6. 计算$$ES$$|$$ES_{0.97}=\frac{\sigma_{5d}V\phi(z)}{1-0.97}=\frac{0.01802776\times200000\times0.06804195}{0.03}=\$8,177.62$$|正态$$ES$$公式|
<!-- bilingual-en:start -->
| **Step** | **Key calculation** | **Purpose** |
| --- | --- | --- |
| 1. Inputs | Total value USD 200,000; weights 0.5 and 0.5; daily volatilities 1%; [[相关系数|correlation]] 0.3 | State the inputs |
| 2. Daily portfolio volatility | $\sigma_p=\sqrt{0.5^2(0.01)^2+0.5^2(0.01)^2+2(0.5)(0.5)(0.3)(0.01)^2}=0.00806226$ | Daily volatility is 0.806226% |
| 3. Five-day volatility | $\sigma_{5d}=0.00806226\sqrt{5}=0.01802776$ | Square-root-of-time scaling |
| 4. Critical value and density | $z_{0.97}=1.88079$ and $\phi(z)=0.06804195$ | One-sided normal 97% inputs |
| 5. VaR | $1.88079(0.01802776)(200000)=\$6,781.30$ | Loss quantile |
| 6. ES | $0.01802776(200000)(0.06804195)/(0.03)=\$8,177.62$ | Normal ES formula |
<!-- bilingual-en:end -->

 **结论**
<!-- bilingual-en:start -->
**Conclusion**
<!-- bilingual-en:end -->

- 组合 **5 天、97% [[风险度量口径|置信水平]]** 下$$\boxed{VaR=\$6,781.30}$$
- 同期$$\boxed{ES=\$8,177.62}$$
<!-- bilingual-en:start -->
- At a **five-day horizon and 97% [[风险度量口径|confidence level]]**, portfolio VaR is USD 6,781.30.
- ES over the same horizon is USD 8,177.62.
<!-- bilingual-en:end -->

## 14.4

>[!question] 
>一家金融机构拥有一个标的变量为 USD/GBP 汇率的期权投资组合，投资组合相对于汇率单位比例变动的 Delta 为 390 万；如果汇率日收益率的波动率为 0.7%，请问 10 天展望期、99% 置信度的 VaR 为多少？
><!-- bilingual-en:start -->
>A financial institution has an option portfolio on the USD/GBP exchange rate. Its Delta is 3.9 million currency units per unit proportional change in the exchange rate. If daily exchange-rate-return volatility is 0.7%, what is ten-day 99% VaR?
><!-- bilingual-en:end -->


| **步骤**      | **公式与计算**                                                                                                | **说明**        |
| ----------- | -------------------------------------------------------------------------------------------------------- | ------------- |
| 1. 参数列示     | $$\Delta=3.9\text{ 百万}$$（相对于汇率**单位比例变动**，即收益率变动 $1.0$ 的损益敏感度）$$\sigma_d=0.7\%=0.007$$（汇率日波动率）$$h=10$$（天数）$$z_{0.99}=2.33$$ | 明确 Delta 的计量口径          |
| 2. 日收益标准差   | $$\sigma_P=\Delta\sigma_d=3.9\times0.007=0.0273$$（百万）                                                    | $\Delta$-正态近似 |
| 3. 10 日标准差  | $$\sigma_{10}=\sigma_P\sqrt{h}=0.0273\sqrt{10}=0.0273\times3.1623=0.0863$$（百万）                           | $\sqrt{h}$缩放  |
| 4.$$VaR$$计算 | $$VaR_{0.99}=z_{0.99}\sigma_{10}=2.33\times0.0863\approx0.201\text{ 百万}$$                               | 单边$$99\%$$    |
| 5. 答案       | $$VaR\approx0.201\text{ 百万}$$                                                                                      | 结果呈现          |
<!-- bilingual-en:start -->
| **Step** | **Formula and calculation** | **Purpose** |
| --- | --- | --- |
| 1. Inputs | $\Delta=3.9$ million per unit proportional exchange-rate change; daily volatility $0.7\%=0.007$; $h=10$; $z_{0.99}=2.33$ | State the Delta convention |
| 2. Daily P&L standard deviation | $\sigma_P=3.9(0.007)=0.0273$ million | Delta–Normal approximation |
| 3. Ten-day standard deviation | $\sigma_{10}=0.0273\sqrt{10}=0.0863$ million | Square-root-of-time scaling |
| 4. VaR | $2.33(0.0863)\approx0.201$ million | One-sided 99% VaR |
| 5. Answer | VaR is approximately 0.20 million | Report the result |
<!-- bilingual-en:end -->

## 14.6

>[!question] 
>投资组合有2年期利率和5年期利率的风险敞口。2年期利率每增加一个基点，会使投资组合的价值增加10000美元。5年期利率每增加一个基点，投资组合的价值就会减少8 000美元。2年期利率和5年期利率的每日标准差分别为7个基点和8个基点，两者之问的相关系数为0.8。当置信水平为98%、展望期为5天时，投资组合的ES是多少?
><!-- bilingual-en:start -->
>A portfolio is exposed to two-year and five-year interest rates. A one-basis-point rise in the two-year rate increases portfolio value by USD 10,000, while a one-basis-point rise in the five-year rate decreases it by USD 8,000. Their daily standard deviations are 7 and 8 basis points, and their correlation is 0.8. Find five-day ES at the 98% confidence level.
><!-- bilingual-en:end -->

| **步骤**          | **计算公式**                                                                                                                                                                                             | **结果 / 说明**                    |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 1. 列示已知         |$$\Delta_1=+10,000\ \text{USD/bp},\quad \Delta_2=-8,000\ \text{USD/bp}$$$$\sigma_1=7\ \text{bp},\quad \sigma_2=8\ \text{bp},\quad \rho=0.8$$                                                        | 2Y 与 5Y 利率风险敞口                 |
| 2. 组合**日**方差    |$$\sigma_P^2=(\Delta_1\sigma_1)^2+(\Delta_2\sigma_2)^2+2\rho\Delta_1\Delta_2\sigma_1\sigma_2$$                                                                                                     |$$\sigma_P^2=1.828\times10^9$$|
| 3. 组合**日**标准差   |$$\sigma_P=\sqrt{\sigma_P^2}\approx42,755\ \text{USD}$$                                                                                                                                             | 正态近似                           |
| 4. 5 天标准差       |$$\sigma_{5d}=\sigma_P\sqrt{5}\approx95,600\ \text{USD}$$                                                                                                                                           |$$\sqrt{h}$$缩放                |
| 5.$$ES$$公式与计算 |$$z_{0.98}=2.054,\quad \phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\approx0.0484$$$$ES_{0.98}=\frac{\sigma_{5d}\phi(z_{0.98})}{1-0.98}\approx\frac{95,600\times0.0484}{0.02}\approx231,000\ \text{USD}$$|$$\phi$$为标准正态密度               |
以期末考试答卷的过程解答这个题目,要足够简洁但是要包含所有关键步骤
<!-- bilingual-en:start -->
| **Step** | **Calculation** | **Result or purpose** |
| --- | --- | --- |
| 1. Inputs | $\Delta_1=+10,000$ USD/bp; $\Delta_2=-8,000$ USD/bp; $\sigma_1=7$ bp; $\sigma_2=8$ bp; $\rho=0.8$ | Two- and five-year rate exposures |
| 2. Daily variance | $(70000)^2+(-64000)^2+2(0.8)(70000)(-64000)$ | $1.828\times10^9$ USD² |
| 3. Daily standard deviation | $\sqrt{1.828\times10^9}$ | Approximately USD 42,755 |
| 4. Five-day standard deviation | $42,755\sqrt{5}$ | Approximately USD 95,600 |
| 5. ES | $z_{0.98}=2.054$, $\phi(z)=0.0484$; $ES=95,600(0.0484)/(0.02)$ | Approximately USD 231,000 |

For an exam answer, show the signed exposures, covariance term, horizon scaling, normal ES formula, and final units.
<!-- bilingual-en:end -->
<!-- GB-39:start -->
> [!note] GB-39｜14.1、14.4、14.6 的正态与时距条件
> 这些数值沿用零均值正态线性损益、冻结敞口及跨日独立同分布增量。14.1 的 $6781.30/8177.62$ 和14.4约 $0.201$ 百万的计算可保留；14.6 的正负基点敞口与协方差项也正确，高精度正态 ES 约 $231446.79$ 美元，原 $231000$ 是粗略报价。只有日波动率和相关系数而没有分布、跨日及近似条件，不能唯一确定多日 VaR/ES。见 [[方差协方差VaR]]、[[VaR时间缩放]]、[[ES定义]]。
> <!-- bilingual-en:start -->
> These exercises assume zero-mean normal linear P&L, frozen exposures and iid daily increments. The results in 14.1 and the approximately 0.201 million in 14.4 are retained. The signed basis-point exposures and covariance in 14.6 are also correct; the unrounded normal ES is about USD 231,446.79, while USD 231,000 is a coarse quotation. Volatilities and correlations alone do not determine multi-day VaR or ES without the distributional and temporal assumptions. See [[方差协方差VaR|normal linear VaR]], [[VaR时间缩放|time scaling]] and [[ES定义|ES]].
> <!-- bilingual-en:end -->
<!-- GB-39:end -->

## 14.8
>[!question] 
假设某投资组合的每天价值变化与主成分分析（PCA）法所计算出的两个因子呈最好的线性关系。  
投资组合对于**第一个因子**的 *delta* 为 **6**，对于**第二个因子**的 *delta* 为 **-4**。  
两个因子的标准差分别为 **20** 与 **8**。  
试求该投资组合 **5 天展望期、90 % 置信水平** 的 VaR 为多少？  
<!-- bilingual-en:start -->
Suppose a portfolio's daily value change is best represented as a linear function of two factors from principal component analysis (PCA). Its *delta* is **6** with respect to the first factor and **-4** with respect to the second. The factor standard deviations are **20** and **8**. Find five-day VaR at the **90% confidence level**.
<!-- bilingual-en:end -->

| 步骤           | 关键公式                                                                              | 计算                                                                | 说明                      |
| ------------ | --------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------- |
| 1️⃣ 因子方差贡献   | $\sigma_{P,\text{day}}^{2} = ( \Delta_1\sigma_1 )^{2} + ( \Delta_2\sigma_2 )^{2}$ | $(6×20)^2 + (-4×8)^2 = 120^2 + 32^2 = 14\,400 + 1\,024 = 15\,424$ | **PCA 因子彼此正交 ⇒ 协方差为 0** |
| 2️⃣ 每日波动率    | $\sigma_{P,\text{day}} = \sqrt{15\,424} = 124.1$                                  | 单位同投资组合货币                                                         |                         |
| 3️⃣ 5-天波动率   | $\sigma_{P,5d} = \sigma_{P,\text{day}}\sqrt{5} = 124.1×2.236 = 277.2$             | 令天际独立同分布(IID)                                                     |                         |
| 4️⃣ 90 % VaR | $\text{VaR}_{0.90,5d} = z_{0.90}\, \sigma_{P,5d}$；$z_{0.90}=1.281$                | $1.281×277.2 = 355.7$                                             |                         |
| 5️⃣ 结论       | **5 天、90 % VaR ≈ 356**                                                            | 取绝对值表示潜在损失                                                        |                         |
<!-- bilingual-en:start -->
| Step | Key formula | Calculation | Explanation |
| --- | --- | --- | --- |
| 1. Factor contributions to variance | $\sigma_{P,\text{day}}^{2} = ( \Delta_1\sigma_1 )^{2} + ( \Delta_2\sigma_2 )^{2}$ | $(6×20)^2 + (-4×8)^2 = 120^2 + 32^2 = 14\,400 + 1\,024 = 15\,424$ | PCA factors are orthogonal, so covariance is 0 |
| 2. Daily standard deviation | $\sigma_{P,\text{day}} = \sqrt{15\,424} = 124.1$ | Same currency units as portfolio value | |
| 3. Five-day standard deviation | $\sigma_{P,5d} = \sigma_{P,\text{day}}\sqrt{5} = 124.1×2.236 = 277.2$ | Assume independent and identically distributed (IID) daily changes | |
| 4. 90% VaR | $\text{VaR}_{0.90,5d} = z_{0.90}\, \sigma_{P,5d}$; $z_{0.90}=1.281$ | $1.281×277.2 = 355.7$ | |
| 5. Conclusion | **Five-day 90% VaR ≈ 356** | Report the potential loss as a positive amount | |
<!-- bilingual-en:end -->

> **一句话记忆**：因子正交 → 方差相加；多日 VaR = 单日 σ × √天数 × z-score。
> <!-- bilingual-en:start -->
> **One-sentence reminder:** orthogonal factors imply that their variance contributions add; multi-day VaR equals daily $\sigma$ times the square root of the horizon times the relevant z-score.
> <!-- bilingual-en:end -->

<!-- GB-40:start -->
> [!note] GB-40｜14.8 的正交口径与连续舍入
> PCA 得分的零协方差针对构造它的中心化样本协方差；不能自动承诺预测期仍不相关，更不能在没有联合正态等条件时推出独立。见 [[PCA稳定性]]、[[联合高斯独立判据]]。沿本题零协方差、零均值正态及跨日 IID 的线性近似，复算为
> $$\sigma_d=\sqrt{15424}=124.19339757\ldots,\quad \sigma_5=\sqrt{77120}=277.70487932\ldots,\quad \operatorname{VaR}_{0.90}=355.89312286\ldots.$$
> 因此约 $356$ 的末尾结论保留，但 $124.1\to277.2\to355.7$ 不是一致的精确计算。一般 [[VaR定义]] 是损失分位数，不是把任意分位数取绝对值；这里零均值正态的高置信损失分位数恰为正。
> <!-- bilingual-en:start -->
> PCA scores are uncorrelated under the centred covariance used to construct them, not automatically under a future covariance; independence requires additional conditions such as joint normality. See [[PCA稳定性|PCA stability]] and [[联合高斯独立判据|the Gaussian independence criterion]]. Under this exercise's stated approximation, the corrected intermediate values are displayed above and still round to a final VaR of 356. The original intermediate chain is numerically inconsistent. General [[VaR定义|VaR]] is a loss quantile, not an absolute-value operation; this high-confidence zero-mean normal loss quantile happens to be positive.
> <!-- bilingual-en:end -->
<!-- GB-40:end -->

## 14.10
>[!question] 
一家银行拥有某资产的多个期限权投资组合，期权组合的 *delta* 为 **-30**，*gamma* 为 **-5**。  1. 先解释这两个数字的含义。资产现价为 **20**，其**每日价格变化的波动率为 1 %**。   2. 采用 **Isserlis 定理**计算投资组合价值变化的前三阶矩；再结合 **Cornish–Fisher 展开**，分两种情形求 **1 天展望期、99 % 置信水平** 的 VaR：   (a) 仅使用前 **二阶矩**；   (b) 使用 **前三阶矩**。  
<!-- bilingual-en:start -->
A bank holds a portfolio of options on one asset. Portfolio *delta* is **-30** and *gamma* is **-5**. First explain these sensitivities. The asset price is **20**, and its daily price volatility is **1%**, so the daily standard deviation of the price change is 0.2. Next use **Isserlis' theorem** to calculate the first three moments of portfolio value change. Then use the **Cornish–Fisher expansion** to estimate one-day 99% VaR (a) from the first two moments and (b) from the first three moments.
<!-- bilingual-en:end -->

**完整数值汇总（金额单位）**
<!-- bilingual-en:start -->
**Complete numerical summary, in currency units**
<!-- bilingual-en:end -->

Delta $=-30$ 表示标的价格每上升 1 个金额单位，组合价值的一阶近似减少 30；Gamma $=-5$ 表示标的价格每上升 1 个金额单位，Delta 约再下降 5。令标的一天价格变化为
$$
X\sim N(0,0.2^2),\qquad Y=\Delta X+\frac12\Gamma X^2=-30X-2.5X^2,
$$
其中 $Y$ 是组合的一天损益。
<!-- bilingual-en:start -->
Delta $=-30$ means that a one-currency-unit increase in the asset price reduces portfolio value by approximately 30 at first order. Gamma $=-5$ means that Delta decreases by approximately 5 for a one-unit increase in the asset price. Let the one-day asset-price change and portfolio P&L be
$$
X\sim N(0,0.2^2),\qquad Y=\Delta X+\frac12\Gamma X^2=-30X-2.5X^2.
$$
<!-- bilingual-en:end -->

| 符号 / 指标 | Isserlis 定理下的计算式 | 数值 |
|--------------|--------|------|
| 均值 $\mu$ | $E[Y]=-2.5E[X^2]=-2.5(0.04)$ | $-0.1$ |
| 方差 $\operatorname{Var}(Y)$ | $(-30)^2(0.04)+(-2.5)^2\,2(0.04)^2$ | $36.02$ |
| 标准差 $\sigma_Y$ | $\sqrt{36.02}$ | $6.001666$ |
| 三阶中心矩 $\mu_3$ | $6(-30)^2(-2.5)(0.04)^2+8(-2.5)^3(0.04)^3$ | $-21.608$ |
| 偏度 $\gamma_1$ | $\mu_3/\operatorname{Var}(Y)^{3/2}$ | $-0.099954$ |
| 左尾正态分位数 $z_{0.01}$ | — | $-2.326348$ |
| 调整后分位数 $z'_{0.01}$ | $z+\dfrac{z^2-1}{6}\gamma_1$ | $-2.399845$ |
<!-- bilingual-en:start -->
| Symbol or measure | Calculation using Isserlis' theorem | Value |
| --- | --- | ---: |
| Mean $\mu$ | $E[Y]=-2.5E[X^2]=-2.5(0.04)$ | $-0.1$ |
| Variance $\operatorname{Var}(Y)$ | $(-30)^2(0.04)+(-2.5)^2\,2(0.04)^2$ | $36.02$ |
| Standard deviation $\sigma_Y$ | $\sqrt{36.02}$ | $6.001666$ |
| Third central moment $\mu_3$ | $6(-30)^2(-2.5)(0.04)^2+8(-2.5)^3(0.04)^3$ | $-21.608$ |
| Skewness $\gamma_1$ | $\mu_3/\operatorname{Var}(Y)^{3/2}$ | $-0.099954$ |
| Normal left-tail quantile $z_{0.01}$ | — | $-2.326348$ |
| Adjusted quantile $z'_{0.01}$ | $z+\dfrac{z^2-1}{6}\gamma_1$ | $-2.399845$ |
<!-- bilingual-en:end -->

---

| 方案 | 1-Day · 99 % VaR |
|------|-----------------|
| (a) 仅二阶矩（正态）：$-(\mu+\sigma_Yz_{0.01})$ | **14.06196** |
| (b) 含三阶矩（Cornish–Fisher）：$-(\mu+\sigma_Yz'_{0.01})$ | **14.50307** |
<!-- bilingual-en:start -->
| Method | One-day 99% VaR |
| --- | ---: |
| (a) First two moments, normal approximation: $-(\mu+\sigma_Yz_{0.01})$ | **14.06196** |
| (b) First three moments, Cornish–Fisher: $-(\mu+\sigma_Yz'_{0.01})$ | **14.50307** |
<!-- bilingual-en:end -->

> VaR 以正数报告损失。这里 $Y$ 是损益，负偏度使左尾损益分位数更负，因此 Cornish–Fisher VaR 高于正态 VaR：负损益偏度恶化了左尾风险。
> <!-- bilingual-en:start -->
> VaR reports loss as a positive amount. Because $Y$ is P&L, its negative skewness makes the left-tail P&L quantile more negative, so Cornish–Fisher VaR is higher than normal VaR: negative P&L skewness worsens left-tail risk.
> <!-- bilingual-en:end -->

## 14.13
>[!question] 
 假定在过去的某一时间，某家公司签署了一项远期合约，约定在未来某时以 **150 万美元** 买入 **100 万英镑**。
该远期合约 **6 个月后到期**。   6 个月 **零息英国国债**（以美元计价后）的**每日波动率为 0.06 %**；  
-6 个月期限 **零息美国国债** 的**每日波动率为 0.05 %**；  - 两只债券回报的相关系数为 **0.8**。  
当时的即期汇率为 **1.53 USD/GBP**。  
请计算该远期合约 **1 天（以美元计）价值变化的标准差**。
<!-- bilingual-en:start -->
A company entered a forward contract to pay **USD 1.5 million** and receive **GBP 1 million** in six months. The dollar value of the six-month sterling zero-coupon bond has daily volatility **0.06%**; the six-month US dollar zero-coupon bond has daily volatility **0.05%**; and their returns have correlation **0.8**. The spot exchange rate is **1.53 USD/GBP**. Calculate the standard deviation of the forward contract's **one-day value change in US dollars**.
<!-- bilingual-en:end -->

**已知数据**  
- 远期合同：未来支付 \$1 500 000，收取 £1 000 000  
- 即期汇率 $S_0 = 1.53\ \text{USD/GBP}$  
- 6 M 英镑零息债日波动 $0.06\%$  
- 6 M 美元零息债日波动 $0.05\%$  
- 两债收益相关系数 $\rho = 0.8$
<!-- bilingual-en:start -->
**Given**
- Forward contract: pay \$1 500 000 and receive £1 000 000 at maturity
- Spot exchange rate: $S_0 = 1.53\ \text{USD/GBP}$
- Daily volatility of the 6 M sterling zero-coupon bond: $0.06\%$
- Daily volatility of the 6 M dollar zero-coupon bond: $0.05\%$
- Correlation between the two bond returns: $\rho = 0.8$
<!-- bilingual-en:end -->

**1. 等效持仓（美元计）**  
- 英镑债市值：$S_0 \times £1\,000\,000$  
- 美元债市值：\$1 500 000
<!-- bilingual-en:start -->
**1. Equivalent positions in US dollars**
- Sterling-bond value: $S_0 \times £1\,000\,000$
- Dollar-bond value: \$1 500 000
<!-- bilingual-en:end -->


**2. 一天价值变动写作**  
$$
\Delta V
  = S_0\,£1{,}000{,}000\,\epsilon_{\text{GBP}}
    \;-\;
    1{,}500{,}000\,\epsilon_{\text{USD}}
$$  
其中  
$\sigma(\epsilon_{\text{GBP}})=0.0006$,  
$\sigma(\epsilon_{\text{USD}})=0.0005$,  
$\operatorname{Corr}(\epsilon_{\text{GBP}},\epsilon_{\text{USD}})=0.8$.
<!-- bilingual-en:start -->
**2. Write the one-day value change**
where
$\sigma(\epsilon_{\text{GBP}})=0.0006$,
$\sigma(\epsilon_{\text{USD}})=0.0005$,
$\operatorname{Corr}(\epsilon_{\text{GBP}},\epsilon_{\text{USD}})=0.8$.
<!-- bilingual-en:end -->

**3. 定义系数**  
$$
a = S_0 \times 1{,}000{,}000 \times 0.0006 = 918, \qquad
b = 1{,}500{,}000 \times 0.0005 = 750
$$  
**4. 价值变化的方差与标准差**  
$$
\begin{aligned}
\sigma_{\Delta V}^2
  &= a^2 + b^2 - 2\rho\,a\,b \\[4pt]
  &= 918^2 + 750^2 - 2(0.8)(918)(750) \\[4pt]
  &= 303\,624
\end{aligned}
$$  
$$
\sigma_{\Delta V} = \sqrt{303\,624} \;\approx\; 551.02\ \text{USD}
$$
**5. 结论**  
远期合约 1 天（美元计）价值变化的标准差  
**≈ \$551.02**
<!-- bilingual-en:start -->
**3. Define the scaled exposures**
The sterling leg has one-standard-deviation exposure $a=1.53(1{,}000{,}000)(0.0006)=918$ dollars; the dollar leg has $b=1{,}500{,}000(0.0005)=750$ dollars.

**4. Calculate variance and standard deviation**
Because the two legs enter the forward with opposite signs, variance is $a^2+b^2-2\rho ab=303{,}624$, so the standard deviation is $\sqrt{303{,}624}\approx551.02$ dollars.

**5. Conclusion**
The one-day standard deviation of the forward's dollar value change is **approximately USD 551.02**.
<!-- bilingual-en:end -->
<!-- GB-41:start -->

> [!note] GB-41｜14.13 缺少零息债的现值输入
> 到期收付金额不是当前零息债市值。设半年英镑、美元贴现因子分别为 $D_{GBP},D_{USD}$，则两条腿的美元现值应为 $1530000D_{GBP}$ 与 $1500000D_{USD}$。沿题给两条腿美元收益波动率，正确的一般式是
> $$a=918D_{GBP},\qquad b=750D_{USD},\qquad \sigma_{\Delta V}=\sqrt{a^2+b^2-2(0.8)ab}.$$
> 原 $551.02$ 的方差算术正确，但采用了忽略贴现、令两因子近似为1的额外近似；当前题面未给出贴现因子或对应利率，不能把它当作完整估值后的唯一数值答案。英镑腿的波动率已是美元计价后的波动率，不另重复叠加汇率冲击。复制关系见 [[外汇远期估值]] 及 [NYU §1，第8页](https://math.nyu.edu/~kohn/derivative.securities/2007/section1.pdf#page=8)。
> <!-- bilingual-en:start -->
> Maturity payments are not current zero-coupon-bond values. With half-year discount factors $D_{GBP}$ and $D_{USD}$, the dollar values of the legs are $1530000D_{GBP}$ and $1500000D_{USD}$. Their scaled exposures and resulting standard deviation are shown above. The USD 551.02 calculation additionally approximates both discount factors by one. Since rates or discount factors are missing, the question as recorded does not determine a unique fully valued answer. The sterling leg's quoted volatility already includes conversion into dollars; do not add another FX shock. See [[外汇远期估值|FX forward valuation]] and the NYU source.
> <!-- bilingual-en:end -->
<!-- GB-41:end -->


# 1. VaR的定义与方差-协方差法原理（含正态分布假设）
<!-- bilingual-en:start -->
*1. Definition of VaR and the Variance–Covariance Method, Including the Normality Assumption*
<!-- bilingual-en:end -->

和前面一样[[VaR]]
<!-- bilingual-en:start -->
This is the same [[VaR|VaR]] concept introduced earlier.
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

# 4. 利率类资产的VaR估算：现金流映射、[[PCA|主成分分析]]（[[PCA|PCA]]）
<!-- bilingual-en:start -->
*4. VaR for Interest-Rate Positions: Cash-Flow Mapping and [[PCA|Principal Component Analysis]] ([[PCA|PCA]])*
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

 **2  为什么要拆到“标准桶”？**
<!-- bilingual-en:start -->
**2. Why map the cash flows into standard buckets?**
<!-- bilingual-en:end -->

- 银行／资产负债管理系统通常只设 **标准期限节点**（3 M、6 M、1 Y、2 Y …）。
- 将任意 $t$ 的现金流 **线性插值** 到最近上下两个节点，可以让后续的久期、缺口、[[Stress Testing|压力测试]] **统一口径**。
<!-- bilingual-en:start -->
- Bank and asset–liability-management systems generally use only **standard maturity nodes**, such as 3 M, 6 M, 1 Y, and 2 Y.
- Linearly interpolating a cash flow at any maturity $t$ between its two nearest nodes creates a common basis for duration, gap analysis, and [[Stress Testing|stress testing]].
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
- 每个桶内都视作 **零息债**，后续贴现、[[duration|久期]]/[[Convexity|凸性]]、缺口分析即可直接调用系统已有工具。
<!-- bilingual-en:start -->
- The mapped face values still total $1,100,000$, so the amount of cash flow is conserved.
- Treat each bucketed amount as a zero-coupon bond. Existing tools can then be used directly for discounting, [[duration|duration]], [[Convexity|convexity]], and gap analysis.
<!-- bilingual-en:end -->

**主成分分析法（[[PCA|PCA]]）**：提取收益率曲线主要因子（平移、斜率、[[Curvature|曲率]]），用少数主成分近似全曲线变动。~~略~~
<!-- bilingual-en:start -->
**Principal component analysis ([[PCA|PCA]]):** Extract the main yield-curve factors—level shifts, slope, and [[Curvature|curvature]]—so that a small number of principal components approximate movements in the full curve. ~~Details omitted in the original outline.~~
<!-- bilingual-en:end -->

## 5. 非线性资产（期权等）的VaR估算方法：[[Delta Approximation|Delta近似法]]、二次模型与Cornish-Fisher展开
<!-- bilingual-en:start -->
*5. VaR for Nonlinear Positions: [[Delta Approximation|Delta Approximation]], Quadratic Models, and the Cornish–Fisher Expansion*
<!-- bilingual-en:end -->

**[[Delta Approximation|Delta近似法]]（线性模型）**：对于期权等非线性衍生品，价格变动和基础资产之间的关系非线性，直接应用方差-协方差法容易失准。**Delta法**通过泰勒展开一阶项，将小幅价格变动下的期权损益近似为线性函数：$\[[Delta|Delta]] P \approx \delta\,\[[Delta|Delta]] S$，其中$\delta$为期权的Delta（即价格对标的资产价格变化的一阶敏感度），$\[[Delta|Delta]] S$是标的资产价格变动这相当于将期权头寸视作持有$\delta$股标的资产的等效头寸，然后用线性组合的方法计算VaR 。[[Delta|Delta]]-Normal法的实现步骤：先求出组合对各基础风险因子的Delta敞口，再将$\delta_i S_i$作为等效资产价值，利用协方差矩阵求取组合方差并计算VaR。这种方法计算简便，适用于Delta支配风险、Gamma和其他高阶效应可以忽略的情况。然而，Delta近似忽略了期权的非线性特征，当价格大幅波动时误差增大，对于深度价内/价外期权或持有大量期权的组合，线性假设可能低估尾部风险。
<!-- bilingual-en:start -->
**[[Delta Approximation|Delta approximation]] (linear model):** An option's price is a nonlinear function of the underlying asset, so a variance–covariance calculation applied directly to the option can be inaccurate. The Delta method keeps the first-order Taylor term and approximates a small price change by $\Delta P \approx \delta\,\Delta S$. Here $\delta$ is the option's Delta, its first-order sensitivity to the underlying price, and $\Delta S$ is the change in that price. The option is therefore treated as an equivalent position of $\delta$ units of the underlying. In Delta–Normal VaR, first calculate each Delta exposure, treat $\delta_i S_i$ as an equivalent linear position, and then use the covariance matrix to obtain portfolio variance and VaR. The method is quick and works when Delta dominates and Gamma and other higher-order effects are negligible. It becomes unreliable for large market moves, deeply in- or out-of-the-money options, or portfolios with substantial optionality, where it may understate tail risk.
<!-- bilingual-en:end -->

**二次模型（[[Delta|Delta]]-Gamma法）**：为提高非线性资产VaR估计精度，可在泰勒展开中保留二阶项。对单一标的资产期权，有$\[[Delta|Delta]] P \approx \delta\,\[[Delta|Delta]] S + \frac{1}{2}\gamma\,(\[[Delta|Delta]] S)^2$，其中$\gamma$为期权的Gamma（二阶敏感度。对于多因子组合，可将每个期权价值变化展开为对其相关单一风险因子的$\delta$和$\gamma$项的和（假设不同资产依赖独立的风险因子）。二次项引入了$\[[Delta|Delta]] S^2$使损益分布不再对称。当标的资产收益$\[[Delta|Delta]] S$近似正态时，$\delta\,\[[Delta|Delta]] S$项呈正态分布，而$\gamma\,(\[[Delta|Delta]] S)^2$项会产生偏度和峰度，使组合损益分布相对于正态出现**偏斜和厚尾**。具体而言：
- 如果$\gamma>0$（组合具有正Gamma，例如持有看涨期权），损益分布左尾比相应正态分布更窄。极端负收益出现概率降低，因此直接用正态假设算出的VaR会偏保守（偏大）
- 如果$\gamma<0$（组合Gamma为负，如卖出期权），损益分布左尾比正态更厚。出现巨大损失的概率高于正态预测，正态假设下计算的VaR将会偏低，低估尾部风险 
<!-- bilingual-en:start -->
**Quadratic model (Delta–Gamma method):** Retaining the second-order Taylor term gives $\Delta P \approx \delta\,\Delta S + \frac{1}{2}\gamma\,(\Delta S)^2$, where $\gamma$ is Gamma, the second-order sensitivity. For a multi-factor portfolio, each option's change can be expanded into Delta and Gamma terms for its relevant risk factors. The squared term $(\Delta S)^2$ makes profit and loss asymmetric. If $\Delta S$ is approximately normal, $\delta\,\Delta S$ is normal but $\gamma\,(\Delta S)^2$ creates skewness and excess kurtosis.
- If $\gamma>0$, as for a long option position, the left tail of profit and loss can be thinner than its linear-normal approximation, so linear-normal VaR may be conservative.
- If $\gamma<0$, as for a short option position, the left tail can be heavier than the normal approximation, so linear-normal VaR may understate tail risk.
<!-- bilingual-en:end -->

Major [[Gamma|Gamma]] combination: Right tail thickening > [[Linear Combination|Linear]] VaR tends to be overestimated T toi [EEL A (normal approx.) 0.40 to ti A+T(+T) ! i --- linear 99% VaR = -2.33 0.35 ti ss real 99% VaR = -1.79 0.30 |i ti ri 0.25 i ri 0.20 ti iG ti ti 0.15 i ri 0.10 fa 13 1 1 0.05 1 1 0.00 -4 -2 0 2 4 6
Major [[Gamma|Gamma]] combination: right tail thickening. [[Linear Combination|Linear]] VaR tends to be **overestimated**.\n- Normal-approx linear 99% VaR: $-2.33$\n- Real 99% VaR: $-1.79$

Minor [[Gamma|Gamma]] combination: Left tail thickening > [[Linear Combination|Linear]] VaR tends to be underestimated 7 [EE A (normal approx.) 1 0.40 A+T(CD 1 --- linear 99% VaR = -2.33 ! 0.35 | …… real 99% VaR = -2.87 | 0.30 po i po pa 0.25 po po pa po 0.20 poy pa po pa 0.15 pa pa pa it 0.10 i pa 0.05 pot iat 0.00 -6 -4 -2 0 2 4
Minor [[Gamma|Gamma]] combination: left tail thickening. [[Linear Combination|Linear]] VaR tends to be **underestimated**.\n- Normal-approx linear 99% VaR: $-2.33$\n- Real 99% VaR: $-2.87$

二次模型要求计算组合损益分布的二阶矩、三阶矩等统计量，然后根据偏度、峰度对VaR进行修正。
<!-- bilingual-en:start -->
A quadratic model requires the second and third moments, and sometimes higher moments, of portfolio profit and loss. VaR can then be adjusted for the resulting skewness and kurtosis.
<!-- bilingual-en:end -->

**[[Cornish-Fisher Expansion|Cornish-Fisher展开]]**：这是一种利用分布矩（矩阶）来近似求解分位数的方法 。在VaR计算中，常用Cornish-Fisher展开根据分布的偏度和峰度对正态分位数进行调整，从而估计非正态分布的VaR。基本思想是：设损益分布的标准化偏度为$\gamma_3$（即三阶中心矩），则调整后的$\alpha$分位数近似为：
<!-- bilingual-en:start -->
**[[Cornish-Fisher Expansion|Cornish–Fisher expansion]]:** This method approximates a distribution's quantiles from its moments. In VaR work, it adjusts a normal quantile for skewness and kurtosis to approximate a non-normal quantile. If standardised skewness is $\gamma_3$, the adjusted $\alpha$-quantile is approximately:
<!-- bilingual-en:end -->

$$ z_{\text{adj}} = z_{\alpha} + \frac{1}{6}(z_{\alpha}^2 - 1)\,\gamma_3 + \cdots $$

（上式省略了涉及峰度的高阶项）。其中$z_{\alpha}$为正态分布的$\alpha$分位数，$z_{\text{adj}}$为考虑偏度修正后的等效分位数。如果分布偏度$\gamma_3$为负（左偏，厚尾在左侧），则$(z_{\alpha}^2-1)\gamma_3$项为负，使$z_{\text{adj}} < z_{\alpha}$，表明实际分位数在左尾更极端，VaR应比正态估计更大；反之，$\gamma_3$为正（右偏），$z_{\text{adj}} > z_{\alpha}$，对应VaR降低。通过Cornish-Fisher公式，可以在已知组合损益的一、二、三阶矩的情况下，调整正态VaR的结果以更贴近真实分布的VaR 。需要注意当分布偏度、峰度很大时，该近似的精度可能降低，但它提供了一个相对简单的修正思路。
<!-- bilingual-en:start -->
The displayed formula omits higher-order terms involving kurtosis. Here $z_{\alpha}$ is the normal $\alpha$-quantile and $z_{\text{adj}}$ is its skewness-adjusted counterpart. The sign must be interpreted consistently: for a profit-and-loss distribution, VaR uses a lower-tail quantile, whereas a positive loss variable uses an upper-tail quantile. Negative skewness moves the lower P&L quantile farther into the left tail and generally raises loss VaR; under the loss convention, the corresponding skewness sign is reversed. Cornish–Fisher can improve on normal VaR when the first few moments are estimated reliably, but the approximation may become non-monotonic or inaccurate when skewness or kurtosis is large.
<!-- bilingual-en:end -->

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

## 6. [[Monte Carlo Simulation Method|蒙特卡罗模拟法]]：原理、步骤与优缺点
<!-- bilingual-en:start -->
*6. [[Monte Carlo Simulation Method|Monte Carlo Simulation]]: Principle, Procedure, Advantages, and Limitations*
<!-- bilingual-en:end -->

## 7. 不同VaR方法的对比分析（优劣、适用场景）
<!-- bilingual-en:start -->
*7. Comparing VaR Methods: Strengths, Limitations, and Appropriate Uses*
<!-- bilingual-en:end -->

常用的VaR计量方法主要有参数法（[[Variance-Covariance Method|方差-协方差法]]）、历史模拟法和蒙特卡罗模拟法。它们各有优缺点，在不同情境下适用性不同：
<!-- bilingual-en:start -->
The main VaR methods are the parametric [[Variance-Covariance Method|variance–covariance method]], historical simulation, and Monte Carlo simulation. Each has different strengths, weaknesses, and suitable applications:
<!-- bilingual-en:end -->

- **[[Variance-Covariance Method|方差-协方差法]]（[[Variance-Covariance Method|参数法]]）**：计算快速，理解和实现简单。只需估计均值、方差和相关系数等参数，就能得到风险值，便于日常风险监控和报告。VaR提供了统一的风险度量语言，管理者和投资者易于理解，对监管资本计算也有参考价值 。然而，该方法**假定收益分布形状**（通常正态），存在模型风险。当资产收益呈现厚尾或偏态时，正态假设会低估极端风险。另外参数法主要基于**线性近似**，无法准确处理期权等非线性产品（Gamma风险、波动率风险被忽略）。**适用场景**：组合以线性资产为主、收益分布接近正态，例如股票+债券的传统投资组合在正常市场波动情况下，可采用参数法快速估计VaR；也常用于高频实时风险估计（因计算简便）。对于包含少量期权的组合，可在参数法基础上做Delta近似，但需警惕误差。
<!-- bilingual-en:start -->
- **[[Variance-Covariance Method|Variance–covariance method]] ([[Variance-Covariance Method|parametric method]]):** Fast, transparent, and easy to implement. Once means, variances, and correlations have been estimated, VaR can be produced quickly for routine monitoring and reporting. Its main weaknesses are model risk and linearity. A normal distribution can understate risk when returns are skewed or heavy-tailed, and a linear approximation misses option Gamma, volatility risk, and other nonlinear effects. **Best suited to:** portfolios dominated by linear assets with approximately elliptical return distributions, such as conventional stock-and-bond portfolios in ordinary market conditions. A Delta approximation can accommodate limited optionality, but the approximation error must be monitored.
<!-- bilingual-en:end -->

- **[[Historical Simulation Method|历史模拟法]]**：不对收益分布作特定假设，直接使用历史实际的数据计算VaR，因而**能够自然反映真实分布的胖尾和异常波动**。实现起来也相对简单：将过去一段时间每一天的组合收益按照当前持仓重算，再从历史损益分布中取所需分位数作为VaR。历史法**优点**是直观透明，结果容易解释，同时能自动涵盖组合的非线性因素（因为每个历史情景下都对组合重新定价，无需线性近似）。**缺点**在于完全依赖历史样本：如果历史数据有限或不具代表性（例如从未经历过某些极端事件），VaR估计可能不准确甚至具有误导性。历史模拟假设“未来风险等同于过去风险”，当市场环境发生结构性变化时（比如制度变迁、黑天鹅事件），历史法就失效了。此外，历史模拟对置信水平越高所需的数据量越大，例如要估计99% VaR通常需要至少100天以上的历史数据，否则分位数估计很不稳定。**适用场景**：当有足够长且具有代表性的历史数据，且组合包含明显的非线性因素时，历史模拟是比参数法更稳健的选择。例如，用于捕捉股票指数在金融危机时期的极端波动风险，或者包含复杂期权组合但希望避免模型假设，则历史模拟能提供基于真实分布的VaR。不过对于全新交易策略（缺乏历史）或市场状态明显不同于过去的情况，历史法并不适用。
<!-- bilingual-en:start -->
- **[[Historical Simulation Method|Historical simulation]]:** Reprices today's portfolio under actual historical market moves and takes a quantile of the resulting empirical profit-and-loss distribution. It imposes no parametric distribution and can preserve observed skewness, heavy tails, and cross-factor dependence. Full repricing also captures nonlinear positions. Its central weakness is dependence on the chosen historical window: unobserved shocks cannot appear, structural change can make the sample unrepresentative, and high-confidence quantiles are noisy unless the sample is sufficiently large. A bare minimum of 100 observations leaves only one observation in the 1% tail, so practical 99% estimation normally requires substantially more data. **Best suited to:** portfolios for which a long, representative history exists and full repricing under historical shocks is feasible; unsuitable for genuinely new strategies or regimes unlike the past.
<!-- bilingual-en:end -->

- **[[Monte Carlo Simulation Method|蒙特卡罗模拟法]]**：具备**最广泛的适用性和灵活性**。通过选择合适的随机模型，蒙特卡罗能模拟出**超越历史样本范围**的情景，包括假想的极端市场冲击，从而评估“极端但有可能”事件的风险敞口。这一方法能够**严格地对复杂衍生品组合进行全重估**，自然地处理非线性和路径依赖风险。对于要求计算高置信度VaR（例如99.9%）的场景，历史数据往往不足，但模拟法可以通过扩大量进行估计。蒙特卡罗的**缺点**主要是计算资源的高要求和模型设置的复杂性，如前所述，需要平衡模拟次数与精度，并承担模型假设错误的风险。**适用场景**：当组合包含大量非线性或路径依赖产品、或者希望评估在历史未出现过的极端情形下的风险，蒙特卡罗方法是首选。例如，大型金融机构的交易盘包含复杂衍生品时，通常使用蒙特卡罗VaR来符合监管要求（如计入非线性风险）。又如需评估某特定假想情景对组合的冲击，可以将该情景嵌入模拟中。需要注意在实际应用中，蒙特卡罗往往结合方差缩减和并行计算技术以提高效率。
<!-- bilingual-en:start -->
- **[[Monte Carlo Simulation Method|Monte Carlo simulation]]:** The most flexible approach. A chosen stochastic model generates scenarios beyond the historical sample, including hypothetical severe shocks, and full revaluation captures nonlinearity and path dependence. Large simulations can estimate very high confidence levels when historical observations are scarce. The costs are computation, implementation complexity, and sensitivity to model assumptions and calibration. **Best suited to:** large portfolios of nonlinear or path-dependent products, or analyses of plausible scenarios not seen historically. Variance reduction and parallel computation are commonly used to improve efficiency.
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
| 1. 配置与已知  | 组合总市值$$V=100000+100000=200000$$权重$$w_A=w_B=\frac{100000}{200000}=0.5$$日波动率$$\sigma_A=\sigma_B=0.01$$[[Correlation Coefficient|相关系数]]$$\rho=0.3$$                                               | 写明输入参数               |
| 2. 组合日方差  |$$\sigma_p=\sqrt{w_A^2\sigma_A^2+w_B^2\sigma_B^2+2w_Aw_B\rho\sigma_A\sigma_B}$$$$=\sqrt{0.25\cdot0.0001+0.25\cdot0.0001+0.5\cdot0.5\cdot0.3\cdot2\cdot0.0001}=0.00806$$| 得出日波动率$$0.806\%$$  |
| 3. 5 天波动率 |$$\sigma_{5d}=\sigma_p\sqrt{5}=0.00806\times2.23607=0.01803$$                                                                                                          | √时距缩放                |
| 4. 查分位点   |$$z_{0.97}=1.881$$                                                                                                                                                     | 正态分布$97\%$单侧$z$值 |
|5. 计算$$VaR$$|$$VaR_{0.97}=z_{0.97},\sigma_{5d},V=1.881\times0.01803\times200000\approx$6,780$$|负向损失界|
|6. 计算$$[[ES|ES]]$$|$$\phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2}=0.0682$$$$ES_{0.97}=\frac{\sigma_{5d}V,\phi(z)}{1-0.97}=\frac{0.01803\times200000\times0.0682}{0.03}\approx$8,220$$|正态$$[[ES|ES]]$$公式|
<!-- bilingual-en:start -->
| **Step** | **Key calculation** | **Purpose** |
| --- | --- | --- |
| 1. Inputs | Total value USD 200,000; weights 0.5 and 0.5; daily volatilities 1%; [[Correlation Coefficient|correlation]] 0.3 | State the inputs |
| 2. Daily portfolio volatility | $\sigma_p=\sqrt{0.5^2(0.01)^2+0.5^2(0.01)^2+2(0.5)(0.5)(0.3)(0.01)^2}=0.00806$ | Daily volatility is 0.806% |
| 3. Five-day volatility | $\sigma_{5d}=0.00806\sqrt{5}=0.01803$ | Square-root-of-time scaling |
| 4. Critical value | $z_{0.97}=1.881$ | One-sided normal 97% quantile |
| 5. VaR | $1.881(0.01803)(200000)\approx\$6,780$ | Loss quantile |
| 6. ES | $0.01803(200000)\phi(1.881)/(0.03)\approx\$8,220$ | Normal ES formula |
<!-- bilingual-en:end -->

 **结论**
<!-- bilingual-en:start -->
**Conclusion**
<!-- bilingual-en:end -->

- 组合 **5 天、97% [[Confidence Level|置信水平]]** 下$$\boxed{VaR\approx6.8\times10^3}$$
- 同期$$\boxed{[[ES|ES]]\approx.2\times10^3}$$
<!-- bilingual-en:start -->
- At a **five-day horizon and 97% [[Confidence Level|confidence level]]**, portfolio VaR is approximately USD 6,800.
- ES over the same horizon is approximately USD 8,200.
<!-- bilingual-en:end -->

## 14.4

>[!question] 
>一家金融机构拥有一个标的变量为USD/CBP汇率的期权投资组合，投资组合相对于汇率变化百分比的de1ta为3.9，如果汇率每天变化的波动率为0.7%，请问10天展望期、99%置信度的VaR为多少?
><!-- bilingual-en:start -->
>A financial institution has an option portfolio on the USD/GBP exchange rate. The portfolio's Delta with respect to the percentage change in the exchange rate is 3.9 million currency units. If daily exchange-rate volatility is 0.7%, what is ten-day 99% VaR?
><!-- bilingual-en:end -->


| **步骤**      | **公式与计算**                                                                                                | **说明**        |
| ----------- | -------------------------------------------------------------------------------------------------------- | ------------- |
| 1. 参数列示     | $$\[[Delta|Delta]]=3.9$$（单位：若汇率变动$1\%$，组合价值变动$3.9$百万）$$\sigma_d=0.7\%=0.007$$（汇率日波动率）$$h=10$$（天数）$$z_{0.99}=2.33$$ | 明确已知          |
| 2. 日收益标准差   | $$\sigma_P=\[[Delta|Delta]]\sigma_d=3.9\times0.007=0.0273$$（百万）                                                    | $\[[Delta|Delta]]$-正态近似 |
| 3. 10 日标准差  | $$\sigma_{10}=\sigma_P\sqrt{h}=0.0273\sqrt{10}=0.0273\times3.1623=0.0863$$（百万）                           | $\sqrt{h}$缩放  |
| 4.$$VaR$$计算 | $$VaR_{0.99}=z_{0.99},\sigma_{10}=2.33\times0.0863\approx0.201\text{ 百万}$$                               | 单边$$99\%$$    |
| 5. 答案       | $$VaR\approx$0.20$$                                                                                      | 结果呈现          |
<!-- bilingual-en:start -->
| **Step** | **Formula and calculation** | **Purpose** |
| --- | --- | --- |
| 1. Inputs | $\Delta=3.9$ million; daily volatility $0.7\%=0.007$; $h=10$; $z_{0.99}=2.33$ | State the inputs |
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
| 2. 组合**日**方差    |$$\sigma_P^2=(\Delta_1\sigma_1)^2+(\Delta_2\sigma_2)^2+2\rho,\Delta_1\Delta_2\sigma_1\sigma_2$$                                                                                                     |$$\sigma_P^2=1.828\times10^9$$|
| 3. 组合**日**标准差   |$$\sigma_P=\sqrt{\sigma_P^2}\approx42,755\ \text{USD}$$                                                                                                                                             | 正态近似                           |
| 4. 5 天标准差       |$$\sigma_{5d}=\sigma_P\sqrt{5}\approx95,600\ \text{USD}$$                                                                                                                                           |$$\sqrt{h}$$缩放                |
| 5.$$[[ES|ES]]$$公式与计算 |$$z_{0.98}=2.054,\quad \phi(z)=\frac{1}{\sqrt{2\pi}}e^{-z^2/2}\approx0.0484$$$$ES_{0.98}=\frac{\sigma_{5d},\phi(z_{0.98})}{1-0.98}\approx\frac{95,600\times0.0484}{0.02}\approx231,000\ \text{USD}$$|$$\phi$$为标准正态密度               |
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
## 14.8
>[!question] 
假设某投资组合的每天价值变化与主成分分析（[[PCA|PCA]]）法所计算出的两个因子呈最好的线性关系。  
投资组合对于**第一个因子**的 *delta* 为 **6**，对于**第二个因子**的 *delta* 为 **-4**。  
两个因子的标准差分别为 **20** 与 **8**。  
试求该投资组合 **5 天展望期、90 % [[Confidence Level|置信水平]]** 的 VaR 为多少？  
<!-- bilingual-en:start -->
Suppose a portfolio's daily value change is best represented as a linear function of two factors from principal component analysis ([[PCA|PCA]]). Its *delta* is **6** with respect to the first factor and **-4** with respect to the second. The factor standard deviations are **20** and **8**. Find five-day VaR at the **90% [[Confidence Level|confidence level]]**.
<!-- bilingual-en:end -->

| 步骤           | 关键公式                                                                              | 计算                                                                | 说明                      |
| ------------ | --------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------- |
| 1️⃣ 因子方差贡献   | $\sigma_{P,\text{day}}^{2} = ( \Delta_1\sigma_1 )^{2} + ( \Delta_2\sigma_2 )^{2}$ | $(6×20)^2 + (-4×8)^2 = 120^2 + 32^2 = 14\,400 + 1\,024 = 15\,424$ | **[[PCA|PCA]] 因子彼此正交 ⇒ 协方差为 0** |
| 2️⃣ 每日波动率    | $\sigma_{P,\text{day}} = \sqrt{15\,424} = 124.1$                                  | 单位同投资组合货币                                                         |                         |
| 3️⃣ 5-天波动率   | $\sigma_{P,5d} = \sigma_{P,\text{day}}\sqrt{5} = 124.1×2.236 = 277.2$             | 令天际独立同分布([[IID|IID]])                                                     |                         |
| 4️⃣ 90 % VaR | $\text{VaR}_{0.90,5d} = z_{0.90}\, \sigma_{P,5d}$；$z_{0.90}=1.281$                | $1.281×277.2 = 355.7$                                             |                         |
| 5️⃣ 结论       | **5 天、90 % VaR ≈ 356**                                                            | 取绝对值表示潜在损失                                                        |                         |
<!-- bilingual-en:start -->
| Step | Key formula | Calculation | Explanation |
| --- | --- | --- | --- |
| 1. Factor contributions to variance | $\sigma_{P,\text{day}}^{2} = ( \Delta_1\sigma_1 )^{2} + ( \Delta_2\sigma_2 )^{2}$ | $(6×20)^2 + (-4×8)^2 = 120^2 + 32^2 = 14\,400 + 1\,024 = 15\,424$ | [[PCA|PCA]] factors are orthogonal, so covariance is 0 |
| 2. Daily standard deviation | $\sigma_{P,\text{day}} = \sqrt{15\,424} = 124.1$ | Same currency units as portfolio value | |
| 3. Five-day standard deviation | $\sigma_{P,5d} = \sigma_{P,\text{day}}\sqrt{5} = 124.1×2.236 = 277.2$ | Assume independent and identically distributed ([[IID|IID]]) daily changes | |
| 4. 90% VaR | $\text{VaR}_{0.90,5d} = z_{0.90}\, \sigma_{P,5d}$; $z_{0.90}=1.281$ | $1.281×277.2 = 355.7$ | |
| 5. Conclusion | **Five-day 90% VaR ≈ 356** | Report the potential loss as a positive amount | |
<!-- bilingual-en:end -->

> **一句话记忆**：因子正交 → 方差相加；多日 VaR = 单日 σ × √天数 × z-score。
> <!-- bilingual-en:start -->
> **One-sentence reminder:** orthogonal factors imply that their variance contributions add; multi-day VaR equals daily $\sigma$ times the square root of the horizon times the relevant z-score.
> <!-- bilingual-en:end -->

## 14.10
>[!question] 
一家银行拥有某资产的多个期限权投资组合，期权组合的 *delta* 为 **-30**，*gamma* 为 **-5**。  1. 先解释这两个数字的含义。资产现价为 **20**，其**每日价格变化的波动率为 1 %**。   2. 采用 **Isserlis 定理**计算投资组合价值变化的前三阶矩；再结合 **[[Cornish-Fisher Expansion|Cornish]]–Fisher 展开**，分两种情形求 **1 天展望期、99 % [[Confidence Level|置信水平]]** 的 VaR：   (a) 仅使用前 **二阶矩**；   (b) 使用 **前三阶矩**。  
<!-- bilingual-en:start -->
A bank holds a portfolio of options on one asset. Portfolio *delta* is **-30** and *gamma* is **-5**. First explain these sensitivities. The asset price is **20**, and its daily price volatility is **1%**, so the daily standard deviation of the price change is 0.2. Next use **Isserlis' theorem** to calculate the first three moments of portfolio value change. Then use the **[[Cornish-Fisher Expansion|Cornish]]–Fisher expansion** to estimate one-day 99% VaR (a) from the first two moments and (b) from the first three moments.
<!-- bilingual-en:end -->

**完整数值汇总（金额单位）**
<!-- bilingual-en:start -->
**Complete numerical summary, in currency units**
<!-- bilingual-en:end -->

| 符号 / 指标 | 计算式 | 数值 |
|--------------|--------|------|
| 均值 $\mu$ | $\tfrac12\[[Gamma|Gamma]]\,\sigma_S^2$ | $-0.10$ |
| 方差 $\sigma_P^2$ | $\[[Delta|Delta]]^2\sigma_S^2 + \bigl(\tfrac12\[[Gamma|Gamma]]\bigr)^2 3\sigma_S^4$ | $36.02$ |
| 标准差 $\sigma_P$ | $\sqrt{36.02}$ | $6.00$ |
| 三阶中心矩 $\mu_3$ | $3\[[Delta|Delta]]^2(\tfrac12\[[Gamma|Gamma]]) 3\sigma_S^4 + (\tfrac12\[[Gamma|Gamma]])^3 15\sigma_S^6$ | $-21.61$ |
| 偏度 $\gamma_1$ | $\mu_3/\sigma_P^3$ | $-0.100$ |
| $z_{0.99}$ (正态) | — | $2.326$ |
| 调整后 $z'$ | $z + \dfrac{z^2-1}{6}\gamma_1$ | $2.25$ |
<!-- bilingual-en:start -->
| Symbol or measure | Calculation | Value |
| --- | --- | ---: |
| Mean $\mu$ | $\tfrac12\Gamma\,\sigma_S^2$ | $-0.10$ |
| Variance $\sigma_P^2$ | $\Delta^2\sigma_S^2 + \tfrac12\Gamma^2\sigma_S^4$ | $36.02$ |
| Standard deviation $\sigma_P$ | $\sqrt{36.02}$ | $6.00$ |
| Third central moment $\mu_3$ | $3\Delta^2\Gamma\sigma_S^4 + \Gamma^3\sigma_S^6$ | $-21.61$ |
| Skewness $\gamma_1$ | $\mu_3/\sigma_P^3$ | $-0.100$ |
| Normal $z_{0.99}$ | — | $2.326$ |
| Adjusted $z'$ | $z + \dfrac{z^2-1}{6}\gamma_1$ | $2.25$ |
<!-- bilingual-en:end -->

---

| 方案 | 1-Day · 99 % VaR |
|------|-----------------|
| (a) 仅二阶矩（正态） | **13.86** |
| (b) 含三阶矩（[[Cornish-Fisher Expansion|Cornish]]–Fisher） | **13.42** |
<!-- bilingual-en:start -->
| Method | One-day 99% VaR |
| --- | ---: |
| (a) First two moments only, normal approximation | **13.86** |
| (b) First three moments, [[Cornish-Fisher Expansion|Cornish]]–Fisher | **13.42** |
<!-- bilingual-en:end -->

> 正向为损失：两种方法均表明，在 99 % 置信水平下，组合 1 天潜在最大损失约 13.4–13.9 单位。负偏度 ($\gamma_1<0$) 使 [[Cornish-Fisher Expansion|Cornish]]–Fisher 修正的 VaR 略低于正态估计。
> <!-- bilingual-en:start -->
> With losses reported as positive amounts, both methods put one-day 99% VaR at roughly 13.4–13.9 currency units. Under the convention used in this worked example, negative skewness ($\gamma_1<0$) makes the [[Cornish-Fisher Expansion|Cornish]]–Fisher estimate slightly smaller than the normal estimate; always check whether skewness is defined for P&L or for positive loss before applying that sign rule.
> <!-- bilingual-en:end -->

## 14.13
>[!question] 
 假定在过去的某一时间，某家公司签署了一项远期合约，约定在未来某时以 **100 万英镑** 买入 **150 万美元**。  
该远期合约 **6 个月后到期**。   6 个月 **零息英国国债**（以美元计价后）的**每日波动率为 0.06 %**；  
-6 个月期限 **零息美国国债** 的**每日波动率为 0.05 %**；  - 两只债券回报的相关系数为 **0.8**。  
当时的即期汇率为 **1.53 USD/GBP**。  
请计算该远期合约 **1 天（以美元计）价值变化的标准差**。
<!-- bilingual-en:start -->
A company entered a forward contract to pay **GBP 1 million** and receive **USD 1.5 million** in six months. The dollar value of the six-month sterling zero-coupon bond has daily volatility **0.06%**; the six-month US dollar zero-coupon bond has daily volatility **0.05%**; and their returns have correlation **0.8**. The spot exchange rate is **1.53 USD/GBP**. Calculate the standard deviation of the forward contract's **one-day value change in US dollars**.
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
\[[Delta|Delta]] V
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
\sigma_{\[[Delta|Delta]] V}^2
  &= a^2 + b^2 - 2\rho\,a\,b \\[4pt]
  &= 918^2 + 750^2 - 2(0.8)(918)(750) \\[4pt]
  &= 303\,624
\end{aligned}
$$  
$$
\sigma_{\[[Delta|Delta]] V} = \sqrt{303\,624} \;\approx\; 5.52 \times 10^{2}
$$
**5. 结论**  
远期合约 1 天（美元计）价值变化的标准差  
**≈ \$552 000（约 \$0.55 million）**
<!-- bilingual-en:start -->
**3. Define the scaled exposures**
The sterling leg has one-standard-deviation exposure $a=1.53(1{,}000{,}000)(0.0006)=918$ dollars; the dollar leg has $b=1{,}500{,}000(0.0005)=750$ dollars.

**4. Calculate variance and standard deviation**
Because the two legs enter the forward with opposite signs, variance is $a^2+b^2-2\rho ab=303{,}624$, so the standard deviation is $\sqrt{303{,}624}\approx552$ dollars.

**5. Conclusion**
The one-day standard deviation of the forward's dollar value change is **approximately USD 552**. The Chinese source's final “USD 552,000” is a factor-of-1,000 error relative to its own inputs and displayed calculation.
<!-- bilingual-en:end -->

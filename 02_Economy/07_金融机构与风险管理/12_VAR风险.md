# 1. VaR 定义及解释
<!-- bilingual-en:start -->
*1. VaR Definition and Explanation*
<!-- bilingual-en:end -->

[[VaR]]（[[VaR-hub|Value at Risk]]，[[VaR-hub|风险价值]]）表示在正常市场条件下、给定置信水平（如95%或99%）和持有期内，金融资产或投资组合可能遭受的最大潜在损失。换言之，VaR回答的是：“在未来特定时期内，有一定概率（如99%）损失不超过多少。” 
<!-- bilingual-en:start -->
[[VaR|VaR]] ([[VaR-hub|value at risk]]) is the loss quantile for a financial asset or portfolio over a specified holding period and at a specified confidence level, usually under ordinary market conditions. It is often described informally as a “maximum loss,” but it is not the worst loss that can ever occur. Rather, VaR answers: “Over this horizon, what loss threshold will be exceeded only with a stated small probability, such as 1%?”
<!-- bilingual-en:end -->

从定义来看，如果记$X_t$为时刻$t$资产价值，$L_{t\to T}$为从$t$到$T$期间资产的损失（损失取正值），那么在置信水平$\alpha$下、持有期为$T-t$的VaR可以表示为该损失分布的$\alpha$分位数：
$$
\text{VaR}_{\alpha}(T-t) = F^{-1}_{L}(\alpha),
$$
其中$F^{-1}_L$表示损失随机变量$L$的分位数函数。此定义意味着：
$$
P(L \le \text{VaR}_{\alpha}) = \alpha,\quad P(L > \text{VaR}_{\alpha}) = 1-\alpha,
$$ 
即有$(1-\alpha)$的概率损失将超过VaR。【通常将损失以正值表示，VaR本身取一个非负货币额。】
<!-- bilingual-en:start -->
Formally, let $X_t$ be the asset value at time $t$ and let $L_{t\to T}$ be the positive loss between $t$ and $T$. At confidence level $\alpha$ and holding period $T-t$, VaR is the $\alpha$ quantile of the loss distribution. Here, $F^{-1}_L$ is the quantile function of the loss variable $L$. Equivalently, the probability that loss exceeds VaR is $1-\alpha$. Losses are normally recorded as positive amounts, so VaR is reported as a non-negative monetary figure.
<!-- bilingual-en:end -->

直观解释：例如某投资组合10日、99%置信水平的VaR为6400万元，意味着在未来10天内，该组合有99%的把握损失不超过6400万元（损失超过6400万的概率仅为1%）。==VaR常以绝对金额表示这一最大可能损失。==
<!-- bilingual-en:start -->
Intuition: if a portfolio has a ten-day 99% VaR of CNY 64 million, the model says that its loss over the next ten days will not exceed CNY 64 million with 99% probability; the probability of a larger loss is 1%. ==VaR usually reports this threshold as an absolute monetary amount.==
<!-- bilingual-en:end -->

# 2. 三种 VaR 测算方法及优缺点
<!-- bilingual-en:start -->
*2. Three VaR Methods and Their Advantages and Disadvantages*
<!-- bilingual-en:end -->

计算VaR常用的三种方法为：**[[Variance-Covariance Method|方差-协方差法]]（[[Variance-Covariance Method|参数法]]）**、**[[Historical Simulation Method|历史模拟法]]**和**[[Monte Carlo Simulation Method|蒙特卡罗模拟法]]**。
<!-- bilingual-en:start -->
Three common methods for calculating VaR are **[[Variance-Covariance Method|variance-covariance method]] ([[Variance-Covariance Method|parametric method]])**, **[[Historical Simulation Method|historical simulation method]]** and **[[Monte Carlo Simulation Method|Monte Carlo simulation method]]**.
<!-- bilingual-en:end -->
## 2.1 [[方差-协方差法]]（正态分布假设法）
<!-- bilingual-en:start -->
*2.1 [[方差-协方差法|Variance–Covariance Method]] (Normal Parametric Method)*
<!-- bilingual-en:end -->

计算方法：方差-协方差法是假定资产或投资组合收益（或损失）服从某种已知分布（通常假设正态分布），利用收益的期望和方差-协方差等参数直接计算VaR。该方法也称“[[Variance-Covariance Method|参数法]]”。在最常见的正态分布假设下，若已知组合在单位时间的预期收益$\mu$和标准差$\sigma$，则$\alpha$置信水平下的一天VaR为：
<!-- bilingual-en:start -->
**Method:** The variance–covariance method assumes a known distribution for asset or portfolio returns, usually a normal distribution, and calculates VaR from parameters such as the mean and covariance matrix. It is therefore also called the [[Variance-Covariance Method|parametric method]]. Under the standard normal specification, if one-period portfolio return has mean $\mu$ and standard deviation $\sigma$, the one-day VaR at confidence level $\alpha$ is given by the displayed formula.
<!-- bilingual-en:end -->

$$
\text{VaR}_{\alpha,1\text{天}} = -\left(\mu + z_{\alpha}\sigma\right),
$$ 
其中$z_{\alpha}$是标准正态分布的$\alpha$分位点（例如$\alpha=99\%$时$z_{0.99}\approx 2.33$；$\alpha=95\%$时$z_{0.95}\approx 1.645$）。当假设$\mu\approx0$时，VaR近似$z_{\alpha}\sigma$。如果持有期为$N$天且每天损失独立，则$N$天VaR可按$\text{VaR}_{\alpha,N\text{天}} \approx z_{\alpha}\sigma\sqrt{N}$进行尺度调整（详见后文持有期的影响）。
<!-- bilingual-en:start -->
Here, $z_{\alpha}$ is the $\alpha$ quantile of the standard normal distribution: for example, $z_{0.99}\approx 2.33$ when $\alpha=99\%$, and $z_{0.95}\approx 1.645$ when $\alpha=95\%$. If $\mu\approx0$, VaR is approximately $z_{\alpha}\sigma$. If daily losses are independent and the holding period is $N$ days, scale by $\sqrt{N}$: $\text{VaR}_{\alpha,N\text{ days}} \approx z_{\alpha}\sigma\sqrt{N}$.
<!-- bilingual-en:end -->

**优点：** 
- 计算快速简便，只需估计期望、方差等参数即可算出VaR，适合日常风险监控。  
- 容易理解和实现，基于成熟的统计模型（如正态分布）有解析解。  
- 对于线性投资组合（如仅包含线性金融产品），在正态等假定下计算结果明确可靠。
<!-- bilingual-en:start -->
**Advantages:**
- Fast: only a small set of parameters, such as means, variances, and covariances, must be estimated, which is useful for routine risk monitoring.
- Easy to implement: common distributions such as the normal distribution yield closed-form results.
- Well suited to linear portfolios when the distributional and covariance assumptions are reasonable.
<!-- bilingual-en:end -->

**缺点：** 
- 分布假设局限：通常假设收益正态分布，实际金融资产常呈现厚尾、偏态，正态假设可能低估极端风险。  
- 非线性风险低估：对包含期权等非线性衍生品的组合，正态参数法须近似处理（如Delta-Gamma方法），否则VaR误差较大。  
- 相关性假设简化：通常用历史协方差矩阵，假定相关关系稳定，但市场相关性在危机时可能剧变，参数法难以捕捉。
<!-- bilingual-en:start -->
**Disadvantages:**
- **Distributional risk:** Financial returns are often skewed and heavy-tailed, so a normal model can understate extreme losses.
- **Nonlinearity:** Portfolios containing options or other nonlinear instruments require an approximation such as Delta–Gamma; a purely linear approximation can be inaccurate.
- **Unstable dependence:** Historical covariance estimates treat dependence as sufficiently stable, even though correlations can change sharply in a crisis.
<!-- bilingual-en:end -->

## 2.2 [[历史模拟法]]
<!-- bilingual-en:start -->
*2.2 [[历史模拟法|historical simulation method]]*
<!-- bilingual-en:end -->

计算方法：历史模拟法无需对收益分布作特定假设，而直接利用过去观察到的历史数据来估计VaR。步骤为：收集过去一段观察期内资产或组合的实际收益/损失数据，假设未来风险与历史相似；将这些收益按从最坏损失到最好的结果排序，取相应分位数处的损失作为VaR。例如，有100天历史日损益数据，99% VaR 对应将损益从最亏到最盈排序后第~~第~~2 大亏损（因为 1% 的尾部对应最差 1 天）。历史模拟直接以过去实际波动来评估未来风险。
<!-- bilingual-en:start -->
**Method:** Historical simulation makes no parametric assumption about return distributions. Revalue today's portfolio under a window of observed historical market moves, sort the resulting profit-and-loss observations from worst to best, and select the relevant loss quantile. With 100 daily observations, a 99% VaR is determined by the extreme 1% tail; the precise order statistic depends on the chosen sample-quantile convention. The method therefore asks what today's portfolio would have lost under past market moves.
<!-- bilingual-en:end -->

**优点：** 
- 直观易行：不需复杂模型，基于真实历史损益数据，计算过程简单明了。  
- 无需分布假设：属于非参数方法，可自然反映厚尾等特征，极端事件如果在历史中出现过会体现在结果中。  
- 适用广泛：可直接用于任意资产组合（包括非线性产品），只要有足够历史数据，统统按照实际盈亏重算组合损失即可。
<!-- bilingual-en:start -->
**Advantages:**
- Intuitive: it uses observed historical market moves and requires no complex distributional model.
- Nonparametric: skewness and heavy tails are retained if they occurred in the sample.
- Flexible valuation: nonlinear instruments can be fully revalued under each historical scenario, provided adequate market histories exist.
<!-- bilingual-en:end -->

**缺点：** 
- 对历史数据依赖强：假设“历史重演”，如果未来市场状况与过去不同（结构性变化、未知风险），则VaR估计会失准。尤其对于未出现过的极端事件，历史模拟无法捕捉，可能低估尾部风险。  
- 数据量要求高：置信水平越高（如99.9%），需要越长历史样本才能可靠估计分位数，否则结果不稳定。  
- 无法反映新组合：对于新的或剧变的投资组合，历史数据可能不足以代表当前风险（例如新增资产缺乏历史数据）。此外，历史模拟隐含假设近期波动可代表未来，缺少对波动时变性的建模。
<!-- bilingual-en:start -->
**Disadvantages:**
- **Dependence on history:** Structural changes and events absent from the sample cannot be represented, so tail risk may be understated.
- **Large sample requirement:** Very high confidence levels, such as 99.9%, need long histories to estimate a tail quantile with tolerable sampling error.
- **Poor fit for new or changing portfolios:** New instruments may lack histories, and an unweighted historical window does not model time-varying volatility unless it is modified.
<!-- bilingual-en:end -->

## 2.3 [[蒙特卡罗模拟方法]]
<!-- bilingual-en:start -->
*2.3 [[蒙特卡罗模拟方法|Monte Carlo Simulation]]*
<!-- bilingual-en:end -->

计算方法：蒙特卡罗模拟法通过对风险因素进行随机抽样模拟大量可能的未来情景，从而估计组合损益分布并计算VaR。一般步骤：根据假定的概率分布模型（可考虑更复杂的分布、相关结构），用计算机生成成千上万次模拟的价格路径或收益样本，每次重新估值组合得到一次收益/损失结果；然后对所有模拟损失取$\alpha$分位数作为VaR。此方法灵活强大，适用于复杂组合。
<!-- bilingual-en:start -->
**Method:** Monte Carlo simulation draws risk-factor shocks from a calibrated model to create many possible future scenarios. The portfolio is revalued in every scenario, producing a simulated loss distribution; its $\alpha$ quantile is the VaR. Because both the risk-factor model and valuation function can be complex, the method is especially useful for nonlinear or path-dependent portfolios.
<!-- bilingual-en:end -->

**优点：** 
- 适用复杂情况：能够处理高维、多因素的组合风险，适用于包含非线性、路径依赖的衍生品，突破了解析法的局限。  
- 分布灵活：可以假设和校准更贴近现实的收益分布（如带厚尾、偏度），或者采用历史数据重抽样（Bootstrapping）来模拟，从而更准确反映风险。  
- 可计算任意风险指标：除了VaR，还可同时估计预期损失等其他风险度量，获得整个损失分布信息。
<!-- bilingual-en:start -->
**Advantages:**
- Handles high-dimensional, nonlinear, and path-dependent portfolios.
- Allows flexible marginal distributions, tail behavior, and dependence structures, including calibrated heavy-tailed models or bootstrap resampling.
- Produces an entire simulated loss distribution, from which VaR, [[ES|ES]], and other risk measures can be estimated.
<!-- bilingual-en:end -->

**缺点：** 
- 计算量大：需大量模拟运行，计算过程耗时，尤其对大型组合需要高性能计算资源。模拟次数不足时，结果会有随机误差。  
- 模型风险：模拟结果依赖于对风险因子分布和相关结构的假设，若模型假设不当（例如相关性危机时被低估），VaR结果仍可能有偏差。  
- 实现较复杂：需构建全面的估值模型和随机发生器，对技术和数据要求较高，不如参数法和历史法直观。
<!-- bilingual-en:start -->
**Disadvantages:**
- Computationally expensive; too few simulations leave material Monte Carlo error, particularly in the tail.
- Exposed to model risk because results depend on the assumed risk-factor distributions, dynamics, and correlations.
- Operationally demanding: robust valuation models, data pipelines, random-number generation, and validation are all required.
<!-- bilingual-en:end -->

# 3. VaR 关键参数设定与影响分析
<!-- bilingual-en:start -->
*3. VaR Key Parameter Setting and Impact Analysis*
<!-- bilingual-en:end -->

VaR的数值取决于所采用的**[[Confidence Level|置信水平]]、[[Holding Period|持有期]]**（或称展望期）**以及观察期**（历史数据窗口）等关键参数的设定。这些参数的选择会显著影响VaR结果，需要根据监管要求和实际风险偏好进行设定。
<!-- bilingual-en:start -->
VaR depends on several design choices: the **[[Confidence Level|confidence level]]**, **[[Holding Period|holding period]]** or horizon, and the **observation window** used to estimate the model. Changing any of them can materially change the reported number, so they must match the decision, regulatory rule, and risk appetite for which VaR is being used.
<!-- bilingual-en:end -->

## 3.1 [[Confidence Level|置信水平]]
<!-- bilingual-en:start -->
*3.1 [[Confidence Level|confidence level]]*
<!-- bilingual-en:end -->

**[[Confidence Level|置信水平]]**（[[Confidence Level|Confidence Level]]）指计算VaR时要有多大的把握不超过该损失金额，常用的有90%、95%、99%、99.9%等。置信水平越高，意味着要求对极端损失也有更高的把握度，因此**VaR值会越大**（因为需要覆盖更极端的不利情形）。 
<!-- bilingual-en:start -->
The **[[Confidence Level|confidence level]]** is the probability with which modeled loss should not exceed VaR. Common choices include 90%, 95%, 99%, and 99.9%. A higher confidence level reaches farther into the adverse tail, so the corresponding **VaR is larger**, all else equal.
<!-- bilingual-en:end -->

## 3.2 [[Holding Period|持有期]]（展望期）
<!-- bilingual-en:start -->
*3.2 [[Holding Period|Holding Period]] (Risk Horizon)*
<!-- bilingual-en:end -->

**[[Holding Period|持有期]]**（[[Holding Period|Holding Period]]，也称展望期）是指VaR所针对的未来时间长度，即假定头寸在多长时间内可能无法或不方便调整，从而暴露于市场风险。典型持有期可以是1天、10天、2周、1个月等。持有期应根据资产流动性和风险管理需要确定：**流动性高**的头寸（如日交易的股票、外汇头寸）常用**1天**VaR，因为可以每日调整仓位；**长期投资**（如养老基金、大型投资组合）可用**更长周期**（如10天、1个月）VaR评估中长期风险。
<!-- bilingual-en:start -->
The **[[Holding Period|holding period]]**, or risk horizon, is the future interval over which the position is assumed to remain exposed before it can be adjusted or unwound. Typical horizons include one day, ten days, two weeks, and one month. Highly liquid trading positions often use a one-day horizon, while less liquid or longer-term positions may require longer horizons. The chosen horizon should reflect both liquidity and the purpose of the risk measure.
<!-- bilingual-en:end -->

**监管要求：巴塞尔委员会为市场风险计量规定了**10天**的持有期，即监管VaR通常指10日99% VaR；信用和操作风险则采用一年（约252交易日）的99.9% VaR来计算所需资本。这些较长持有期是出于审慎考虑，因为在压力情况下可能需要较长时间平仓或对冲风险。**
<!-- bilingual-en:start -->
**Regulatory context:** The earlier Basel market-risk VaR framework used a ten-day, 99% VaR for capital calculations. The current FRTB internal-model framework instead uses 97.5% [[ES|ES]] with prescribed liquidity-horizon adjustments. Credit and operational-risk capital are commonly framed over a one-year horizon at very high confidence levels. Longer horizons reflect the possibility that positions take longer to close or hedge under stress.
<!-- bilingual-en:end -->

**时间扩展影响：**若假定每天的损益分布独立同分布，则持有期越长，VaR通常随时间增长，大致符合**平方根规律**：$N$天VaR约等于$1$天VaR乘以$\sqrt{N}$（因为标准差随时间$\sqrt{N}$增长）。例如，若1天99% VaR为$X$，则10天99% VaR$\approx X\sqrt{10}$。然而，**独立性假设**并不总成立：实际中损益常存在**自相关或波动聚集性**，这会导致多日组合损失分布的方差增长快于线性累加。简单地说，若日间收益存在正相关（$\rho>0$），则$N$天VaR会**大于**$1$天VaR的$\sqrt{N}$倍；反之若有负相关则增长慢于$\sqrt{N}$。 
<!-- bilingual-en:start -->
**Time scaling:** If daily profit and loss is independent and identically distributed, standard deviation grows with the square root of time, so $N$-day VaR is approximately one-day VaR multiplied by $\sqrt{N}$. Thus, if one-day 99% VaR is $X$, ten-day 99% VaR is approximately $X\sqrt{10}$. This rule fails when returns are serially correlated or volatility clusters. Positive serial correlation, $\rho>0$, makes multi-day risk exceed the simple $\sqrt{N}$ scaling; negative serial correlation has the opposite effect.
<!-- bilingual-en:end -->

>[!example] 正相关的例子
> 示例：假设某组合每日收益标准差为$\sigma=300$万美元，日间收益相关系数约$\rho=0.1$（存在轻微正相关）。则按照精确公式计算5天损益标准差:
> $$
> \sigma_{5} = \sqrt{\sigma^2 \left(5 + 2\sum_{k=1}^{4}(5-k)\rho^k\right)} \approx 726\text{万美元},
> $$ 
> 而$1$天标准差$=300$万，$\sqrt{5}\times300=670$万。可以看到因收益正相关，实际5天风险（726万）高于独立假设下的670万。**结论：**持有期越长VaR越大，但需注意收益非独立时VaR不能简单按$\sqrt{N}$放大，应考虑相关性和波动聚集影响。
><!-- bilingual-en:start -->
>Suppose daily portfolio profit and loss has a standard deviation of USD 3 million and adjacent daily returns have correlation $\rho=0.1$. The exact five-day variance formula shown above gives a standard deviation of about USD 7.26 million, whereas independence would give $\sqrt{5}\times3\approx USD 6.70$ million. Positive serial correlation therefore raises five-day risk. **Conclusion:** VaR generally grows with the holding period, but $\sqrt{N}$ scaling is justified only when the dependence and volatility assumptions support it.
><!-- bilingual-en:end -->

## 3.3 [[Observation Window|观察期]]（数据窗口）
<!-- bilingual-en:start -->
*3.3 [[Observation Window|Observation Window]] (Data Window)*
<!-- bilingual-en:end -->

**[[Observation Window|观察期]]**（[[Observation Window|Observation]] Period，又称数据窗口）指用于计算VaR时历史数据取样的时间长度。例如我们可能采用过去1年的每日收益数据来估计波动率、分位数等。观察期的选择存在权衡：**数据越长**，包含更多市场环境，统计估计更稳定，能平滑短期异常；但**窗口过长**可能引入过时信息，若市场结构发生变化，久远数据会降低准确性。
<!-- bilingual-en:start -->
The **[[Observation Window|observation window]]** is the span of historical data used to estimate volatility, correlations, or quantiles. A longer window includes more market regimes and usually stabilizes estimates, but it may give substantial weight to stale observations after the market structure changes. A shorter window is more responsive but noisier.
<!-- bilingual-en:end -->

一般而言，为平衡统计可靠性与现实适用性，**1年左右**的历史数据较常用（约250个交易日）。巴塞尔委员会规定内部模型法计算VaR至少用**一年**历史数据。此外，有些金融机构采用**加权历史数据**（赋予近期更大权重）以兼顾新旧信息。
<!-- bilingual-en:start -->
A window of roughly one year, or about 250 trading days, is common when balancing statistical reliability and current relevance. Earlier Basel internal-model VaR rules required at least one year of historical observations. Some institutions weight recent observations more heavily so that the estimate reacts faster without discarding older information entirely.
<!-- bilingual-en:end -->

观察期的影响：窗口过短，VaR易受偶发事件左右，不稳定；窗口过长，可能掩盖近期风险上升的趋势，导致VaR反应迟缓。例如，在平稳时期用10年的数据估计波动可能低估近期的剧烈波动风险。因此实际应用中，常针对不同目的选择不同长度窗口，或者采用滚动窗口并辅以压力情景补充，使VaR既有充分数据支撑又能反映当前风险水平。
<!-- bilingual-en:start -->
A window that is too short produces unstable VaR estimates dominated by individual observations. A window that is too long can conceal a recent rise in risk; for example, a ten-year volatility estimate may react slowly to a newly turbulent regime. In practice, institutions use windows suited to the purpose, often rolling them through time and supplementing them with stress scenarios.
<!-- bilingual-en:end -->

# 4. 绝对 VaR 与相对 VaR 的区别
<!-- bilingual-en:start -->
*4. Difference between absolute VaR and relative VaR*
<!-- bilingual-en:end -->

VaR可以分为**[[Absolute VaR|绝对VaR]]**和**[[Relative VaR|相对VaR]]**两种口径：
<!-- bilingual-en:start -->
VaR can be divided into **[[Absolute VaR|absolute VaR]]** and **[[Relative VaR|relative VaR]]**:
<!-- bilingual-en:end -->

- **[[Absolute VaR|绝对VaR]]**（[[Absolute VaR|Absolute VaR]]）指相对于当前持仓价值本身可能遭受的最大损失，直接以头寸价值减少的金额表示。这通常是我们默认的VaR概念，即假设初始资产价值为$W$，在置信水平下未来价值降至$W-\text{VaR}$（或损失$\text{VaR}$）。
<!-- bilingual-en:start -->
- **[[Absolute VaR|Absolute VaR]]** measures the loss relative to the portfolio's current value. If current value is $W$, the adverse quantile of future value is $W-\text{VaR}$, so the reported VaR is the corresponding loss from today's value.
<!-- bilingual-en:end -->

- **[[Relative VaR|相对VaR]]**（[[Relative VaR|Relative VaR]]）则是指相对于某基准收益水平（通常是持有期间的期望收益或无风险收益）的最大损失。换言之，先考虑资产在持有期内的**期望增长**，相对VaR表示实际收益相对期望值下降的最大幅度。若资产预期收益为正，**[[Relative VaR|相对VaR]]**将略大于绝对VaR（因为包含放弃掉的那部分正期望收益）；若预期收益为负，相对VaR反而小一些（因为基准期望本就亏损）。
<!-- bilingual-en:start -->
- **[[Relative VaR|Relative VaR]]** measures underperformance relative to a benchmark, usually expected return or a risk-free return. It therefore includes the return that the portfolio was expected to earn but did not. With positive expected return, relative VaR is slightly larger than absolute VaR; with negative expected return, it can be smaller.
<!-- bilingual-en:end -->

在实践中，**[[Relative VaR|相对VaR]]**常用于衡量相对于某基准（如相对于平均收益、或者相对于零收益）的风险。例如一些机构假定短期内期望收益接近0，则相对VaR与绝对VaR数值几乎相同。还有一种情形，**跟踪误差VaR**可视作相对VaR的一种，即衡量投资组合相对于基准指数的超额损失风险。
<!-- bilingual-en:start -->
In practice, **[[Relative VaR|relative VaR]]** is used for downside risk relative to a return target or benchmark. Tracking-error VaR is one example: it measures an adverse quantile of active return relative to an index. When short-horizon expected return is treated as zero, absolute and relative VaR are nearly identical.
<!-- bilingual-en:end -->

>[!example] 绝对VaR和相对VaR的例子
> 例子：当前投资组合价值$W=1亿美元$，预期每天收益$\mu=0.1\%$（即盈利10万美元），每日波动率$\sigma=2\%$（200万美元）。计算95%置信水平下的一天VaR：
> - [[Absolute VaR|绝对VaR]]：直接计算收益分布5%分位数。收益~$N(0.1\%,2\%)$，5%分位数约为$\mu - 1.645\sigma = 0.1\% - 1.645\times2\% \approx -3.19\%$，即日损失3.19%。[[Absolute VaR|绝对VaR]] = $3.19\% \times 1$亿 = **319万美元**（损失）。
> - [[Relative VaR|相对VaR]]：相对期望收益计算的损失幅度，即考虑期望本应盈利10万，现在不仅没赚还亏。可以用$\text{[[Relative VaR|相对VaR]]} = -(\text{5\%分位收益} - \mu)$得到：$-( -3.19\% - 0.1\%) = 3.29\%$，对应**329万美元**。这个数比绝对VaR略大，是因为包含了本应赚的那10万收益的落差。
><!-- bilingual-en:start -->
>A portfolio is currently worth USD 100 million. Its expected daily return is $\mu=0.1\%$, or USD 0.1 million, and its daily volatility is $\sigma=2\%$, or USD 2 million. At 95% confidence:
>- **[[Absolute VaR|Absolute VaR]]:** The 5th percentile of return is $\mu - 1.645\sigma = 0.1\% - 1.645\times2\% \approx -3.19\%$. The loss relative to current value is therefore about USD 3.19 million.
>- **[[Relative VaR|Relative VaR]]:** Relative to the expected gain, the adverse deviation is $1.645\sigma=3.29\%$, or USD 3.29 million. Equivalently, it is the USD 3.19 million absolute loss plus the USD 0.1 million expected gain that was forgone.
><!-- bilingual-en:end -->

通常风险管理报告中直接给出的VaR都指**绝对金额**的VaR（默认为相对当前价值）。相对VaR更多在特定分析（如相对于平均业绩的下行风险，或相对于基准的跟踪误差）中使用。在数值上，两者差异取决于资产期望收益的大小，相对于短期市场波动往往很小，所以经常忽略不计而直接使用绝对VaR。
<!-- bilingual-en:start -->
Risk reports usually use absolute monetary VaR by default. Relative VaR is more common in performance or benchmark-relative analysis. Because short-horizon expected return is generally small compared with volatility, the numerical difference is often modest, but the two measures answer different questions.
<!-- bilingual-en:end -->

# 5. VaR 与 [[ES|ES]]（预期损失）的比较及一致性问题
<!-- bilingual-en:start -->
*5. VaR versus [[ES|Expected Shortfall]] and the Question of Coherence*
<!-- bilingual-en:end -->

[[ES]]（Expected Shortfall，预期损失）也称条件VaR（CVaR）或尾部期望损失，是另一种风险度量，定义为在损失超过VaR条件下的平均损失。换言之，ES关心的是“最坏的$(1-\alpha)$的情况平均会损失多少”。举例来说，在99%置信水平下的10日ES是指未来10天内，处于最糟糕1%情形下的平均损失金额。
<!-- bilingual-en:start -->
[[ES|Expected shortfall]] (ES), also called conditional VaR or expected tail loss, measures average loss in the worst $1-\alpha$ fraction of outcomes. For example, ten-day ES at 99% confidence is the average ten-day loss among the worst 1% of modeled scenarios. For discontinuous loss distributions, this tail-average definition is preferable to conditioning mechanically on $L>\text{VaR}$ because probability mass may sit exactly at the VaR threshold.
<!-- bilingual-en:end -->

**VaR vs [[ES|ES]]:**
- VaR提供的是**分位点信息**：它告诉我们损失分布在某高置信度下的阈值，但对更坏的情况并无涉及。比如99% VaR = 6400万，说明99%时候损失不超过6400万，但万一超过了6400万，可能是6500万也可能是1亿，都不体现。
- ES提供**尾部均值信息**：它进一步回答“如果损失超出了VaR，这些极端情况下平均会亏多少钱”。因此ES考虑了损失分布尾部的厚度，比VaR更全面反映极端风险的严重程度。
<!-- bilingual-en:start -->
**VaR versus [[ES|ES]]:**
- VaR supplies a **quantile threshold**. A 99% VaR of CNY 64 million says that 99% of modeled losses are no greater than CNY 64 million, but it says nothing about how large losses are after the threshold is crossed.
- ES supplies a **tail average**. It asks how much is lost on average in the worst 1% of cases and therefore reflects the severity of the distribution beyond VaR.
<!-- bilingual-en:end -->

一致性（Coherence）问题：在风险度量理论中，[[一致性风险度量]]指满足一组合理性的公理（详见下一节）。ES被证明是**一致风险度量**，而VaR并不满足所有一致性要求（特别是次可加性）。这使得ES在理论上更受青睐，因为它不会像VaR那样可能违反分散化原则。例如，对于两个子组合，[[ES|ES]](A+B)总是≤ [[ES|ES]](A)+[[ES|ES]](B)，而VaR在某些罕见情况下可能出现组合风险>单独风险之和的反直觉结果。
<!-- bilingual-en:start -->
**Coherence:** A [[一致性风险度量|coherent risk measure]] satisfies a set of economically meaningful axioms. ES is coherent under standard conditions, whereas VaR can violate subadditivity. Thus, for two portfolios, [[ES|ES]] obeys $\mathrm{ES}(A+B)\le \mathrm{ES}(A)+\mathrm{ES}(B)$, while VaR can sometimes report more risk for the diversified combination than the sum of the separate VaRs.
<!-- bilingual-en:end -->

监管选择：由于ES在极端风险衡量和一致性方面的优势，巴塞尔新规（如FRTB框架）已从VaR转向使用**97.5% [[ES|ES]]**作为市场风险资本计量标准，部分原因正是VaR不一致而ES更合理。此外，ES难以被交易员“投机性规避”——因为它关注尾部平均，交易员若试图通过降低VaR而将风险集中到极端尾部（“赌小概率大亏”）会被ES捕捉到。例如有交易员每日将99% VaR控制在1000万，但留下0.5%概率出现5000万损失，这种策略下VaR看似合规，实际尾部风险巨大；ES则会将那0.5%的巨大损失平均进来，显示一个高得多的风险值，阻止此类规避行为。
<!-- bilingual-en:start -->
**Regulatory choice:** The FRTB internal-model approach uses **97.5% [[ES|expected shortfall]]** for market-risk capital because ES captures losses beyond a single quantile and is sensitive to tail severity. It also makes it harder to hide a very small probability of a very large loss just beyond the VaR cutoff. A strategy can keep 99% VaR low while concentrating losses in the worst 1%; ES brings those losses into the reported tail average.
<!-- bilingual-en:end -->

总结：VaR直观易懂且便于计算，但它仅给出“不超过”的损失界限；ES进一步告诉我们“超出界限时有多糟”。在关注极端事件的风险管理中，ES被视为比VaR更有信息量的指标。不过ES计算相对复杂且对极值敏感度高，需要更多数据或假设支撑，这也是实际应用中曾长期沿用VaR的原因。
<!-- bilingual-en:start -->
In summary, VaR is intuitive and easy to communicate, but it reports only a loss threshold. ES describes the average severity beyond that threshold and is therefore more informative about extreme loss. Its tail estimate is also more data- and model-intensive, which helps explain the long historical use of VaR.
<!-- bilingual-en:end -->

# 6. VaR 的一致性争议、CVaR 的优势、[[Spectral Risk Measure|光谱风险度量]]
<!-- bilingual-en:start -->
*6. VaR, Coherence, CVaR, and [[Spectral Risk Measure|Spectral Risk Measures]]*
<!-- bilingual-en:end -->

1997年，Artzner等人提出了[[一致性风险度量]]的概念，认为合理的风险度量应满足以下公理：
<!-- bilingual-en:start -->
In 1997, Artzner and co-authors formalized the concept of a [[一致性风险度量|coherent risk measure]]. A coherent measure satisfies the following axioms:
<!-- bilingual-en:end -->

1. **单调性（Monotonicity）**
2. **次可加性（Subadditivity）**
3. **正齐次性（[[Positive Definite Matrix|Positive]] Homogeneity）**
4. **平移不变性（Translation Invariance）**
<!-- bilingual-en:start -->
1. **Monotonicity**
2. **Subadditivity**
3. **[[Positive Definite Matrix|Positive]] Homogeneity**
4. **Translation Invariance**
<!-- bilingual-en:end -->

VaR 的不一致性：VaR一般满足单调性、正齐次、平移不变性，但**不保证满足次可加性**。
<!-- bilingual-en:start -->
VaR generally satisfies monotonicity, positive homogeneity, and translation invariance, but it **need not satisfy subadditivity**. This is the source of its failure to be coherent in general.
<!-- bilingual-en:end -->

CVaR/[[ES|ES]] 的优势：CVaR（[[ES|ES]]）满足上述所有一致性公理，包括次可加性，因而是一致风险度量。ES不仅规避了VaR的理论问题，还提供了尾部损失信息，因此在衡量极端风险时更可靠。
<!-- bilingual-en:start -->
CVaR, or [[ES|ES]], satisfies the four coherence axioms, including subadditivity. It also reports information about losses beyond the VaR threshold, making it more informative for extreme-risk analysis.
<!-- bilingual-en:end -->

>[!quote] [[Spectral Risk Measure|光谱风险度量]] 
> [[Spectral Risk Measure|光谱风险度量]]：为了广泛刻画风险厌恶程度，Acerbi等提出光谱风险度量（[[Spectral Risk Measure|Spectral Risk Measure]]）概念。光谱风险度量将损失分布的各个分位损失按照某种**权重函数**加权求和：
> $$
> \rho_{\phi}(X) = \int_{0}^{1} \phi(q) F^{-1}_X(q)\,\mathrm{d}q,
> $$ 
> 其中$\phi(q)$为对第$q$分位损失的权重函数。如果$\phi(q)$是非递减的（对更大损失给予不小于前面的权重），则该风险度量满足次可加性，是一致的。直观理解：光谱风险度量按照预先设定的权重关注不同置信水平的损失，权重越偏向尾部，高损失得到更大强调，该度量就越保守。
> 
> VaR和ES都属于特殊的光谱度量：
> - VaR只关注单一分位点（如99%分位损失），可视为在该点权重100%、其他损失权重0%的极端光谱。这种权重函数在尾部突然跃升，不满足单调递增的平滑条件，导致VaR不是一致风险度量。
> - ES对超出VaR阈值以上的尾部损失赋予**均等权重**，对更糟糕情形并不再区别对待，因此它对应的权重函数在VaR分位点后是常数（对所有尾部损失一视同仁），这是一个非下降函数，满足一致性要求。
> 
> 通过光谱视角我们看到，**VaR的问题在于其风险权重集中在临界分位点，无法体现更极端损失，且权重函数不满足递增性**；而ES平滑地考虑了VaR之后的所有损失，因此更符合一致性。进一步的光谱度量族，如**指数光谱风险度量**（对越极端损失赋予指数增长的权重），允许根据风险厌恶程度调整尾部权重，所有这些光谱风险度量只要权重递增，都满足一致性，比VaR更一般化。
><!-- bilingual-en:start -->
>A [[Spectral Risk Measure|spectral risk measure]] combines loss quantiles using a **weight function**:
>$\rho_{\phi}(X)=\int_0^1\phi(q)F_X^{-1}(q)\,\mathrm dq$.
>Here, $\phi(q)$ is the weight assigned to the $q$th loss quantile. If the weights are non-negative, integrate to one, and are non-decreasing toward worse outcomes, the measure is coherent. Greater weight in the far tail represents greater aversion to catastrophic loss.
>
>ES is a spectral risk measure: it assigns equal positive weight to every quantile beyond the VaR cutoff and zero weight below it. VaR can be represented informally as a point mass at one quantile, but that generalized “spectrum” is not an admissible increasing integrable weight function, which is why VaR is not a coherent spectral risk measure. Exponential and other increasing spectra allow tail weights to reflect different degrees of risk aversion.
><!-- bilingual-en:end -->

# 7. VaR 分解（[[Marginal VaR|边际VaR]]、[[Component VaR|成分VaR]]、[[Incremental VaR|递增VaR]]）及欧拉分解法
<!-- bilingual-en:start -->
*7. VaR Decomposition ([[Marginal VaR|marginal VaR]], [[Component VaR|component VaR]], [[Incremental VaR|incremental VaR]]) and Euler Decomposition*
<!-- bilingual-en:end -->

看看得了.
<!-- bilingual-en:start -->
Skim this section.
<!-- bilingual-en:end -->

## 7.1 [[Marginal VaR|边际VaR]]
<!-- bilingual-en:start -->
*7.1 [[Marginal VaR|marginal VaR]]*
<!-- bilingual-en:end -->

**[[边际VaR]]（[[Marginal VaR|Marginal VaR]]）**：边际VaR定义为组合VaR对某资产头寸的变化率，直观上是**组合VaR对单个资产头寸的偏导数**。边际VaR表示在当前组合中，若第$i$项资产持仓增加一微小单位，组合VaR增加多少。公式上，资产$i$的边际VaR = $\partial \text{VaR}_{\text{组合}} / \partial w_i$（$w_i$为资产权重或金额）。边际VaR反映了每增加一元资产$i$所带来的风险增量。
<!-- bilingual-en:start -->
**[[边际VaR|Marginal VaR]] ([[Marginal VaR|Marginal VaR]]):** Marginal VaR is the derivative of portfolio VaR with respect to a position. It answers how much total VaR changes when position $i$ is increased by an infinitesimal amount:
$\partial \text{VaR}_{\text{portfolio}}/\partial w_i$,
where $w_i$ is an asset weight or monetary position. It is a local sensitivity, not the VaR of asset $i$ in isolation.
<!-- bilingual-en:end -->

  边际VaR与资产在组合中的**系统性风险贡献**有关。对于高度分散的组合，在正态参数法框架下，可以证明资产$i$的边际VaR与其在组合中的Beta系数成正比——Beta越高（与组合高度正相关，波动贡献大），边际VaR越大；反之，如果某资产与组合低相关甚至负相关，增加它反而可能降低组合风险，此时边际VaR可以为负值（意味着增持该资产会降低总VaR）。因此，边际VaR提供了调整组合的指引：增加边际VaR小甚至为负的资产有助于降低整体风险。
<!-- bilingual-en:start -->
Marginal VaR reflects how an asset co-moves with the rest of the portfolio. Under a linear normal model, it is related to the asset's covariance, or beta, with the portfolio. A highly positively correlated asset tends to have high marginal VaR; an effective hedge can have negative marginal VaR, meaning that a small increase in the hedge reduces total portfolio VaR.
<!-- bilingual-en:end -->

## 7.2 [[Incremental VaR|递增VaR]]
<!-- bilingual-en:start -->
*7.2 [[Incremental VaR|incremental VaR]]*
<!-- bilingual-en:end -->

 **[[递增VaR]]（[[Incremental VaR|Incremental VaR]]）**：递增VaR指**新增或剔除一笔交易对组合VaR的影响**，即比较“有该交易”和“无该交易”两种组合VaR之差。例如，计算将资产$j$从组合中去掉后VaR的变化量，或者新增一个头寸后VaR的增量。递增VaR实际上是有限幅度（非无限小）的VaR变化评估，适用于评估一项具体投资决策对整体风险的影响。对于相对小的新增头寸，递增VaR与边际VaR近似相等；对于较大调整，需重新计算组合VaR来获得精确增量。
<!-- bilingual-en:start -->
**[[递增VaR|Incremental VaR]] ([[Incremental VaR|Incremental VaR]]):** Incremental VaR is the finite change in portfolio VaR caused by adding, removing, or resizing a trade. It compares VaR with and without the change. For a very small adjustment it is approximated by marginal VaR times the position change; for a large adjustment, the portfolio must be fully re-evaluated.
<!-- bilingual-en:end -->

## 7.3 [[Component VaR|成分VaR]]
<!-- bilingual-en:start -->
*7.3 [[Component VaR|component VaR]]*
<!-- bilingual-en:end -->

 **[[成分VaR]]（[[Component VaR|Component VaR]]）**：成分VaR又称风险贡献度，是将组合总VaR划分到各组成资产的一种分摊，使各部分成分VaR之和等于整体VaR。成分VaR旨在回答：“组合总VaR中有多少是由资产$i$贡献的？” 根据风险分配理论，若风险度量满足正齐次性，可应用**欧拉分解法（Euler’s Theorem）**：对于组合风险$V$，如果对任意$\lambda>0$有$V(\lambda \mathbf{x}) = \lambda V(\mathbf{x})$（线性齐次），则
  $$
  V(\mathbf{x}) = \sum_{i=1}^{N} x_i \frac{\partial V}{\partial x_i}(\mathbf{x}),
  $$
  其中$x_i \frac{\partial V}{\partial x_i}$可以解释为第$i$项的风险成分。套用于VaR，若将组合各资产头寸$w_i$均放大$\lambda$倍，VaR也放大$\lambda$倍（VaR的一阶齐次性在正态等模型下成立），则有：
  $$
  \text{VaR}_{\text{组合}} = \sum_{i} w_i \frac{\partial \text{VaR}}{\partial w_i}.
  $$
  右侧每一项正是资产$i$的持仓规模乘以其边际VaR，定义为资产$i$的**[[Component VaR|成分VaR]]**。这一定义下，各资产成分VaR之和正好等于组合VaR，实现了VaR的可加分解。
<!-- bilingual-en:start -->
**[[成分VaR|Component VaR]] ([[Component VaR|Component VaR]]):** Component VaR allocates total portfolio VaR across positions. For a positively homogeneous differentiable risk measure $V$, Euler's theorem gives the decomposition shown above. Applied to VaR, position $i$ contributes $w_i\,\partial\text{VaR}/\partial w_i$. The component contributions sum exactly to portfolio VaR when the required homogeneity and differentiability conditions hold.
<!-- bilingual-en:end -->

  性质：成分VaR具有如下特性：(1) 对大型组合，成分VaR与单一资产被视作“小幅变化”时的递增VaR非常接近，即成分VaR近似等于将该资产从组合中移除一点点的VaR差异；(2) 将所有资产的成分VaR相加，正好得到组合总VaR（由欧拉定理保证）。
<!-- bilingual-en:start -->
Component VaR has two useful interpretations. First, for a large portfolio and a small proportional reduction in one position, its risk contribution is closely related to incremental VaR. Second, summing the component VaRs across all positions reproduces total portfolio VaR by Euler's theorem.
<!-- bilingual-en:end -->

# 8. VaR 的聚合方法
<!-- bilingual-en:start -->
*8. Aggregating VaR across Business Units*
<!-- bilingual-en:end -->

大型金融机构往往从不同业务单元获得各自的VaR估计，希望将它们合并为整个机构的总VaR。这需要考虑各部分之间的相关性。VaR的聚合并非简单相加，而应考虑**损失的相关性**带来的分散效应。若用$VaR_i$表示第$i$单元在相同置信水平和持有期下的VaR，$\rho_{ij}$表示单元$i$与$j$损失的相关系数，则**总VaR**可用近似公式：
$$
VaR_{\text{total}} = \sqrt{\sum_{i}\sum_{j} \rho_{ij}\,VaR_i\,VaR_j}\,,
$$
对于两个业务的特例，上式化简为：
$$
VaR_{总} = \sqrt{VaR_1^2 + VaR_2^2 + 2\,\rho_{12}\,VaR_1\,VaR_2}\,.
$$
<!-- bilingual-en:start -->
A financial institution may receive VaR estimates from several business units and need an institution-wide number. Simple addition ignores diversification. If $VaR_i$ is unit $i$'s VaR at a common confidence level and holding period, and $\rho_{ij}$ is the correlation between the units' losses, total VaR can be approximated by the displayed covariance-style formula. For two units, it reduces to the second formula shown above.
<!-- bilingual-en:end -->

这一公式在各单元损失近似正态且均值为0的情形下严格成立，在一般情况下也被认为是合理的近似。它体现了相关系数对组合风险的影响： 
- 如果$\rho_{ij}=1$（完全正相关），则$VaR_{总} = \sum_i VaR_i$，组合VaR等于各部分VaR之和（没有分散效应）。 
- 如果相关性小于1，$VaR_{总}$将小于直接相加，相关性越低，组合分散化效益越明显。极端地，若所有单元独立（$\rho=0$），则$VaR_{总} = \sqrt{\sum VaR_i^2}$，明显小于简单求和。
- 若存在负相关（极少见，比如对冲头寸），则总VaR可能更低。
<!-- bilingual-en:start -->
The formula is exact for jointly normal, zero-mean losses when each unit's VaR is proportional to its standard deviation. It is only an approximation more generally.
- If $\rho_{ij}=1$, then $VaR_{\text{total}}=\sum_iVaR_i$: there is no diversification benefit.
- If correlations are below one, total VaR is below the simple sum. If units are independent, $VaR_{\text{total}}=\sqrt{\sum_iVaR_i^2}$.
- Negative dependence, as with an effective hedge, can reduce total VaR further.
<!-- bilingual-en:end -->

>[!example] 示例
> 假设业务A和B的10日99% VaR分别为6000万和1亿，二者损失相关系数$\rho=0.4$。则合并后的集团整体VaR：
> $$
> VaR_{合} = \sqrt{(60)^2 + (100)^2 + 2\times0.4\times60\times100}\ (\text{百万元}) = \sqrt{3600 + 10000 + 4800} = \sqrt{18400} \approx 135.6\text{百万元},
> $$
> 约为1.356亿元。可见小于直接相加的1.6亿，体现了相关性小于1带来的风险抵消效应。
><!-- bilingual-en:start -->
>Suppose businesses A and B have ten-day 99% VaRs of CNY 60 million and CNY 100 million, and their losses have correlation $\rho=0.4$. The displayed aggregation formula gives approximately CNY 135.6 million. This is below the undiversified sum of CNY 160 million because imperfect correlation creates a diversification benefit.
><!-- bilingual-en:end -->

在实际聚合VaR时，需要注意不同业务VaR可能基于不同假设或数据口径，直接应用上述公式前要确保VaR计算的置信度、持有期一致，并对不同市场风险类别（如利率 vs 信用）可能的非正态厚尾作出调整。有时机构也会采用更加保守的方法（如假定更高相关性）来聚合，以避免低估总体风险。
<!-- bilingual-en:start -->
Before aggregating VaR, ensure that business-unit estimates use the same confidence level, holding period, valuation date, loss convention, and compatible data. The covariance formula may be unreliable across risks with heavy tails or state-dependent dependence, such as market and credit risk. Institutions sometimes impose stressed or conservative correlations to reduce the chance of overstating diversification.
<!-- bilingual-en:end -->

# 9. VaR 模型的检验方法（[[Backtesting|回溯检验]]、[[Kupiec Test|Kupiec检验]]、聚束效应等）
<!-- bilingual-en:start -->
*9. Validating VaR Models: [[Backtesting|Backtesting]], the [[Kupiec Test|Kupiec Test]], and Exception Clustering*
<!-- bilingual-en:end -->

VaR模型需要通过**[[Backtesting|回溯检验]]（[[Backtesting|Backtesting]]）**来评估其准确性。回溯检验是将模型预测的VaR与实际损益数据对比，统计实际损失超过VaR的次数（称为“例外”或“突破”，exception）的频率，以及这些异常是否随机分布。
<!-- bilingual-en:start -->
A VaR model is assessed through **[[Backtesting|backtesting]]**: compare each forecast VaR with the subsequently realized profit and loss, record every day on which loss exceeds VaR, and examine both the number and timing of these “exceptions.”
<!-- bilingual-en:end -->

## 9.1 违反频率检验（Kupiec比例检验）
<!-- bilingual-en:start -->
*9.1 Unconditional Coverage: the Kupiec Proportion-of-Failures Test*
<!-- bilingual-en:end -->

**[[Kupiec Test|Kupiec检验]]**是一种检验例外率是否与标称概率一致的方法，又称“比例违约检验”（Proportion of Failures, POF）。假设模型计算的一日VaR置信水平为$\alpha$（例如99%，则$p=1-\alpha=0.01$为单日超VaR的理论概率）。在$n$天中，我们期望例外次数$X$服从$\text{Binomial}(n, p)$的二项分布，期望值为$np$。
<!-- bilingual-en:start -->
The **[[Kupiec Test|Kupiec test]]**, or proportion-of-failures (POF) test, checks whether the observed exception rate is consistent with the nominal rate. For one-day VaR at confidence level $\alpha$, the model-implied exception probability is $p=1-\alpha$. Over $n$ independent days, the exception count $X$ follows $\text{Binomial}(n,p)$ under the null, with expected value $np$.
<!-- bilingual-en:end -->

检验的原假设$H_0$：实际例外发生概率等于$p$，备择假设$H_1$：实际例外概率高于$p$（模型低估风险，异常过多）。统计量可以采用似然比形式：
$$
LR_{\text{POF}} = -2 \ln\left[\frac{(1-p)^{(n-X)}p^X}{(1-\hat{p})^{(n-X)}\hat{p}^X}\right],
$$
其中$\hat{p} = X/n$是观测例外率。该统计量近似服从$\chi^2(1)$（自由度1的卡方分布）。简化来说，也可以直接利用二项分布计算出现$X$个及以上例外的概率$p_{\text{obs}} = P(X_{\text{bin}} \ge X)$作为$p$值。
<!-- bilingual-en:start -->
The null hypothesis $H_0$ is that the exception probability equals $p$. A one-sided alternative asks whether it is greater than $p$, meaning that the model understates risk. The displayed likelihood-ratio statistic compares the null likelihood with the likelihood at the observed rate $\hat p=X/n$ and is asymptotically $\chi^2(1)$ for the standard two-sided POF test. For a one-sided exact test, use the binomial upper-tail probability $P(X_{\text{bin}}\ge X)$.
<!-- bilingual-en:end -->

- 若$p$值很小（如<0.05），说明实际例外次数远高于模型预期，上述统计量显著，拒绝$H_0$，认为VaR模型低估了风险（不准确）。
- 若实际例外次数远低于预期（例如理论应有5次而实际0次），也可构造左尾检验（检验$H_1$: 实际例外概率 < $p$），表明模型可能过于保守。
<!-- bilingual-en:start -->
- A small upper-tail $p$-value, such as less than 0.05, means that too many exceptions occurred under the model; reject $H_0$ and investigate risk understatement.
- A separate lower-tail test can identify an unusually small number of exceptions, which may indicate an excessively conservative model.
<!-- bilingual-en:end -->

**Kupiec双尾检验**同时考虑例外过多或过少，两侧偏离均显著时拒绝模型。实务中更关注例外过多的情况，因为那意味着风险被低估。
<!-- bilingual-en:start -->
A two-sided coverage test rejects for either too many or too few exceptions. Risk management often pays particular attention to the upper tail because too many exceptions imply that VaR is too low.
<!-- bilingual-en:end -->

>[!example] 示例
> 某模型声称99%置信度VaR（日频），在600个交易日回溯中实际出现了$m=9$天损失超过VaR。理论期望例外$np = 600\times0.01 = 6$天。利用二项分布计算$P(X\ge 9)$，或使用Kupiec统计量，都可得出$p\approx0.15$（15%）。由于$p$值 > 0.05，无法拒绝模型，认为9次异常尚在随机波动范围内。反之，如果600天内异常发生了15次（明显多于6次），计算将得到极小的$p$，应拒绝模型有效性。
><!-- bilingual-en:start -->
>A daily 99% VaR model produces $m=9$ exceptions over 600 trading days. The expected count is $np=600\times0.01=6$. The exact binomial upper-tail probability is $P(X\ge9)\approx0.152$, so a one-sided 5% test does not reject the model. Nine exceptions are still plausible sampling variation. Fifteen exceptions would yield a much smaller upper-tail probability and much stronger evidence of risk understatement.
><!-- bilingual-en:end -->

监管部门也给出了方便的“信号灯”标准：例如99%VaR在250个样本日中，例外0~4次为“绿色”（模型合理），5~9次为“黄色”（轻度超标，需要关注），≥10次为“红色”（模型显著低估风险，需要整改）。这些门槛值本质上对应了一定的统计置信区间（比如95%或99%）。
<!-- bilingual-en:start -->
The Basel backtesting traffic-light framework for 250 observations of one-day 99% VaR classifies 0–4 exceptions as green, 5–9 as amber, and 10 or more as red. These zones prescribe increasing supervisory consequences; they are not simply informal labels for “accurate” and “inaccurate” models.
<!-- bilingual-en:end -->

## 9.2 序列独立性检验（[[Clustering Test|聚束效应检验]]）
<!-- bilingual-en:start -->
*9.2 Exception Independence and the [[Clustering Test|Clustering Test]]*
<!-- bilingual-en:end -->

除了例外频率正确，**独立性**也是重要假设：理想模型下，超VaR事件在时间上不应有系统性规律，即昨天发生异常不应提高今天异常的概率 —— 换言之，异常应独立分布。如果异常现象**聚束在某些时期**，则说明风险模型未能捕捉时变的波动性或相关性。
<!-- bilingual-en:start -->
Correct unconditional frequency is not enough. Under a well-calibrated dynamic model, VaR exceptions should not be predictably clustered through time: an exception yesterday should not systematically raise the probability of another today. Clustering suggests that the model is slow to capture changing volatility or dependence.
<!-- bilingual-en:end -->

[[Bunching|聚束效应]]（[[Bunching|Bunching]]）指异常发生往往成堆出现的现象，例如在市场剧烈波动的一段时间内连续多日VaR被突破，然后平静时期很久无异常。这表明模型没有及时反映波动率的跃升（例如GARCH效应未建模)。
<!-- bilingual-en:start -->
[[Bunching|Bunching]], or exception clustering, occurs when breaches arrive in groups—for example, several consecutive exceptions during a volatile market followed by a long quiet interval. This pattern often indicates that the VaR forecast failed to adjust quickly enough to a volatility regime change.
<!-- bilingual-en:end -->

**[[Christoffersen Test|Christoffersen检验]]**用于检测异常序列的独立性。其构造一个2×2转移矩阵，统计：
- $N_{00}$：今天不异常、明天不异常的次数
- $N_{01}$：今天不异常、明天异常的次数
- $N_{10}$：今天异常、明天不异常的次数
- $N_{11}$：今天异常、明天异常的次数
<!-- bilingual-en:start -->
The **[[Christoffersen Test|Christoffersen test]]** examines independence through a $2\times2$ transition table:
- $N_{00}$: no exception followed by no exception.
- $N_{01}$: no exception followed by an exception.
- $N_{10}$: an exception followed by no exception.
- $N_{11}$: an exception followed by an exception.
<!-- bilingual-en:end -->

在独立性假设下，明日异常概率应不依赖今日状态，所以$P(\text{异常}|前日异常) = P(\text{异常}|前日正常) = p$。检验可采用似然比：
$$
LR_{\text{ind}} = -2 \ln\left[\frac{(1-p)^{N_{00}}p^{N_{01}}(1-p)^{N_{10}}p^{N_{11}}}{(1-\hat{p}_0)^{N_{00}} \hat{p}_0^{N_{01}}(1-\hat{p}_1)^{N_{10}} \hat{p}_1^{N_{11}}}\right],
$$
其中$\hat{p}_0 = \frac{N_{01}}{N_{00}+N_{01}}$，$\hat{p}_1 = \frac{N_{11}}{N_{10}+N_{11}}$分别是前日正常和异常条件下观察到的次日异常率。该统计量服从$\chi^2(1)$。如果$LR_{\text{ind}}$显著，则拒绝独立性假设，认为存在异常聚集。
<!-- bilingual-en:start -->
Under independence, tomorrow's exception probability does not depend on today's state:
$P(\text{exception}\mid\text{previous exception})=P(\text{exception}\mid\text{previous non-exception})=p$.
The likelihood-ratio statistic shown above compares this restriction with separate transition probabilities $\hat p_0=N_{01}/(N_{00}+N_{01})$ and $\hat p_1=N_{11}/(N_{10}+N_{11})$. It is asymptotically $\chi^2(1)$. A significant result rejects independence and indicates exception clustering.
<!-- bilingual-en:end -->

含义：异常聚束往往意味着市场波动性有时段性上升，而VaR模型可能假设波动率恒定未能及时调整。例如，若发现$N_{11}$明显偏多（一旦发生异常，次日也异常的情况频繁），说明模型低估了危机期间风险的持续性。改进措施包括引入波动率动态模型（如GARCH）或针对集群风险设置情景VaR。
<!-- bilingual-en:start -->
A large $N_{11}$ means exceptions tend to persist once they begin, suggesting that the model understates risk during volatile regimes. Possible remedies include a dynamic volatility model such as GARCH, faster-moving inputs, and complementary scenario or stress analysis.
<!-- bilingual-en:end -->

## 9.3 综合回溯评价
<!-- bilingual-en:start -->
*9.3 Overall Backtesting Assessment*
<!-- bilingual-en:end -->

完整的VaR回溯检验通常同时考察**异常比例**和**异常独立性**（称为条件覆盖率检验，[[Conditional Heteroskedasticity|Conditional]] Coverage）。只有当异常次数与理论匹配且无聚集特征时，模型才能算通过检验。若频率不符，应调整VaR模型风险因子的估计（提高波动率预测或厚尾性）；若频率尚可但聚集显著，则可能需要引入时变风险因子模型，或分析风险在不同市场状态下的行为。
<!-- bilingual-en:start -->
A complete backtest examines both the **exception rate** and **exception independence**. Their joint assessment is called a conditional-coverage test. A model passes only when the frequency is compatible with the nominal rate and the sequence shows no material clustering. Frequency failure calls for recalibration of the loss distribution; clustering calls for more responsive time-varying dynamics.
<!-- bilingual-en:end -->

回溯检验是监管要求的重要部分，每日VaR模型需要不断以最新数据验证。在实际操作中，若模型未通过回溯测试，可能需要增加附加资本、调整模型参数甚至更换模型方法，以确保VaR可信可靠。
<!-- bilingual-en:start -->
Backtesting is an ongoing regulatory and internal-control process. A failed model may trigger investigation, model redevelopment, parameter changes, use restrictions, or additional capital, depending on the applicable framework.
<!-- bilingual-en:end -->

最后要指出，VaR模型只能捕捉一定置信度内的损失范围，对于超出VaR的极端风险及其他类型风险（流动性风险、模型风险等）并不能完全覆盖。因此，回溯检验通过也不意味着万无一失，风险管理仍需结合压力测试和情景分析来补充VaR的不足。
<!-- bilingual-en:start -->
Passing a VaR backtest does not mean that all risks are covered. VaR says little about loss severity beyond its quantile and may omit liquidity, basis, or model risk. Stress testing and scenario analysis therefore remain essential complements.
<!-- bilingual-en:end -->

# 模拟计算题与解答
<!-- bilingual-en:start -->
*Practice Calculations and Worked Solutions*
<!-- bilingual-en:end -->

以下提供若干VaR相关的计算例题，以及详细的解答步骤，帮助理解上述概念的应用。
<!-- bilingual-en:start -->
A number of VaR-related calculation examples are provided below, along with detailed solution steps to help understand the application of the above concepts.
<!-- bilingual-en:end -->

**例题 1：正态参数法计算单日 VaR**  
某投资组合市值为1亿元人民币，其每日收益近似服从正态分布，期望为0，日标准差为2%。假设正态分布成立，求95%置信水平下该组合1天的VaR是多少？  
<!-- bilingual-en:start -->
**Example 1: One-Day VaR under the Normal Parametric Method**

A portfolio is worth CNY 100 million. Daily return is approximately normal with mean zero and standard deviation 2%. Calculate one-day VaR at 95% confidence.
<!-- bilingual-en:end -->

**解答：**  
- 第一步，明确参数：$W=1$亿元，$\mu=0$，$\sigma=2\%$。95%置信水平对应正态分布的临界值$z_{0.95}=1.645$。  
- 由于$\mu=0$，VaR可直接近似为$z_{0.95}\sigma$（以收益的绝对下降幅度表示）。计算：$1.645 \times 2\% = 3.29\%$。  
- 将比例转化为金额：$3.29\% \times 1$亿 = **329万元**。这表示在95%置信度下，该组合1日最大可能损失为329万元人民币。  
- 换言之，100天中大约有5天的损失会超过329万元（符合5%的尾部概率）。
<!-- bilingual-en:start -->
**Solution:**
- Parameters: $W=$ CNY 100 million, $\mu=0$, and $\sigma=2\%$. The 95% standard normal quantile is $z_{0.95}=1.645$.
- With zero mean, the VaR return is $1.645\times2\%=3.29\%$.
- In money terms, $3.29\%\times$ CNY 100 million = **CNY 3.29 million**.
- Over repeated one-day observations, roughly 5% of losses would be expected to exceed this threshold if the model is correct.
<!-- bilingual-en:end -->

**例题 2：离散分布下的 VaR（历史模拟/情景法）**  
某一年期投资项目有三种可能结果：以98%的概率获得盈利200万美元；1.5%的概率损失400万美元；0.5%的概率损失1000万美元。试根据这一离散分布计算该项目1年期的99% VaR。  
<!-- bilingual-en:start -->
**Example 2: VaR for a Discrete Loss Distribution**

A one-year project earns USD 2 million with 98% probability, loses USD 4 million with 1.5% probability, and loses USD 10 million with 0.5% probability. Calculate its one-year 99% VaR.
<!-- bilingual-en:end -->

**解答：**  
- 先列出所有可能结果及其概率，并找出损失分布的分位：  
  - 盈利200万记作损失 -200万（即负的损失）概率98%  
  - 损失400万（记作+400万）概率1.5%  
  - 损失1000万（+1000万）概率0.5%  
- 按损失大小从小到大排序及累计概率：  
  - 最好情形：-200万（盈利） —— 累计概率98%  
  - 中间情形：+400万（亏损400万） —— 累计概率98% + 1.5% = 99.5%  
  - 最差情形：+1000万（亏损1000万） —— 累计概率99.5% + 0.5% = 100%  
- 我们寻找99%置信水平下的VaR，即找损失不超过该值的概率为99%。从累计概率看：98%时损失为-200万，到了99.5%时损失跳到+400万。因此**99%分位损失介于-200万和+400万之间**。严格来说，99%分位应取400万元损失（因为在99%这个位置上，损失已经可能达到400万，超过它的概率≤1%）。  
- 因此，该项目99% VaR = **400万美元**。这表示一年内有99%的把握损失不超过400万，只有极少数情况下（1%概率内）会损失更大（最大可能1000万）。  
<!-- bilingual-en:start -->
**Solution:**
- Record profit as negative loss. The outcomes are: a USD 2 million gain, recorded as a loss of $-2$ million, with probability 98%; a USD 4 million loss with probability 1.5%; and a USD 10 million loss with probability 0.5%.
- In ascending loss order, cumulative probability is 98% at $-2$ million, 99.5% at USD 4 million, and 100% at USD 10 million.
- Using the left-quantile definition $\inf\{l:P(L\le l)\ge0.99\}$, the 99% VaR is therefore **USD 4 million**.
<!-- bilingual-en:end -->

*(注：此例中采用保守做法将99% VaR取为400万，即使99%落在离散分布的间隙上，也通常选择下一个可能损失水平。实际中可根据具体分布插值估算。)*
<!-- bilingual-en:start -->
*Note: For this discrete distribution, the 99th percentile is USD 4 million under the standard left-quantile convention. Interpolation would define a different statistic and should not be introduced silently.*
<!-- bilingual-en:end -->

**例题 3：持有期与置信度对 VaR 的影响**  
某交易组合已知：每日95% VaR为150万元，且假设每日损益服从正态分布且均值为0。求：  
1. 该组合每日99% VaR；  
2. 10天持有期的99% VaR；  
3. 250天（约一年）持有期的99% VaR。  
<!-- bilingual-en:start -->
**Example 3: Effects of Confidence Level and Holding Period**

A portfolio's one-day 95% VaR is CNY 1.5 million. Daily profit and loss is normal with mean zero. Find:
1. one-day 99% VaR;
2. ten-day 99% VaR;
3. 250-day 99% VaR.
<!-- bilingual-en:end -->

**解答：**  
- (1) **计算1日99% VaR：** 已知1日95% VaR = 150万。在正态假设且$\mu=0$下，$95\%$VaR = $1.645\sigma = 150$万，可求得$\sigma = 150/1.645 \approx 91.16$万元。99% VaR对应$z_{0.99}=2.33$，因此1日99% VaR = $2.33 \times \sigma \approx 2.33 \times 91.16 = 212.4$万元，约**212万元**（四舍五入）。  
- (2) **计算10日99% VaR：** 在假定每天损益独立同分布的情况下，10日标准差$\approx \sigma\sqrt{10} \approx 91.16 \times 3.162 = 288.5$万元。沿用$z_{0.99}=2.33$（持有期只影响尺度不影响置信度），10日99% VaR = $2.33 \times 288.5 \approx 672$万元，约合**672万元**。  
- (3) **计算250日99% VaR：** 250天约为一年交易日数。同理标准差$\approx 91.16 \times \sqrt{250} \approx 91.16 \times 15.811 = 1440$万元左右。乘以2.33得VaR $\approx 2.33 \times 1440 = 3355$万元，约**3355万元**。  
<!-- bilingual-en:start -->
**Solution:**
- **One-day 99% VaR:** Since $1.645\sigma=$ CNY 1.5 million, $\sigma=1.5/1.645\approx$ CNY 0.9116 million. Thus $2.33\sigma\approx$ **CNY 2.124 million**.
- **Ten-day 99% VaR:** Under independent, identically distributed daily P&L, multiply by $\sqrt{10}$: $2.124\sqrt{10}\approx$ **CNY 6.72 million**.
- **250-day 99% VaR:** Multiply by $\sqrt{250}$: $2.124\sqrt{250}\approx$ **CNY 33.55 million**.
<!-- bilingual-en:end -->

上述结果表明，在独立正态假设下，持有期从1天延长到10天，VaR增加约4.5倍；延长到250天，VaR增至最初的20多倍，反映时间跨度对风险累计效应。另外，将置信水平从95%提高到99%，VaR也增加显著（从150万增至212万，约提升41%）。这定量展示了置信度和持有期对VaR的影响。
<!-- bilingual-en:start -->
The 99% VaR rises with the square root of the holding period under the independent normal model. Relative to the original one-day **95%** VaR of CNY 1.5 million, ten-day 99% VaR is about 4.5 times as large and 250-day 99% VaR is more than twenty times as large. This comparison combines both a higher confidence level and a longer horizon.
<!-- bilingual-en:end -->

**例题 4：VaR 模型回溯检验（[[Kupiec Test|Kupiec]] 检验）**  
某投资银行采用99%置信水平的VaR模型来监控每日交易风险。在过去250个交易日中，有8天的实际损失超过了VaR预测值。问：该模型在95%置信水平的Kupiec检验下是否通过？（假设检验原假设$H_0$：模型准确，即日超额损失概率为1%）  
<!-- bilingual-en:start -->
**Example 4: VaR [[Kupiec Test|Kupiec Backtest]]**

A bank uses one-day 99% VaR and observes eight exceptions over 250 trading days. Does the model pass a 5% test of the null that the daily exception probability is 1%?
<!-- bilingual-en:end -->

**解答：**  
- 在99%VaR模型下，理论上**超VaR例外率**$p=0.01$。样本天数$n=250$，则**期望例外次数**$np=2.5$天。实际观察到$X=8$天超过VaR。直觉上8天明显高于2.5天，模型可能低估了风险。  
- 定量检验：例外次数$X$服从$\text{Bin}(250,0.01)$。计算出现$\ge 8$次例外的概率$p_{\text{obs}} = P(X\ge 8)$。可使用二项累积或Poisson近似：$\lambda=np=2.5$，$P(X\ge8) \approx 1 - \sum_{k=0}^{7} e^{-\lambda}\frac{\lambda^k}{k!}$。经过计算，$P(X\ge8) \approx 0.0040$（约0.4%）。  
- 因此$p$值约0.4%，远小于显著性水平5%。拒绝原假设$H_0$。即**检验结论：模型未通过Kupiec 95%检验**，实际异常频率显著高于1%，VaR模型低估了风险。  
- 实际意义：250天8次异常，相当于异常率3.2%，明显高于标称1%。风险管理应审视VaR模型，可能需要提高波动估计或使用更保守分布，以使异常率降至合理范围。
<!-- bilingual-en:start -->
**Solution:**
- Under the model, $p=0.01$, $n=250$, and the expected exception count is $np=2.5$. The observed count is $X=8$.
- The exact one-sided binomial probability is $P(X\ge8)\approx0.0040$.
- Because $0.0040<0.05$, reject $H_0$: eight exceptions are too many for a correctly calibrated 99% VaR model.
- The observed exception rate is 3.2%, so the model and its volatility, tail, and data assumptions should be investigated.
<!-- bilingual-en:end -->

*(注：Kupiec检验的卡方近似亦可验证此结论。$LR=-2\ln[(0.99)^{242}(0.01)^8/(0.968)^{242}(0.032)^8]\approx7.73$，临界值$\chi^2_{0.95}(1)=3.84$，显著超出，拒绝模型。）*
<!-- bilingual-en:start -->
*The Kupiec likelihood-ratio approximation reaches the same conclusion: $LR=-2\ln[(0.99)^{242}(0.01)^8/(0.968)^{242}(0.032)^8]\approx7.73$, which exceeds the 5% $\chi^2_1$ critical value of 3.84.*
<!-- bilingual-en:end -->

**例题 5：绝对 VaR 与 相对 VaR 计算比较**  
投资组合当前价值1000万元，未来一周预期收益5万元（0.5%），预期收益标准差为50万元。假设一周投资回报近似正态分布。问：99%置信水平下，该组合一周的绝对VaR和相对VaR分别是多少？二者有何区别？  
<!-- bilingual-en:start -->
**Example 5: Absolute versus Relative VaR**

A portfolio is worth CNY 10 million. Its expected one-week profit is CNY 50,000 and the standard deviation of weekly profit is CNY 500,000. Assuming normality, calculate one-week absolute and relative VaR at 99% confidence.
<!-- bilingual-en:end -->

**解答：**  
- 首先计算正态分布99%分位数：$z_{0.99}=2.33$。一周收益标准差$\sigma=50$万元，期望$\mu=5$万元。  
- **[[Absolute VaR|绝对VaR]]：**直接计算收益分布1%分位点的损失额。收益的1%分位$= \mu - 2.33\sigma = 5 - 2.33\times50 = 5 - 116.5 = -111.5$万元。负号表示亏损111.5万。因此99%VaR（绝对）为**111.5万元**，意味着有99%把握损失不超过111.5万（投资价值最多降至888.5万）。  
- **[[Relative VaR|相对VaR]]：**相对VaR强调相对于期望收益的偏离，即= 期望收益 - 分位收益。这里= $5 - (-111.5) = 116.5$万元。也可理解为绝对VaR再加上期望收益5万，因为相对于原本要赚5万，现在不但没赚还亏了111.5万，相对预期少了116.5万。得到**相对VaR约116.5万元**。  
- **区别：**本例中相对VaR = 116.5万稍大于绝对VaR = 111.5万，原因是组合原本有正期望收益5万，绝对VaR计算损失相对于初始1000万，而相对VaR把没赚到的钱也视为损失的一部分。如果预期收益为0，则两者相等；如果预期收益为负（预期亏损），相对VaR会小于绝对VaR。  
<!-- bilingual-en:start -->
**Solution:**
- The 99% normal quantile is $z_{0.99}=2.33$, with $\sigma=$ CNY 500,000 and $\mu=$ CNY 50,000.
- **[[Absolute VaR|Absolute VaR]]:** The 1st percentile of profit is $\mu-2.33\sigma=50{,}000-1{,}165{,}000=-1{,}115{,}000$. Absolute VaR is therefore **CNY 1.115 million**.
- **[[Relative VaR|Relative VaR]]:** Relative to expected profit, the adverse deviation is $2.33\sigma=$ **CNY 1.165 million**.
- Relative VaR exceeds absolute VaR by the expected profit of CNY 50,000. If expected profit were zero, the two would coincide.
<!-- bilingual-en:end -->

通常在报表中使用绝对VaR，即直接报告可能损失多少本金。而相对VaR更多用于绩效评估等场景。在绝大多数风险管理场合，短期$\mu$相对$\sigma$很小，可以忽略不计，因此绝对VaR与相对VaR数值差别不大。这个例子主要是强调概念差异。
<!-- bilingual-en:start -->
Absolute VaR is the usual reporting convention because it states the loss of principal relative to today's value. Relative VaR is useful for performance and benchmark analysis. At short horizons, $\mu$ is often small relative to $\sigma$, so the numerical difference is modest even though the interpretation differs.
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->

## 12.5

>[!question]
> 假设某两项投资中的任何一项都有0.9%的可能引发1000万美元损失，而有99.1%的可能引发100万美元损失，这两项投资相互独立
> (a)对应于在99%的置信水平，任意一项投资的VaR是多少
> (b)选定99%的置信水平，任意一项投资的ES是多少?
> (c)将两项投资迭加在一起所产生的投资组合对应于99%置信水平的VaR是多少?
> (d)将两项投资迭加在一起所产生的投资组合在99%置信平的ES是多少?
> (e)请说明此例中的VaR不满足次可加性条件，但是ES满足次可加性条件?
> <!-- bilingual-en:start -->
> Each of two independent investments loses USD 10 million with probability 0.9% and loses USD 1 million with probability 99.1%.
> (a) Find the 99% VaR of either investment.
> (b) Find the 99% ES of either investment.
> (c) Find the 99% VaR of the combined portfolio.
> (d) Find the 99% ES of the combined portfolio.
> (e) Show that VaR violates subadditivity in this example whereas ES satisfies it.
> <!-- bilingual-en:end -->
**题设回顾及符号约定**
<!-- bilingual-en:start -->
**Set-up and notation**
<!-- bilingual-en:end -->

### (a) 单项投资的$VaR_{0.99}$
<!-- bilingual-en:start -->
*(a) Single-Investment $VaR_{0.99}$*
<!-- bilingual-en:end -->

$$
VaR_{0.99}(L_i)=1,000,000.
$$
**理由**：
$P(L_i\le1,000,000)=0.991\ge0.99$，而$P(L_i\le l)<0.99$对所有$l<1,000,000$成立；故满足定义的最小$l$即为$1,000,000$。
<!-- bilingual-en:start -->
**Reason:** $P(L_i\le1{,}000{,}000)=0.991\ge0.99$, while $P(L_i\le l)<0.99$ for every $l<1{,}000{,}000$. Hence the smallest qualifying threshold is $1{,}000{,}000$.
<!-- bilingual-en:end -->

### (b) 单项投资的$ES_{0.99}$
<!-- bilingual-en:start -->
*(b) Single-Investment $ES_{0.99}$*
<!-- bilingual-en:end -->

尾部概率$1\%$中包含
- $0.9\%$概率的$10,000,000$损失，
- 以及$0.1\%$概率的$1,000,000$损失（用来“凑满”尾部的$1\%$）。
<!-- bilingual-en:start -->
The worst 1% of outcomes consists of:
- the 0.9% probability mass at a USD 10 million loss; and
- 0.1% probability mass from the USD 1 million outcome, which completes the 1% tail.
<!-- bilingual-en:end -->

因此
$$

ES_{0.99}(L_i)=\frac{0.009\times10,000,000+0.001\times1,000,000}{0.01}=9,100,000.

$$
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

### (c) 组合$L=L_1+L_2$的$VaR_{0.99}$
<!-- bilingual-en:start -->
*(c) Combined-Portfolio $VaR_{0.99}$ for $L=L_1+L_2$*
<!-- bilingual-en:end -->

组合的可能取值与概率
<!-- bilingual-en:start -->
The combined loss distribution is:
<!-- bilingual-en:end -->

| **组合损失$L$**  | **概率**                             |
| ------------ | ---------------------------------- |
| $2,000,000$  | $0.991^2=0.982081$                 |
| $11,000,000$ | $2\times0.991\times0.009=0.017838$ |
| $20,000,000$ | $0.009^2=0.000081$                 |
<!-- bilingual-en:start -->
| **Combined loss $L$** | **Probability** |
| --- | --- |
| $2,000,000$ | $0.991^2=0.982081$ |
| $11,000,000$ | $2\times0.991\times0.009=0.017838$ |
| $20,000,000$ | $0.009^2=0.000081$ |
<!-- bilingual-en:end -->

累积概率到$2,000,000$仅$98.2081\%<99\%$，加上$11,000,000$即达$99.9919\%>99\%$。
故
<!-- bilingual-en:start -->
Cumulative probability is only $98.2081\%<99\%$ at USD 2 million, but reaches $99.9919\%>99\%$ at USD 11 million. Therefore, the combined 99% VaR is USD 11 million.
<!-- bilingual-en:end -->

$$
VaR_{0.99}(L)=11,000,000.
$$

### (d) 组合的$ES_{0.99}$
<!-- bilingual-en:start -->
*(d) Combined-Portfolio $ES_{0.99}$*
<!-- bilingual-en:end -->

尾部$1\%$由
- $20,000,000$——概率$0.000081$（占尾部$0.81\%$），
- $11,000,000$——再取$0.009919$概率（占尾部$99.19\%$） 
组成。于是
$$
ES_{0.99}(L)=\frac{0.000081\times20,000,000+0.009919\times11,000,000}{0.01}\approx11,072,900.
$$
<!-- bilingual-en:start -->
The worst 1% consists of:
- the entire USD 20 million outcome, with probability $0.000081$; and
- probability $0.009919$ from the USD 11 million outcome.
Thus the displayed tail average is USD 11.0729 million.
<!-- bilingual-en:end -->

### (e) $VaR$与$[[ES|ES]]$的次可加性对比
<!-- bilingual-en:start -->
*(e) Comparing Subadditivity of $VaR$ and $[[ES|ES]]$*
<!-- bilingual-en:end -->

[[一致性风险度量]]
<!-- bilingual-en:start -->
See [[一致性风险度量|coherent risk measures]]. Here, individual VaRs sum to USD 2 million, while combined VaR is USD 11 million, so VaR violates subadditivity. Individual ES values sum to USD 18.2 million, which exceeds combined ES of USD 11.0729 million, so ES satisfies subadditivity.
<!-- bilingual-en:end -->

## 12.6

>[!question] 
> 每日收益（以损失为正）$ΔV\sim N(0,σ^2)$，其中$σ=2,000,000$（美元）。
> (a)1天展望期的97.5vaR为多少?
> (b)5天展望期的97.5%VaR为多少?
> (c)5天展望期的99%VaR为多少?
><!-- bilingual-en:start -->
>Daily loss is $\Delta V\sim N(0,\sigma^2)$ with $\sigma=$ USD 2 million.
>(a) Find one-day 97.5% VaR.
>(b) Find five-day 97.5% VaR.
>(c) Find five-day 99% VaR.
><!-- bilingual-en:end -->

$$VaR_{c,T}=z_cσ\sqrt{T}$$，$z_c$ 为标准正态分位数、$T$ 为展望期（以“天”为单位，若$T=1$则$\sqrt{T}=1$）。
<!-- bilingual-en:start -->
The formula is shown above, where $z_c$ is the standard normal quantile and $T$ is the horizon in days. When $T=1$, $\sqrt{T}=1$.
<!-- bilingual-en:end -->

| **[[Confidence Level|置信水平]]$c$** | **$z_c$** | **展望期$T$** | **公式**                                              | **数值结果**              |
| ----------- | --------- | ---------- | --------------------------------------------------- | --------------------- |
| $97.5\%$    | $1.96$    | $1$        | $$VaR_{0.975,1}=1.96\times2,000,000$$               | $$3,920,000$$         |
| $97.5\%$    | $1.96$    | $5$        | $$VaR_{0.975,5}=1.96\times2,000,000\times\sqrt{5}$$ | $$\approx8,765,000$$  |
| $99\%$      | $2.326$   | $5$        | $$VaR_{0.99,5}=2.326\times2,000,000\times\sqrt{5}$$ | $$\approx10,406,000$$ |
<!-- bilingual-en:start -->
| **[[Confidence Level|Confidence level]] $c$** | **$z_c$** | **Horizon $T$** | **Formula** | **Result** |
| --- | --- | --- | --- | --- |
| $97.5\%$ | $1.96$ | $1$ | $1.96\times2{,}000{,}000$ | USD 3.920 million |
| $97.5\%$ | $1.96$ | $5$ | $1.96\times2{,}000{,}000\times\sqrt5$ | approximately USD 8.765 million |
| $99\%$ | $2.326$ | $5$ | $2.326\times2{,}000{,}000\times\sqrt5$ | approximately USD 10.406 million |
<!-- bilingual-en:end -->

- > (a) **1 天 $97.5\%$ VaR**：约 $3.92,\text{million}$
- > (b) **5 天 $97.5\%$ VaR**：约 $8.77\text{million}$
- > (c) **5 天 $99\%$ VaR**：约 $10.41\text{million}$
计算中默认日收益独立同分布，因而使用$\sqrt{T}$规则对波动率进行时间缩放。
<!-- bilingual-en:start -->
- (a) **One-day 97.5% VaR:** approximately USD 3.92 million.
- (b) **Five-day 97.5% VaR:** approximately USD 8.77 million.
- (c) **Five-day 99% VaR:** approximately USD 10.41 million.
The square-root-of-time rule assumes independent, identically distributed daily losses.
<!-- bilingual-en:end -->

## 12.9

>[!question] 
>假定我们采用1000个历史数据来对VaR模型进行回溯测试，VaR所采用的置信度为99%，在观察日中我们共发现了17个例外，选用5%的置信水平,我们是否应该拒绝模型?在测试中请采用单向检测。
><!-- bilingual-en:start -->
>A 99% VaR model is backtested over 1,000 observations and produces 17 exceptions. At a 5% significance level, should the model be rejected in a one-sided test?
><!-- bilingual-en:end -->

**解答：**
<!-- bilingual-en:start -->
**Answer:**
<!-- bilingual-en:end -->

- 已知：
  - 历史观测天数 $n = 1000$，
  - $VaR$ 置信度 $99\%$，单日超额损失概率 $p = 1\% = 0.01$，
  - 观测到 $x = 17$ 个例外，
  - 显著性水平 $\alpha = 5\%$，采用**单尾检验**。
<!-- bilingual-en:start -->
- Given:
- number of observations $n=1000$;
- VaR confidence level $99\%$, so the daily exception probability is $p=1\%=0.01$;
- observed exceptions $x=17$;
- significance level $\alpha=5\%$, using a **one-sided test**.
<!-- bilingual-en:end -->

**1. 零假设 $H_0$：**
> 模型正确，例外发生概率为 $p = 0.01$。
<!-- bilingual-en:start -->
**1. Null hypothesis $H_0$:**

The model is correctly calibrated, so the exception probability is $p=0.01$.
<!-- bilingual-en:end -->

**2. 统计量分布：**
> 例外次数 $X \sim \mathrm{Binomial}(n=1000,\,p=0.01)$。
<!-- bilingual-en:start -->
**2. Sampling distribution:**

Under $H_0$, the exception count is $X\sim\mathrm{Binomial}(n=1000,p=0.01)$.
<!-- bilingual-en:end -->

**3. 临界值/拒绝域计算：**
<!-- bilingual-en:start -->
**3. Critical region:**
<!-- bilingual-en:end -->

- 查二项分布上尾，求最小 $k_{0.05}$ 使
  $$
  P(X \geq k_{0.05}) \leq 0.05
  $$
- 查表或用计算器，可得 $k_{0.05} = 16$，即当 $X \geq 17$ 时拒绝 $H_0$。
<!-- bilingual-en:start -->
- Find the smallest integer $k$ whose binomial upper tail satisfies $P(X\ge k)\le0.05$.
- The exact threshold is $k=16$, because $P(X\ge16)\approx0.0479$. Thus an exact 5% upper-tail test rejects when $X\ge16$. Using $X\ge17$ is a slightly more conservative rejection rule.
<!-- bilingual-en:end -->

**4. 实际观测 $x=17$，$p$ 值计算：**
$$
\begin{align*}
p\text{-value} &= P(X \geq 17) \\
&= 1 - \sum_{k=0}^{16}\binom{1000}{k}0.01^{k}0.99^{1000-k} \\
&\approx 0.0264 < 0.05
\end{align*}
$$
<!-- bilingual-en:start -->
**4. Observed result $x=17$ and its $p$-value:**
<!-- bilingual-en:end -->

**5. 结论：**
<!-- bilingual-en:start -->
**5. Conclusion:**
<!-- bilingual-en:end -->

- 由于 $p$ 值 $< 5\%$ 或 $x=17 > k_{0.05}=16$，
- **在 $5\%$ 显著性水平下，**我们**应拒绝**该 $VaR$ 模型，
- 说明模型低估了尾部风险。
$$
\boxed{
\text{应拒绝绝对模型。}
}
$$
<!-- bilingual-en:start -->
- The exact one-sided $p$-value is $P(X\ge17)\approx0.0264<0.05$.
- Therefore, **reject** the VaR model at the 5% significance level.
- The exception frequency provides evidence that the model understates tail risk.
<!-- bilingual-en:end -->

## 12.12

>[!question] 
>交易组合在1个月内的变化服从正态分布，均值为0，标准差为200万美元，计算98%置信度下，3个月展望期的VaR和ES。
><!-- bilingual-en:start -->
>A trading portfolio's one-month change is normal with mean zero and standard deviation USD 2 million. Calculate three-month VaR and ES at 98% confidence.
><!-- bilingual-en:end -->


若假定各月独立，3个月累计收益服从 $N(0, 3\times(200)^2)$，即标准差：
$$
\sigma_{3m} = 200 \times \sqrt{3} \approx 346.41\,\text{万美元}
$$
<!-- bilingual-en:start -->
Assuming independent monthly changes, the three-month total is $N(0,3\times(200)^2)$ when amounts are measured in ten-thousands of dollars. Its standard deviation is shown above, equal to approximately USD 3.4641 million.
<!-- bilingual-en:end -->

- $98\%$置信度：$z_{0.98} = 2.054$（正态分布上$98\%$分位点）
- $[[ES|ES]]$公式（正态分布）：
  $$
  ES_\alpha = \frac{\phi(z_\alpha)}{1-\alpha} \sigma
  $$
  其中 $\phi(z_\alpha)$ 为标准正态密度在 $z_\alpha$ 处的值。    $$
  \phi(2.054) = \frac{1}{\sqrt{2\pi}}e^{-2.054^2/2}\approx 0.0484
  $$
  故
  $$
  ES_{0.98} = \frac{0.0484}{1-0.98}\times \sigma_{3m} \approx 2.42 \times \sigma_{3m}
  $$
$VaR_{0.98, 3m}$
$$
VaR_{0.98} = z_{0.98} \cdot \sigma_{3m} = 2.054 \times 346.41 \approx 712.0\;\text{万美元}
$$
$ES_{0.98, 3m}$
$$
ES_{0.98} = 2.42 \times 346.41 \approx 838.3\;\text{万美元}
$$
<!-- bilingual-en:start -->
- At 98% confidence, $z_{0.98}=2.054$.
- For a zero-mean normal loss, $[[ES|ES]]_\alpha=\phi(z_\alpha)\sigma/(1-\alpha)$, where $\phi(z_\alpha)$ is the standard normal density.
- Here, $\phi(2.054)\approx0.0484$, so the ES multiplier is approximately $0.0484/0.02=2.42$.
- $VaR_{0.98,3m}\approx2.054\times3.4641=$ **USD 7.12 million**.
- $ES_{0.98,3m}\approx2.42\times3.4641=$ **USD 8.38 million**.
<!-- bilingual-en:end -->

## 12.13
>[!question] 
> 假定某两项投资的任何一项都有4%的概率会引发1000万美元损失，有2%的概率引发100万美元损失，有94%的概率盈利100万美元，两项投资相互独立:
> (a)对应于95%的置信水平，任意一项投资的VaR是多少?
> (b)选定95%的置信水平，任意一项投资的ES是多少?
> (c)将两项投资迭加在一起所产生的投资组合对应于95%置信水平的VaR是多少?
> (d)将两项投资迭加在一起所产生的投资组合对应于95%置信水平的ES是多少?
> (e)请说明此例的VaR不满足次可加性条件但是ES满足次可加性条件。
><!-- bilingual-en:start -->
>Each of two independent investments has a 4% probability of losing USD 10 million, a 2% probability of losing USD 1 million, and a 94% probability of earning USD 1 million.
>(a) Find the 95% VaR of either investment.
>(b) Find the 95% ES of either investment.
>(c) Find the 95% VaR of the combined portfolio.
>(d) Find the 95% ES of the combined portfolio.
>(e) Show that VaR violates subadditivity here whereas ES satisfies it.
><!-- bilingual-en:end -->

同 12.5 ~~略~~ 要点提示：关注相同置信度与持有期下 VaR 的可比性、并留意厚尾与相关性变化对聚合的影响。
<!-- bilingual-en:start -->
The source refers back to Exercise 12.5 without showing the calculation. Applying the same discrete-tail method gives: single-investment 95% VaR = USD 1 million and ES = USD 8.2 million; combined 95% VaR = USD 9 million and ES = USD 9.416 million. Thus combined VaR exceeds the sum of individual VaRs, USD 2 million, while combined ES remains below the sum of individual ES values, USD 16.4 million. Compare measures only at the same confidence level and horizon, and remember that heavy tails and dependence affect aggregation.
<!-- bilingual-en:end -->

## 12.14

>[!question] 
> 假定一个交易组合的每天价值变化的一阶自相关系数为0.12，由1天的VaR乘以$\sqrt{10}$ 的10天的VaR为200万美元，将自相关考虑在内时，VaR的最佳估计为多少?
><!-- bilingual-en:start -->
>A portfolio's daily value change has first-order autocorrelation 0.12. The ten-day VaR obtained by multiplying one-day VaR by $\sqrt{10}$ is USD 2 million. What is the best ten-day VaR estimate after accounting for autocorrelation?
><!-- bilingual-en:end -->

当收益序列有一阶自相关（即$t$和$t-1$相关），标准差放大系数为：
<!-- bilingual-en:start -->
If returns follow an AR(1)-type correlation pattern, so that lag-$k$ correlation is $\rho^k$, the standard-deviation scaling factor is the expression shown above. With $T=10$ and $\rho=0.12$, it equals approximately 3.5238 rather than $\sqrt{10}=3.1623$. Therefore, adjusted ten-day VaR is approximately $2.0\times3.5238/3.1623=$ **USD 2.23 million**.
<!-- bilingual-en:end -->

$$
\sqrt{T + 2\sum_{k=1}^{T-1}(T-k)\rho^k}
$$


# 补充
<!-- bilingual-en:start -->
*Supplement*
<!-- bilingual-en:end -->

## **1.** **常见风险度量方法简介**
<!-- bilingual-en:start -->
*1. Overview of Common Risk Measures*
<!-- bilingual-en:end -->

| **方法**     | **公式或定义**                                       | **主要衡量内容**           | **适用场景/优缺点**                                   |
| ---------- | ----------------------------------------------- | -------------------- | ---------------------------------------------- |
| β (Beta)   | $$\beta = \frac{Cov(r_i, r_m)}{Var(r_m)}$$      | 系统性风险，对市场的敏感度        | 只适合衡量相对市场波动（如CAPM），反映与市场的相关风险，**不反映绝对风险或极端尾部** |
| $\sigma$   | $$\sigma = \sqrt{Var(r)}$$                      | 总波动性                 | 容易理解和计算，假设风险对称、正态，**不能捕捉尾部和偏度**                |
| $\sigma^2$ | $$Var(r) = E[(r-E(r))^2]$$                      | 波动率的平方               | 便于理论推导，缺点同上                                    |
| $VaR$      | $$VaR_\alpha$$使得$$P(loss>VaR_\alpha)=1-\alpha$$ | 指定置信度下最大损失           | 易于理解和监管采用，**不能反映分布尾部极端损失、不可加**，对非正态/厚尾分布失真     |
| $[[ES|ES]]$       | $$ES_\alpha = E[loss                            | loss > VaR_\alpha]$$ | 超过VaR的平均损失（尾部期望损失）                             |
<!-- bilingual-en:start -->
| **Measure** | **Formula or definition** | **What it measures** | **Uses and limitations** |
| --- | --- | --- | --- |
| Beta | $\beta=\operatorname{Cov}(r_i,r_m)/\operatorname{Var}(r_m)$ | Sensitivity to systematic market risk | Useful in CAPM and relative market-risk analysis; does not measure total or tail risk. |
| $\sigma$ | $\sigma=\sqrt{\operatorname{Var}(r)}$ | Total volatility | Simple, but treats upside and downside symmetrically and does not by itself describe tail shape. |
| $\sigma^2$ | $\operatorname{Var}(r)=E[(r-E(r))^2]$ | Variance, or squared volatility | Convenient analytically; has the same tail limitations as standard deviation. |
| $VaR$ | $P(L>VaR_\alpha)\le1-\alpha$ | A loss quantile at confidence $\alpha$ | Intuitive and widely reported; does not describe losses beyond the threshold and can fail subadditivity. |
| $[[ES|ES]]$ | Average loss in the worst $1-\alpha$ fraction of outcomes | Tail-average loss | Tail-sensitive and coherent under standard conditions; statistically demanding in small samples. |
<!-- bilingual-en:end -->

---

## **2.**  **度量方法详细对比与优缺点**
<!-- bilingual-en:start -->
*2. Detailed Comparison of the Measures*
<!-- bilingual-en:end -->

### **(1) β（Beta系数）**
<!-- bilingual-en:start -->
*(1) Beta Coefficient*
<!-- bilingual-en:end -->

- 衡量：资产对市场整体的敏感度（**系统性风险**）。
<!-- bilingual-en:start -->
- Measures an asset's sensitivity to the overall market, or **systematic risk**.
<!-- bilingual-en:end -->
    
- 优点：用于资产定价、投资组合管理。
<!-- bilingual-en:start -->
- Useful in asset pricing and portfolio management.
<!-- bilingual-en:end -->
    
- 缺点：只关注相对风险，**不关心独立风险/极端风险**，假定市场收益为正态。
<!-- bilingual-en:start -->
- Does not measure idiosyncratic or tail risk. Its definition does not require normal market returns, although some related inference or modeling assumptions may.
<!-- bilingual-en:end -->


### **(2) 标准差（σ）/方差（σ²）**
<!-- bilingual-en:start -->
*(2) Standard Deviation ($\sigma$) and Variance ($\sigma^2$)*
<!-- bilingual-en:end -->

- 衡量：**整体波动性**，视为风险代理。
<!-- bilingual-en:start -->
- Measure **overall dispersion** and are often used as volatility-based risk proxies.
<!-- bilingual-en:end -->
    
- 优点：简单直观，金融理论常用。
<!-- bilingual-en:start -->
- Simple, intuitive, and widely used in financial theory.
<!-- bilingual-en:end -->
    
- 缺点：**只考虑波动，忽略方向和极端事件**，对正态假设敏感。
<!-- bilingual-en:start -->
- Treat favorable and adverse deviations symmetrically and do not reveal skewness or extreme-tail severity by themselves.
<!-- bilingual-en:end -->


### **(3) VaR（[[VaR-hub|风险价值]]）**
<!-- bilingual-en:start -->
*(3) VaR ([[VaR-hub|Value at Risk]])*
<!-- bilingual-en:end -->

- 衡量：给定置信度下，最大可能损失。
<!-- bilingual-en:start -->
- Measures a loss quantile at a specified confidence level, not the absolute worst possible loss.
<!-- bilingual-en:end -->
    
- 优点：易于监管，行业标准，直观。
<!-- bilingual-en:start -->
- Intuitive, easy to communicate, and embedded in many risk-management practices.
<!-- bilingual-en:end -->
    
- 缺点：**不关注VaR之外的损失（尾部风险）**，不满足次可加性，不一致，非正态下失效。
<!-- bilingual-en:start -->
- Does not report severity beyond the VaR threshold and can violate subadditivity. It does not automatically “fail” for every non-normal distribution, but a normal-parametric implementation can be badly misspecified for skewed or heavy-tailed losses.
<!-- bilingual-en:end -->


### **(4) [[ES|ES]]（期望损失，尾部期望，CVaR）**
<!-- bilingual-en:start -->
*(4) [[ES|Expected Shortfall]], Expected Tail Loss, or CVaR*
<!-- bilingual-en:end -->

- 衡量：在超过VaR的情况下的平均损失。
<!-- bilingual-en:start -->
- Measures average loss in the worst $1-\alpha$ fraction of outcomes.
<!-- bilingual-en:end -->
    
- 优点：**关注尾部风险**，一致风险度量，**满足可加性**（合组合不会增加风险估计）。
<!-- bilingual-en:start -->
- Tail-sensitive and coherent; **subadditivity**, not additivity, formalizes the diversification property.
<!-- bilingual-en:end -->
    
- 缺点：计算复杂，对极端数据敏感，需大样本。
<!-- bilingual-en:start -->
- More difficult to estimate precisely because it depends on sparse extreme observations and is sensitive to tail modeling.
<!-- bilingual-en:end -->
    

---

## **3.**  **评判标准**
<!-- bilingual-en:start -->
*3. Evaluation Criteria*
<!-- bilingual-en:end -->

**好的风险度量方法通常应满足：**
<!-- bilingual-en:start -->
**A useful risk measure should be assessed against:**
<!-- bilingual-en:end -->

1. **一致性（Coherence）**
<!-- bilingual-en:start -->
1. **Coherence**
<!-- bilingual-en:end -->
    
    - 次可加性（subadditivity）：$\rho(X+Y)\leq\rho(X)+\rho(Y)$，分散能降低风险。
    - 单调性（monotonicity）：风险高的资产度量值更高。
    - 正齐次性（positive homogeneity）。
    - 零风险资产风险度量为0（translation invariance）。
<!-- bilingual-en:start -->
- Subadditivity: $\rho(X+Y)\leq\rho(X)+\rho(Y)$.
- Monotonicity: a position with no smaller loss in every state should not receive a lower risk measure.
- Positive homogeneity: scaling every loss by a positive constant scales the risk measure by the same constant.
- Translation invariance under the loss convention: $\rho(X+c)=\rho(X)+c$ for a certain added cash loss $c$. Normalization, $\rho(0)=0$, follows separately.
<!-- bilingual-en:end -->
    
2. **尾部敏感性**：能否反映极端损失（厚尾）。
<!-- bilingual-en:start -->
2. **Tail sensitivity:** Does the measure respond to the severity of extreme losses?
<!-- bilingual-en:end -->
    
3. **可操作性**：计算方便、直观易解释。
<!-- bilingual-en:start -->
3. **Operational usability:** Can it be estimated, explained, validated, and acted upon?
<!-- bilingual-en:end -->
    
4. **适用性**：适合实际业务、可用于组合或监管。
<!-- bilingual-en:start -->
4. **Fitness for purpose:** Does it suit the portfolio, decision, and regulatory context?
<!-- bilingual-en:end -->

---

## **4.**  **哪个方法最好？**
<!-- bilingual-en:start -->
*4. Which Measure Is Best?*
<!-- bilingual-en:end -->

- **理论上**：$[[ES|ES]]$（期望损失）最优，因为它**满足所有一致性公理，能反映极端损失，组合可加**，也是巴塞尔新资本协议主推方法。
<!-- bilingual-en:start -->
- **There is no universally best measure.** Within this comparison, $[[ES|ES]]$ is preferable when tail severity and coherence matter, and 97.5% ES is central to the FRTB internal-model framework.
<!-- bilingual-en:end -->
    
- **实际中**：监管、机构常用$VaR$，因其直观、便于操作。
<!-- bilingual-en:start -->
- **In practice,** $VaR$ remains widely used because it is intuitive, operationally familiar, and useful as a quantile diagnostic.
<!-- bilingual-en:end -->
    
- $β$适合做市场比较，标准差/方差适合波动性描述但不适合极端风险衡量。
<!-- bilingual-en:start -->
- Beta is useful for relative market exposure; standard deviation and variance summarize volatility but do not by themselves measure tail severity.
<!-- bilingual-en:end -->

---

## **5.** **简要总结**
<!-- bilingual-en:start -->
*5. Brief Summary*
<!-- bilingual-en:end -->

- > **$VaR$**：监管常用，直观但不能反映极端尾部损失，不满足次可加性。
<!-- bilingual-en:start -->
- > **$VaR$:** an intuitive and widely used quantile measure, but it does not describe loss severity beyond the threshold and may violate subadditivity.
<!-- bilingual-en:end -->
    
- > **$[[ES|ES]]$**：理论最优，能度量极端尾部损失，满足一致性公理，是未来主流方向。
<!-- bilingual-en:start -->
- > **$[[ES|ES]]$:** a coherent, tail-sensitive measure and a central market-risk measure in the FRTB framework.
<!-- bilingual-en:end -->
    
- > **$\sigma$/$\sigma^2$**：只描述波动，不关注尾部，正态假设下才合理。
<!-- bilingual-en:start -->
- > **$\sigma$ and $\sigma^2$:** measures of dispersion, not complete descriptions of downside or tail risk.
<!-- bilingual-en:end -->
    
- > **$β$**：只度量市场相关风险，不能度量总风险或尾部风险。
<!-- bilingual-en:start -->
- > **Beta:** measures systematic market sensitivity, not total or tail risk.
<!-- bilingual-en:end -->
    

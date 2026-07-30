---
aliases:
  - "Historical and Monte Carlo Risk Simulation"
  - "风险模拟"
  - "历史模拟"
  - "Monte Carlo 风险模拟"
status: source-checked
---

# 历史模拟与 Monte Carlo 风险模拟
<!-- bilingual-en:start -->
*Historical and Monte Carlo Risk Simulation*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 从风险因子情景生成当前组合的完整损益分布，尤其处理期权等不能只靠线性方差描述的头寸。
> **具体锚点：** 历史模拟把过去每一天的利率、价格和波动共同变化施加到今天组合；Monte Carlo 则从显式边际、依赖和动态模型生成许多新路径。
> **核心难点：** 历史模拟把代表性风险放在窗口选择，Monte Carlo 把风险放在模型设定；一个“精确”分位可能只是精确估计了错误模型。
> **为什么重要：** VaR、ES、PFE 和非线性风险都需要损失分布，方法选择必须匹配头寸、数据、期限和计算预算。
> **继续：** 先画风险因子—价值映射并决定全重估或近似；极端尾部外推见 [[极值理论 EVT 与尾部风险]]，最终验证见 [[VaR、ES 与回测]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** It generates a full P&L distribution for the current portfolio from risk-factor scenarios, especially for options and other positions that cannot be described by linear variance alone.
> **Concrete anchor:** Historical simulation applies each past day's joint changes in rates, prices, and volatility to today's portfolio. Monte Carlo generates many new paths from explicit marginal, dependence, and dynamic models.
> **Central difficulty:** Historical simulation places representativeness risk in window choice, while Monte Carlo places it in model specification. A precise quantile may precisely estimate the wrong model.
> **Why it matters:** VaR, ES, PFE, and nonlinear risk require a loss distribution, and method choice must match positions, data, horizon, and computation budget.
> **Continue:** Draw the risk-factor–value map first and decide between full revaluation and approximation. For extrapolating extreme tails, see [[极值理论 EVT 与尾部风险|Extreme Value Theory and Tail Risk]]; for final validation, see [[VaR、ES 与回测|VaR, Expected Shortfall, and Backtesting]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
<!-- bilingual-en:start -->
> [!source] Basis for this section
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
<!-- bilingual-en:end -->

## 方差—协方差基准
<!-- bilingual-en:start -->
*The variance–covariance benchmark*
<!-- bilingual-en:end -->

线性头寸且因子近椭圆/正态时，组合 P&L 均值方差可解析得到 VaR。它快速透明，但厚尾、偏态、相关动态和非线性会失真。Delta–Gamma 近似扩展曲率，却仍依赖局部展开。
<!-- bilingual-en:start -->
For linear positions with approximately elliptical or normal factors, portfolio P&L mean and variance yield analytic VaR. The method is fast and transparent but misrepresents heavy tails, skewness, changing dependence, and nonlinearity. A delta–gamma approximation adds curvature but remains a local expansion.
<!-- bilingual-en:end -->

这个解析基准有价值，因为它能快速核对模拟方向和数量级。若全重估模拟与线性结果差异巨大，应先判断是期权曲率、非正态、路径依赖还是代码错误，而不是直接相信更复杂结果。
<!-- bilingual-en:start -->
The analytic benchmark is valuable for checking the direction and order of magnitude of simulation. If full revaluation differs greatly from the linear result, first determine whether option curvature, non-normality, path dependence, or code error explains the gap rather than trusting complexity automatically.
<!-- bilingual-en:end -->

## 历史模拟
<!-- bilingual-en:start -->
*Historical simulation*
<!-- bilingual-en:end -->

把历史风险因子共同变化应用于当前头寸并重估，天然保留历史横截面依赖和非正态。它假设未来可由所选历史窗口代表，尾部分辨率受样本量限制；加权/过滤历史模拟改变时变波动假设。
<!-- bilingual-en:start -->
Historical simulation applies past joint risk-factor changes to current positions and revalues them, naturally preserving historical cross-sectional dependence and non-normality. It assumes the selected historical window represents the future, and tail resolution is limited by sample size; weighted and filtered variants alter the time-varying-volatility assumption.
<!-- bilingual-en:end -->

实施顺序是：对每个历史日提取同口径风险因子变化；把整行共同变化施加到当前因子；全重估或用经验证近似得到 P&L；排序损失取分位和尾均值。不能分别抽取每个因子的历史日，否则会破坏当日共同依赖。
<!-- bilingual-en:start -->
Implementation proceeds by extracting consistently defined factor changes for each historical date, applying each whole row jointly to current factors, fully revaluing or using a validated approximation, and sorting losses for quantiles and tail means. Sampling a different historical date for each factor destroys contemporaneous dependence.
<!-- bilingual-en:end -->

500 个日观测的经验 99% 尾部只有约 5 个点，ES 几乎由这几个损失决定。增加窗口提高分辨率，却可能引入过时 regime；缩短窗口反应当前波动，却让尾部更稀疏。
<!-- bilingual-en:start -->
With 500 daily observations, the empirical 99% tail contains only about five points, so ES is almost entirely determined by those losses. A longer window improves resolution but may import obsolete regimes; a shorter window responds to current volatility but leaves an even thinner tail.
<!-- bilingual-en:end -->

## Monte Carlo
<!-- bilingual-en:start -->
*Monte Carlo simulation*
<!-- bilingual-en:end -->

指定边际、相关/ copula 和动态，模拟风险因子路径，再重估头寸。它可处理复杂非线性和新情景，但模型校准、随机误差和计算成本重要。使用共同随机数和收敛误差可提高比较可靠性。
<!-- bilingual-en:start -->
Specify marginals, dependence or a copula, and dynamics; simulate risk-factor paths and revalue positions. Monte Carlo can handle complex nonlinearity and new scenarios, but calibration, sampling error, and computational cost matter. Common random numbers and convergence-error reporting improve comparison reliability.
<!-- bilingual-en:end -->

一套最小 Monte Carlo 流程包括：估计真实世界动态；生成独立随机数并通过相关或 copula 形成联合冲击；按时间递推因子；在每条路径重估；聚合 P&L；用不同种子与样本数检查分位和 ES 稳定性。价格模型与风险因子模型要分开记录。
<!-- bilingual-en:start -->
A minimum Monte Carlo workflow estimates real-world dynamics, generates independent random numbers and imposes joint shocks through correlation or a copula, propagates factors through time, revalues on every path, aggregates P&L, and checks quantile and ES stability across seeds and sample sizes. Record the pricing model separately from the risk-factor model.
<!-- bilingual-en:end -->

模拟标准误大致随 $1/\sqrt{N}$ 下降，因此把误差减半通常需要约四倍路径。尾部分位与 ES 的有效样本更少，收敛可能比均值慢；只报告路径数而不报告重复运行稳定性不足以说明精度。
<!-- bilingual-en:start -->
Simulation standard error falls approximately as $1/\sqrt{N}$, so halving error usually requires about four times as many paths. Tail quantiles and ES have fewer effective observations and can converge more slowly than means. Reporting path count without stability across repeated runs does not establish precision.
<!-- bilingual-en:end -->

## 验证与组合
<!-- bilingual-en:start -->
*Validation and hybrid methods*
<!-- bilingual-en:end -->

比较 out-of-sample exceptions、尾损失、不同窗口/分布和压力场景。混合方法常合理：GARCH 过滤波动、经验/coupla 保留依赖、EVT 修正尾部，但每层增加模型风险。
<!-- bilingual-en:start -->
Compare out-of-sample exceptions, tail losses, alternative windows and distributions, and stress scenarios. Hybrid methods can be reasonable: GARCH filters volatility, empirical methods or copulas preserve dependence, and EVT adjusts tails, but each layer adds model risk.
<!-- bilingual-en:end -->

## 从原主题保留的全局定位
<!-- bilingual-en:start -->
*Global orientation retained from the original topic*
<!-- bilingual-en:end -->

> **它解决什么：** 用历史重放、参数化随机生成和专门尾部模型得到组合损失分布，并处理非线性头寸。
> **具体锚点：** 对期权组合，简单方差—协方差法可能漏掉曲率；历史/Monte Carlo 对每个因子情景全重估可保留非线性。
> **核心难点：** 历史模拟受窗口限制，Monte Carlo 受模型限制，EVT 受阈值和尾部样本限制；没有无假设方法。
> **为什么重要：** 方法选择应由头寸非线性、数据量、尾部目标和计算预算共同决定。
> **继续：** 先画风险因子—价值映射，再比较三种方法并做回测/压力。
<!-- bilingual-en:start -->
> **What it solves:** It obtains portfolio loss distributions through historical replay, parametric random generation, and specialized tail models while handling nonlinear positions.
> **Concrete anchor:** For an option portfolio, a simple variance–covariance method can miss curvature; historical or Monte Carlo full revaluation under each factor scenario preserves nonlinearity.
> **Central difficulty:** Historical simulation is limited by its window, Monte Carlo by its model, and EVT by its threshold and tail sample. No method is assumption-free.
> **Why it matters:** Method choice should jointly reflect position nonlinearity, data quantity, tail objective, and computational budget.
> **Continue:** Draw the risk-factor–value map first, then compare the three methods and conduct backtesting and stress testing.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 将历史因子逐列独立打乱，错误消除共同冲击。
- 历史窗口没有危机便声称未来尾部很薄。
- Monte Carlo 路径很多便认为模型可靠，未验证边际、依赖与动态。
- 只检查均值和波动收敛，不检查目标分位与 ES 的抽样稳定性。
<!-- bilingual-en:start -->
- Independently shuffling historical factors by column incorrectly removes common shocks.
- Claiming thin future tails because the historical window contains no crisis.
- Treating a Monte Carlo model as reliable because it has many paths without validating marginals, dependence, and dynamics.
- Checking convergence of mean and volatility only, not sampling stability of the target quantile and ES.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 历史模拟真的‘无模型’吗？
<!-- bilingual-en:start -->
*Is historical simulation truly model-free?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不是。它隐含历史窗口代表未来、历史共同变化可重用于当前头寸等强假设。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. It assumes that the historical window represents the future and that past joint changes can be applied to current positions.
<!-- bilingual-en:end -->

### Monte Carlo 结果为何可能很精确却很错？
<!-- bilingual-en:start -->
*Why can a Monte Carlo result be very precise yet wrong?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 模拟误差可很小，但若边际、依赖或动态模型错设，结果会精确地逼近错误模型。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Sampling error can be small, but if marginals, dependence, or dynamics are misspecified, the simulation precisely approximates the wrong model.
<!-- bilingual-en:end -->

### 用自己的话解释：历史模拟为什么必须把同一天的因子变化一起重放？
<!-- bilingual-en:start -->
*Explain in your own words: why must historical simulation replay all factor changes from the same day together?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 同一天的利率、价格、波动和汇率变化包含历史依赖；分开抽样会制造从未出现的组合并抹去共同压力，使组合尾部失真。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Same-day moves in rates, prices, volatility, and exchange rates contain historical dependence. Sampling them separately creates combinations that never occurred and removes common stress, distorting portfolio tails.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 已逐项核验方差—协方差、历史重放与 Monte Carlo 的假设和实施顺序；经验尾部样本数与 $1/\sqrt{N}$ 收敛关系重新计算。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- Assumptions and implementation order for variance–covariance, historical replay, and Monte Carlo were checked item by item; empirical-tail sample size and the $1/\sqrt{N}$ convergence relation were recalculated.
<!-- bilingual-en:end -->

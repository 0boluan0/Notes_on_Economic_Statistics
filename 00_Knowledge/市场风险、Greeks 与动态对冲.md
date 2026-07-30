---
aliases:
  - "Market Risk, Greeks, and Dynamic Hedging"
  - "Greeks"
  - "市场风险管理"
status: source-checked
---

# 市场风险、Greeks 与动态对冲
<!-- bilingual-en:start -->
*Market Risk, Greeks, and Dynamic Hedging*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 把组合价值映射到价格、利率、汇率和波动率等风险因子，使用局部一阶与二阶敏感度解释损益并设计动态对冲。
> **具体锚点：** 期权 Delta 为 0.5 只表示在当前点、其他条件不变且标的小幅变动时，标的涨 1 期权约涨 0.5；大幅变动时 Gamma 与波动率变化会使近似失真。
> **核心难点：** Greeks 是会随市场和时间变化的局部导数；对冲一个因子会留下曲率、基差、跳跃、交易成本与模型风险。
> **为什么重要：** 交易敞口、对冲、VaR 与损益归因都从风险因子—价值函数开始，名义金额本身不能说明风险。
> **继续：** 小变动用 Delta/Gamma 近似并持续再平衡；严重多因子变化交给 [[压力测试与逆向压力测试]]，尾部分布聚合见 [[VaR、ES 与回测]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** It maps portfolio value to risk factors such as prices, rates, exchange rates, and volatility and uses local first- and second-order sensitivities to explain P&L and design dynamic hedges.
> **Concrete anchor:** An option delta of 0.5 means only that at the current point, holding other inputs fixed and for a small underlying move, a rise of 1 changes option value by about 0.5. Large moves make gamma and volatility changes invalidate the approximation.
> **Central difficulty:** Greeks are local derivatives that change with market state and time. Hedging one factor leaves curvature, basis, jumps, transaction costs, and model risk.
> **Why it matters:** Trading exposure, hedging, VaR, and P&L attribution all begin with a risk-factor–value function; notional amount alone does not describe risk.
> **Continue:** Use delta and gamma approximations for small moves and rebalance continually. Send severe multi-factor changes to [[压力测试与逆向压力测试|Stress Testing and Reverse Stress Testing]], and aggregate tail distributions in [[VaR、ES 与回测|VaR, Expected Shortfall, and Backtesting]].
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

## 风险因子与价值映射
<!-- bilingual-en:start -->
*Risk factors and the value map*
<!-- bilingual-en:end -->

先把头寸价值写成利率、价格、波动率、汇率和信用利差等因子的函数 $V(x)$。名义头寸不等于风险敞口；同一名义在不同到期、期权性和相关结构下敏感度不同。
<!-- bilingual-en:start -->
First write position value as a function $V(x)$ of factors such as rates, prices, volatility, exchange rates, and credit spreads. Notional position is not risk exposure; equal notionals with different maturities, optionality, and dependence structures have different sensitivities.
<!-- bilingual-en:end -->

风险因子选择要能解释价格变化，又不能重复计算同一变化。例如外币股票可以拆成本地股价与汇率；固定收益可拆曲线节点、信用利差与波动。残差持续有结构时，说明映射遗漏因子或定价模型不一致。
<!-- bilingual-en:start -->
Risk factors should explain price changes without counting the same movement twice. A foreign equity can be decomposed into local share price and exchange rate; fixed income can be decomposed into curve nodes, credit spread, and volatility. Persistently structured residuals indicate missing factors or an inconsistent pricing model.
<!-- bilingual-en:end -->

## Delta、Gamma、Vega、Theta、Rho
<!-- bilingual-en:start -->
*Delta, gamma, vega, theta, and rho*
<!-- bilingual-en:end -->

Delta 是对标的的一阶导数，Gamma 是 Delta 的变化率，Vega 对隐含波动率敏感，Theta 对时间流逝，Rho 对利率。符号和单位依产品/报价约定；Vega 常按一个波动率百分点报告，而数学导数可能按 1.00 变化。
<!-- bilingual-en:start -->
Delta is the first derivative with respect to the underlying, gamma is the rate of change of delta, vega is sensitivity to implied volatility, theta to passage of time, and rho to interest rates. Signs and units depend on product and quotation conventions; vega is often reported per one volatility percentage point even when the mathematical derivative uses a 1.00 change.
<!-- bilingual-en:end -->

Greek 本身不是独立风险源，而是价值函数对选定输入的导数。若使用不同波动率曲面参数化或不同利率曲线，Vega 和 Rho 的含义会变；报告数字必须连同 bump 大小、单位和保持不变的其他输入。
<!-- bilingual-en:start -->
A Greek is not an independent source of risk; it is a derivative of the value function with respect to a chosen input. Vega and rho change meaning under different volatility-surface or yield-curve parameterizations. Report the number with bump size, unit, and other inputs held fixed.
<!-- bilingual-en:end -->

## Taylor 近似
<!-- bilingual-en:start -->
*Taylor approximation*
<!-- bilingual-en:end -->

$\Delta V\approx\delta^T\Delta x+\frac12\Delta x^T\Gamma\Delta x$。一阶适合小变动和近线性产品，二阶捕捉曲率但仍是局部。离散跳跃、障碍、提前行权和大变动应全重估。
<!-- bilingual-en:start -->
$\Delta V\approx\delta^T\Delta x+\frac12\Delta x^T\Gamma\Delta x$. The first-order term suits small changes and nearly linear products; the second-order term captures curvature but remains local. Discrete jumps, barriers, early exercise, and large moves require full revaluation.
<!-- bilingual-en:end -->

### 数值锚点
<!-- bilingual-en:start -->
*Numerical anchor*
<!-- bilingual-en:end -->

一只期权 Delta 为 0.5、Gamma 为 0.04，标的上涨 2，忽略其他因子时，一阶预测价值增加 1；Delta–Gamma 预测为 $0.5\times2+\frac12\times0.04\times2^2=1.08$。若实际增加 1.20，余下 0.12 可能来自更高阶曲率、隐含波动变化、时间或模型误差。
<!-- bilingual-en:start -->
An option has delta 0.5 and gamma 0.04, and the underlying rises by 2. Ignoring other factors, the first-order prediction is a value increase of 1; the delta–gamma prediction is $0.5\times2+\frac12\times0.04\times2^2=1.08$. If actual value rises by 1.20, the remaining 0.12 may come from higher-order curvature, an implied-volatility move, time, or model error.
<!-- bilingual-en:end -->

## 对冲
<!-- bilingual-en:start -->
*Hedging*
<!-- bilingual-en:end -->

Delta-neutral 只在当前点对小标的变动中性，时间和市场变化后需再平衡；Gamma/Vega 对冲常需要其他期权。对冲减少特定因子风险，却引入基差、交易成本、流动性和模型风险。
<!-- bilingual-en:start -->
Delta neutrality applies only at the current point for a small underlying move and requires rebalancing as time and markets change; hedging gamma or vega often requires other options. Hedging reduces selected factor risk but introduces basis, transaction-cost, liquidity, and model risk.
<!-- bilingual-en:end -->

动态对冲结果依赖路径：连续时间、无摩擦复制是理论极限，现实离散再平衡会产生 hedging error。再平衡更频繁可降低局部暴露，却增加成本与市场冲击；最优频率取决于 Gamma、波动、流动性和风险容忍度。
<!-- bilingual-en:start -->
Dynamic-hedging outcomes depend on the path. Continuous-time, frictionless replication is a theoretical limit; discrete real-world rebalancing creates hedging error. More frequent rebalancing can reduce local exposure but raises costs and impact; optimal frequency depends on gamma, volatility, liquidity, and risk tolerance.
<!-- bilingual-en:end -->

## 真实世界与风险中性
<!-- bilingual-en:start -->
*Real-world and risk-neutral measures*
<!-- bilingual-en:end -->

风险中性测度用于无套利定价，漂移由资金成本等决定；真实世界测度用于预测实际 P&L、资本和压力。用风险中性情景估真实频率风险，或反过来直接定价，都会混淆目标。
<!-- bilingual-en:start -->
The risk-neutral measure supports no-arbitrage pricing, with drift governed by funding and related inputs; the real-world measure is used to forecast actual P&L, capital, and stress. Using risk-neutral scenarios to estimate real-world frequencies, or real-world probabilities directly for arbitrage-free pricing, confuses objectives.
<!-- bilingual-en:end -->

## 从原主题保留的全局定位
<!-- bilingual-en:start -->
*Global orientation retained from the original topic*
<!-- bilingual-en:end -->

> **它解决什么：** 把组合价值对市场因子的小幅变化和大幅情景变化分开度量，并据此对冲或设置限额。
> **具体锚点：** 期权 Delta 为 0.5 只近似说明标的小涨 1 时价值涨 0.5；若标的大动，Gamma 和波动率变化会让线性近似失真。
> **核心难点：** Greeks 是局部敏感度且彼此随市场变化；风险中性定价参数和真实世界损益分布服务不同问题。
> **为什么重要：** 交易敞口、对冲、VaR 和压力测试都从因子—价值映射开始。
> **继续：** 小变动用 Delta/Gamma，大变动用全重估情景；聚合尾部见 [[VaR、ES 与回测|VaR、ES、回测与压力测试]]。
<!-- bilingual-en:start -->
> **What it solves:** It measures small market-factor changes and large scenario changes separately and uses them for hedging or limits.
> **Concrete anchor:** An option delta of 0.5 only approximates a value rise of 0.5 for a small underlying rise of 1; a large move makes gamma and volatility changes invalidate the linear approximation.
> **Central difficulty:** Greeks are local sensitivities and change together with markets; risk-neutral pricing inputs and real-world P&L distributions serve different questions.
> **Why it matters:** Trading exposure, hedging, VaR, and stress testing all begin with a factor–value map.
> **Continue:** Use delta and gamma for small moves and fully revalued scenarios for large moves. For aggregated tails, see [[VaR、ES 与回测|VaR, Expected Shortfall, and Backtesting]].
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 把名义本金当作市场风险，未映射期限与非线性。
- 报告 Vega=10 却不说明是每 1% 还是每 100% 波动变化。
- Delta 对冲完成后称组合无风险，遗漏 Gamma、Vega、跳跃和再平衡。
- 用风险中性漂移模拟一年实际损益频率，混淆定价测度与预测测度。
<!-- bilingual-en:start -->
- Treating notional principal as market risk without mapping maturity and nonlinearity.
- Reporting vega of 10 without stating whether it is per 1% or per 100% volatility change.
- Calling a portfolio riskless after delta hedging while omitting gamma, vega, jumps, and rebalancing.
- Simulating one-year actual P&L frequencies with risk-neutral drift, confusing pricing and forecasting measures.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### Delta-neutral 为什么不等于没有市场风险？
<!-- bilingual-en:start -->
*Why does delta-neutral not mean free of market risk?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> Delta 会随价格、时间和波动变化，仍有 Gamma、Vega、跳跃、基差、流动性和再平衡成本。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Delta changes with price, time, and volatility; gamma, vega, jumps, basis, liquidity, and rebalancing cost remain.
<!-- bilingual-en:end -->

### 什么时候 Delta–Gamma 近似仍不够？
<!-- bilingual-en:start -->
*When is a delta–gamma approximation still insufficient?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 大幅/离散冲击、强路径依赖、障碍或提前行权附近，应对每个情景全重估。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Under large or discrete shocks, strong path dependence, barriers, or near early-exercise boundaries, fully revalue each scenario.
<!-- bilingual-en:end -->

### 风险中性与真实世界情景各用于什么？
<!-- bilingual-en:start -->
*What are risk-neutral and real-world scenarios used for?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 前者主要用于无套利定价，后者用于实际损益分布、风险、资本和情景概率判断。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The former is primarily for no-arbitrage pricing; the latter is for actual P&L distributions, risk, capital, and scenario-probability judgment.
<!-- bilingual-en:end -->

### 用自己的话解释：动态 Delta 对冲为什么仍会有损益？
<!-- bilingual-en:start -->
*Explain in your own words: why can dynamic delta hedging still generate profit or loss?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> Delta 只在瞬时局部成立；价格跳跃、Gamma、波动率变化、离散再平衡和交易成本使复制不完整，模型与实际路径的差异形成对冲损益。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Delta is instantaneous and local. Price jumps, gamma, volatility changes, discrete rebalancing, and transaction costs make replication incomplete, so differences between model and realized path create hedge P&L.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 已按 Taylor 展开复核 Delta–Gamma 算例，并分别核对风险中性定价与真实世界风险预测的用途边界。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- The delta–gamma example was checked against the Taylor expansion, and the purpose boundaries between risk-neutral pricing and real-world risk forecasting were reviewed separately.
<!-- bilingual-en:end -->

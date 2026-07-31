---
aliases:
  - "Capital Asset Pricing Model"
  - "CAPM"
  - "资本资产定价模型"
status: source-checked
---

# CAPM、系统风险与资本成本
<!-- bilingual-en:start -->
*CAPM, Systematic Risk, and the Cost of Capital*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在一组明确假设下，把一项资产对市场共同风险的暴露转换成要求回报，并为股权资本成本提供基准。
> **具体锚点：** 无风险利率 3%、市场风险溢价 5%、股票 beta 为 1.2 时，CAPM 要求回报为 $3\%+1.2\times5\%=9\%$。
> **核心难点：** beta 不是总风险，市场组合不可完全观察，估计区间、杠杆与所选市场代理都会改变结果。
> **为什么重要：** 它把分散化直觉连接到估值贴现率、项目资本成本、绩效 alpha 与风险归因，但只能作为模型基准。
> **继续：** beta 的组合基础见 [[均值—方差投资组合理论]]；alpha 与有效性检验见 [[有效市场假说与事件研究]]；估值应用见 [[股票与企业价值评估]]。
> <!-- bilingual-en:start -->
> **What it solves:** Under a stated set of assumptions, it translates an asset's exposure to common market risk into a required return and supplies a benchmark cost of equity.
> **Concrete anchor:** With a 3% risk-free rate, a 5% market risk premium, and stock beta of 1.2, CAPM required return is $3\%+1.2\times5\%=9\%$.
> **Central difficulty:** Beta is not total risk, the market portfolio is not fully observable, and the estimation window, leverage, and chosen market proxy all change the result.
> **Why it matters:** It connects diversification to valuation discount rates, project costs of capital, performance alpha, and risk attribution—but only as a model benchmark.
> **Continue:** For beta's portfolio foundation, see [[均值—方差投资组合理论|Mean–Variance Portfolio Theory]]. For alpha and efficiency tests, see [[有效市场假说与事件研究|The Efficient-Market Hypothesis and Event Studies]]. For valuation use, see [[股票与企业价值评估|Equity and Enterprise Valuation]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[02_Economy/06_证券投资学/证券投资学.pdf]]：支持证券工具、市场、基本面、技术分析、组合与资产定价的课程范围。
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
> - Sharpe（1964）《[Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk](https://doi.org/10.1111/j.1540-6261.1964.tb02865.x)》：核验资本市场线、市场均衡与系统风险定价的原始推导。
> <!-- bilingual-en:start -->
> - [[02_Economy/06_证券投资学/证券投资学.pdf|Securities Investment Textbook]] supports the course scope for securities, markets, fundamental analysis, technical analysis, portfolios, and asset pricing.
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
> - Sharpe (1964), “[Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk](https://doi.org/10.1111/j.1540-6261.1964.tb02865.x),” verifies the original derivation of the capital market line, market equilibrium, and pricing of systematic risk.
> <!-- bilingual-en:end -->

## 从分散化到系统风险定价
<!-- bilingual-en:start -->
*From diversification to pricing systematic risk*
<!-- bilingual-en:end -->

若投资者能持有充分分散的组合，单项资产可被组合抵消的特有波动不会单独要求均衡补偿；资产影响组合风险的关键是它与市场组合共同变化多少。CAPM 把这一边际贡献写成 beta，而不是用资产自身标准差定价。
<!-- bilingual-en:start -->
If investors can hold well-diversified portfolios, an individual asset's idiosyncratic volatility that can be offset within the portfolio does not receive separate equilibrium compensation. What matters is how much the asset co-moves with the market portfolio. CAPM expresses this marginal contribution as beta rather than pricing the asset's own standard deviation.
<!-- bilingual-en:end -->

## CAPM
<!-- bilingual-en:start -->
*The capital asset pricing model*
<!-- bilingual-en:end -->

$E(R_i)-R_f=\beta_i[E(R_m)-R_f]$，$\beta_i=Cov(R_i,R_m)/Var(R_m)$。均衡中只补偿不可分散的市场风险，SML 描述 beta—期望收益关系。市场组合不可完全观察，实证检验与代理选择联合。
<!-- bilingual-en:start -->
$E(R_i)-R_f=\beta_i[E(R_m)-R_f]$, where $\beta_i=Cov(R_i,R_m)/Var(R_m)$. In equilibrium, only non-diversifiable market risk is compensated, and the security market line relates beta to expected return. The full market portfolio is unobservable, so empirical tests are joint with the chosen proxy.
<!-- bilingual-en:end -->

CAPM 的核心假设包括单期均值—方差决策、同质预期、无摩擦交易、可借贷无风险资产以及所有相关资产可交易。不是每次使用都要相信现实完全满足它们，但偏离假设时要知道估计在近似什么，并做替代模型或情景检验。
<!-- bilingual-en:start -->
Core CAPM assumptions include single-period mean–variance choice, homogeneous expectations, frictionless trading, borrowing and lending at a risk-free rate, and tradability of all relevant assets. Using CAPM does not require believing reality satisfies them exactly, but departures should be understood and tested with alternative models or scenarios.
<!-- bilingual-en:end -->

## beta 的估计与杠杆调整
<!-- bilingual-en:start -->
*Estimating beta and adjusting for leverage*
<!-- bilingual-en:end -->

历史 beta 常由 $R_i-R_f=\alpha+\beta(R_m-R_f)+\varepsilon$ 回归得到。结果依赖市场指数、收益频率、样本窗口、非同步交易和结构变化。低标准误不代表未来业务与杠杆不变。
<!-- bilingual-en:start -->
Historical beta is often estimated from $R_i-R_f=\alpha+\beta(R_m-R_f)+\varepsilon$. The result depends on the market index, return frequency, sample window, non-synchronous trading, and structural change. A small standard error does not imply that future business mix and leverage will remain unchanged.
<!-- bilingual-en:end -->

比较公司时通常先把观测权益 beta 去杠杆，近似 $\beta_A=\beta_E/[1+(1-T)D/E]$，再按目标资本结构加杠杆。这个公式依赖债务 beta 近似为零等假设；高风险债务或动态杠杆下应使用更完整分解。
<!-- bilingual-en:start -->
Comparable-company analysis often unleveres observed equity beta using the approximation $\beta_A=\beta_E/[1+(1-T)D/E]$ and then releveres it at the target capital structure. This formula assumes, among other things, that debt beta is near zero; risky debt or dynamic leverage requires a fuller decomposition.
<!-- bilingual-en:end -->

## alpha、beta 与绩效
<!-- bilingual-en:start -->
*Alpha, beta, and performance*
<!-- bilingual-en:end -->

alpha 是相对选定因子模型的平均异常回报，不是无条件技能。beta 随业务、杠杆和样本变化。显著 alpha 需考虑费用、交易成本、多重检验和模型遗漏。
<!-- bilingual-en:start -->
Alpha is average abnormal return relative to a selected factor model, not unconditional skill. Beta changes with business mix, leverage, and sample. A statistically significant alpha must be assessed after fees, transaction costs, multiple testing, and omitted factors.
<!-- bilingual-en:end -->

## 模型边界
<!-- bilingual-en:start -->
*Model boundaries*
<!-- bilingual-en:end -->

多因子模型允许价值、规模、动量等系统维度；行为和套利限制提供另一解释。CAPM 仍是有用基准，但不应独占风险解释。
<!-- bilingual-en:start -->
Multifactor models allow systematic dimensions such as value, size, and momentum; behavior and limits to arbitrage provide another class of explanations. CAPM remains a useful benchmark but should not monopolize risk explanation.
<!-- bilingual-en:end -->

资本成本应用尤其要防止“公司 beta 直接用于所有项目”。新项目若业务风险、地区、经营杠杆或融资结构不同，应从可比项目或资产 beta 出发，而不是沿用公司的历史权益 beta。
<!-- bilingual-en:start -->
Cost-of-capital applications must especially avoid using one company beta for every project. If a new project differs in business risk, geography, operating leverage, or financing structure, begin with comparable-project or asset beta rather than the company's historical equity beta.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 把高标准差资产自动判为高 beta，忽略其波动可能主要是特有风险。
- 用本国大盘指数估计全球业务公司的市场 beta，却不讨论市场代理。
- 把显著正 alpha 直接称为能力，未扣费、未控制因子或未校正反复试验。
- 用当前高杠杆权益 beta 折现全股权融资项目，会把融资结构差异误作经营风险。
<!-- bilingual-en:start -->
- Automatically assigning high beta to a high-volatility asset ignores the possibility that most of its volatility is idiosyncratic.
- Estimating a global firm's market beta with one domestic broad index without discussing the market proxy.
- Calling statistically positive alpha skill without fees, factor controls, or correction for repeated testing.
- Discounting an all-equity-financed project with the firm's currently highly levered equity beta mistakes financing structure for operating risk.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### CAPM 的 beta 与总波动有什么区别？
<!-- bilingual-en:start -->
*How does CAPM beta differ from total volatility?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> beta 只度量与市场组合共同变化的系统暴露，总波动还含可分散的特有风险。
> <!-- bilingual-en:start -->
> Beta measures only systematic exposure that co-moves with the market portfolio; total volatility also contains diversifiable idiosyncratic risk.
> <!-- bilingual-en:end -->

### 用自己的话解释：为什么可分散风险在 CAPM 中没有单独风险溢价？
<!-- bilingual-en:start -->
*Explain in your own words: why does diversifiable risk receive no separate premium in CAPM?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 投资者能以持有其他资产低成本抵消这种风险；市场无需为可以在组合中消除的边际波动支付均衡补偿，只有共同市场风险无法这样消除。
> <!-- bilingual-en:start -->
> Investors can offset that risk cheaply by holding other assets. The market need not pay equilibrium compensation for marginal volatility that disappears in a portfolio; common market risk cannot be eliminated that way.
> <!-- bilingual-en:end -->

### 一只 beta 为 0.8、特有波动很大的股票是否一定比 beta 为 1.2 的股票要求回报高？
<!-- bilingual-en:start -->
*Must a stock with beta 0.8 and high idiosyncratic volatility have a higher required return than a stock with beta 1.2?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 按 CAPM 不一定；基准要求回报由 beta 决定，特有波动可分散。但现实模型可能包含其他系统因子、摩擦和集中持仓成本，所以还需检验模型边界。
> <!-- bilingual-en:start -->
> Not under CAPM; benchmark required return is determined by beta, and idiosyncratic volatility is diversifiable. Realistic models may include other systematic factors, frictions, and concentration costs, so model boundaries still require testing.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[02_Economy/06_证券投资学/证券投资学.pdf]]：支持证券工具、市场、基本面、技术分析、组合与资产定价的课程范围。
- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- Sharpe（1964）《[Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk](https://doi.org/10.1111/j.1540-6261.1964.tb02865.x)》：逐项核验无风险借贷、切点组合、市场均衡和 beta—期望收益关系；算例按证券市场线复算。
<!-- bilingual-en:start -->
- [[02_Economy/06_证券投资学/证券投资学.pdf|Securities Investment Textbook]] supports the course scope for securities, markets, fundamental analysis, technical analysis, portfolios, and asset pricing.
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- Sharpe (1964), “[Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk](https://doi.org/10.1111/j.1540-6261.1964.tb02865.x),” was checked for risk-free borrowing and lending, the tangency portfolio, market equilibrium, and the beta–expected-return relation; the numerical example was recomputed from the security market line.
<!-- bilingual-en:end -->

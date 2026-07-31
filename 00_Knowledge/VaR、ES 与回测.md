---
aliases:
  - "VaR, Expected Shortfall, and Backtesting"
  - "Value at Risk"
  - "Expected Shortfall"
  - "VaR 与 ES"
status: source-checked
---

# VaR、ES 与回测
<!-- bilingual-en:start -->
*VaR, Expected Shortfall, and Backtesting*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用给定期限内的损失分位点和尾部均值概括损失分布，并用实际损益检验这些预测是否覆盖正确、是否在时间上聚集。
> **具体锚点：** 一日 99% VaR 为 100 万表示模型下约 1% 的交易日损失超过 100 万；它绝不表示最大只会亏 100 万。
> **核心难点：** VaR 不告诉你越过门槛后亏多少，ES 尾部信息更完整却更难估准；两者都依赖损失符号、期限、置信度、数据与模型。
> **为什么重要：** 风险限额、资本和跨组合沟通需要共同口径，但只有把预测和回测闭环，数字才具有可诊断意义。
> **继续：** 分布估计方法见 [[历史模拟与 Monte Carlo 风险模拟]] 和 [[极值理论 EVT 与尾部风险]]；模型覆盖不到的结构断点见 [[压力测试与逆向压力测试]]。
> <!-- bilingual-en:start -->
> **What it solves:** It summarizes the loss distribution through a horizon-specific quantile and tail mean and uses realized profit and loss to test whether forecasts cover at the right rate and whether breaches cluster over time.
> **Concrete anchor:** A one-day 99% VaR of one million means that under the model roughly 1% of trading days lose more than one million. It never means the maximum possible loss is one million.
> **Central difficulty:** VaR says nothing about loss magnitude beyond the threshold, while ES provides richer tail information but is harder to estimate. Both depend on loss sign, horizon, confidence, data, and model.
> **Why it matters:** Limits, capital, and cross-portfolio communication need common conventions, but the number becomes diagnostically meaningful only when prediction is closed with backtesting.
> **Continue:** For distribution-estimation methods, see [[历史模拟与 Monte Carlo 风险模拟|Historical and Monte Carlo Risk Simulation]] and [[极值理论 EVT 与尾部风险|Extreme Value Theory and Tail Risk]]. For structural breaks outside model coverage, see [[压力测试与逆向压力测试|Stress Testing and Reverse Stress Testing]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
> - Artzner 等（1999）《[Coherent Measures of Risk](https://doi.org/10.1111/1467-9965.00068)》：核验一致风险度量与 VaR 次可加性的边界。
> <!-- bilingual-en:start -->
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
> - Artzner et al. (1999), “[Coherent Measures of Risk](https://doi.org/10.1111/1467-9965.00068),” verifies coherent risk measures and the boundary of VaR subadditivity.
> <!-- bilingual-en:end -->

## 先固定随机变量与口径
<!-- bilingual-en:start -->
*Fix the random variable and conventions first*
<!-- bilingual-en:end -->

令 $L=-\text{P\&L}$ 为损失，因此更大的 $L$ 表示更坏结果。必须同时报告估值时点、持有期、置信水平、币种、覆盖头寸、损益定义和数据窗口。若一方以收益为正、一方以损失为正，公式方向会完全反转。
<!-- bilingual-en:start -->
Let $L=-\text{P\&L}$ be loss, so larger $L$ means a worse outcome. Report valuation time, holding horizon, confidence level, currency, covered positions, P&L definition, and data window together. If one party uses positive returns and another positive losses, formula directions reverse.
<!-- bilingual-en:end -->

## VaR 与 ES 定义
<!-- bilingual-en:start -->
*Definitions of VaR and ES*
<!-- bilingual-en:end -->

令 L 为损失，$VaR_\alpha$ 是其 $\alpha$ 分位点；$ES_\alpha=E[L\mid L>VaR_\alpha]$ 在连续分布下为超尾平均。离散分布需用更一般分位积分定义 ES。损益和损失符号混用会让公式方向反转。
<!-- bilingual-en:start -->
Let L denote loss. $VaR_\alpha$ is its $\alpha$-quantile, and for a continuous distribution $ES_\alpha=E[L\mid L>VaR_\alpha]$ is the average beyond that tail threshold. Discrete distributions require the more general quantile-integral definition of ES. Mixing profit-and-loss and loss signs reverses formula directions.
<!-- bilingual-en:end -->

更一般地，$ES_\alpha=\frac{1}{1-\alpha}\int_\alpha^1 VaR_u\,du$。这个定义在分位点有概率质量时仍能正确处理只部分落入尾部的状态，而简单条件期望可能误计。
<!-- bilingual-en:start -->
More generally, $ES_\alpha=\frac{1}{1-\alpha}\int_\alpha^1 VaR_u\,du$. This definition correctly allocates probability mass that only partly falls in the tail when the quantile has an atom, whereas a simple conditional expectation may miscount it.
<!-- bilingual-en:end -->

### 离散算例
<!-- bilingual-en:start -->
*Discrete worked example*
<!-- bilingual-en:end -->

设 98% 情况损失为 0，1% 情况损失为 100，1% 情况损失为 1,000。99% VaR 按常用左分位定义为 100，但最坏 1% 的 ES 为 1,000。VaR 只标出进入尾部的门槛，完全看不见 100 与 1,000 的差距。
<!-- bilingual-en:start -->
Suppose loss is zero with probability 98%, 100 with probability 1%, and 1,000 with probability 1%. Under a common left-quantile convention, 99% VaR is 100, while ES over the worst 1% is 1,000. VaR marks the entrance to the tail and completely misses the gap between 100 and 1,000.
<!-- bilingual-en:end -->

## 置信水平、持有期与窗口
<!-- bilingual-en:start -->
*Confidence level, horizon, and window*
<!-- bilingual-en:end -->

99% 一日与 97.5% 十日不可直接比较。平方根时间缩放依赖 IID/线性等近似，在波动聚集、跳跃和非线性头寸中失效。短窗口响应快但噪声大，长窗口稳定却可能忽略 regime 变化。
<!-- bilingual-en:start -->
One-day 99% and ten-day 97.5% measures are not directly comparable. Square-root-of-time scaling relies on approximations such as IID changes and linear positions and fails with volatility clustering, jumps, and nonlinear positions. A short window responds quickly but is noisy; a long window is stable but may miss regime change.
<!-- bilingual-en:end -->

## 一致风险度量与局限
<!-- bilingual-en:start -->
*Coherent risk measures and limitations*
<!-- bilingual-en:end -->

在一般分布下 VaR 可能违反次可加，出现组合 VaR 大于分项和；ES 在适当定义下是一致风险度量。风险度量一致不等于估计准确或易回测。
<!-- bilingual-en:start -->
For general distributions, VaR can violate subadditivity, so portfolio VaR can exceed the sum of component VaRs; properly defined ES is coherent. Coherence does not guarantee accurate estimation or easy backtesting.
<!-- bilingual-en:end -->

次可加性表达分散后风险不应超过分别持有风险之和。VaR 在连续椭圆分布等特定条件下可表现良好，但离散违约损失或高度非线性组合会出现反例。不要把“VaR 不是一致度量”误写成“VaR 在任何场景都不能用”。
<!-- bilingual-en:start -->
Subadditivity states that risk after combining positions should not exceed the sum of stand-alone risks. VaR behaves well under particular conditions such as continuous elliptical distributions, but discrete default loss or highly nonlinear portfolios can produce counterexamples. “VaR is not coherent” does not mean VaR is unusable in every setting.
<!-- bilingual-en:end -->

## VaR 回测
<!-- bilingual-en:start -->
*VaR backtesting*
<!-- bilingual-en:end -->

记录实际损失超过预测 VaR 的 exceptions，检查无条件覆盖率和时间独立性；异常聚集说明波动动态不足。反复使用同一数据校准和回测会降低检验含义。
<!-- bilingual-en:start -->
Record exceptions where realized loss exceeds forecast VaR and test unconditional coverage plus temporal independence. Clustered exceptions indicate inadequate volatility dynamics. Repeatedly using the same data for calibration and backtesting weakens the test's meaning.
<!-- bilingual-en:end -->

250 个独立交易日在 99% VaR 下期望约有 2.5 次 exception，但“恰有两三次”不是充分合格。若三次连续出现在一周内，覆盖总数看似合理，独立性却明显可疑；还要检查损益是否含费、是否使用实际或假设 P&L，以及模型变更是否事后追随 exception。
<!-- bilingual-en:start -->
Across 250 independent trading days, 99% VaR implies about 2.5 expected exceptions, but observing two or three is not sufficient for acceptance. If three occur in one week, total coverage may look reasonable while independence is doubtful. Also inspect whether P&L includes fees, whether actual or hypothetical P&L is used, and whether model changes followed exceptions ex post.
<!-- bilingual-en:end -->

## ES 验证
<!-- bilingual-en:start -->
*ES validation*
<!-- bilingual-en:end -->

ES 与对应 VaR 联合评估，比较超越日损失和预测尾均值。尾部样本少，单一组合/时期的检验功效有限，应结合 desk-level 诊断与模型比较。
<!-- bilingual-en:start -->
Evaluate ES jointly with the corresponding VaR by comparing exception losses with the predicted tail mean. Tail observations are sparse, so a test on one portfolio or period has limited power and should be combined with desk-level diagnostics and model comparisons.
<!-- bilingual-en:end -->

## 沟通
<!-- bilingual-en:start -->
*Communication*
<!-- bilingual-en:end -->

同时报告方法、数据期、期限、置信度、覆盖资产、主要假设和近期 exceptions。VaR 数值无这些上下文不可解释。
<!-- bilingual-en:start -->
Report method, data period, horizon, confidence, covered assets, major assumptions, and recent exceptions together. A VaR number without this context is uninterpretable.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 把 99% VaR 当作最坏损失，忽略剩余 1% 尾部。
- 把独立同分布下的一日 VaR 乘 $\sqrt{10}$ 用于有跳跃、波动聚集和期权的十日组合。
- exception 总数接近期望便宣布模型通过，未检验聚集与超越幅度。
- 在看到损失后反复改窗口，再用同一时期声称回测成功。
<!-- bilingual-en:start -->
- Treating 99% VaR as worst loss and ignoring the remaining 1% tail.
- Multiplying one-day VaR by $\sqrt{10}$ for a ten-day portfolio with jumps, volatility clustering, and options.
- Declaring a model passed because the exception count is near expectation without testing clustering or breach magnitude.
- Repeatedly changing the window after observing losses and then claiming success on the same period.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 99% VaR 为 100 万最常见的错误解释是什么？
<!-- bilingual-en:start -->
*What is the most common wrong interpretation of a 99% VaR of one million?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 把它说成最大损失；正确是模型下约 1% 情况损失会超过该阈值，超出幅度未由 VaR 给出。
> <!-- bilingual-en:start -->
> Calling it the maximum loss. Correctly interpreted, about 1% of model outcomes exceed the threshold, and VaR does not specify by how much.
> <!-- bilingual-en:end -->

### ES 相比 VaR 增加了什么信息？
<!-- bilingual-en:start -->
*What information does ES add beyond VaR?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它描述进入最坏尾部后平均损失大小，而不仅是进入尾部的门槛。
> <!-- bilingual-en:start -->
> It describes the average loss after entering the worst tail rather than only the threshold for entering it.
> <!-- bilingual-en:end -->

### 回测 exception 数量正确为何仍可能模型有问题？
<!-- bilingual-en:start -->
*Why can a model still be wrong when the number of backtesting exceptions is correct?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> exception 可能时间聚集、特定因子集中或超越损失过大；还需独立性和尾部大小诊断。
> <!-- bilingual-en:start -->
> Exceptions may cluster in time, concentrate in one factor, or have excessive magnitude; independence and tail-size diagnostics are also needed.
> <!-- bilingual-en:end -->

### 用自己的话解释：为什么 ES 更关注尾部却不必更容易验证？
<!-- bilingual-en:start -->
*Explain in your own words: why can ES focus more on the tail yet be harder to validate?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> ES 需要估计罕见超越事件的平均大小；有效样本远少于中部观测，少数极端值又影响很大，所以估计方差和检验不确定性更高。
> <!-- bilingual-en:start -->
> ES estimates the average magnitude of rare exceedances. There are far fewer effective tail observations than central observations, and a few extremes have large influence, so estimation variance and testing uncertainty are higher.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- Artzner et al. (1999), *Coherent Measures of Risk*：核验一致风险度量性质及 VaR 的边界。
- 已按分位积分定义复核离散 ES 算例，并按二项覆盖与 exception 独立性重新检查回测解释。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- Artzner et al. (1999), *Coherent Measures of Risk*, verifies coherent-risk-measure properties and VaR's boundary.
- The discrete ES example was checked against the quantile-integral definition, and the backtest interpretation was rechecked using binomial coverage and exception independence.
<!-- bilingual-en:end -->

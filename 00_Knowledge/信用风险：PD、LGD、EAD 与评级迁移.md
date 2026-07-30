---
aliases:
  - "Credit Risk: PD, LGD, EAD, and Rating Migration"
  - "信用风险参数"
status: source-checked
---

# 信用风险：PD、LGD、EAD 与评级迁移
<!-- bilingual-en:start -->
*Credit Risk: PD, LGD, EAD, and Rating Migration*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 把单个借款人的信用损失拆成违约可能性、违约时损失比例和违约时敞口，并跟踪违约前的评级恶化。
> **具体锚点：** 一年 PD 2%、LGD 40%、EAD 100 万时，预期损失为 8,000；这不表示实际损失会稳定等于 8,000，而是大量同类敞口的平均。
> **核心难点：** PD、LGD 与 EAD 必须共享违约定义、期限和经济情景，并可能在衰退中同时恶化；市场隐含与真实世界参数也不可混用。
> **为什么重要：** 信贷定价、拨备、限额、抵押品与资本都建立在这些参数上，错误口径会在整个系统中重复放大。
> **继续：** 市场转移信用风险见 [[CDS、信用利差与基差]]；多个借款人共同违约见 [[信用组合风险与 Credit VaR]]；衍生品未来敞口见 [[对手方信用风险、CVA 与 DVA]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** It decomposes one borrower's credit loss into probability of default, loss severity at default, and exposure at default, while tracking deterioration before default through rating migration.
> **Concrete anchor:** With one-year PD of 2%, LGD of 40%, and EAD of one million, expected loss is 8,000. Actual loss does not remain near 8,000; this is an average across many comparable exposures.
> **Central difficulty:** PD, LGD, and EAD must share a default definition, horizon, and economic scenario and can all worsen together in a downturn. Market-implied and real-world parameters cannot be mixed casually.
> **Why it matters:** Credit pricing, provisioning, limits, collateral, and capital all build on these parameters, so inconsistent conventions propagate through the system.
> **Continue:** For transferring credit risk in markets, see [[CDS、信用利差与基差|CDS, Credit Spreads, and Basis]]. For common default across borrowers, see [[信用组合风险与 Credit VaR|Credit Portfolio Risk and Credit VaR]]. For dynamic derivative exposure, see [[对手方信用风险、CVA 与 DVA|Counterparty Credit Risk, CVA, and DVA]].
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

## PD、LGD、EAD 与预期损失
<!-- bilingual-en:start -->
*PD, LGD, EAD, and expected loss*
<!-- bilingual-en:end -->

在指定期限和口径下，PD 是违约概率，LGD 是违约时损失比例，EAD 是违约时敞口。$EL=PD\times LGD\times EAD$ 是期望；实际定价还含资金、运营、风险溢价和资本成本。三项在压力中可能相关。
<!-- bilingual-en:start -->
For a specified horizon and convention, PD is probability of default, LGD is loss proportion conditional on default, and EAD is exposure at default. $EL=PD\times LGD\times EAD$ is an expectation; actual pricing also includes funding, operating cost, risk premium, and capital cost. The three components can be dependent under stress.
<!-- bilingual-en:end -->

- PD 的分母是尚未违约的借款人，必须说明一年或终身、point-in-time 或 through-the-cycle。
- LGD 应基于经济回收，包含回收金额、回收时点、处置成本和折现；“有抵押”不等于 LGD 为零。
- EAD 对定期贷款可能接近余额，对循环额度还需估计违约前继续提款的信用转换。
<!-- bilingual-en:start -->
- The denominator for PD is borrowers not yet in default; state whether it is one-year or lifetime and point-in-time or through-the-cycle.
- LGD should use economic recovery, including amount, timing, workout cost, and discounting. Secured does not mean zero LGD.
- EAD may be near current balance for a term loan, while a revolving facility requires estimating additional drawdown before default through a credit-conversion factor.
<!-- bilingual-en:end -->

### 算例与情景依赖
<!-- bilingual-en:start -->
*Worked example and scenario dependence*
<!-- bilingual-en:end -->

100 笔各 EAD 100 万的同类贷款，基准 PD 2%、LGD 40%，组合期望损失为 $100\times100万\times2\%\times40\%=80万$。衰退中若 PD 升到 5%、房产抵押品下跌使 LGD 升到 55%、额度提款使 EAD 平均升到 110 万，期望损失变为 302.5 万，不是只把原损失按 PD 比例放大。
<!-- bilingual-en:start -->
For 100 comparable loans each with EAD one million, base PD 2%, and LGD 40%, portfolio expected loss is $100\times1\text{ million}\times2\%\times40\%=0.8\text{ million}$. In recession, if PD rises to 5%, falling property collateral raises LGD to 55%, and line drawdowns raise average EAD to 1.1 million, expected loss becomes 3.025 million rather than merely scaling the original loss by PD.
<!-- bilingual-en:end -->

## 评级与迁移
<!-- bilingual-en:start -->
*Ratings and migration*
<!-- bilingual-en:end -->

评级是对信用质量的离散摘要，迁移矩阵给一段时间内等级变化。through-the-cycle 与 point-in-time 评级对周期反应不同。历史迁移率受样本、定义和 regime 影响，不能当固定自然常数。
<!-- bilingual-en:start -->
A rating is a discrete summary of credit quality, and a migration matrix gives grade changes over a period. Through-the-cycle and point-in-time ratings respond differently to the business cycle. Historical migration rates depend on sample, definition, and regime and are not fixed natural constants.
<!-- bilingual-en:end -->

迁移不仅影响是否违约，也会在违约前改变贷款或债券市值、保证金和限额。矩阵每行概率应加总为一，期限转换不能机械重复乘矩阵，除非接受 Markov 与时间同质等假设。
<!-- bilingual-en:start -->
Migration affects loan or bond value, margin, and limits before default, not only whether default occurs. Each transition-matrix row should sum to one. Converting horizons by repeated matrix multiplication is valid only under assumptions such as Markov behavior and time homogeneity.
<!-- bilingual-en:end -->

## 结构与强度模型
<!-- bilingual-en:start -->
*Structural and intensity models*
<!-- bilingual-en:end -->

Merton 型结构模型把公司资产低于债务阈值视为违约，连接股权期权性；reduced-form/强度模型直接建违约到达率。前者机制强但资产不可观测，后者校准灵活但经济结构较弱。
<!-- bilingual-en:start -->
A Merton-style structural model defines default through firm assets falling below a debt threshold and links credit to equity optionality. Reduced-form or intensity models specify default arrival directly. The former has a strong mechanism but unobservable firm assets; the latter calibrates flexibly but has weaker economic structure.
<!-- bilingual-en:end -->

结构模型最适合回答资本结构、资产波动与距离违约怎样连接；强度模型最适合匹配信用利差期限结构与随机违约时点。选择应由问题决定，不应因一种模型能拟合价格就断言其真实违约机制正确。
<!-- bilingual-en:start -->
Structural models are most useful for connecting capital structure, asset volatility, and distance to default. Intensity models are most useful for matching credit-spread term structures and random default timing. The question should determine the model; fitting prices does not prove that a model's real default mechanism is correct.
<!-- bilingual-en:end -->

## 从原主题保留的全局定位
<!-- bilingual-en:start -->
*Global orientation retained from the original topic*
<!-- bilingual-en:end -->

> **它解决什么：** 把借款人不履约的可能性、违约时损失和组合共同违约转成定价、限额和资本指标。
> **具体锚点：** 两笔预期损失相同的贷款，若其中一笔违约与经济衰退高度相关，它的意外损失和资本需求可能更高。
> **核心难点：** 预期损失 $PD\times LGD\times EAD$ 与尾部意外损失不同；评级、市场利差和真实违约概率也不是同一口径。
> **为什么重要：** 信贷定价、拨备、组合集中、CDS 和银行资本都依赖这些区分。
> **继续：** 先分解 PD/LGD/EAD，再看迁移、相关和组合模型；对手方随市场变化的敞口另见 [[对手方信用风险、CVA 与 DVA]]。
<!-- bilingual-en:start -->
> **What it solves:** It turns a borrower's probability of non-performance, loss at default, and joint portfolio default into pricing, limit, and capital measures.
> **Concrete anchor:** Two loans can have equal expected loss, yet the one whose default is more correlated with recession may have greater unexpected loss and capital need.
> **Central difficulty:** Expected loss $PD\times LGD\times EAD$ differs from tail unexpected loss; ratings, market spreads, and real-world default probabilities are also different conventions.
> **Why it matters:** Credit pricing, provisions, portfolio concentration, CDS, and bank capital depend on these distinctions.
> **Continue:** Decompose PD, LGD, and EAD first, then examine migration, dependence, and portfolio models. For counterparty exposure that changes with markets, see [[对手方信用风险、CVA 与 DVA|Counterparty Credit Risk, CVA, and DVA]].
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 混用一年 PD 与终身 LGD/EAD，却没有共同期限。
- 用正常期平均 LGD 压力测试衰退违约，忽略抵押品共同下跌和处置拥堵。
- 把评级当连续精确概率，遗漏评级目标、更新时间和组内异质性。
- 循环额度按当前余额当 EAD，忽略客户在违约前提款。
<!-- bilingual-en:start -->
- Combining one-year PD with lifetime LGD or EAD without a common horizon.
- Using normal-period average LGD for recession stress while ignoring joint collateral decline and workout congestion.
- Treating a rating as an exact continuous probability while ignoring rating objective, update frequency, and within-grade heterogeneity.
- Setting EAD on a revolving facility equal to current balance and ignoring pre-default drawdown.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用自己的话解释：为什么预期损失不是一笔贷款“最可能亏损的金额”？
<!-- bilingual-en:start -->
*Explain in your own words: why is expected loss not the most likely loss on one loan?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 单笔贷款通常要么不违约损失接近零，要么违约产生较大损失；PD×LGD×EAD 是这些状态按概率加权的平均，不必等于任何实际状态。
<!-- bilingual-en:start -->
> [!answer]- Answer
> One loan normally either survives with near-zero credit loss or defaults with a substantial loss. PD times LGD times EAD is the probability-weighted average of those states and need not equal any realized state.
<!-- bilingual-en:end -->

### 有抵押品为什么仍不能把 LGD 设为零？
<!-- bilingual-en:start -->
*Why does collateral not justify setting LGD to zero?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 抵押品价值会变化，处置有时间与成本，法律优先权可能不完整，且违约往往与抵押品价格下跌同时发生。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Collateral value changes, workout takes time and costs money, legal priority may be incomplete, and default often coincides with falling collateral prices.
<!-- bilingual-en:end -->

### 评级迁移矩阵为什么不能无条件重复乘来预测多年？
<!-- bilingual-en:start -->
*Why can a rating-transition matrix not be multiplied repeatedly without qualification to forecast many years?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 重复乘隐含转移只依赖当前等级且概率随时间不变；周期、评级历史和政策变化会破坏 Markov 与时间同质假设。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Repeated multiplication assumes transition depends only on current grade and probabilities remain constant. Cycles, rating history, and policy change violate Markov and time-homogeneity assumptions.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 已按共同期限与违约定义复核 PD、LGD、EAD 和迁移口径；新增衰退算例按三参数联合变化重新计算。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- PD, LGD, EAD, and migration conventions were checked under common horizons and default definitions; the added recession example was recomputed with joint movement in all three parameters.
<!-- bilingual-en:end -->

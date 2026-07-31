---
aliases:
  - "Counterparty Credit Risk"
  - "CVA"
  - "DVA"
  - "对手方信用风险"
status: source-checked
---

# 对手方信用风险、CVA 与 DVA
<!-- bilingual-en:start -->
*Counterparty Credit Risk, CVA, and DVA*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 衍生品未来市值会随市场变化，交易对手又可能在不利时违约；本主题把动态敞口与违约共同定价和管理。
> **具体锚点：** 利率互换今天价值为零不等于无信用敞口，未来利率变化可能使其对我方大幅为正，恰逢对手违约就有损失。
> **核心难点：** 敞口、PD、LGD 和市场因子可能相关；净额、抵押品和 margin period of risk 必须按法律集合建模。
> **为什么重要：** CVA 把对手信用成本纳入公允价值，资本和限额还需覆盖其波动与尾部。
> **继续：** 先建 exposure profile，再加入 default 与 recovery；监管口径见 [[巴塞尔银行资本与流动性监管|巴塞尔资本监管与 OTC 清算]]。
> <!-- bilingual-en:start -->
> **What it solves:** A derivative's future value changes with markets while the counterparty may default in adverse states; this topic jointly prices and manages dynamic exposure and default.
> **Concrete anchor:** A swap worth zero today is not free of credit exposure. Future rate changes can make it substantially positive to us precisely when the counterparty defaults.
> **Central difficulty:** Exposure, PD, LGD, and market factors can be dependent. Netting, collateral, and the margin period of risk must be modeled by legally enforceable sets.
> **Why it matters:** CVA incorporates counterparty credit cost into fair value, while capital and limits must also cover its volatility and tail.
> **Continue:** Build an exposure profile first, then add default and recovery. For regulatory conventions, see [[巴塞尔银行资本与流动性监管|Basel Bank Capital and Liquidity Regulation]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
> <!-- bilingual-en:start -->
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
> <!-- bilingual-en:end -->

## 信用敞口为何随市场变化
<!-- bilingual-en:start -->
*Why credit exposure changes with markets*
<!-- bilingual-en:end -->

贷款的 EAD 常主要由余额与额度决定；衍生品对手敞口则取决于未来重置价值。若交易对我方为负，对手违约不会让我们损失这笔负价值；若交易对我方为正，对手违约使我们只能按回收率获得价值。因此敞口用正部 $\max(V,0)$。
<!-- bilingual-en:start -->
A loan's EAD is often driven mainly by balance and facility size; derivative counterparty exposure depends on future replacement value. If a trade has negative value to us, counterparty default does not make us lose that negative value. If it has positive value, default leaves us recovering only part of it. Exposure therefore uses the positive part $\max(V,0)$.
<!-- bilingual-en:end -->

## 当前与潜在未来敞口
<!-- bilingual-en:start -->
*Current and potential future exposure*
<!-- bilingual-en:end -->

当前敞口是正的 replacement cost，负市值通常截为零信用敞口；PFE 是未来敞口分布高分位，EE 是期望正敞口，EPE 再按时间平均。指标服务限额、定价和资本的口径不同。
<!-- bilingual-en:start -->
Current exposure is positive replacement cost, with negative market value normally floored at zero for credit exposure. PFE is a high quantile of the future-exposure distribution, EE is expected positive exposure, and EPE averages EE over time. The measures serve different limit, pricing, and capital conventions.
<!-- bilingual-en:end -->

PFE 不应与 VaR 混为一谈：PFE 是未来某时点正敞口的分位，尚未乘违约与 LGD；VaR 通常是组合损失分布的分位。一个交易可以有高 PFE 但对手 PD 很低，也可以敞口较小却与违约高度 wrong-way。
<!-- bilingual-en:start -->
PFE should not be confused with VaR. PFE is a quantile of positive exposure at a future date before multiplying default and LGD; VaR is usually a quantile of portfolio loss. A trade can have high PFE with very low counterparty PD, or modest exposure with strong wrong-way dependence.
<!-- bilingual-en:end -->

## 净额、抵押品与补救期
<!-- bilingual-en:start -->
*Netting, collateral, and the margin period of risk*
<!-- bilingual-en:end -->

法律可执行的 netting set 内正负交易可抵消。variation margin 降低当前敞口，initial margin 覆盖违约到平仓期间风险；threshold、minimum transfer、频率和 dispute 会留下 gap。模拟必须在组合层应用协议。
<!-- bilingual-en:start -->
Positive and negative trades can offset within a legally enforceable netting set. Variation margin reduces current exposure; initial margin covers risk from default through closeout. Thresholds, minimum transfer amounts, frequency, and disputes leave a gap. Simulation must apply agreements at portfolio level.
<!-- bilingual-en:end -->

先逐交易模拟再把每笔正部相加会丢失净额效应；正确顺序通常是同一净额集合内先加总市值，应用抵押品和阈值，再取剩余正敞口。不同法律实体或协议不能因经济上相关便随意净额。
<!-- bilingual-en:start -->
Simulating each trade and then adding its positive part loses netting. The usual correct order is to aggregate market values within one netting set, apply collateral and thresholds, and only then take residual positive exposure. Different legal entities or agreements cannot be netted merely because they are economically related.
<!-- bilingual-en:end -->

## CVA
<!-- bilingual-en:start -->
*Credit valuation adjustment*
<!-- bilingual-en:end -->

简化单边 CVA 是未来各期 discounted expected exposure × marginal default probability × LGD 的和。实践使用风险中性市场/信用输入用于公允价值，并考虑 wrong-way risk、净额和 collateral。
<!-- bilingual-en:start -->
Simplified unilateral CVA is the sum across future periods of discounted expected exposure times marginal default probability times LGD. Fair-value practice uses risk-neutral market and credit inputs and incorporates wrong-way risk, netting, and collateral.
<!-- bilingual-en:end -->

在离散时点下可写为
<!-- bilingual-en:start -->
At discrete dates it can be written as
<!-- bilingual-en:end -->

$$
CVA\approx(1-R)\sum_i DF(t_i)\,EE^*(t_i)\,[PD(t_{i-1})-PD(t_i)],
$$

其中 $PD(t)$ 在这里表示生存概率时需按具体记号调整，$EE^*$ 是与违约联合的风险中性敞口。若一年内贴现 EE 为 10、边际违约概率 2%、LGD 60%，单期近似 CVA 为 0.12；但独立假设下的乘积在 wrong-way risk 时会低估。
<!-- bilingual-en:start -->
where notation must be adjusted if $PD(t)$ denotes survival probability, and $EE^*$ is risk-neutral exposure jointly considered with default. If one-year discounted EE is 10, marginal default probability 2%, and LGD 60%, a one-period approximation gives CVA 0.12; the product under independence understates CVA under wrong-way risk.
<!-- bilingual-en:end -->

## DVA 与双边调整
<!-- bilingual-en:start -->
*DVA and bilateral adjustment*
<!-- bilingual-en:end -->

DVA 反映自身信用恶化使本方负债公允价值下降的会计/定价调整，经济解释有争议且无法无摩擦兑现。双边 CVA/DVA 需一致处理谁先违约和 close-out。
<!-- bilingual-en:start -->
DVA reflects the accounting or valuation effect by which one's own credit deterioration reduces the fair value of liabilities. Its economic interpretation is controversial and it cannot be realized frictionlessly. Bilateral CVA and DVA must treat first-to-default and closeout consistently.
<!-- bilingual-en:end -->

## wrong-way risk
<!-- bilingual-en:start -->
*Wrong-way risk*
<!-- bilingual-en:end -->

当对手更可能违约时我方敞口也更高，称 wrong-way risk；例如商品生产商在商品价格暴跌时既信用恶化又对某衍生品负担加重。独立假设会低估风险。
<!-- bilingual-en:start -->
Wrong-way risk occurs when our exposure is higher precisely when the counterparty is more likely to default. For example, after a commodity-price collapse, a producer's credit quality can deteriorate while its liability on a derivative increases. An independence assumption understates risk.
<!-- bilingual-en:end -->

特定 wrong-way risk 来自交易与对手的直接经济联系，通常能设计针对性情景；一般 wrong-way risk 来自共同宏观状态，需要因子相关或联合模型。仅用历史线性相关可能在危机样本稀少时看不见两者。
<!-- bilingual-en:start -->
Specific wrong-way risk comes from a direct economic link between trade and counterparty and can often be tested with targeted scenarios. General wrong-way risk comes from shared macro states and requires factor dependence or joint models. Historical linear correlation may miss both when crisis observations are sparse.
<!-- bilingual-en:end -->

## Monte Carlo 与 Greeks
<!-- bilingual-en:start -->
*Monte Carlo and Greeks*
<!-- bilingual-en:end -->

模拟市场路径、重估交易、应用 collateral/netting，再与违约模型整合得到 exposure 和 CVA。CVA Greeks 衡量利率、信用利差和波动变化，产生 CVA market risk 和 hedge basis。
<!-- bilingual-en:start -->
Simulate market paths, revalue trades, apply collateral and netting, and then combine with default models to obtain exposure and CVA. CVA Greeks measure sensitivity to rates, credit spreads, and volatility, creating CVA market risk and hedge basis.
<!-- bilingual-en:end -->

## 限额与治理
<!-- bilingual-en:start -->
*Limits and governance*
<!-- bilingual-en:end -->

同时管理当前/PFE、wrong-way、集中、评级触发和 collateral liquidity。模型校准、法律意见和数据质量与公式同等重要。
<!-- bilingual-en:start -->
Manage current exposure and PFE, wrong-way risk, concentration, rating triggers, and collateral liquidity together. Model calibration, legal opinions, and data quality are as important as formulas.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 因交易当前价值为零便认为没有对手风险。
- 在交易层分别取正敞口再相加，漏掉法律净额集合。
- 抵押品每天交换便把敞口设为零，遗漏门槛、争议和补救期跳跃。
- 用 EE×PD×LGD 独立乘积处理与违约同向变化的敞口。
<!-- bilingual-en:start -->
- Assuming no counterparty risk because a trade's current value is zero.
- Taking positive exposure trade by trade before aggregation and omitting the legal netting set.
- Setting exposure to zero because collateral is exchanged daily while omitting thresholds, disputes, and margin-period jumps.
- Using the independent product EE times PD times LGD when exposure rises with default likelihood.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 互换初始价值为零为什么仍有对手风险？
<!-- bilingual-en:start -->
*Why does a swap with zero initial value still have counterparty risk?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 未来市场变化会使 replacement value 为正，而对手可能在那时违约；风险来自未来敞口分布。
> <!-- bilingual-en:start -->
> Future market changes can make replacement value positive, and the counterparty may default then; risk comes from the future exposure distribution.
> <!-- bilingual-en:end -->

### wrong-way risk 是什么？
<!-- bilingual-en:start -->
*What is wrong-way risk?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 对手违约可能性升高的状态恰好也是我方对其敞口升高的状态。
> <!-- bilingual-en:start -->
> It is the state in which counterparty default likelihood rises at the same time as our exposure to that counterparty rises.
> <!-- bilingual-en:end -->

### 净额与抵押品是否把敞口降到零？
<!-- bilingual-en:start -->
*Do netting and collateral reduce exposure to zero?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 通常不会；估值变化、门槛、转移滞后、争议和补救期会留下 gap，且法律可执行性是前提。
> <!-- bilingual-en:start -->
> Usually not. Valuation change, thresholds, transfer delays, disputes, and the margin period leave a gap, and legal enforceability is a prerequisite.
> <!-- bilingual-en:end -->

### 用自己的话解释：为什么 PFE 不等于信用损失分位？
<!-- bilingual-en:start -->
*Explain in your own words: why is PFE not a credit-loss quantile?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> PFE 只描述未来正重置价值的高分位，尚未加入对手是否违约和违约时回收；信用损失还需把敞口与违约、LGD 联合。
> <!-- bilingual-en:start -->
> PFE is only a high quantile of future positive replacement value before counterparty default and recovery. Credit loss must combine exposure jointly with default and LGD.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 已逐项核验 current exposure、EE/EPE/PFE、净额与抵押品顺序、CVA/DVA 和 wrong-way risk；单期 CVA 算例按贴现敞口×边际违约×LGD 复算。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- Current exposure, EE, EPE, PFE, netting and collateral order, CVA, DVA, and wrong-way risk were checked item by item; the one-period CVA example was recomputed as discounted exposure times marginal default probability times LGD.
<!-- bilingual-en:end -->

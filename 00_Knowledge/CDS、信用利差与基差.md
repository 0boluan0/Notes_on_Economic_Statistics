---
aliases:
  - "Credit Default Swaps, Credit Spreads, and Basis"
  - "CDS"
  - "信用违约互换"
status: source-checked
---

# CDS、信用利差与基差
<!-- bilingual-en:start -->
*Credit Default Swaps, Credit Spreads, and Basis*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 解释信用保护买卖双方怎样通过 CDS 转移参考实体的信用事件风险，以及 CDS 利差、债券利差和违约概率如何连接又为何会偏离。
> **具体锚点：** 保护买方定期支付 spread；参考实体发生合同定义的信用事件并结算时，保护卖方补偿合格债务的损失。
> **核心难点：** CDS 报价隐含风险中性违约与回收假设，还含流动性、融资、对手方和技术供需；不能用 spread/LGD 直接当真实 PD。
> **为什么重要：** CDS 用于对冲、交易和价格发现，也会产生基差、wrong-way risk 与结算条款风险。
> **继续：** 单体参数基础见 [[信用风险：PD、LGD、EAD 与评级迁移]]；对手方与 CVA 见 [[对手方信用风险、CVA 与 DVA]]。
> <!-- bilingual-en:start -->
> **What it solves:** It explains how CDS protection buyers and sellers transfer credit-event risk on a reference entity and how CDS spreads, bond spreads, and default probability are connected yet can diverge.
> **Concrete anchor:** The protection buyer pays a periodic spread; if a contractually defined credit event occurs and settles, the protection seller compensates loss on eligible reference obligations.
> **Central difficulty:** A CDS quote embeds risk-neutral default and recovery assumptions plus liquidity, funding, counterparty effects, and technical supply and demand. Spread divided by LGD is not a real-world PD.
> **Why it matters:** CDS support hedging, trading, and price discovery while creating basis, wrong-way risk, and settlement-term risk.
> **Continue:** For single-name parameter foundations, see [[信用风险：PD、LGD、EAD 与评级迁移|Credit Risk: PD, LGD, EAD, and Rating Migration]]. For counterparty risk and CVA, see [[对手方信用风险、CVA 与 DVA|Counterparty Credit Risk, CVA, and DVA]].
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

## 信用利差与 CDS
<!-- bilingual-en:start -->
*Credit spreads and CDS*
<!-- bilingual-en:end -->

CDS protection buyer 支付 spread，违约/信用事件时获得补偿。简化下 spread 与风险中性 PD、LGD 相关，但还含流动性、对手风险和技术供需。cash bond spread 与 CDS spread 的 basis 可偏离零。
<!-- bilingual-en:start -->
A CDS protection buyer pays a spread and receives compensation after default or another covered credit event. In a simplification, spread is related to risk-neutral PD and LGD, but it also reflects liquidity, counterparty risk, and technical supply and demand. The basis between a cash-bond spread and a CDS spread can differ from zero.
<!-- bilingual-en:end -->

CDS 有两条腿：premium leg 是存续期间定期支付及违约时应计保费；protection leg 是信用事件后的损失补偿。公平 spread 使两腿风险中性现值相等。期限结构校准需要逐期生存概率、贴现和回收，而不是只看一个年化利差。
<!-- bilingual-en:start -->
A CDS has two legs: the premium leg consists of periodic payments while the name survives plus accrued premium at default; the protection leg pays credit-event loss. The fair spread equates the two risk-neutral present values. Term-structure calibration requires period-by-period survival probabilities, discounting, and recovery rather than one annualized spread.
<!-- bilingual-en:end -->

### hazard-rate 的简化近似
<!-- bilingual-en:start -->
*A simplified hazard-rate approximation*
<!-- bilingual-en:end -->

若违约强度与回收率近似恒定、利差较小且忽略应计等细节，$s\approx\lambda(1-R)$。例如 spread 200bp、回收 40%，得到风险中性 hazard 约 $2\%/60\%=3.33\%$ 每年。这只是初始量级，不能称为真实世界一年 PD。
<!-- bilingual-en:start -->
If default intensity and recovery are approximately constant, spreads are small, and accrual details are ignored, $s\approx\lambda(1-R)$. A 200-basis-point spread with 40% recovery gives a risk-neutral hazard rate of about $2\%/60\%=3.33\%$ per year. This is only an initial order-of-magnitude estimate, not a real-world one-year PD.
<!-- bilingual-en:end -->

## 债券—CDS 基差
<!-- bilingual-en:start -->
*The bond–CDS basis*
<!-- bilingual-en:end -->

粗略定义 basis 为 CDS spread 减去可比债券信用利差。理论复制要求同一参考实体、期限、优先级、回收、融资与可交割条款；现实中的 repo 融资、债券稀缺、流动性、对手方、结算选择和资产负债表成本使基差持续偏离零。
<!-- bilingual-en:start -->
Roughly define basis as CDS spread minus the credit spread on a comparable bond. The theoretical replication requires the same reference entity, maturity, seniority, recovery, financing, and deliverability terms. Repo funding, bond scarcity, liquidity, counterparty risk, settlement options, and balance-sheet cost can keep real-world basis away from zero.
<!-- bilingual-en:end -->

负基差交易通常买债券、买 CDS 保护并融资债券，试图锁定债券较宽利差；但它不是无风险套利。融资可以到期或提价，可交割券和结算价值会变化，CDS 对手也可能违约，债券与 CDS 的定义并非完全匹配。
<!-- bilingual-en:start -->
A negative-basis trade commonly buys the bond, buys CDS protection, and finances the bond to capture its wider spread. It is not riskless arbitrage. Funding can mature or reprice, deliverables and settlement value can change, the CDS counterparty can default, and bond and CDS definitions are not perfectly identical.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 用 CDS spread 除 LGD 直接报告历史真实违约概率。
- 对冲债券却没有匹配期限、优先级、币种和可交割义务。
- 把基差偏离零称无风险套利，忽略融资、流动性和对手风险。
- 只按名义本金对冲，不按违约损失、回收和市场价值敏感度校准。
<!-- bilingual-en:start -->
- Dividing CDS spread by LGD and reporting the result as historical real-world default probability.
- Hedging a bond without matching maturity, seniority, currency, and deliverable obligations.
- Calling a nonzero basis riskless arbitrage while ignoring funding, liquidity, and counterparty risk.
- Hedging only by notional rather than calibrating default loss, recovery, and market-value sensitivity.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### CDS spread 能否直接除以 LGD 得真实违约概率？
<!-- bilingual-en:start -->
*Can CDS spread be divided directly by LGD to obtain real-world default probability?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 只能在很强简化下近似风险中性概率；现实还含期限结构、流动性、风险溢价和对手因素。
> <!-- bilingual-en:start -->
> Only as an approximation to a risk-neutral probability under strong simplifications; reality also contains term structure, liquidity, risk premium, and counterparty effects.
> <!-- bilingual-en:end -->

### 用自己的话解释：保护买方支付 spread 换到了什么？
<!-- bilingual-en:start -->
*Explain in your own words: what does the protection buyer receive in exchange for paying the spread?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在参考实体发生合同覆盖的信用事件时，按结算机制获得合格债务价值损失的补偿；保护不是对所有市场跌价或任何借款人问题无条件赔付。
> <!-- bilingual-en:start -->
> If the reference entity experiences a contractually covered credit event, the buyer receives compensation for loss in value on eligible obligations under the settlement mechanism. Protection does not cover every market decline or every borrower problem unconditionally.
> <!-- bilingual-en:end -->

### 为什么相同公司的债券利差与 CDS spread 可以不同？
<!-- bilingual-en:start -->
*Why can the bond spread and CDS spread of the same company differ?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 两市场的融资、流动性、供需、对手方、合同条款和资产负债表成本不同，理论复制条件不完整，所以基差不必为零。
> <!-- bilingual-en:start -->
> Funding, liquidity, supply and demand, counterparty risk, contractual terms, and balance-sheet costs differ across the two markets, so theoretical replication is incomplete and basis need not be zero.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 已逐项核验 CDS premium/protection leg、风险中性校准与 cash–CDS basis；hazard 近似按公平两腿的一阶关系复算。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- CDS premium and protection legs, risk-neutral calibration, and cash–CDS basis were checked item by item; the hazard approximation was recomputed from the first-order fair-leg relation.
<!-- bilingual-en:end -->

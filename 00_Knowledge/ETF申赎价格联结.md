---
aliases:
  - 可执行的ETF创设赎回交易能约束市价偏离资产池价值但不保证零折溢价
  - ETF creation-redemption price linkage
student_os: knowledge-atom
atom_id: FI-SEC-027
atom_type: mechanism
status: source-checked
requires:
  - "[[ETF]]"
  - "[[基金折溢价]]"
related:
  - "[[套利限制]]"
  - "[[套利与价差交易]]"
part_of:
  - "[[证券工具与现金流权利.canvas]]"
---

# 可执行的ETF创设赎回交易能约束市价偏离资产池价值但不保证零折溢价
<!-- bilingual-en:start -->
*Executable ETF creation and redemption can constrain deviations from pool value without guaranteeing zero premiums or discounts*
<!-- bilingual-en:end -->

[[ETF]]的创设、赎回提供了份额与申赎对价之间的转换渠道。若有资格的参与者能够完成交易、锁定两边价格且收益超过全部成本，溢价时可买入所需资产并创设份额后卖出；折价时可买入份额并赎回资产后出售。新增卖盘或买盘以及份额供给变化，形成纠偏力量。
<!-- bilingual-en:start -->
[[ETF]] creation and redemption connect shares to the required consideration. If an eligible participant can complete the exchange, lock in both sides, and cover all costs, a premium can support buying the required assets, creating shares, and selling them; a discount can support buying shares, redeeming them, and selling the received assets. The resulting orders and changes in share supply provide corrective pressure.
<!-- bilingual-en:end -->

这里应比较可执行的申赎篮子及现金调整，不是机械比较某个公布的[[基金份额净值]]。用相同数量份额计量，在实物模式下，溢价交易的毛差额是“份额可售收入减创设对价可购成本”；折价交易的毛差额是“赎回所得资产可售收入减份额可购成本”。实际净收益还要扣尚未计入这些价格的申赎费、佣金、融资、借券及交收等成本；若已用可成交买卖价，不能再重复扣其价差。现金模式按其实际现金定价与到账条件核算。
<!-- bilingual-en:start -->
The relevant comparison uses executable basket values and cash adjustments, not a mechanical subtraction from published [[基金份额净值|NAV]]. For a matched number of shares in an in-kind exchange, the premium trade's gross spread is share-sale proceeds less the purchase cost of creation consideration; the discount trade's is sale proceeds from redeemed assets less the cost of buying shares. Net returns deduct creation/redemption fees, commissions, financing, borrowing, and settlement costs not already included in those prices. Spreads must not be deducted again when executable bids and asks already incorporate them. Cash exchanges require their actual cash-pricing and payment terms.
<!-- bilingual-en:end -->

课程的“市价 27 元、净值 28 元、10 万份”例，若这 10 万份满足赎回单位与资格要求，能按每份 27 元买入且赎回所得确能按每份 28 元变现，并锁定相关价格，则毛差额可锁定为 $100{,}000(28-27)=100{,}000$ 元。若尚未计入的全部成本为 2 万元，净差额为 8 万元；若净值已过时、资产不能成交或任一价格尚未锁定，10 万元就只是纸面差额，不是已保证的套利利润。
<!-- bilingual-en:start -->
In the course's price-27, NAV-28, 100,000-share example, suppose the shares meet redemption-size and eligibility requirements, can be bought at 27, and deliver assets that can actually be sold at 28 per share, with those prices locked in. The locked gross spread is then $100{,}000(28-27)=100{,}000$. All remaining costs of 20,000 reduce the net spread to 80,000. Stale NAV, untradeable assets, or unlocked prices turn the quoted spread into a paper comparison rather than guaranteed arbitrage profit.
<!-- bilingual-en:end -->

因此该机制约束的是满足执行条件后的偏离空间，不是证明 $P=v$ 永远成立。市场分割、暂停申赎、底层流动性下降或参与者资本约束可削弱联结；不能仅从 AP 有交易渠道就推出每个状态下都会实施纠偏。若需承担未对冲风险，应按[[套利与价差交易|风险价差交易]]理解，而非自动视为无风险套利。
<!-- bilingual-en:start -->
The mechanism constrains deviations after execution conditions are considered; it does not prove $P=v$ at every instant. Market segmentation, suspended creations or redemptions, illiquid underlying assets, or participant capital constraints can weaken the link. Access alone does not establish that an AP will correct deviations in every state. When unhedged risk remains, the position is a [[套利与价差交易|risky spread trade]], not automatically riskless arbitrage.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [SEC, Updated Investor Bulletin: ETFs, 2023，How ETFs Work / Trading Prices](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-24)：已重开，支持 AP 创设赎回、现金/实物对价及价格联结机制；未把其通常接近 NAV 的描述读成恒等式。[SEC 2019 ETF 最终规则说明，PDF／印刷第 11–14 页](https://www.sec.gov/files/rules/final/2019/33-10695.pdf#page=11)亦已实际读取，支持篮子、费用、双向纠偏和对冲日内风险的条件。
<!-- bilingual-en:start -->
- The reopened bulletin supports AP creation/redemption, cash or in-kind consideration, and price linkage without turning usual proximity to NAV into an identity. The 2019 final-rule explanation, PDF / printed pp. 11–14, was also read for baskets, fees, corrective trades in both directions, and intraday hedging conditions.
<!-- bilingual-en:end -->
- [[02_Economy/06_证券投资学/证券投资学.pdf#page=147|证券投资学课程 PDF 第 147–149 页]]：已重开，第 149 页算例已目视。保留原 27、28、10 万份参数，补齐成交、资格、单位与成本条件；2 万元成本为自拟对照并复算。执行和融资约束沿用已核验的[[套利限制]]，不另建重复机制。
<!-- bilingual-en:start -->
- The course mechanism and example were reopened, with the numerical slide visually checked. Its original parameters are retained with explicit execution, eligibility, unit-size, and cost conditions; the 20,000 cost is a recomputed illustrative extension. Execution and funding qualifications reuse [[套利限制|limits to arbitrage]].
<!-- bilingual-en:end -->

---
aliases:
  - "OTC Derivatives Clearing, Margin, and CCP Risk"
  - "Central Counterparty Risk"
  - "CCP 风险"
  - "OTC 清算"
status: source-checked
---

# OTC 衍生品清算、保证金与 CCP 风险
<!-- bilingual-en:start -->
*OTC Derivatives Clearing, Margin, and CCP Risk*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 解释中央清算、双边/多边净额、变动与初始保证金怎样重组 OTC 对手风险，以及风险为何从双边网络集中到 CCP 与流动性需求。
> **具体锚点：** 变动保证金补上已经发生的市值变化，初始保证金覆盖成员违约到平仓期间的潜在变化；两者功能不同。
> **核心难点：** CCP 不消灭风险，而是改变净额、抵押品、违约瀑布和共同依赖；压力期同步追缴保证金可放大流动性冲击。
> **为什么重要：** 衍生品基础设施决定违约如何传播、谁先承担损失，以及市场在成员失败时能否继续运行。
> **继续：** 先读 [[对手方信用风险、CVA 与 DVA]] 理解动态敞口，再沿“成员 → 保证金 → 违约基金 → recovery/resolution”检查 CCP。
> <!-- bilingual-en:start -->
> **What it solves:** It explains how central clearing, bilateral or multilateral netting, and variation and initial margin reorganize OTC counterparty risk and why risk migrates from a bilateral network into CCP concentration and liquidity demand.
> **Concrete anchor:** Variation margin covers market-value changes already realized, while initial margin covers potential change from member default through closeout. They serve different functions.
> **Central difficulty:** A CCP does not eliminate risk; it changes netting, collateral, the default waterfall, and common dependence. Synchronized margin calls can amplify liquidity stress.
> **Why it matters:** Derivatives infrastructure determines how default propagates, who absorbs loss first, and whether markets continue operating after a member failure.
> **Continue:** First read [[对手方信用风险、CVA 与 DVA|Counterparty Credit Risk, CVA, and DVA]] for dynamic exposure, then audit the chain member → margin → default fund → recovery or resolution.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
> - CPMI–IOSCO《[Principles for financial market infrastructures](https://www.bis.org/cpmi/publ/d101a.pdf)》：核验 CCP 风险管理、违约资源与治理原则。
> <!-- bilingual-en:start -->
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
> - CPMI–IOSCO's “[Principles for financial market infrastructures](https://www.bis.org/cpmi/publ/d101a.pdf)” verifies CCP risk management, default resources, and governance principles.
> <!-- bilingual-en:end -->

## 双边网络与中央清算
<!-- bilingual-en:start -->
*Bilateral networks and central clearing*
<!-- bilingual-en:end -->

双边 OTC 中，每对交易方依法律协议净额并交换抵押品；中央清算后，CCP 通过 novation 成为每个卖方的买方、每个买方的卖方。多边净额可降低总敞口与网络复杂度，但所有成员对同一基础设施产生共同依赖。
<!-- bilingual-en:start -->
In bilateral OTC markets, each pair of counterparties nets under legal agreements and exchanges collateral. After central clearing, novation makes the CCP buyer to every seller and seller to every buyer. Multilateral netting can reduce gross exposure and network complexity while creating common dependence on one infrastructure.
<!-- bilingual-en:end -->

中央清算是否降低总风险取决于产品集合、跨产品净额损失、保证金、成员质量和 CCP 治理。将原可跨产品双边净额的头寸分到多个 CCP 可能增加抵押品需求；不能只比较交易名义本金。
<!-- bilingual-en:start -->
Whether central clearing reduces total risk depends on product scope, loss of cross-product netting, margin, member quality, and CCP governance. Splitting positions that previously netted bilaterally across multiple CCPs can increase collateral needs. Notional comparison alone is insufficient.
<!-- bilingual-en:end -->

## OTC、CCP 与保证金
<!-- bilingual-en:start -->
*OTC, CCPs, and margin*
<!-- bilingual-en:end -->

标准化衍生品中央清算、交易报告和双边保证金减少不透明双边网络。variation margin 覆盖已发生市值变化，initial margin 覆盖违约到平仓的潜在变化。净额结算和抵押品降低敞口但产生流动性需求。
<!-- bilingual-en:start -->
Central clearing of standardized derivatives, trade reporting, and bilateral margin reduce opaque bilateral networks. Variation margin covers realized market-value change, while initial margin covers potential change from default through closeout. Net settlement and collateral reduce exposure but create liquidity demand.
<!-- bilingual-en:end -->

若成员一天亏损 8，变动保证金通常将 8 转给盈利方；若成员随后违约，初始保证金用于覆盖上次收取 VM 到头寸完成平仓期间的新损失。模型若只覆盖正常期波动，市场跳跃和流动性折价会穿透 IM。
<!-- bilingual-en:start -->
If a member loses 8 in one day, variation margin normally transfers the 8 to the winning side. If the member then defaults, initial margin covers new loss from the last VM collection through completion of closeout. If the model reflects only normal-period volatility, market jumps and liquidity discounts can breach IM.
<!-- bilingual-en:end -->

## CCP 风险与 waterfall
<!-- bilingual-en:start -->
*CCP risk and the default waterfall*
<!-- bilingual-en:end -->

CCP 集中风险并用违约基金、成员出资和 recovery waterfall 分摊损失。它降低网络复杂度但可能成为系统关键节点；错误模型、集中头寸和同时追缴保证金可放大压力。
<!-- bilingual-en:start -->
A CCP concentrates risk and allocates loss through default funds, member contributions, and a recovery waterfall. It reduces network complexity but can become a systemically critical node; model error, concentrated positions, and simultaneous margin calls can amplify stress.
<!-- bilingual-en:end -->

具体顺序依 CCP 规则，但概念上先使用违约成员的保证金和违约基金出资，再由 CCP 自有资金与非违约成员共同资源按规则承担，极端时进入 assessment、variation-margin gains haircutting、合约 tear-up 或 resolution。分析必须以实际规则为准，不能背一个通用瀑布。
<!-- bilingual-en:start -->
Exact order depends on CCP rules, but conceptually the defaulter's margin and default-fund contribution are used first, followed under specified rules by CCP equity and mutualized resources of non-defaulting members. Extreme cases may invoke assessments, variation-margin-gains haircutting, contract tear-up, or resolution. Analysis must use actual rules rather than one generic waterfall.
<!-- bilingual-en:end -->

## 流动性与顺周期
<!-- bilingual-en:start -->
*Liquidity and procyclicality*
<!-- bilingual-en:end -->

市场剧烈变化时 VM 现金需求立即出现，IM 模型也可能提高要求；多个 CCP 和双边对手同时追缴会迫使成员出售资产。保证金降低信用敞口，却可能把信用风险转成流动性与火售风险，因此需预置高质量抵押品和转换渠道。
<!-- bilingual-en:start -->
During sharp market moves, cash demand from VM arrives immediately and IM models may also raise requirements. Simultaneous calls from multiple CCPs and bilateral counterparties can force asset sales. Margin reduces credit exposure but can transform it into liquidity and fire-sale risk, requiring prepositioned high-quality collateral and reliable transformation channels.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 中央清算后把对手风险设为零，未分析 CCP 与成员共同资源。
- 把 VM 和 IM 都称为违约损失缓冲，未区分已发生与未来补救期变化。
- 只算净敞口下降，未算跨 CCP 抵押品与同步追缴。
- 用“违约基金很大”判断安全，未将其与成员集中、压力损失和 waterfall 规则比较。
<!-- bilingual-en:start -->
- Setting counterparty risk to zero after central clearing without analyzing the CCP and mutualized member resources.
- Calling both VM and IM buffers against the same default loss without distinguishing realized change from future margin-period change.
- Calculating lower net exposure without collateral fragmentation across CCPs and synchronized calls.
- Judging safety from a large default fund without comparing it with member concentration, stressed loss, and waterfall rules.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### CCP 是否消除了对手风险？
<!-- bilingual-en:start -->
*Does a CCP eliminate counterparty risk?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 没有。它通过净额和保证金重组、集中风险，并引入对 CCP 模型、成员和流动性 waterfall 的依赖。
> <!-- bilingual-en:start -->
> No. It reorganizes and concentrates risk through netting and margin and creates dependence on CCP models, members, and the liquidity waterfall.
> <!-- bilingual-en:end -->

### 用自己的话解释：VM 与 IM 各覆盖哪段风险？
<!-- bilingual-en:start -->
*Explain in your own words: which interval of risk do VM and IM cover?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> VM 结算截至当前已经发生的市值变化；IM 为成员违约后到头寸平仓或重新对冲期间仍可能发生的额外变化提供缓冲。
> <!-- bilingual-en:start -->
> VM settles market-value change already realized through the present; IM buffers additional change that can occur after member default until positions are closed or rehedged.
> <!-- bilingual-en:end -->

### 为什么提高保证金既降低信用风险又可能增加系统压力？
<!-- bilingual-en:start -->
*Why can higher margin reduce credit risk yet increase systemic stress?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 更多抵押品降低违约后未覆盖敞口，但压力期同步追加会抽走现金、迫使火售并把冲击传给其他市场。
> <!-- bilingual-en:start -->
> More collateral reduces uncovered exposure after default, but synchronized calls under stress drain cash, force fire sales, and transmit shocks to other markets.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- CPMI–IOSCO《[Principles for financial market infrastructures](https://www.bis.org/cpmi/publ/d101a.pdf)》：逐项核验法律基础、信用/流动性风险、保证金、违约资源和治理；waterfall 明确保留为依 CCP 规则而变。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- CPMI–IOSCO's “[Principles for financial market infrastructures](https://www.bis.org/cpmi/publ/d101a.pdf)” was checked for legal basis, credit and liquidity risk, margin, default resources, and governance; the waterfall is explicitly left contingent on actual CCP rules.
<!-- bilingual-en:end -->

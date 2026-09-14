# **OTC衍生产品清算与监管学习笔记**
<!-- bilingual-en:start -->
*Study Notes on Clearing and Regulation of OTC Derivatives*
<!-- bilingual-en:end -->

> [!summary] 课程地图
> 本章依次区分交易执行、中央清算、保证金、违约管理、恢复、处置和资本处理。详细边界见 [[OTC清算与CCP风险.canvas|OTC清算与CCP风险]]。
<!-- bilingual-en:start -->
> This chapter separates trade execution, central clearing, margin, default management, recovery, resolution, and capital treatment. See [[OTC清算与CCP风险.canvas|OTC clearing and CCP risk]] for the detailed map.
<!-- bilingual-en:end -->

## **1. OTC衍生品清算机制**
<!-- bilingual-en:start -->
*1. OTC Derivatives Clearing Mechanisms*
<!-- bilingual-en:end -->

- **双边清算（bilateral clearing）**：双方在同一可执行主协议与净额集合内管理付款、担保品和违约后的 close-out netting。ISDA 主协议和 CSA 很常见，但具体权利取决于所选版本、交易确认书、适用法律与法律意见；不能把所有 OTC 交易都假定为同一份 ISDA 条款。
<!-- bilingual-en:start -->
- **Bilateral clearing:** The parties manage payments, collateral, and close-out netting within a legally enforceable master agreement and netting set. An ISDA Master Agreement and CSA are common, but the precise rights depend on the chosen documents, confirmations, governing law, and legal opinions; not every OTC trade has identical ISDA terms.
<!-- bilingual-en:end -->

- **中央对手方清算（CCP clearing）**：CCP 不是撮合者。交易场所或其他执行系统先决定交易是否成交；交易符合接纳条件后，CCP 才通过 **novation（更新契约）**、**open offer（公开要约）**或其他有约束力的法律机制，成为每个卖方的买方和每个买方的卖方。交易执行、中央清算和证券上市是三个不同判断，即使交易所与 CCP 属于同一集团也不能混写。见 [[中央清算与多边净额]]。
<!-- bilingual-en:start -->
- **Central counterparty clearing:** A CCP is not a matching engine. A venue or another execution system first determines whether a trade is executed; only after the trade satisfies acceptance conditions does the CCP become buyer to every seller and seller to every buyer through **novation**, **open offer**, or another legally binding mechanism. Execution, central clearing, and exchange listing are distinct questions even when an exchange and CCP belong to the same group. See [[中央清算与多边净额|Central clearing and multilateral netting]].
<!-- bilingual-en:end -->

## **2. 保证金制度**
<!-- bilingual-en:start -->
*2. The Margin System*
<!-- bilingual-en:end -->

- **变动保证金（variation margin, VM）**：VM 根据组合市值变化发生支付或担保品转移，但不保证当前敞口被全额、即时覆盖。若协议把 VM 作为 collateralised-to-market（CTM）担保品，计算时要避免与市值重复扣减；若把 VM 作为 settled-to-market（STM）结算付款，它会重置合约价值，不能再次当作抵押品扣除。threshold、minimum transfer amount（MTA）、估值争议、付款频率与结算时差都可能留下 gap。
<!-- bilingual-en:start -->
- **Variation margin (VM):** VM transfers payments or collateral as portfolio values change, but it does not guarantee complete and instantaneous coverage of current exposure. Under collateralised-to-market (CTM), VM remains collateral and must not be double-counted against value; under settled-to-market (STM), the payment resets contract value and must not also be deducted as collateral. Thresholds, minimum transfer amounts (MTAs), valuation disputes, call frequency, and settlement gaps can leave residual exposure.
<!-- bilingual-en:end -->

- **初始保证金（initial margin, IM）**：IM 为最后一次 VM 交换后到完成平仓、替换或重新对冲之间的潜在变化提供预付缓冲。保证金风险期（margin period of risk, MPOR）、置信水平、压力校准、产品范围、组合抵销、集中度附加、合格抵押品和隔离方式都依产品、组合、CCP 规则或当地未清算保证金规则而异；不能把“场内 3–5 天、双边 10 天、统一 99%”写成全球通则。现金 IM 或 VM 是否计息也由合同、账户和当地规则决定，没有统一的“期货/OTC”计息规则。见 [[净额与抵押品]]。
<!-- bilingual-en:start -->
- **Initial margin (IM):** IM is prefunded protection against potential change between the last VM exchange and completion of close-out, replacement, or re-hedging. The margin period of risk (MPOR), confidence level, stress calibration, product scope, portfolio offsets, concentration add-ons, eligible collateral, and segregation vary by product, portfolio, CCP rules, or local uncleared-margin rules. “Three to five days for exchange trades, ten days bilaterally, and 99% everywhere” is not a global rule. Interest on cash IM or VM also depends on the contract, account, and local rules rather than a universal futures/OTC convention. See [[净额与抵押品|Netting and collateral]].
<!-- bilingual-en:end -->

- **流动性含义**：VM 通常把已经发生的市值变化迅速转成短期付款；IM 上升通常占用更多抵押品，而不是第二笔交易损失。压力期的 VM 流出、IM 上调和抵押品折价可能共同放大融资与火售压力，但这种反馈取决于市场和风险管理条件。见 [[保证金顺周期]]。
<!-- bilingual-en:start -->
- **Liquidity meaning:** VM generally turns realised mark-to-market change into a short-dated payment; an IM increase normally ties up more collateral rather than creating a second trading loss. In stress, VM outflows, IM increases, and collateral haircuts can amplify funding and fire-sale pressure, but the feedback depends on market conditions and risk management. See [[保证金顺周期|Margin procyclicality]].
<!-- bilingual-en:end -->

## **3. CCP 结构与风险承担**
<!-- bilingual-en:start -->
*3. CCP Structure and Loss Bearing*
<!-- bilingual-en:end -->

- **CCP 结构**：CCP 接纳交易后，以两条法律头寸替代或直接形成原双边关系，并按具体清算服务、账户和规则收取 VM、IM 与违约基金出资。它把原双边对手风险重组为对 CCP 的敞口和成员间共同资源风险，而不是消灭风险。
<!-- bilingual-en:start -->
- **CCP structure:** Once a trade is accepted, the CCP replaces or directly forms the bilateral relationship with two legal positions and collects VM, IM, and default-fund contributions under the relevant service, account, and rules. It reorganises bilateral counterparty risk into exposures to the CCP and mutualised member-resource risk; it does not eliminate risk.
<!-- bilingual-en:end -->

- **违约瀑布（default waterfall）**：全球没有统一的完整顺序。最后一次已结算 VM 已经影响损失基数，不能再列作预融资层；到期未付 VM 是违约义务，VM gains haircutting 则是可能的恢复工具。应分别识别违约成员 IM、违约成员违约基金出资、指定的 CCP 自有资金（skin in the game）、非违约成员共同违约基金，以及追加出资或其他后续资源。CCP 自有资金可能位于共同资源之前、同时或之后并可分层使用，必须读具体 rulebook。见 [[CCP违约瀑布]]。
<!-- bilingual-en:start -->
- **Default waterfall:** There is no globally uniform complete sequence. VM already settled affects the loss base and is not a prefunded layer; unpaid VM is a default obligation, while haircutting VM gains may be a recovery tool. Distinguish the defaulter's IM, the defaulter's default-fund contribution, designated CCP own resources or “skin in the game,” non-defaulting members' mutualised default fund, and assessments or other later resources. CCP capital may rank before, alongside, or after mutualised resources and may be split into tranches, so the actual rulebook controls. See [[CCP违约瀑布|CCP default waterfall]].
<!-- bilingual-en:end -->

## **4. ISDA协议与净额结算**
<!-- bilingual-en:start -->
*4. ISDA Agreements and Netting*
<!-- bilingual-en:end -->

- **ISDA 主协议及 CSA**：主协议规定交易关系、违约和提前终止框架；CSA 规定担保品调用、threshold、MTA、合格担保品、haircut、争议和返还等。经济上方向相反的交易只有在同一对手、同一可执行净额集合和相关法律意见支持下，才可在 close-out 时抵销。
<!-- bilingual-en:start -->
- **ISDA Master Agreement and CSA:** The Master Agreement provides the framework for the trading relationship, default, and early termination; the CSA governs collateral calls, thresholds, MTAs, eligible collateral, haircuts, disputes, and return. Economically offsetting trades can be combined at close-out only within the same counterparty relationship and legally enforceable netting set supported by the relevant legal analysis.
<!-- bilingual-en:end -->

- **多边净额（multilateral netting）**：中央清算后，只能在同一 CCP、同一清算服务、同一账户及同一法律净额集合允许的边界内汇总。$45\rightarrow15$ 只是“所有头寸确实进入同一集合、暂忽略抵押品与 MPOR”的算术示意；它既不证明总市场敞口下降，也不证明 IM、违约基金或抵押品需求下降。跨 CCP、跨服务或隔离账户的反向头寸不能自动抵销。见 [[中央清算与多边净额]] 与 [[净额与抵押品]]。
<!-- bilingual-en:start -->
- **Multilateral netting:** After central clearing, aggregation is permitted only within the same CCP, clearing service, account, and legally enforceable netting set. The $45\rightarrow15$ result is merely arithmetic under the assumptions that all positions enter one set and collateral and MPOR are ignored. It proves neither a fall in total market exposure nor a reduction in IM, default-fund contributions, or collateral demand. Opposite positions across CCPs, services, or segregated accounts do not offset automatically. See [[中央清算与多边净额|Central clearing and multilateral netting]] and [[净额与抵押品|Netting and collateral]].
<!-- bilingual-en:end -->

## **5. 违约处理流程与案例分析**
<!-- bilingual-en:start -->
*5. Default Management and an Illustrative Close-Out*
<!-- bilingual-en:end -->

- **违约、终止与中止权**：发生 event of default 不代表所有交易自动 close-out。是否需要通知、是否有补救或宽限期、是否适用 Automatic Early Termination（AET），以及监管 stay、破产中止和金融担保品法律会否限制终止或执行，都取决于协议和适用法。满足条件后才指定或发生提前终止，并按有效净额集合计算一个 close-out amount。
<!-- bilingual-en:start -->
- **Default, termination, and stays:** An event of default does not make every trade close out automatically. Notice, cure or grace periods, Automatic Early Termination (AET), regulatory stays, insolvency stays, and financial-collateral law may affect termination or enforcement. Only when the contractual and legal conditions are met is early termination designated or triggered, after which one close-out amount is determined for the valid netting set.
<!-- bilingual-en:end -->

- **Close-out amount**：金额依具体协议的估值条款和可用市场信息确定，可能考虑报价、市场数据、替代交易成本和损失；ISDA 并没有一条普遍的“mid ± half spread”算法。下文第 3 题保留中间价与买卖价的计算，只是教学报价算例，不能外推为 ISDA 通则或具体法律结论。
<!-- bilingual-en:start -->
- **Close-out amount:** The amount depends on the agreement's valuation terms and available market information, which may include quotations, market data, replacement costs, and losses. ISDA does not prescribe a universal “mid ± half spread” algorithm. Exercise 3 retains the bid–ask arithmetic only as a teaching quotation example, not as a general ISDA rule or legal conclusion.
<!-- bilingual-en:end -->

## **6. 金融危机后监管改革措施**
<!-- bilingual-en:start -->
*6. Post-Crisis Regulatory Reforms*
<!-- bilingual-en:end -->

- **G20 改革承诺**：2009 年 G20 承诺推动适当的标准化 OTC 衍生品在交易场所或电子平台执行、通过 CCP 清算，并向交易数据库报告；非集中清算合约应受到更高资本要求。这些承诺受法域、产品定义、强制范围、豁免、跨境认可和本地实施约束，不能写成全球所有交易都适用的单一规则。
<!-- bilingual-en:start -->
- **G20 reform commitments:** In 2009, the G20 committed to move appropriate standardised OTC derivatives to exchanges or electronic platforms, central clearing, and trade-repository reporting, with higher capital requirements for non-centrally cleared contracts. The commitments are conditioned by jurisdiction, product definitions, mandate scope, exemptions, cross-border recognition, and local implementation; they are not one universal rule for every trade.
<!-- bilingual-en:end -->

- **三项义务彼此独立**：交易执行回答“在哪里、怎样成交”；中央清算回答“CCP 是否依法成为对手方”；交易报告回答“谁向哪个 repository 报告哪些数据”。某交易被报告不代表已清算，被清算也不代表在交易所上市或必须在某一平台成交。
<!-- bilingual-en:start -->
- **Three distinct obligations:** Execution asks where and how a trade is made; clearing asks whether a CCP legally becomes the counterparty; reporting asks who reports which data to which repository. A reported trade is not necessarily cleared, and a cleared trade is not necessarily exchange-listed or required to execute on a particular venue.
<!-- bilingual-en:end -->

## **7. 未清算交易及其保证金规定**
<!-- bilingual-en:start -->
*7. Uncleared Transactions and Their Margin Rules*
<!-- bilingual-en:end -->

- **适用范围**：未集中清算交易不一定只是“不够标准化”的交易，也可能因产品、对手类型、法域、清算义务豁免或 CCP 接纳条件而保持双边。监管保证金只适用于规则定义的 covered entities 与 covered instruments，并存在对手、产品、集团内部、主权、央行、外汇等类别的具体范围或豁免；必须按当地实施核对。
<!-- bilingual-en:start -->
- **Scope:** A non-centrally cleared trade is not necessarily “insufficiently standardised”; it may remain bilateral because of the product, counterparty type, jurisdiction, mandate exemption, or CCP acceptance conditions. Regulatory margin applies only to defined covered entities and covered instruments, with jurisdiction-specific scope or exemptions for counterparties, products, intragroup trades, sovereigns, central banks, foreign exchange, and other categories. Local implementation must be checked.
<!-- bilingual-en:end -->

- **VM 与 IM 的阈值不同**：适用关系中的 VM 通常有其 own threshold 和 MTA 规则；IM 则另有集团层面的 collection threshold、MTA、模型或 schedule、托管和隔离要求。达到某个 AANA 门槛通常是进入范围或实施阶段的筛选，不等于每笔交易都要“全额 VM + 全额 IM”，也不能把 VM threshold、IM collection threshold 与 MTA 混成一个数。
<!-- bilingual-en:start -->
- **Different VM and IM thresholds:** VM in a covered relationship has its own threshold and MTA rules; IM has a separate group-level collection threshold, MTA, model or schedule, custody, and segregation requirements. Crossing an AANA threshold is generally a scope or phase-in test, not proof that every trade requires “full VM plus full IM.” VM thresholds, IM collection thresholds, and MTAs are different concepts.
<!-- bilingual-en:end -->

- **历史阶段示例**：欧盟口径中 2021 年第 5 阶段的 EUR 50bn 和 2022 年第 6 阶段的 EUR 8bn AANA 可作为历史实施示例，但币种、计算期、实体范围和现行门槛要按报告日与法域重新核对，不能当作全球当前规则。
<!-- bilingual-en:start -->
- **Historical phase-in example:** The EUR 50bn AANA threshold for the EU's 2021 Phase 5 and EUR 8bn for its 2022 Phase 6 can be retained as historical implementation examples. Currency, calculation period, entity scope, and current thresholds must be rechecked for the reporting date and jurisdiction; these figures are not a current global rule.
<!-- bilingual-en:end -->

## **8. 初始保证金模型（SIMM）及公式解析**
<!-- bilingual-en:start -->
*8. The Standard Initial Margin Model (SIMM)*
<!-- bilingual-en:end -->

- **SIMM 概念**：ISDA SIMM 是版本化、风险敏感的未清算初始保证金模型，不是全局适用的 $\text{normal VaR}\times2.33\times\sqrt{10}$。模型按当前版本处理 Delta、Vega、curvature 与 concentration 等组成，将敏感度映射到利率、信用、股票、商品和外汇等风险类别及其 buckets，再按规定的类内、桶内、桶间和跨风险类别规则聚合。参数、产品映射、集中度门槛和校准会随版本变化，实际计算必须使用适用版本的方法文件与许可实现。
<!-- bilingual-en:start -->
- **SIMM:** ISDA SIMM is a versioned, risk-sensitive model for uncleared IM, not a globally applicable $\text{normal VaR}\times2.33\times\sqrt{10}$ formula. The applicable version treats components such as Delta, Vega, curvature, and concentration, maps sensitivities into risk classes and buckets for interest rates, credit, equity, commodities, and foreign exchange, and then applies prescribed within-class, within-bucket, between-bucket, and cross-risk-class aggregation. Parameters, product mapping, concentration thresholds, and calibration change by version, so an actual calculation must use the applicable methodology and licensed implementation.
<!-- bilingual-en:end -->

- **教学公式边界**：第 4 题改为“一因子 normal-VaR 教学算例”，只演示在正态、独立同分布和平方根时间缩放假设下如何计算 VaR。它不使用 SIMM 风险权重或聚合规则，结果不能代表 SIMM 保证金。
<!-- bilingual-en:start -->
- **Boundary of the teaching formula:** Exercise 4 is renamed a “one-factor normal-VaR teaching example.” It only illustrates VaR under normality, independent and identically distributed returns, and square-root-of-time scaling. It uses neither SIMM risk weights nor SIMM aggregation, and its result cannot represent SIMM IM.
<!-- bilingual-en:end -->

## **9. CCP间互操作、抵押品再使用与流动性风险**
<!-- bilingual-en:start -->
*9. CCP Interoperability, Collateral Reuse, and Liquidity Risk*
<!-- bilingual-en:end -->

- **互操作（interoperability）**：CCP 间建立互联，会在 CCP 之间产生新的法律与信用敞口，并要求专门的风险管理、保证金和违约安排。互联本身不会自动授权参与者跨 CCP 净额或 cross-margin，也不会给出保证金折扣。只有合同明确授权、法律意见支持且风险模型认可具体 offset 时，才可计算折后金额。第 6 题因此只能回答“信息不足”。
<!-- bilingual-en:start -->
- **Interoperability:** A link between CCPs creates legal and credit exposures between the CCPs and requires specific risk management, margin, and default arrangements. A link does not automatically authorise participant-level cross-CCP netting or cross-margining, nor does it determine a margin discount. A reduced amount can be calculated only when the contracts authorise it, legal opinions support it, and the risk model recognises the particular offset. Exercise 6 therefore has the answer “insufficient information.”
<!-- bilingual-en:end -->

- **抵押品再使用（rehypothecation, repledge or reuse）**：收到的担保品能否再次提供给第三方，取决于所有权转移或质押结构、协议、隔离和适用法。未清算 IM 的一次再使用只在 BCBS–IOSCO 框架列明的严格条件下可能获准，且法域可以更严格；不能概括为“IM 普遍可再抵押一次”。VM 也不是无限再使用：其可用性受合同、财产权、破产法、结算安排和机构流动性管理约束。本章不采用“危机期间平均循环 4 次”的无来源通则。
<!-- bilingual-en:start -->
- **Collateral reuse (rehypothecation, repledge, or reuse):** Whether received collateral can be passed to a third party depends on title-transfer or security-interest structure, the agreement, segregation, and applicable law. One-time reuse of uncleared IM may be permitted only under the strict conditions listed in the BCBS–IOSCO framework, and jurisdictions may be more restrictive; it is not a general entitlement to reuse IM once. VM is not infinitely reusable either: availability is constrained by contract, property and insolvency law, settlement arrangements, and the firm's liquidity management. This chapter does not retain an unsupported universal claim that crisis collateral circulated four times on average.
<!-- bilingual-en:end -->

- **保证金顺周期**：短期 VM 调用、新增 IM、抵押品 haircut 和同时发生的融资需求可能迫使机构变现资产，并通过价格与波动反馈引起新调用。反顺周期工具、透明的模拟和充足流动性缓冲可以减弱但不能消除该机制。见 [[保证金顺周期]]。
<!-- bilingual-en:start -->
- **Margin procyclicality:** Short-dated VM calls, additional IM, collateral haircuts, and simultaneous funding needs may force asset sales and feed prices and volatility back into new calls. Anti-procyclicality tools, transparent simulation, and adequate liquidity buffers can mitigate but not eliminate this mechanism. See [[保证金顺周期|Margin procyclicality]].
<!-- bilingual-en:end -->

## **10. OTC与场内交易的融合趋势**
<!-- bilingual-en:start -->
*10. Convergence Between OTC and Exchange-Traded Markets*
<!-- bilingual-en:end -->

- **基础设施趋同不等于法律角色合并**：标准化 OTC 合约可能在电子场所执行、提交 CCP 清算并接受更统一的保证金流程；交易所也可能提供更灵活的产品。但 execution venue、exchange listing、trade repository、CCP、清算成员和托管人仍承担不同法律功能。
<!-- bilingual-en:start -->
- **Infrastructure convergence is not legal-role merger:** Standardised OTC contracts may execute electronically, be submitted for CCP clearing, and use more standardised margin processes, while exchanges may offer more flexible products. Execution venues, exchange listing, trade repositories, CCPs, clearing members, and custodians still perform different legal functions.
<!-- bilingual-en:end -->

- **比较必须逐项进行**：判断 OTC 与场内市场是否“趋同”，应分别比较合约标准化、成交方式、清算安排、账户隔离、保证金模型、透明度和监管义务，不能只凭“通过 CCP”就把产品称为上市合约。
<!-- bilingual-en:start -->
- **Compare dimension by dimension:** Assess convergence separately for contract standardisation, execution, clearing, account segregation, margin models, transparency, and regulatory obligations. A product does not become exchange-listed merely because it clears through a CCP.
<!-- bilingual-en:end -->

## **11. CCP失效的系统性风险、恢复与处置**
<!-- bilingual-en:start -->
*11. Systemic Risk, Recovery, and Resolution of a CCP*
<!-- bilingual-en:end -->

- **风险集中**：中央清算可以简化部分双边网络并集中违约管理，但也形成对少数 CCP、清算成员、结算银行和抵押品渠道的共同依赖。系统风险是否下降取决于净额边界、保证金、流动性、成员集中和可执行的违约管理，不能只看名义头寸迁移。
<!-- bilingual-en:start -->
- **Risk concentration:** Central clearing can simplify parts of the bilateral network and centralise default management, but it also creates common dependence on a small number of CCPs, clearing members, settlement banks, and collateral channels. Whether systemic risk falls depends on netting boundaries, margin, liquidity, member concentration, and executable default management—not merely on the migration of notional positions.
<!-- bilingual-en:end -->

- **恢复与处置必须分开**：recovery 由 CCP 依事先制定并披露的计划，在监管监督下使用 assessment、VM gains haircutting、强制分配或 tear-up 等工具，以补充资源、分配损失并恢复 matched book。resolution 由法定处置机关依当地法律介入，以维持关键功能和金融稳定。处置不必等所有恢复工具耗尽；若恢复不可行、不够及时或会危及金融稳定，机关可以更早介入。见 [[CCP恢复与处置]]。
<!-- bilingual-en:start -->
- **Recovery and resolution must remain separate:** In recovery, the CCP acts under a pre-established and disclosed plan, subject to supervision, and may use assessments, VM gains haircutting, forced allocation, or tear-up to replenish resources, allocate losses, and restore a matched book. In resolution, a statutory resolution authority intervenes under local law to maintain critical functions and financial stability. Resolution need not wait until every recovery tool is exhausted; the authority may intervene earlier if recovery is infeasible, untimely, or destabilising. See [[CCP恢复与处置|CCP recovery and resolution]].
<!-- bilingual-en:end -->

- **QCCP 资本连接**：QCCP 资格不表示零风险。按 Basel 的基本分类，清算成员自营的 QCCP **交易 EAD**适用 2% 风险权重；提交抵押品按持有与 bankruptcy-remoteness 条件处理；违约基金出资使用独立公式，不能一起乘 2%。非 QCCP 不享受同一交易敞口优惠，其违约基金也有更严厉的单独处理。具体资本结果仍受客户清算结构、报告日和当地实施约束。见 [[QCCP资本处理]]。
<!-- bilingual-en:start -->
- **QCCP capital link:** QCCP status does not mean zero risk. Under the basic Basel classification, a clearing member's own-account **trade EAD** to a QCCP receives a 2% risk weight; posted collateral is treated according to holding and bankruptcy-remoteness conditions; and default-fund contributions use a separate formula rather than the same 2%. A non-QCCP does not receive the same trade-exposure concession, and its default fund has a more punitive separate treatment. The result still depends on client-clearing structure, reporting date, and local implementation. See [[QCCP资本处理|QCCP capital treatment]].
<!-- bilingual-en:end -->

> [!info] 一手来源定位
> 本章口径以 [[中央清算与多边净额]]、[[CCP违约瀑布]]、[[CCP恢复与处置]]、[[保证金顺周期]]、[[QCCP资本处理]] 与 [[净额与抵押品]] 中列明的 Basel Framework、BCBS–IOSCO、CPMI–IOSCO、FSB 与 ISDA 一手资料为定位依据；具体交易、CCP 和报告日仍需查适用协议、rulebook 与当地法。
<!-- bilingual-en:start -->
> The chapter is anchored to the Basel Framework and the BCBS–IOSCO, CPMI–IOSCO, FSB, and ISDA primary materials listed in [[中央清算与多边净额|Central clearing and multilateral netting]], [[CCP违约瀑布|CCP default waterfall]], [[CCP恢复与处置|CCP recovery and resolution]], [[保证金顺周期|Margin procyclicality]], [[QCCP资本处理|QCCP capital treatment]], and [[净额与抵押品|Netting and collateral]]. A specific trade, CCP, and reporting date still require the applicable agreement, rulebook, and local law.
<!-- bilingual-en:end -->

## **计算题**
<!-- bilingual-en:start -->
*Calculation Exercises*
<!-- bilingual-en:end -->

1. **题目：** A、B 的一笔 OTC 衍生品今日对 A 增值 USD 100,000、对 B 减值 USD 100,000。假设适用协议要求零 VM threshold、MTA 已满足、无争议且今日按 CTM 转移现金，应发生什么操作？
<!-- bilingual-en:start -->

&nbsp;
**1. Question:** An OTC derivative gains USD 100,000 for A and loses USD 100,000 for B today. Assume the agreement has a zero VM threshold, the MTA is met, there is no dispute, and cash is transferred today under CTM. What happens?<br>
<!-- bilingual-en:end -->

    **解答：** 在这些明确假设下，B 向 A 转移 USD 100,000 VM。若有 threshold、未满足 MTA、争议或结算时差，实际当日转移额可能不同；若采用 STM，该现金是重置合约价值的结算付款，不能又从未重置的市值中重复扣除。
<!-- bilingual-en:start -->
**Solution:** Under the stated assumptions, B transfers USD 100,000 of VM to A. A threshold, unmet MTA, dispute, or settlement gap could change the amount transferred that day. Under STM, the cash would be a settlement payment that resets contract value and must not be deducted again from an unreset value.
<!-- bilingual-en:end -->

2. **题目：** 某交易商的头寸市值为 +10M、+15M、−20M、+5M、−10M、+10M、+5M。（a）若它们分别面对七个对手且不可跨对手净额，逐对手正敞口之和是多少？（b）仅假设所有头寸已被同一 CCP、同一清算服务、同一账户和同一可执行净额集合接纳，且忽略抵押品、VM、IM 与 MPOR，该集合的净重置价值是多少？
<!-- bilingual-en:start -->

&nbsp;
**2. Question:** A dealer has positions worth +10M, +15M, −20M, +5M, −10M, +10M, and +5M. (a) If they face seven separate counterparties and cannot be netted across counterparties, what is the sum of positive exposure? (b) Only on the assumption that all positions are accepted into the same CCP, clearing service, account, and enforceable netting set, with collateral, VM, IM, and MPOR ignored, what is the set's net replacement value?<br>
<!-- bilingual-en:end -->

    **解答：** (a) $10+15+5+10+5=45$M。(b) $[45-(20+10)]^+=15$M。$45\rightarrow15$ 只是题设集合内的当前重置价值示意；若负头寸落在另一 CCP、服务或隔离账户，就不能抵销，也不能由此推断 IM 或总抵押品下降。
<!-- bilingual-en:start -->
**Solution:** (a) $10+15+5+10+5=45$M. (b) $[45-(20+10)]^+=15$M. The $45\rightarrow15$ result is only an illustration of current replacement value within the stated set. A negative position in another CCP, service, or segregated account cannot offset the positives, and the result says nothing by itself about IM or total collateral.
<!-- bilingual-en:end -->

3. **题目：教学报价算例（不是 ISDA 通则）。** 一笔替代交易的中间价为 USD 20M，bid 为 USD 18M，ask 为 USD 22M。若题设规定买入替代头寸使用 ask，报价是多少？若反向替代头寸的题设可执行报价为 −USD 18M，报价是多少？
<!-- bilingual-en:start -->

&nbsp;
**3. Question: Teaching quotation example, not a general ISDA rule.** A replacement trade has a USD 20M mid, USD 18M bid, and USD 22M ask. If the exercise says that buying the replacement uses the ask, what quote is used? If the exercise gives −USD 18M as the executable quote for the opposite replacement, what quote is used?<br>
<!-- bilingual-en:end -->

    **解答：** 第一种题设报价为 USD 22M，第二种为 −USD 18M。算术只读取题设的可执行买卖报价；实际 close-out amount 仍须依具体协议、市场信息、notice/grace/AET、stay 与适用法确定，不能套用“mid ± half spread”。
<!-- bilingual-en:start -->
**Solution:** The exercise quotes are USD 22M and −USD 18M. The arithmetic merely reads the executable bid or ask specified in the question. An actual close-out amount still depends on the agreement, market information, notice or grace periods, AET, stays, and applicable law; “mid ± half spread” cannot be applied as a universal rule.
<!-- bilingual-en:end -->

4. **题目：一因子 normal-VaR 教学算例，不是 SIMM 计算。** 某头寸对单一收益因子的线性金额敞口为 150 个货币单位，日收益波动率为 1.5%。在独立同分布、平方根时间缩放与正态分布假设下，用 10 天和单侧 99% 分位数 2.33 计算 VaR。
<!-- bilingual-en:start -->

&nbsp;
**4. Question: One-factor normal-VaR teaching example, not a SIMM calculation.** A position has a linear monetary exposure of 150 units to one return factor and daily return volatility of 1.5%. Under i.i.d. returns, square-root-of-time scaling, and normality, calculate ten-day VaR using the one-sided 99% quantile of 2.33.<br>
<!-- bilingual-en:end -->

    **解答：** 教学模型只有

    $$
    \operatorname{VaR}_{10,99\%}=2.33\times150\times0.015\times\sqrt{10}\approx16.58.
    $$

    因此结果约为 16.58 个货币单位。这里没有使用 SIMM 的 Delta、Vega、curvature、concentration、风险类别、bucket 或相关聚合，16.58 不能代表 SIMM IM。
<!-- bilingual-en:start -->
**Solution:** The teaching model is only

$$
\operatorname{VaR}_{10,99\%}=2.33\times150\times0.015\times\sqrt{10}\approx16.58.
$$

The result is about 16.58 monetary units. No SIMM Delta, Vega, curvature, concentration, risk-class, bucket, or correlation aggregation is used, so 16.58 cannot represent SIMM IM.
<!-- bilingual-en:end -->

5. **题目：题设规则下的瀑布算例。** 只为本题，假设某 CCP 规则明确规定顺序为：违约成员 IM 50M、违约成员违约基金出资 100M、非违约成员共同违约基金 200M、CCP 指定自有资源 500M。已结算 VM 已反映在损失基数中。完成对冲和处置后的待分配损失为 250M。如何分摊？
<!-- bilingual-en:start -->

&nbsp;
**5. Question: A waterfall example under a stipulated rule.** For this exercise only, assume the CCP rules expressly order resources as follows: 50M of defaulter IM, 100M of the defaulter's default-fund contribution, 200M of non-defaulting members' mutualised default fund, and 500M of designated CCP own resources. Settled VM is already reflected in the loss base. The loss remaining after hedging and disposal is 250M. How is it allocated?<br>
<!-- bilingual-en:end -->

    **解答：** 违约成员 IM 承担 50M，剩余 200M；其违约基金出资承担 100M，剩余 100M；非违约成员共同基金承担 100M，剩余 0。按**本题规定的顺序**，CCP 自有资源未使用。这个答案不表示 CCP 资本永远最后；其他 rulebook 可能把 skin in the game 放在共同基金之前、同时或分成多个层次。
<!-- bilingual-en:start -->
**Solution:** Defaulter IM absorbs 50M, leaving 200M. The defaulter's default-fund contribution absorbs 100M, leaving 100M. The mutualised fund absorbs 100M, leaving zero. Under the **order stipulated by this question**, CCP own resources are unused. This does not mean CCP capital is always last; another rulebook may put skin in the game before or alongside the mutualised fund or split it into tranches.
<!-- bilingual-en:end -->

6. **题目：** 某参与者在 CCP1 持有多头、在 CCP2 持有等量空头，两边单独 IM 各为 USD 0.5M。已知两 CCP 存在互操作链接，但题目未给跨 CCP 净额授权、法律意见、风险模型、CCP 间保证金或最低保证金。能否算出链接后的总 IM？
<!-- bilingual-en:start -->

&nbsp;
**6. Question:** A participant is long at CCP1 and equally short at CCP2, with standalone IM of USD 0.5M at each CCP. The CCPs have an interoperability link, but the question gives no cross-CCP netting authority, legal opinion, risk model, inter-CCP margin, or margin floor. Can total IM after the link be calculated?<br>
<!-- bilingual-en:end -->

    **解答：** 不能。无链接时可算出 $0.5M+0.5M=1.0M$；仅知道“互操作”不足以授权 cross-netting 或 cross-margining，链接还会产生 CCP 间敞口。只有给出合同授权、法律可执行性、认可 offset 的风险模型及相关附加/最低保证金，才可算折后金额；不能凭等量多空断言 0、0.5M 或节省 50%。
<!-- bilingual-en:start -->
**Solution:** No. Without the link, standalone IM totals $0.5M+0.5M=1.0M$. Interoperability alone neither authorises cross-netting nor cross-margining and also creates inter-CCP exposure. A reduced amount requires contractual authority, legal enforceability, a risk model that recognises the offset, and any add-ons or floors. Equal and opposite positions do not justify an answer of zero, USD 0.5M, or a 50% saving.
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->

## 17.3

> [!question]
> 为什么在 2007～2008 年金融危机后引入的监管规定会给某些金融机构带来流动性问题？
<!-- bilingual-en:start -->
> Why could regulations introduced after the 2007–08 financial crisis create liquidity problems for some financial institutions?
<!-- bilingual-en:end -->

中央清算和未清算保证金规则会要求适用机构在较短时间内交付现金或合格高流动性担保品。市场剧烈变动时，VM 流出与 IM 上调可能同时发生；haircut、抵押品集中、托管隔离和结算时差又会限制账面资产的即时可用性。机构可能因此融资或出售资产，并在共同压力下形成 [[保证金顺周期|顺周期反馈]]。但结果取决于产品、法域、豁免、账户和流动性缓冲，不能说所有危机后规则都必然制造同样的流动性问题。
<!-- bilingual-en:start -->
Central-clearing and uncleared-margin rules can require covered firms to deliver cash or eligible liquid collateral quickly. In volatile markets, VM outflows and IM increases may coincide, while haircuts, concentration, custody segregation, and settlement gaps limit the immediate usability of balance-sheet assets. Firms may therefore borrow or sell assets, creating [[保证金顺周期|procyclical feedback]] under common stress. The outcome depends on product, jurisdiction, exemptions, account structure, and liquidity buffers; not every post-crisis rule creates the same liquidity problem.
<!-- bilingual-en:end -->

## 17.4

> [!question]
> 解释一下担保品协议中“折减”的含义。
<!-- bilingual-en:start -->
> Explain what a “haircut” means in a collateral agreement.
<!-- bilingual-en:end -->

haircut 是把资产市场价值下调后才认可为担保品价值的百分比。例如，市价 USD 100、haircut 10% 的资产，只提供 USD 90 的认可担保品价值。折减用于覆盖价格、流动性、期限、币种和处置风险，具体比例依规则或协议而定。
<!-- bilingual-en:start -->
A haircut is the percentage reduction applied to an asset's market value when recognising its collateral value. An asset worth USD 100 with a 10% haircut provides only USD 90 of recognised collateral value. Haircuts address price, liquidity, maturity, currency, and liquidation risk, and the applicable percentage depends on the rules or agreement.
<!-- bilingual-en:end -->

## 17.5

> [!question]
> 解释 ISDA 主协议中“违约事件”和“提前终止”的含义。
<!-- bilingual-en:start -->
> Explain “event of default” and “early termination” under an ISDA Master Agreement.
<!-- bilingual-en:end -->

违约事件是协议列明的触发事实，例如在适用条件下未付款、未交付担保品或破产。它不保证立刻自动终止：要继续检查 notice、grace/cure period、是否选择 AET、监管或破产 stay，以及适用法。只有终止条件满足后，相关未到期交易才在同一可执行净额集合内被终止并计算一个 close-out amount。
<!-- bilingual-en:start -->
An event of default is a contractual trigger such as a qualifying failure to pay, failure to deliver collateral, or insolvency. It does not guarantee immediate automatic termination: notice, grace or cure periods, any election of AET, regulatory or insolvency stays, and applicable law must still be checked. Only once termination conditions are met are the relevant outstanding trades terminated within the enforceable netting set and one close-out amount determined.
<!-- bilingual-en:end -->

## 17.9

> [!question]
> 什么是再抵押？
<!-- bilingual-en:start -->
> What is rehypothecation?
<!-- bilingual-en:end -->

再抵押是担保品接收方在合同和法律允许时，把收到的质押担保品再次质押给第三方；更广义的 reuse 还包括所有权转移结构下的再次使用。能否再用取决于协议、财产权、隔离、破产法与监管限制。未清算 IM 即使允许一次再使用，也必须满足严格条件且当地法可以禁止；VM 的使用同样不受“无限再用”的保证。
<!-- bilingual-en:start -->
Rehypothecation occurs when a collateral taker repledges received pledged collateral to a third party where contract and law permit; broader “reuse” also includes use following a title transfer. Availability depends on the agreement, property law, segregation, insolvency law, and regulation. Even one-time reuse of uncleared IM is subject to strict conditions and may be prohibited locally; VM is likewise not guaranteed to be reusable without limit.
<!-- bilingual-en:end -->

# **OTC衍生产品清算与监管学习笔记**
<!-- bilingual-en:start -->
*Study Notes on Clearing and Regulation of OTC Derivatives*
<!-- bilingual-en:end -->

## **1. OTC衍生品清算机制**
<!-- bilingual-en:start -->
*1. OTC Derivatives Clearing Mechanisms*
<!-- bilingual-en:end -->

- **双边清算（Bilateral Clearing）**：对每一对交易对手签订ISDA主协议及信用支持附件（CSA），明确担保品种类、折减率等。交易双方私下结算衍生品头寸的增值部分，并按ISDA协议约定计算净额清算 。
<!-- bilingual-en:start -->
- **Bilateral clearing:** Each pair of counterparties signs an ISDA Master Agreement and a Credit Support Annex (CSA), specifying eligible collateral, haircuts, and related terms. The parties privately settle changes in the value of their derivatives positions and calculate close-out net amounts under the ISDA agreement.
<!-- bilingual-en:end -->
    
- **中央对手方清算（CCP）**：交易由CCP居中撮合，CCP与交易双方分别建立相互抵消的交易头寸。CCP要求每个成员缴纳初始保证金和变动保证金，并设立互助的**保证基金（guarantee fund）**，当一方违约且保证金不足以覆盖损失时，由保证基金分摊损失 。CCP可视为类似于交易所清算所，对合约进行盯市和抵消，从而显著降低市场参与者间的双边信用风险。
<!-- bilingual-en:start -->
- **Central counterparty (CCP) clearing:** A CCP interposes itself between the parties and establishes offsetting positions with each side. It requires every clearing member to post initial and variation margin and to contribute to a mutual **guarantee or default fund**. If a member defaults and its margin does not cover the loss, the fund absorbs losses according to the CCP's rules. Like an exchange clearing house, a CCP marks contracts to market and nets positions, materially reducing bilateral credit exposures among participants.
<!-- bilingual-en:end -->

## **2. 保证金制度**
<!-- bilingual-en:start -->
*2. The Margin System*
<!-- bilingual-en:end -->

- **变动保证金（Variation Margin, VM）**：根据每日盯市结果结算的担保品。当未平仓合约的市场价值变化时，产生盈利的一方无需支付，而亏损方需按变动幅度补缴担保品。比如若对A方价值增加了$x（B方价值减少$x），B方向A方支付相应的$x担保品 。变动保证金的累计效果确保未偿付合约的当前价值由相应担保覆盖。
<!-- bilingual-en:start -->
- **Variation margin (VM):** Collateral transferred to settle daily mark-to-market changes. When an open contract's value rises for one party, that party receives collateral from the party for whom its value falls. If the value to Party A rises by $x$ while the value to Party B falls by $x$, B transfers $x$ of collateral to A. Cumulative VM therefore collateralises the current replacement value of outstanding contracts.
<!-- bilingual-en:end -->
    
- **初始保证金（Initial Margin, IM）**：除变动保证金外，为防止市场价格大幅波动所造成的潜在损失而预先缴存的担保品。初始保证金通常以覆盖一定展望期（场内一般3-5天、双边10天）的99%置信度的风险价值为目标 。初始保证金会随着持仓规模和市场波动率变化而调整，反映了**极端市场环境下的潜在亏损** 。一般以现金方式存放（或满足条件的高质量资产），期货交易中IM计息、VM不计息；OTC交易中所有现金保证金通常计息 。
<!-- bilingual-en:start -->
- **Initial margin (IM):** Collateral posted in advance, in addition to VM, to cover potential losses caused by adverse price movements during a close-out period. IM is commonly calibrated to a 99% loss measure over a specified margin period of risk—typically three to five days for exchange-traded positions and ten days for bilateral positions in this presentation. It changes with position size and market volatility and represents **potential loss under stressed market conditions**. It is normally posted in cash or eligible high-quality assets. The slides distinguish futures, where IM earns interest and VM does not, from OTC transactions, where cash collateral commonly earns interest.
<!-- bilingual-en:end -->

## **3. CCP 结构与风险承担**
<!-- bilingual-en:start -->
*3. CCP Structure and Loss Bearing*
<!-- bilingual-en:end -->

- **CCP结构**：CCP将交易双方的衍生品头寸相互抵消，并替代原交易对手关系。每个清算成员缴纳**初始保证金**和**变动保证金**，并贡献一定金额组成**保证基金** 。违约时，CCP利用违约成员的保证金和保证基金来覆盖损失，其他成员分担剩余风险。
<!-- bilingual-en:start -->
- **CCP structure:** The CCP replaces the original bilateral relationship with two offsetting derivatives positions. Each clearing member posts **initial margin** and **variation margin** and contributes to a **default fund**. If a member defaults, the CCP applies the defaulting member's resources and the waterfall specified in its rules; mutualised resources can make other members bear residual losses.
<!-- bilingual-en:end -->
    
- **风险承担顺序（违约水位）**：若某会员违约且损失超过其抵押品，损失将按以下顺序分担：
<!-- bilingual-en:start -->
- **Loss waterfall:** In the simplified ordering used by this course note, if a member defaults and its losses exceed its collateral, resources are applied as follows:
<!-- bilingual-en:end -->
    
    1. **违约会员的初始保证金**（首先承担损失）；
<!-- bilingual-en:start -->

&nbsp;
**1.** **The defaulting member's initial margin**, which absorbs losses first;<br>
<!-- bilingual-en:end -->
        
    2. **违约会员缴纳的保证基金**；
<!-- bilingual-en:start -->

&nbsp;
**2.** **The defaulting member's contribution to the default fund**;<br>
<!-- bilingual-en:end -->
        
    3. **其他非违约会员缴纳的保证基金**；
<!-- bilingual-en:start -->

&nbsp;
**3.** **Default-fund contributions from non-defaulting members**;<br>
<!-- bilingual-en:end -->
        
    4. **CCP自有资本或股本**。
<!-- bilingual-en:start -->

&nbsp;
**4.** **The CCP's own capital or equity.**<br>
<!-- bilingual-en:end -->
        
        这一顺序保证先由违约方的资金承担最大部分损失，然后由市场其他成员分摊，最后才动用CCP自有资本 。
<!-- bilingual-en:start -->
Under this simplified sequence, the defaulter's resources absorb the largest first layer, other market participants mutualise the residual loss, and the CCP's own capital is used last. Actual CCP rulebooks can place a tranche of CCP “skin in the game” earlier in the waterfall.
<!-- bilingual-en:end -->
        
    

  

## **4. ISDA协议与净额结算**
<!-- bilingual-en:start -->
*4. ISDA Agreements and [[OTC 衍生品清算、保证金与 CCP 风险|Netting]]*
<!-- bilingual-en:end -->

- **ISDA主协议及CSA**：双边场外衍生品交易通常采用国际掉期与衍生工具协会（ISDA）发布的主协议进行规范，主协议中通过信用支持附件（CSA）明确担保品的交付规则、可接受担保品种类、折减率等条款 。
<!-- bilingual-en:start -->
- **ISDA Master Agreement and CSA:** Bilateral OTC derivatives are commonly governed by the Master Agreement published by the International Swaps and Derivatives Association (ISDA). Its Credit Support Annex (CSA) specifies collateral-delivery rules, eligible collateral, haircuts, and related terms.
<!-- bilingual-en:end -->
    
- **[[OTC 衍生品清算、保证金与 CCP 风险|净额结算]]（Netting）**：ISDA协议和CCP会员协议均包含净额结算条款。净额结算意味着在计算担保品需求和违约清算时，将同一对手间所有交易合并为一笔净头寸来处理 。这样可以避免违约方只对亏损交易违约而保留盈利交易的情形，同时减少对手信用风险和所需保证金。净额结算还允许在多个交易间进行抵消，通常显著降低总体信用曝险。例如，一交易商对7个不同对手的未平仓价值为+10, +15, -20, +5, -10, +10, +5（净值）时，双边风险敞口为45；若通过CCP进行净额结算，整体净敞口仅15 。此外，净额结算使得抵押品要求更加经济：无需为每笔正向合约单独缴存保证金。
<!-- bilingual-en:start -->
- **Netting:** Both ISDA agreements and CCP membership agreements contain netting provisions. For collateral calculations and close-out after default, all transactions with the same counterparty are combined into a single net amount. This prevents a defaulter from disclaiming losing trades while retaining profitable ones, reduces counterparty credit exposure, and lowers collateral requirements. In the note's example, a dealer has values of +10, +15, −20, +5, −10, +10, and +5 against seven different counterparties. Bilateral positive exposure totals 45; if all positions can be novated to and netted through one CCP, the aggregate net exposure is 15. Netting also makes collateral more economical because margin need not be posted separately against every positive-value contract.
<!-- bilingual-en:end -->
    

  

## **5. 违约处理流程与案例分析**
<!-- bilingual-en:start -->
*5. Default Management and an Illustrative Close-Out*
<!-- bilingual-en:end -->

- **违约触发与交易终止**：ISDA协议规定一旦触发违约事件（如宣告破产、到期未支付、未按要求补缴担保品等），在短时间窗口内，非违约方可选择终止与违约方的所有未结交易 。交易终止时，非违约方有权留置违约方已交付的担保品，无需法院许可即可执行留置权。
<!-- bilingual-en:start -->
- **Default event and termination:** An ISDA agreement defines events of default such as bankruptcy, failure to pay when due, or failure to post required collateral. After an event and the applicable notice or grace provisions, the non-defaulting party may designate early termination for outstanding transactions. It may apply collateral delivered by the defaulting party in accordance with the agreement and applicable insolvency and financial-collateral law.
<!-- bilingual-en:end -->
    
- **重设交易价格**：终止时，非违约方将现有交易在市场上重置为等效头寸。具体做法是先计算该交易的市场中间价，然后向不利于违约方（有利于非违约方）调整一个幅度，通常相当于半个买卖价差。实际结算价取被调整后的价格。例如，某交易对非违约方的市场中间价为$20M，市场买价为$18M、卖价为$22M，则结算价格定为$22M ，因为非违约方需要以更高的价格（卖价）重开仓位。同理，如中间价为$-20M，对应买卖价为$-18M和$-22M，则结算价为$-18M 。这一机制确保非违约方不会因违约而蒙受额外损失。
<!-- bilingual-en:start -->
- **Replacing the trade:** On termination, the non-defaulting party establishes the cost of replacing the existing trade. The course illustration starts from the market mid-price and adjusts by half the bid–ask spread in the direction representing the non-defaulting party's replacement cost. If the trade is worth USD 20 million at mid, with an USD 18 million bid and USD 22 million ask, the illustration uses USD 22 million because the party must re-establish the position at the ask. If mid is −USD 20 million, with bid and ask of −USD 18 million and −USD 22 million, it uses −USD 18 million. The intended principle is to leave the non-defaulting party economically whole after replacement, subject to the agreement's valuation terms.
<!-- bilingual-en:end -->
    

  

## **6. 金融危机后监管改革措施**
<!-- bilingual-en:start -->
*6. Post-Crisis Regulatory Reforms*
<!-- bilingual-en:end -->

- **中央对手方清算**：2009年G20峰会后，全球监管机构要求将标准化、可清算的场外衍生品（如普通利率互换和信用指数违约互换）纳入中央对手方清算 。这有助于透明度和风险集中管理。
<!-- bilingual-en:start -->
- **Central clearing:** Following the 2009 G20 summit, regulators required standardised, sufficiently liquid OTC derivatives—such as plain-vanilla interest-rate swaps and index credit-default swaps—to be centrally cleared where subject to a clearing mandate. This improves transparency and centralises risk management.
<!-- bilingual-en:end -->
    
- **电子交易平台**：推动标准化的OTC衍生品在电子平台上交易，以提高市场透明度，并简化交易撮合流程 。例如，许多利率掉期和CDS已在Swap Execution Facilities（SEF）、Swap Execution Platforms（SEP）等场所交易。
<!-- bilingual-en:start -->
- **Electronic trading venues:** Standardised OTC derivatives were moved toward electronic execution to improve transparency and simplify trade matching. Many interest-rate swaps and CDS contracts now trade on venues such as swap execution facilities (SEFs) and comparable platforms in other jurisdictions.
<!-- bilingual-en:end -->
    
- **交易报告与仓库**：要求所有OTC交易（包括定制合约）都必须上报至中央交易登记仓库（trade repository），以便监管机构获得市场规模、集中度和对手方风险的全面信息 。如美国的Dodd-Frank法案和欧盟的EMIR都对交易报告做出类似要求。
<!-- bilingual-en:start -->
- **Trade reporting and repositories:** OTC transactions, including bespoke contracts, must be reported to trade repositories so that authorities can observe market size, concentration, and counterparty exposures. The U.S. Dodd–Frank Act and the EU's EMIR impose related reporting requirements.
<!-- bilingual-en:end -->
    

  

## **7. 未清算交易及其保证金规定**
<!-- bilingual-en:start -->
*7. Uncleared Transactions and Their Margin Rules*
<!-- bilingual-en:end -->

- **未清算交易定义**：G20及全球监管框架（如BCBS/IOSCO）将不具备高度标准化特征、因而无法集中清算的场外衍生品称为“**未清算交易**”。此类合约依然采用双边清算模式。
<!-- bilingual-en:start -->
- **Definition of an uncleared transaction:** Under G20 reforms and frameworks such as the BCBS–IOSCO standards, OTC derivatives that are not sufficiently standardised or otherwise eligible for central clearing remain **uncleared** and continue to be managed bilaterally.
<!-- bilingual-en:end -->
    
- **保证金要求**：监管规定任何未清算交易都必须遵守新的保证金规则，交易双方均需缴纳**初始保证金**和**变动保证金** 。其中，变动保证金可由一方直接支付给另一方；初始保证金则应存放于第三方保管，以避免因对手风险导致的资金安全问题 。
<!-- bilingual-en:start -->
- **Margin requirements:** Covered uncleared transactions are subject to regulatory margin rules under which both parties exchange **variation margin** and, when thresholds and scope conditions are met, **initial margin**. VM can be transferred directly between parties; IM is generally segregated with an independent custodian so that it is protected from counterparty failure.
<!-- bilingual-en:end -->
    
- **阈值和实施阶段**：未清算交易保证金（UMR）分阶段实施。2021年9月起（第5阶段），当非集中清算衍生品平均总名义金额（AANA）超过500亿欧元时，两家交易方需交换保证金；2022年9月起（第6阶段）阈值降至80亿欧元 。达到阈值以上机构的双边OTC交易都必须全额交换VM和IM，否则可能因不合规遭受监管处罚。
<!-- bilingual-en:start -->
- **Thresholds and phase-in:** Uncleared margin rules (UMR) were phased in. From September 2021, Phase 5 covered counterparties whose average aggregate notional amount (AANA) of non-centrally cleared derivatives exceeded EUR 50 billion; from September 2022, Phase 6 reduced the threshold to EUR 8 billion. In-scope bilateral relationships must exchange VM and IM in accordance with the rules, subject to applicable thresholds and minimum-transfer amounts, or face regulatory consequences.
<!-- bilingual-en:end -->
    

  

## **8. 初始保证金模型（SIMM）及公式解析**
<!-- bilingual-en:start -->
*8. The Standard Initial Margin Model (SIMM)*
<!-- bilingual-en:end -->

- **SIMM概念**：为减少外部模型计算差异和抵押品纠纷，ISDA提出了标准初始保证金模型（Standard Initial Margin Model, SIMM） 。SIMM基于风险敏感度方法，将投资组合对风险因子的敏感性乘以监管指定的风险权重后，按照相关系数组合计算总初始保证金。该模型符合BCBS-IOSCO《非集中清算衍生品保证金要求》：保证金需覆盖10天99%置信度下的潜在损失 。
<!-- bilingual-en:start -->
- **SIMM:** ISDA developed the Standard Initial Margin Model (SIMM) to reduce disagreement between proprietary models and associated collateral disputes. SIMM is sensitivity based: portfolio sensitivities are multiplied by prescribed risk weights and aggregated using specified correlations to obtain total IM. It is designed to implement the BCBS–IOSCO margin standard for potential losses over a ten-day horizon at 99% confidence.
<!-- bilingual-en:end -->
    
- **主要公式**：设第$i$个风险因子的敏感度为$S_i$，风险权重为$W_i$，因子间相关系数为$\rho_{ij}$。在10天、99%置信水平下，可近似认为组合损益为正态分布，其标准差$\sigma$由如下协方差模型给出：
<!-- bilingual-en:start -->
- **Core notation:** Let sensitivity to risk factor $i$ be $S_i$, its risk weight be $W_i$, and the correlation between factors be $\rho_{ij}$. Under the note's ten-day, 99% normal approximation, portfolio standard deviation $\sigma$ is obtained from the following covariance aggregation:
<!-- bilingual-en:end -->
    
    $$\sigma = \sqrt{\sum_{i=1}^n\sum_{j=1}^n (W_i S_i)(W_j S_j),\rho_{ij}}.$$
    
    初始保证金约为$\Phi^{-1}(0.99)\sigma\approx2.33,\sigma$。换言之，SIMM先计算加权敏感度$WS_i=W_iS_i$，然后按方差-协方差方式汇总各因子的风险敞口 。该模型同时考虑利率、信用、股权、商品、外汇等类别的风险因子，并通过指定的相关矩阵进行组合。SIMM的具体参数（风险权重、相关系数等）由监管或行业委员会公布（例如ICE衍生品数据的SIMM权重表），以确保一致性。
<!-- bilingual-en:start -->
Initial margin is stated as approximately $\Phi^{-1}(0.99)\sigma\approx2.33,\sigma$; the comma is a typographical error and the intended expression is $2.33\sigma$. SIMM first forms each weighted sensitivity $WS_i=W_iS_i$ and then aggregates factor exposures using a variance–covariance structure. It covers risk factors in interest-rate, credit, equity, commodity, and foreign-exchange classes and combines them through prescribed correlation matrices. Industry governance publishes and calibrates the risk weights and correlations to promote consistent implementation.
<!-- bilingual-en:end -->
    

  

## **9. CCP间互操作、再抵押机制与流动性风险**
<!-- bilingual-en:start -->
*9. CCP Interoperability, Rehypothecation, and Liquidity Risk*
<!-- bilingual-en:end -->

- **互操作（Interoperability）**：当两个或多个CCP之间建立互联机制，使同一交易的买卖双方可在不同CCP清算时，就称为互操作 。互操作通过跨CCP进行抵押品净额结算，提高了竞争并降低了总体保证金需求。例如，一个参与者在CCP1持有多头头寸、在CCP2持有等量空头头寸；若不互操作，需要分别缴存保证金；若互操作则可净额抵消，仅需缴存一笔保证金 。总体上，互操作可以减少参与会员数量、降低交易成本、减少每笔交易的保证金占用，并简化结算 。
<!-- bilingual-en:start -->
- **Interoperability:** Two or more CCPs are interoperable when they are linked so that the buyer and seller to a trade can clear through different CCPs. The course note presents interoperability as enabling cross-CCP collateral netting, increasing competition, and lowering aggregate margin. A participant long at CCP1 and equally short at CCP2 would otherwise post margin at both; with a qualifying interoperable arrangement, offset may reduce the requirement. Such links can reduce trading and collateral costs and simplify settlement, although the precise amount of cross-CCP netting depends on the link's legal and risk-management design.
<!-- bilingual-en:end -->
    
- **再抵押（Re-hypothecation）**：交易商收到的担保品可以再次用作对其他交易对手的担保。然而危机中抵押品被多次重复使用（平均达4次），增加了参与者的隔夜违约风险。BCBS/IOSCO新的UMR规定限制了再抵押：**初始保证金仅允许一次再抵押**，且需满足严格条件；而**变动保证金则不受限制** 。这意味着，IM在未来若被再用，须确保接收方明确同意不再次使用，而VM作为日常结算现金可连续使用。交易商通常也会自行对可接受的再抵押程度加以限制，以控制法律和流动性风险。
<!-- bilingual-en:start -->
- **Rehypothecation:** A dealer may reuse collateral received from one counterparty to secure an obligation to another. Repeated reuse—reported in the note as averaging four times during the crisis—can amplify overnight default and liquidity risk. The BCBS–IOSCO uncleared-margin framework tightly restricts IM reuse: a one-time rehypothecation, repledge, or reuse may be permitted only under strict conditions, while VM is not subject to the same segregation restriction. Any permitted IM reuse must prevent further reuse by the next recipient. Dealers also impose their own limits to manage legal and liquidity risks.
<!-- bilingual-en:end -->
    
- **流动性风险**：新规下市场参与者需要持有大量高流动性担保品来应对可能的保证金追加要求（尤其是IM大幅增长时）。每日盯市造成的变动保证金调用，以及极端行情下IM暴增，都会对机构的资金流动性产生压力 。为了兼顾资本效率，银行需要在自有资本和流动性之间权衡——过高的保证金要求可能导致资金被锁定在中央交易所或托管机构中，从而在市场波动时加剧流动性紧张 。
<!-- bilingual-en:start -->
- **Liquidity risk:** The new framework requires market participants to hold large quantities of liquid collateral for possible margin calls, especially when IM rises sharply. Daily VM calls and stressed increases in IM can strain funding liquidity. Banks therefore balance capital efficiency against liquidity resilience: excessive collateral requirements can lock assets at CCPs or custodians and intensify funding pressure during volatile markets.
<!-- bilingual-en:end -->
    

  

## **10. OTC与场内交易的融合趋势**
<!-- bilingual-en:start -->
*10. Convergence Between OTC and Exchange-Traded Markets*
<!-- bilingual-en:end -->

- **场外交易电子化与清算所化**：OTC衍生品正逐步向电子化和交易所化转变。越来越多的标准化OTC合约在电子交易平台（如Swap Execution Facilities）上撮合，并通过交易所监管的CCP进行清算 。金融机构间的双向托管结构使得传统双边清算架构更接近于交易所清算所的模式，标准化合约占比提升。
<!-- bilingual-en:start -->
- **Electronic execution and clearing-house features:** OTC derivatives have become more electronic and exchange-like. A growing share of standardised OTC contracts is matched on venues such as SEFs and cleared through regulated CCPs. Bilateral custody and margin arrangements have likewise become more formalised, while standardised products account for a larger share of activity.
<!-- bilingual-en:end -->
    
- **交易所提供场外产品**：同时，交易所也在扩大产品线，推出更多非标准化的衍生品以满足不同需求。许多市场已经出现了交易所与CCP业务整合的趋势，在保证金要求和清算架构上进行协同。总体来看，OTC与场内衍生品市场正在趋同：标准OTC交易越来越采用交易所模式清算，而交易所也引入了更多灵活合约。
<!-- bilingual-en:start -->
- **Exchanges offering flexible products:** Exchanges have also broadened their product ranges to accommodate more specialised derivatives. Many markets show closer integration between exchanges and CCPs in margin and clearing. The two market forms are therefore converging: standard OTC transactions increasingly use exchange-like execution and clearing, while exchanges offer a wider range of flexible contracts.
<!-- bilingual-en:end -->
    

  

## **11. CCP倒闭的系统性风险**
<!-- bilingual-en:start -->
*11. Systemic Risk from CCP Failure*
<!-- bilingual-en:end -->

- **风险转移**：将衍生品交易清算集中到CCP后，系统性风险从银行体系转向CCP体系。虽然CCP通过严格的风险管理和多层防线（保证金、保证基金等）来降低对手方违约风险，但如果某CCP发生极端事件（如成员大规模违约且偿付能力不足），潜在损失可能快速蔓延到金融系统。
<!-- bilingual-en:start -->
- **Risk transfer:** Central clearing moves a substantial concentration of derivatives risk from bilateral bank relationships to CCPs. Although CCPs use margin, default funds, and other defences to reduce counterparty risk, an extreme event—such as multiple member defaults that exhaust available resources—could transmit losses rapidly through the financial system.
<!-- bilingual-en:end -->
    
- **监管与可监管性**：CCP的组织结构相对单一（成员资格、交易定价、保证金机制） 。监管机构通过CCP的重组、恢复与处置框架（resolution）来应对可能的CCP危机。目前监管要求CCP制定详尽的恢复计划，包括增资、重置头寸等手段，以避免或减轻倒闭带来的冲击。防止CCP进行自营或经纪业务，以减少风险冲击。虽然将风险集中于CCP可能带来新的挑战，但集中清算也便于监管者监控和管理整体风险，有利于金融系统的整体稳定 。
<!-- bilingual-en:start -->
- **Supervision and resolvability:** A CCP has a comparatively focused structure built around membership, pricing, margin, and default management. Authorities address CCP crises through recovery and resolution frameworks. CCPs must maintain detailed recovery plans that can include replenishing resources, allocating losses, or restoring a matched book. Restrictions on proprietary or brokerage activities can limit additional risk. Centralisation creates new concentration challenges, but it also makes system-wide exposures easier for supervisors to monitor and manage.
<!-- bilingual-en:end -->
    

  

## **计算题**
<!-- bilingual-en:start -->
*Calculation Exercises*
<!-- bilingual-en:end -->

1. **题目：** 两名交易方A、B签订了一份场外衍生品合约，当日盯市显示该合约对A方的价值增加了$100,000（对B方价值减少$100,000）。在既定保证金协议下，应发生什么变动保证金操作？
<!-- bilingual-en:start -->

&nbsp;
**1.** **Question:** Parties A and B enter an OTC derivative. Today's mark-to-market shows that its value to A has risen by USD 100,000 and its value to B has fallen by USD 100,000. Under their margin agreement, what VM transfer should occur?<br>
<!-- bilingual-en:end -->
    
    **解答：** 对A方价值增加$100,000，意味着B方亏损$100,000。根据变动保证金原则，B方应向A方支付$100,000作为追加担保品 。也即B方将$100,000现金支付给A方，恢复双方合约价值平衡。
<!-- bilingual-en:start -->
**Solution:** A's USD 100,000 gain is B's USD 100,000 loss. Under variation-margin settlement, B transfers USD 100,000 of collateral—here, cash—to A, restoring the collateralised balance between the parties.
<!-- bilingual-en:end -->
    
2. **题目：** 某交易商与7个不同对手的未平仓衍生品头寸的市值分别为：+10M、+15M、–20M、+5M、–10M、+10M、+5M。计算（a）在双边清算情况下，该交易商面临的总信用风险敞口；（b）若通过一个中央对手方进行净额结算，合并后的总信用风险敞口。
<!-- bilingual-en:start -->

&nbsp;
**2.** **Question:** A dealer's open derivatives positions against seven different counterparties have market values of +10M, +15M, −20M, +5M, −10M, +10M, and +5M. Calculate (a) total credit exposure under bilateral clearing and (b) aggregate credit exposure if the positions are netted through one CCP.<br>
<!-- bilingual-en:end -->
    
    **解答：** (a) **双边清算**下，对每个对手分别计算曝险。信用风险敞口取所有正向头寸之和：$10+15+5+10+5 = 45$（单位M）。(b) **中心清算（净额结算）**下，所有头寸合并为一个净头寸：正向头寸和45，负向头寸和30，净敞口$45-30=15$（单位M）。即通过CCP抵消交易，整体信用风险仅15M ，显著低于双边场景下的45M。
<!-- bilingual-en:start -->
**Solution:** (a) Under **bilateral clearing**, exposure is calculated separately against each counterparty, so positive exposures sum to $10+15+5+10+5 = 45$ million. (b) Under **central clearing and [[OTC 衍生品清算、保证金与 CCP 风险|netting]]**, all positions are combined: positives total 45 and negatives total 30, leaving $45-30=15$ million. CCP netting therefore reduces the stated exposure from 45M to 15M.
<!-- bilingual-en:end -->
    
3. **题目：** 在ISDA协议中，一笔衍生品交易被标的市场中间价为$20M，市场买价$18M、卖价$22M。若交易的对手方违约，非违约方将这笔交易重设为新的头寸。请根据ISDA规定计算清算时使用的结算价格（non-default方视角）。如果中间价为$-20M（带负号），相应的结算价又是多少？
<!-- bilingual-en:start -->

&nbsp;
**3.** **Question:** Under an ISDA agreement, a derivative has a market mid-value of USD 20M, a bid of USD 18M, and an ask of USD 22M. If the counterparty defaults and the non-defaulting party replaces the trade, what settlement value does the course illustration use? What if the mid-value is −USD 20M, with corresponding prices of −USD 18M and −USD 22M?<br>
<!-- bilingual-en:end -->
    
    **解答：** 根据ISDA违约处理条款，非违约方终止后按对其有利的价格重置头寸。实际操作中取中间价并向不利违约方的方向调整一半买卖价差。例1：中间价$20M，买价18M，卖价22M，故结算价为$22M （非违约方按卖价重开）。例2：若中间价$-20M（表明方向相反），买价为$-18M、卖价为$-22M，则结算价为$-18M （非违约方按买价重开）。因此结算时非违约方分别收到$22M和付出$18M。
<!-- bilingual-en:start -->
**Solution:** The illustration adjusts mid by half the bid–ask spread toward the non-defaulting party's replacement cost. With a USD 20M mid, USD 18M bid, and USD 22M ask, it uses USD 22M because the position is replaced at the ask. With a −USD 20M mid and quotes of −USD 18M and −USD 22M, it uses −USD 18M. The non-defaulting party therefore receives USD 22M in the first case and pays USD 18M in the second.
<!-- bilingual-en:end -->
    
4. **题目：** 某未清算衍生品交易组合的初始保证金由SIMM模型计算而来。假设投资组合对某一风险因子的敏感度为$S=150$（单位与风险权重配合使用），该风险因子的监管风险权重为$W=0.004$，在受压市场下该风险因子的日波动率$\sigma=1.5%$。若采用10天期和99%置信度计算VaR（假设风险因子间独立），求该组合对该因子的初始保证金（近似）。
<!-- bilingual-en:start -->

&nbsp;
**4.** **Question:** SIMM is used to calculate IM for an uncleared derivatives portfolio. Sensitivity to one risk factor is $S=150$, its regulatory risk weight is $W=0.004$, and its daily stressed volatility is $\sigma=1.5%$. Using a ten-day, 99% VaR approximation and assuming independence, estimate IM attributable to this factor.<br>
<!-- bilingual-en:end -->
    
    **解答：** 首先计算加权敏感度：$WS = W \times S = 0.004 \times 150 = 0.6$。10天总风险波动率为$\sigma_{\text{10d}} = \sigma\sqrt{10} = 1.5%\times\sqrt{10} \approx 4.74%$。对应的标准差（敏感度*波动率）$\sigma_p = 0.6 \times 0.0474 = 0.02844$（单位与$S$相同）。在99%置信度下的VaR近似为$\Phi^{-1}(0.99)\sigma_p \approx 2.33\times0.02844\approx0.0663$。因此该风险因子的初始保证金约为0.0663（与$S$相同的货币单位）。该结果表明，为覆盖极端风险敞口，应缴纳约0.0663单位的抵押品 。
<!-- bilingual-en:start -->
**Solution:** First calculate weighted sensitivity: $WS = W \times S = 0.004 \times 150 = 0.6$. Ten-day volatility is $\sigma_{\text{10d}} = \sigma\sqrt{10} = 1.5%\times\sqrt{10} \approx 4.74%$. The position standard deviation is $\sigma_p = 0.6 \times 0.0474 = 0.02844$ in the same units as $S$. At 99% confidence, VaR is approximately $\Phi^{-1}(0.99)\sigma_p \approx 2.33\times0.02844\approx0.0663$. The estimated IM contribution is therefore about 0.0663 units.
<!-- bilingual-en:end -->
    
5. **题目：** 某期货交易所CCP违约水位设计如下：违约会员的初始保证金为50M，违约会员缴纳的保证基金为100M，其他会员的保证基金为200M，CCP自有资本为500M。一会员违约导致损失总额为250M，超过了该会员的初始保证金。计算该CCP如何分摊这250M损失，最后CCP是否需动用自有资本？
<!-- bilingual-en:start -->

&nbsp;
**5.** **Question:** A futures CCP's simplified waterfall contains 50M of the defaulter's IM, 100M of the defaulter's default-fund contribution, 200M of other members' default-fund contributions, and 500M of CCP capital. A member default causes a 250M loss. How is the loss allocated, and is CCP capital used?<br>
<!-- bilingual-en:end -->
    
    **解答：** 按照CCP损失吸收顺序 ：首先用违约会员的初始保证金50M承付，剩余损失$250-50=200$M；接着用违约会员的保证基金100M，剩余$200-100=100$M；然后用其他会员的保证基金来承担剩下的100M，剩余$100-100=0$M；此时损失已完全覆盖，无需动用CCP自有资本（因为自有资本只有在全部保证基金耗尽后才使用）。因此，250M损失依次由50M（违约方IM）、100M（违约方基金）、100M（其他会员基金）分摊，CCP自有资本未被动用。
<!-- bilingual-en:start -->
**Solution:** Apply the simplified sequence. The defaulter's 50M IM leaves $250-50=200$M. Its 100M default-fund contribution leaves $200-100=100$M. Other members' default fund absorbs the final 100M, leaving $100-100=0$M. The 250M loss is therefore allocated as 50M of defaulter IM, 100M of defaulter fund contribution, and 100M of mutualised fund resources; CCP capital is not used.
<!-- bilingual-en:end -->
    
6. **题目：** 某交易参与者在CCP1持有一份多头仓位，在CCP2持有等量的空头仓位。假设该多头仓位在CCP1需缴纳初始保证金$0.5M，空头仓位在CCP2也需$0.5M。若**无**互操作机制，则总保证金为$1.0M；若两家CCP实现互操作，使得该多头和空头可跨CCP抵消，则总保证金将是多少？请说明原因。
<!-- bilingual-en:start -->

&nbsp;
**6.** **Question:** A participant is long at CCP1 and equally short at CCP2. Each CCP requires USD 0.5M of IM. Without interoperability, total IM is USD 1.0M. If an interoperable arrangement recognises the offset, what total margin does the exercise state, and why?<br>
<!-- bilingual-en:end -->
    
    **解答：** 没有互操作时，每个CCP独立要求保证金，总计$0.5M+$0.5M=$1.0M。若实现互操作，则在两个CCP间将持仓视为一个净头寸（多空互抵消），参与者只需对净敞口缴纳保证金。由于多头和空头完全抵消，理论上净头寸为0，因此只需缴纳一个头寸的保证金，约为$0.5M 。即利用互操作机制后，总保证金需求从1.0M降至0.5M，从而节省了50%的抵押品占用。这反映出互操作通过跨CCP净额结算减少了总体保证金占用 。
<!-- bilingual-en:start -->
**Solution:** Without interoperability, the two CCPs independently require $0.5M+$0.5M=$1.0M. The note then treats the positions as one cross-CCP net exposure and states that only one USD 0.5M margin amount remains, reducing collateral use by 50%. There is an internal inconsistency in the source: if the long and short truly offset perfectly and no floor applies, the theoretical net exposure is zero, not USD 0.5M. The USD 0.5M result should therefore be read as an exercise convention or minimum requirement, not as a consequence of perfect netting alone.
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->
## 17.3
>[!question] 
为什么在 2007～2008 年金融危机后引入的监管规定会给某些金融机构带来流动性问题？  
<!-- bilingual-en:start -->
Why could regulations introduced after the 2007–08 financial crisis create liquidity problems for some financial institutions?
<!-- bilingual-en:end -->

金融机构需要为它们的衍生产品账户提供的担保品的数量（一般来说是现金或国债）会增加。
<!-- bilingual-en:start -->
Financial institutions must provide more collateral—normally cash or government securities—against their derivatives portfolios.
<!-- bilingual-en:end -->
## 17.4
>[!question] 
解释一下担保品协议中“折减”的含义。
<!-- bilingual-en:start -->
Explain what a “haircut” means in a collateral agreement.
<!-- bilingual-en:end -->

用于某金融产品的折减率是该产品用作担保品时所计的价值与其市场价格相比所减少的百分比。例如，如果某产品的价格为100 美元，折减率为10%，那么该产品用来满足 90 美元的担保品要求。
<!-- bilingual-en:start -->
A collateral haircut is the percentage by which an asset's market price is reduced when determining its collateral value. If an asset is worth USD 100 and has a 10% haircut, it satisfies only USD 90 of a collateral requirement.
<!-- bilingual-en:end -->
## 17.5
>[!question] 
解释一下 ISDA 主协议中“违约时间”和“提前终止”的含义。 
<!-- bilingual-en:start -->
Explain “event of default” and “early termination” under an ISDA Master Agreement.
<!-- bilingual-en:end -->

如果在衍生产品交易中，签署了 ISDA 主协议的双方中有一方不能缴纳担保品或按照要求付款，那么我们就认为发生了违约。违约发生后，会接着发生提前终止事件。未违约的一方会终止与违约的一方进行的所有未到期交易。
<!-- bilingual-en:start -->
A default occurs when one party to derivatives governed by an ISDA Master Agreement fails to post collateral or make a required payment, subject to the agreement's terms. An early-termination event then follows: the non-defaulting party terminates all outstanding transactions with the defaulting party and determines one net close-out amount.
<!-- bilingual-en:end -->
## 17.9
>[!question] 
什么是再抵押？
<!-- bilingual-en:start -->
What is rehypothecation?
<!-- bilingual-en:end -->

再抵押的含义是，A 方给B方提供的担保品，被B方用来满足C方对B方的担保品要求。
<!-- bilingual-en:start -->
Rehypothecation occurs when Party B reuses collateral received from Party A to satisfy B's own collateral obligation to Party C.
<!-- bilingual-en:end -->
